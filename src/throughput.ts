import { deriveArchitecture } from "./architecture.js";
import { computeKvCacheBytes, computeWeightBytes } from "./memory.js";
import type { DeviceFamily, DeviceSpec, ModelSpec, QuantSpec } from "./types.js";
import { GB_DECIMAL, TFLOP } from "./units.js";

/**
 * Rule 5: throughput.
 *
 * Two regimes, and conflating them is why forum numbers never reproduce.
 *
 *   DECODE  (one token at a time) is memory-bandwidth bound. Every weight the
 *           token routes through is read from memory and multiplied against a
 *           single vector, so arithmetic intensity is ~1 FLOP per byte and the
 *           GPU's TFLOPs are irrelevant. Speed is bytes-read per token divided
 *           into bandwidth.
 *
 *   PREFILL (the whole prompt at once) is compute bound. The same weights are
 *           reused across hundreds of tokens, so the matmuls saturate the
 *           tensor cores and TFLOPs is the limit.
 *
 * That is why a 4090 decodes an 8B model roughly 30x slower than it prefills
 * it, and why a Mac Studio with a tenth of the FLOPs of an H100 decodes at a
 * respectable fraction of its speed.
 *
 * EVERYTHING IN THIS FILE IS AN ESTIMATE. The formulas are exact; the
 * efficiency factors they are multiplied by are empirical, fitted against
 * published llama.cpp benchmark tables, and carry the error bands declared at
 * the bottom of this file. Treat the output as "which order of magnitude, and
 * will changing the quant help", not as a benchmark result.
 */

/* -------------------------------------------------------------------------- */
/* Efficiency factors                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Memory bandwidth utilisation (MBU) at 16-bit weights: the share of a
 * device's peak theoretical bandwidth a decode kernel actually achieves.
 *
 * These are the figures 16-bit decode lands on in llama.cpp's own benchmark
 * tables, where the kernel is doing nothing but streaming weights:
 *
 *   M2 Ultra, LLaMA 7B F16:  42.75 tok/s x 13.2 GB = 564 GB/s of 800  -> 0.71
 *   RTX 4090, LLaMA 7B F16:  ~55   tok/s x 13.2 GB = 726 GB/s of 1008 -> 0.72
 *
 * Datacenter parts with HBM and CUDA-graph-driven runtimes (TensorRT-LLM,
 * vLLM) reach higher; Databricks' MBU write-up reports up to 0.85 there.
 * ROCm's decode kernels are behind CUDA's on the same class of memory.
 */
export const MEMORY_BANDWIDTH_EFFICIENCY: Readonly<Record<DeviceFamily, number>> = {
  "cuda-consumer": 0.75,
  "cuda-datacenter": 0.85,
  rocm: 0.72,
  metal: 0.7,
  cpu: 0.7,
};

/**
 * The correction that the naive bandwidth model misses: quantized weights have
 * to be unpacked before they can be multiplied, and at 4 bits that unpacking
 * is enough work to stop the kernel saturating memory. The same M2 Ultra that
 * hits 0.71 MBU on F16 manages only 0.42 on Q4_0 -- and the gap is a property
 * of the backend's dequantization kernels, not of the memory system, so it is
 * tabulated per family:
 *
 *   M2 Ultra, LLaMA 7B Q4_0: 88.64 tok/s x 3.79 GB = 336 GB/s of 800  -> 0.42
 *   RTX 4090, LLaMA 7B Q4_0: 152.2 tok/s x 3.79 GB = 577 GB/s of 1008 -> 0.57
 *
 * The value below is the multiplier at the 4-bit end; it is interpolated
 * linearly in bits-per-weight up to 1.0 at 16 bits. Reproducing both of the
 * measurements above from one formula is what this table buys.
 */
export const DEQUANT_EFFICIENCY: Readonly<Record<DeviceFamily, number>> = {
  "cuda-consumer": 0.78,
  "cuda-datacenter": 0.85,
  rocm: 0.7,
  metal: 0.62,
  cpu: 0.72,
};

/**
 * Model FLOPs utilisation during prefill. Well below 1.0 because attention,
 * normalisation, RoPE and the dequantization of the weights back to FP16 all
 * cost time that produces no matmul FLOPs, and because llama.cpp's prefill
 * kernels are not TensorRT.
 *
 * Fitted so that prefill of an 8B Q4_K_M comes out near the published
 * `pp512` figures: ~3.7k tok/s on a 4090, ~1.0k tok/s on an M2 Ultra.
 */
export const PREFILL_MFU: Readonly<Record<DeviceFamily, number>> = {
  "cuda-consumer": 0.35,
  "cuda-datacenter": 0.45,
  rocm: 0.22,
  metal: 0.3,
  cpu: 0.4,
};

/** Bits per weight at which `DEQUANT_EFFICIENCY` applies in full. */
const DEQUANT_REFERENCE_BITS = 4;
/** Bits per weight at which there is no dequantization cost at all. */
const NATIVE_BITS = 16;

/** Decode estimates are good to about this fraction, either way. */
export const DECODE_ERROR_BAND = 0.25;
/** Prefill is harder: kernel quality varies far more between backends. */
export const PREFILL_ERROR_BAND = 0.4;

/**
 * Effective share of peak bandwidth for a given backend and weight width.
 * 1.0 would mean a kernel that reads memory at the number on the box.
 */
export function bandwidthEfficiency(family: DeviceFamily, bitsPerWeight: number): number {
  const base = MEMORY_BANDWIDTH_EFFICIENCY[family];
  const atFourBits = DEQUANT_EFFICIENCY[family];
  const span = NATIVE_BITS - DEQUANT_REFERENCE_BITS;
  const narrowness = Math.min(Math.max((NATIVE_BITS - bitsPerWeight) / span, 0), 1);
  return base * (1 - (1 - atFourBits) * narrowness);
}

/* -------------------------------------------------------------------------- */
/* Decode                                                                      */
/* -------------------------------------------------------------------------- */

export interface DecodeEstimate {
  /** Tokens per second for one sequence. This is the number people quote. */
  tokensPerSecond: number;
  /** Tokens per second summed over `batch` concurrent sequences. */
  aggregateTokensPerSecond: number;
  /** Weight bytes read per decoding step (shared by the whole batch). */
  weightBytesPerStep: number;
  /** KV cache bytes read per decoding step, across every sequence. */
  kvBytesPerStep: number;
  /** Total bytes that must cross the memory bus per step. */
  bytesPerStep: number;
  /** Peak bandwidth multiplied by the efficiency below, in bytes/second. */
  effectiveBandwidthBytesPerSecond: number;
  efficiency: number;
}

export interface DecodeInput {
  /** Weight bytes read per token: the active share for a mixture of experts. */
  weightBytesPerStep: number;
  /** KV cache bytes read per step, across all sequences. */
  kvBytesPerStep: number;
  /** Peak bandwidth in bytes/second, before efficiency. */
  peakBandwidthBytesPerSecond: number;
  /** Fraction of peak actually achieved. */
  efficiency: number;
  /** Concurrent sequences sharing one weight read. */
  batch: number;
}

/**
 * Decode from first principles:
 *
 *     seconds_per_step = (active_weight_bytes + kv_bytes) / (bandwidth * eff)
 *     tokens_per_second = batch / seconds_per_step
 *
 * The KV term is the one usually left out, and it is why tok/s falls off as a
 * conversation grows: attention reads the entire cache for every token it
 * emits. A Llama 3.1 8B at Q4_K_M reads 4.6 GB of weights per token; by 32k
 * context it is also reading 4 GB of KV cache, so the rate roughly halves.
 *
 * Weights are read once per step no matter how many sequences are in flight,
 * which is the whole reason batching raises aggregate throughput; each
 * sequence's own cache still has to be read, so the gain is sub-linear.
 */
export function estimateDecodeFrom(input: DecodeInput): DecodeEstimate {
  const batch = Math.max(1, Math.floor(input.batch));
  const effectiveBandwidthBytesPerSecond =
    input.peakBandwidthBytesPerSecond * input.efficiency;
  const bytesPerStep = input.weightBytesPerStep + input.kvBytesPerStep;

  const stepsPerSecond =
    bytesPerStep > 0 && effectiveBandwidthBytesPerSecond > 0
      ? effectiveBandwidthBytesPerSecond / bytesPerStep
      : 0;

  return {
    tokensPerSecond: stepsPerSecond,
    aggregateTokensPerSecond: stepsPerSecond * batch,
    weightBytesPerStep: input.weightBytesPerStep,
    kvBytesPerStep: input.kvBytesPerStep,
    bytesPerStep,
    effectiveBandwidthBytesPerSecond,
    efficiency: input.efficiency,
  };
}

/* -------------------------------------------------------------------------- */
/* Prefill                                                                     */
/* -------------------------------------------------------------------------- */

export interface PrefillEstimate {
  /** Prompt tokens processed per second. */
  tokensPerSecond: number;
  /** FLOPs to process one prompt token at the requested context. */
  flopsPerToken: number;
  /** Share coming from attention rather than from the weight matmuls. */
  attentionFlopsPerToken: number;
  effectiveFlopsPerSecond: number;
  efficiency: number;
}

/** Width of the per-layer attention that the score matmuls actually run at. */
function attentionWidth(model: ModelSpec): number {
  if (model.attention === "mla" && model.mla) {
    // MLA reconstructs full-width heads before attending, so the score matmul
    // is sized by the decompressed head width, not by the cached latent.
    return model.nHeads * (model.mla.qkNopeHeadDim + model.mla.qkRopeHeadDim);
  }
  return model.nHeads * model.headDim;
}

/**
 * Tokens each layer attends over, summed across layers. Sliding-window layers
 * only ever see `windowSize` of them, which is most of why Gemma 3 prefills
 * long prompts far faster than its parameter count suggests.
 */
function attendedTokensAcrossLayers(model: ModelSpec, ctx: number): number {
  const window = model.attentionWindow;
  let total = 0;
  for (let layer = 0; layer < model.nLayers; layer++) {
    const isFull = !window || (layer + 1) % window.fullAttentionEvery === 0;
    total += isFull ? ctx : Math.min(ctx, window.windowSize);
  }
  return total;
}

/**
 * Prefill FLOPs per prompt token.
 *
 *   matmuls:   2 * active_params      (one multiply and one add per weight)
 *   attention: 2 * 2 * attended * width per layer, halved for causal masking
 *
 * `active_params` excludes the token embedding table -- looking a row up is
 * not a matmul -- but includes the output projection, and for a mixture of
 * experts counts only the experts a token is routed to.
 *
 * The causal factor of 1/2 is the average over prompt positions: the first
 * token attends to nothing, the last to the whole prompt.
 */
export function prefillFlopsPerToken(model: ModelSpec, ctx: number): number {
  const arch = deriveArchitecture(model);
  const matmulFlops = 2 * arch.activeMatmulParams;
  // QK^T and (scores x V): two matmuls, 2 FLOPs each per element.
  const attentionFlops = 2 * attendedTokensAcrossLayers(model, ctx) * attentionWidth(model);
  return matmulFlops + attentionFlops;
}

export interface PrefillInput {
  model: ModelSpec;
  ctx: number;
  /** Peak dense FP16 throughput in FLOP/s. */
  peakFlopsPerSecond: number;
  /** Model FLOPs utilisation, 0-1. */
  efficiency: number;
}

export function estimatePrefillFrom(input: PrefillInput): PrefillEstimate {
  const ctx = Math.max(1, Math.floor(input.ctx));
  const arch = deriveArchitecture(input.model);
  const flopsPerToken = prefillFlopsPerToken(input.model, ctx);
  const effectiveFlopsPerSecond = input.peakFlopsPerSecond * input.efficiency;

  return {
    tokensPerSecond:
      flopsPerToken > 0 ? effectiveFlopsPerSecond / flopsPerToken : 0,
    flopsPerToken,
    attentionFlopsPerToken: flopsPerToken - 2 * arch.activeMatmulParams,
    effectiveFlopsPerSecond,
    efficiency: input.efficiency,
  };
}

/* -------------------------------------------------------------------------- */
/* Composition                                                                 */
/* -------------------------------------------------------------------------- */

export interface ThroughputOptions {
  ctx: number;
  batch?: number;
  kvQuant?: string;
  /** Prompt length for time-to-first-token. Defaults to the full context. */
  promptTokens?: number;
  /**
   * Peak read bandwidth in bytes/second, overriding the device's own. Partial
   * offload passes the blended VRAM/system-RAM figure through here.
   */
  peakBandwidthBytesPerSecond?: number;
  /** Override the derived memory-bandwidth efficiency (0-1). */
  efficiency?: number;
  /** Override the derived prefill MFU (0-1). */
  prefillEfficiency?: number;
}

export interface ThroughputEstimate {
  decode: DecodeEstimate;
  prefill: PrefillEstimate;
  /** Seconds to process `promptTokens` before the first token appears. */
  timeToFirstTokenSeconds: number;
  decodeErrorBand: number;
  prefillErrorBand: number;
}

/**
 * Full throughput estimate for a model that fits entirely in device memory.
 *
 * Multi-GPU is deliberately absent here. llama.cpp's default `--split-mode
 * layer` gives each card a contiguous slice of the model and runs them in
 * sequence, so the same total bytes still cross a bus per token and decode
 * speed is unchanged -- adding cards buys capacity, not speed. Tensor-parallel
 * runtimes (vLLM, TensorRT-LLM) do scale decode, roughly 0.7-0.9x per added
 * device once interconnect sync is paid for, but that is a different
 * deployment and is not what this number describes.
 */
export function estimateThroughput(
  model: ModelSpec,
  quant: QuantSpec,
  device: DeviceSpec,
  options: ThroughputOptions,
): ThroughputEstimate {
  const ctx = Math.max(1, Math.floor(options.ctx));
  const batch = Math.max(1, Math.floor(options.batch ?? 1));
  const weights = computeWeightBytes(model, quant);
  const kv = computeKvCacheBytes(model, {
    ctx,
    batch,
    ...(options.kvQuant === undefined ? {} : { kvQuant: options.kvQuant }),
  });

  const peakBandwidth =
    options.peakBandwidthBytesPerSecond ?? device.bandwidthGBs * GB_DECIMAL;
  const efficiency =
    options.efficiency ?? bandwidthEfficiency(device.family, weights.effectiveBitsPerWeight);
  const prefillEff = options.prefillEfficiency ?? PREFILL_MFU[device.family];

  const decode = estimateDecodeFrom({
    weightBytesPerStep: weights.activeBytes,
    kvBytesPerStep: kv.totalBytes,
    peakBandwidthBytesPerSecond: peakBandwidth,
    efficiency,
    batch,
  });

  const prefill = estimatePrefillFrom({
    model,
    ctx,
    peakFlopsPerSecond: device.fp16Tflops * TFLOP,
    efficiency: prefillEff,
  });

  const promptTokens = Math.max(1, Math.floor(options.promptTokens ?? ctx));

  return {
    decode,
    prefill,
    timeToFirstTokenSeconds:
      prefill.tokensPerSecond > 0 ? promptTokens / prefill.tokensPerSecond : Number.POSITIVE_INFINITY,
    decodeErrorBand: DECODE_ERROR_BAND,
    prefillErrorBand: PREFILL_ERROR_BAND,
  };
}
