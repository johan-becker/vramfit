import { computeFootprint, type MemoryFootprint } from "./memory.js";
import {
  blendBandwidth,
  blendCompute,
  planOffload,
  type BlendedBandwidth,
  type BlendedCompute,
  type OffloadPlan,
} from "./offload.js";
import { GGUF_QUANT_FAMILIES, getKvQuant, quantsByQuality } from "./quant.js";
import {
  PREFILL_MFU,
  bandwidthEfficiency,
  estimateThroughput,
  type ThroughputEstimate,
} from "./throughput.js";
import type { DeviceSpec, KvQuantSpec, ModelSpec, QuantSpec } from "./types.js";
import { GB_DECIMAL, GIB, TFLOP } from "./units.js";

/**
 * The question the whole package exists to answer, assembled from the parts:
 * does this configuration fit, how much room is left, what is the largest
 * context that would fit, which quantization should you actually use, and how
 * fast will it go.
 *
 * Rule 7, multi-GPU, lives here too, because it is a capacity question. K
 * identical devices holding a layer-split model share the weights and cache
 * between them but each pay the runtime context and compute buffer in full,
 * so eight cards do not give you eight times the usable memory -- they give
 * you eight times the memory minus eight runtime contexts, which is 5.6 GiB
 * gone on a CUDA setup before a single weight is placed.
 */

/** Assumed system RAM when the caller does not say. Overridable everywhere. */
export const DEFAULT_SYSTEM_RAM_GIB = 32;
/** Dual-channel DDR5-5600: 2 x 64 bit x 5600 MT/s = 89.6 GB/s. */
export const DEFAULT_SYSTEM_RAM_BANDWIDTH_GBS = 89.6;
/** A modern desktop CPU's dense FP16 throughput, for offloaded prefill. */
export const DEFAULT_SYSTEM_COMPUTE_TFLOPS = 1.5;

export interface FitOptions {
  /** Defaults to the model's own default context. */
  ctx?: number;
  /** Concurrent sequences. */
  batch?: number;
  /** llama.cpp `--cache-type-k/-v`. Defaults to f16. */
  kvQuant?: string;
  /** Identical devices sharing the model. Defaults to 1. */
  gpus?: number;
  /** llama.cpp `--ubatch-size`, which sets the compute buffer width. */
  physicalBatch?: number;
  flashAttention?: boolean;
  /** Per-device memory override in GiB, for a variant the database lacks. */
  vramGiB?: number;
  /** System RAM available for offloaded layers, GiB. */
  systemRamGiB?: number;
  /** System RAM bandwidth, decimal GB/s. */
  systemRamBandwidthGBs?: number;
  /** CPU dense FP16 throughput in TFLOP/s, for prefill of offloaded layers. */
  systemComputeTflops?: number;
  /** Prompt length for time-to-first-token. Defaults to the context. */
  promptTokens?: number;
  /** Override the derived memory-bandwidth efficiency. */
  efficiency?: number;
  /** Override the derived prefill MFU. */
  prefillEfficiency?: number;
}

export interface Capacity {
  gpus: number;
  /** Allocatable bytes on one device, after `usableFraction`. */
  perDeviceBytes: number;
  /** Allocatable bytes across every device. */
  totalBytes: number;
  /** Installed bytes across every device, before `usableFraction`. */
  installedBytes: number;
}

export interface OffloadResult {
  plan: OffloadPlan;
  bandwidth: BlendedBandwidth;
  compute: BlendedCompute;
  systemRamRequiredBytes: number;
  systemRamAvailableBytes: number;
  /** False when even system RAM cannot hold the remainder. */
  feasible: boolean;
}

export interface FitResult {
  model: ModelSpec;
  device: DeviceSpec;
  quant: QuantSpec;
  kvQuant: KvQuantSpec;
  ctx: number;
  batch: number;
  gpus: number;
  footprint: MemoryFootprint;
  capacity: Capacity;
  usedBytes: number;
  /** Free bytes when it fits; negative by the shortfall when it does not. */
  headroomBytes: number;
  /** usedBytes / capacity.totalBytes. Above 1 when it does not fit. */
  utilization: number;
  fits: boolean;
  /**
   * Throughput of the deployment as it would actually run: at device speed
   * when it fits, at the blended speed of the offload split when it does not.
   */
  throughput: ThroughputEstimate;
  /** The layer split. Null when the whole model is resident. */
  offload: OffloadResult | null;
  /** Largest context that fits, at this quant and batch. 0 if none does. */
  maxContext: number;
  warnings: string[];
}

function usableBytesPerDevice(device: DeviceSpec, vramGiB?: number): number {
  return (vramGiB ?? device.vramGiB) * GIB * device.usableFraction;
}

function normalizeGpus(gpus: number | undefined): number {
  return Math.max(1, Math.floor(gpus ?? 1));
}

/** Memory capacity of a homogeneous multi-device setup. */
export function computeCapacity(device: DeviceSpec, options: FitOptions = {}): Capacity {
  const gpus = normalizeGpus(options.gpus);
  const perDeviceBytes = usableBytesPerDevice(device, options.vramGiB);
  return {
    gpus,
    perDeviceBytes,
    totalBytes: perDeviceBytes * gpus,
    installedBytes: (options.vramGiB ?? device.vramGiB) * GIB * gpus,
  };
}

function footprintAt(
  model: ModelSpec,
  quant: QuantSpec,
  device: DeviceSpec,
  options: FitOptions,
  ctx: number,
): MemoryFootprint {
  return computeFootprint(model, quant, {
    family: device.family,
    ctx,
    batch: Math.max(1, Math.floor(options.batch ?? 1)),
    deviceCount: normalizeGpus(options.gpus),
    ...(options.kvQuant === undefined ? {} : { kvQuant: options.kvQuant }),
    ...(options.physicalBatch === undefined ? {} : { physicalBatch: options.physicalBatch }),
    ...(options.flashAttention === undefined
      ? {}
      : { flashAttention: options.flashAttention }),
  });
}

/**
 * Largest context that fits, by bisection.
 *
 * Every term of the footprint is non-decreasing in context -- the KV cache
 * linearly, the attention score buffer linearly when flash attention is off,
 * everything else flat -- so the predicate is monotone and bisection is exact
 * rather than approximate. Returns 0 when even a single token does not fit,
 * which is the honest answer for a model that is simply too large.
 */
export function maxContextFor(
  model: ModelSpec,
  quant: QuantSpec,
  device: DeviceSpec,
  options: FitOptions = {},
): number {
  const capacity = computeCapacity(device, options).totalBytes;
  const fitsAt = (ctx: number): boolean =>
    footprintAt(model, quant, device, options, ctx).totalBytes <= capacity;

  if (!fitsAt(1)) return 0;
  if (fitsAt(model.maxCtx)) return model.maxCtx;

  let low = 1;
  let high = model.maxCtx;
  while (high - low > 1) {
    const mid = Math.floor((low + high) / 2);
    if (fitsAt(mid)) low = mid;
    else high = mid;
  }
  return low;
}

function collectWarnings(
  model: ModelSpec,
  quant: QuantSpec,
  device: DeviceSpec,
  ctx: number,
  gpus: number,
  offload: OffloadResult | null,
): string[] {
  const warnings: string[] = [];

  if (ctx > model.maxCtx) {
    warnings.push(
      `Requested context ${ctx} exceeds ${model.name}'s trained maximum of ${model.maxCtx}; quality degrades beyond it even where memory allows.`,
    );
  }
  if (device.unifiedMemory && device.family === "metal") {
    warnings.push(
      `${device.name} shares memory with the OS: only ${Math.round(device.usableFraction * 100)}% is wirable for the GPU by default. Raise it with "sudo sysctl iogpu.wired_limit_mb=N".`,
    );
  }
  if (gpus > 1) {
    warnings.push(
      `Each of the ${gpus} devices pays the runtime context and compute buffer in full, and a layer split does not raise decode speed -- extra devices buy capacity, not tokens per second.`,
    );
  }
  if (device.family === "rocm") {
    warnings.push(
      "ROCm decode kernels trail CUDA on comparable memory; treat the throughput estimate as optimistic.",
    );
  }
  if ((quant.family === "awq" || quant.family === "gptq") && device.family !== "cuda-consumer" && device.family !== "cuda-datacenter") {
    warnings.push(
      `${quant.label} kernels are CUDA-only in practice; on ${device.name} you would want a GGUF quantization instead.`,
    );
  }
  if (offload && model.moe) {
    warnings.push(
      "Offloading a mixture of experts is unusually painful: a token can route to an expert in system RAM on any layer, so the slow path is hit on nearly every token.",
    );
  }
  if (offload && !offload.feasible) {
    warnings.push(
      "Even with every offloadable layer in system RAM the model does not fit; more RAM or a narrower quantization is required.",
    );
  }
  return warnings;
}

/** Does this configuration fit, how much room is left, and how fast is it? */
export function checkFit(
  model: ModelSpec,
  quant: QuantSpec,
  device: DeviceSpec,
  options: FitOptions = {},
): FitResult {
  const ctx = Math.max(1, Math.floor(options.ctx ?? model.defaultCtx));
  const batch = Math.max(1, Math.floor(options.batch ?? 1));
  const gpus = normalizeGpus(options.gpus);
  const kvQuant = getKvQuant(options.kvQuant ?? "f16");

  const footprint = footprintAt(model, quant, device, options, ctx);
  const capacity = computeCapacity(device, options);
  const fits = footprint.totalBytes <= capacity.totalBytes;

  const throughputOptions = {
    ctx,
    batch,
    ...(options.kvQuant === undefined ? {} : { kvQuant: options.kvQuant }),
    ...(options.promptTokens === undefined ? {} : { promptTokens: options.promptTokens }),
    ...(options.efficiency === undefined ? {} : { efficiency: options.efficiency }),
    ...(options.prefillEfficiency === undefined
      ? {}
      : { prefillEfficiency: options.prefillEfficiency }),
  };

  let offload: OffloadResult | null = null;
  let throughput: ThroughputEstimate;

  if (fits) {
    throughput = estimateThroughput(model, quant, device, throughputOptions);
  } else {
    // The fixed overheads have to be on the device before any layer is placed.
    const budget =
      capacity.totalBytes - footprint.runtimeContextBytes - footprint.activationBytes;
    const plan = planOffload(model, footprint.weights, footprint.kv, budget);

    // On a unified-memory machine the "system RAM" a layer spills to is the
    // same physical memory the GPU was reading, so bandwidth is unchanged and
    // the penalty is the CPU's weaker execution, which the cpu efficiency
    // figure stands in for.
    const hostPeak =
      (options.systemRamBandwidthGBs ??
        (device.unifiedMemory ? device.bandwidthGBs : DEFAULT_SYSTEM_RAM_BANDWIDTH_GBS)) *
      GB_DECIMAL;
    const bits = footprint.weights.effectiveBitsPerWeight;
    const bandwidth = blendBandwidth(plan, {
      devicePeakBytesPerSecond: device.bandwidthGBs * GB_DECIMAL,
      deviceEfficiency: options.efficiency ?? bandwidthEfficiency(device.family, bits),
      hostPeakBytesPerSecond: hostPeak,
      hostEfficiency: options.efficiency ?? bandwidthEfficiency("cpu", bits),
    });

    const compute = blendCompute(plan, {
      devicePeakFlopsPerSecond: device.fp16Tflops * TFLOP,
      deviceEfficiency: options.prefillEfficiency ?? PREFILL_MFU[device.family],
      hostPeakFlopsPerSecond:
        (options.systemComputeTflops ?? DEFAULT_SYSTEM_COMPUTE_TFLOPS) * TFLOP,
      hostEfficiency: options.prefillEfficiency ?? PREFILL_MFU.cpu,
    });

    const systemRamAvailableBytes = (options.systemRamGiB ?? DEFAULT_SYSTEM_RAM_GIB) * GIB;
    offload = {
      plan,
      bandwidth,
      compute,
      systemRamRequiredBytes: plan.hostResidentBytes,
      systemRamAvailableBytes,
      feasible: plan.hostResidentBytes <= systemRamAvailableBytes,
    };

    throughput = estimateThroughput(model, quant, device, {
      ...throughputOptions,
      peakBandwidthBytesPerSecond: bandwidth.peakBytesPerSecond,
      efficiency: bandwidth.efficiency,
      peakFlopsPerSecond: compute.peakFlopsPerSecond,
      prefillEfficiency: compute.efficiency,
    });
  }

  return {
    model,
    device,
    quant,
    kvQuant,
    ctx,
    batch,
    gpus,
    footprint,
    capacity,
    usedBytes: footprint.totalBytes,
    headroomBytes: capacity.totalBytes - footprint.totalBytes,
    utilization: capacity.totalBytes > 0 ? footprint.totalBytes / capacity.totalBytes : Number.POSITIVE_INFINITY,
    fits,
    throughput,
    offload,
    maxContext: maxContextFor(model, quant, device, options),
    warnings: collectWarnings(model, quant, device, ctx, gpus, offload),
  };
}

export interface QuantOption {
  quant: QuantSpec;
  fit: FitResult;
  /** Largest context this quantization leaves room for. */
  maxContext: number;
  /** Decode speed at the requested context. */
  decodeTokensPerSecond: number;
}

export interface QuantSearchOptions extends FitOptions {
  /**
   * Restrict candidates to one ecosystem. Defaults to the GGUF families,
   * because mixing them is not a choice a user gets to make at runtime: an
   * AWQ checkpoint will not load in llama.cpp and a GGUF will not load in vLLM.
   */
  families?: ReadonlySet<string>;
}

/** Every candidate quantization, best quality first, each fully evaluated. */
export function evaluateQuants(
  model: ModelSpec,
  device: DeviceSpec,
  options: QuantSearchOptions = {},
): QuantOption[] {
  const families = options.families ?? GGUF_QUANT_FAMILIES;
  return quantsByQuality(families).map((quant) => {
    const fit = checkFit(model, quant, device, options);
    return {
      quant,
      fit,
      maxContext: fit.maxContext,
      decodeTokensPerSecond: fit.throughput.decode.tokensPerSecond,
    };
  });
}

/**
 * The best quantization that still fits: highest quality rank, not smallest
 * file. Undefined when nothing in the family fits, which is the signal to
 * offload or to pick a smaller model.
 */
export function recommendQuant(
  model: ModelSpec,
  device: DeviceSpec,
  options: QuantSearchOptions = {},
): QuantOption | undefined {
  return evaluateQuants(model, device, options).find((option) => option.fit.fits);
}
