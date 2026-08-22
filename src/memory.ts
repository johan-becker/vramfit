import { deriveArchitecture } from "./architecture.js";
import { getKvQuant, quantizedBytes } from "./quant.js";
import type { DeviceFamily, ModelSpec, QuantSpec } from "./types.js";
import { GIB, MIB } from "./units.js";

/* -------------------------------------------------------------------------- */
/* 1. Weights                                                                  */
/* -------------------------------------------------------------------------- */

export interface WeightBreakdown {
  /** Everything the checkpoint occupies. */
  totalBytes: number;
  /** Repeating transformer blocks, at the quant's bits-per-weight. */
  blockBytes: number;
  /** token_embd + output head. Sized with their own, higher, bit rates. */
  embeddingBytes: number;
  /** The token_embd table alone. Zero when embeddings are tied. */
  tokenEmbedBytes: number;
  /** The matrix that produces logits. Read on every decoded token, unlike
   *  the embedding table, so partial offload places it separately. */
  logitMatrixBytes: number;
  /** Bytes read from memory to decode one token: the active share of the
   *  blocks plus the logit matrix. Equals blockBytes + logit matrix for a
   *  dense model; far less for a mixture of experts. */
  activeBytes: number;
  /** totalBytes * 8 / totalParams -- what the file "really" costs per weight. */
  effectiveBitsPerWeight: number;
  /** Block bytes divided by layer count; used by the partial-offload split. */
  bytesPerLayer: number;
}

/**
 * Rule 1: bytes = params * bits_per_weight / 8.
 *
 * With one refinement that matters a great deal for modern, large-vocabulary
 * models: the vocabulary tensors are not stored at the body's width.
 *
 *   - `token_embd` keeps the base rate under GGUF k-quants, but stays FP16
 *     under GPTQ/AWQ/MXFP4, which only quantize linear layers.
 *   - the logit matrix (`output.weight`, or `token_embd` again when the model
 *     ties them) is promoted to Q6_K by every llama.cpp k-quant mix.
 *
 * Worked example -- Llama 3.1 8B at Q4_K_M, 128256 x 4096 untied embeddings:
 *
 *   blocks  (8.03e9 - 2 * 525.3e6) * 4.83 / 8 = 4.214 GB
 *   embed             525.3e6      * 4.83 / 8 = 0.317 GB
 *   output            525.3e6      * 6.56 / 8 = 0.431 GB
 *                                              --------
 *                                               4.962 GB
 *
 * The real Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf is 4.92 GB: 0.9% out.
 * Treating all 8.03e9 parameters as 4.83 bpw would give 4.848 GB, and
 * treating them as a nominal "4 bits" would give 4.015 GB -- 18% low.
 *
 * The published parameter count is authoritative for the total; only the
 * embedding/body *split* is derived from the architecture, so a rounding
 * difference in the derived count can never move the headline size.
 */
export function computeWeightBytes(model: ModelSpec, quant: QuantSpec): WeightBreakdown {
  const arch = deriveArchitecture(model);

  const vocabParams = arch.inputEmbedParams + arch.outputHeadParams;
  const blockParams = Math.max(model.totalParams - vocabParams, 0);

  const blockBytes = quantizedBytes(blockParams, quant.bitsPerWeight);
  // When embeddings are tied there is only one matrix on disk, and it doubles
  // as the output head, so it is stored at the promoted rate.
  const tokenEmbedBytes = model.tiedEmbeddings
    ? 0
    : quantizedBytes(arch.inputEmbedParams, quant.tokenEmbedBits);
  const logitMatrixBytes = quantizedBytes(arch.logitMatrixParams, quant.outputHeadBits);

  const totalBytes = blockBytes + tokenEmbedBytes + logitMatrixBytes;

  // Reconcile the derived block count with the published one before scaling to
  // the active (top-k expert) share, so MoE ratios stay exact even when the
  // published total is rounded to three significant figures.
  const activeShare = arch.blockParams > 0 ? arch.activeBlockParams / arch.blockParams : 1;

  return {
    totalBytes,
    blockBytes,
    embeddingBytes: tokenEmbedBytes + logitMatrixBytes,
    tokenEmbedBytes,
    logitMatrixBytes,
    activeBytes: blockBytes * activeShare + logitMatrixBytes,
    effectiveBitsPerWeight: model.totalParams > 0 ? (totalBytes * 8) / model.totalParams : 0,
    bytesPerLayer: model.nLayers > 0 ? blockBytes / model.nLayers : 0,
  };
}

/* -------------------------------------------------------------------------- */
/* 2. KV cache                                                                 */
/* -------------------------------------------------------------------------- */

export interface KvCacheOptions {
  ctx: number;
  /** Concurrent sequences. Each gets its own cache. */
  batch?: number;
  /** llama.cpp `--cache-type-k` / `-v`; "f16" by default. */
  kvQuant?: string;
}

export interface KvCacheBreakdown {
  totalBytes: number;
  /** One entry per layer, in model order. Sliding-window layers are smaller. */
  bytesPerLayer: number[];
  bytesPerToken: number;
  /** Number of layers whose cache is capped by a sliding window. */
  windowedLayers: number;
  bytesPerElement: number;
}

/**
 * Rule 2: the KV cache.
 *
 * Standard attention (MHA, GQA, MQA -- all the same formula) caches one key
 * and one value vector per KV head per layer per token:
 *
 *     bytes = 2 * n_layers * n_kv_heads * head_dim * ctx * batch * bytes_per_element
 *             ^-- K and V
 *
 * `n_kv_heads`, never `n_heads`. This is the single most expensive mistake in
 * the genre. Llama 3.1 70B has 64 query heads and 8 KV heads: using 64 makes
 * the cache come out exactly 8x too large, which turns "70B at 32k fits in
 * 48 GB" into "you need an H100". MQA (nKvHeads = 1) is the same formula with
 * the group size equal to the head count.
 *
 * Multi-head Latent Attention (DeepSeek V2/V3) does not cache K and V at all.
 * It caches a single compressed latent c_KV of width `kv_lora_rank` plus the
 * decoupled RoPE key of width `qk_rope_head_dim`, shared by every head:
 *
 *     bytes = n_layers * (kv_lora_rank + qk_rope_head_dim) * ctx * batch * bpe
 *
 * -- no factor of 2, no head count. For DeepSeek-V2-Lite that is 576 elements
 * per token per layer instead of 16 heads * 128 dims * 2 = 4096, a 7x saving,
 * and it is why DeepSeek can serve 128k context economically.
 *
 * Interleaved sliding-window attention (Gemma 2/3, gpt-oss) caps most layers
 * at `windowSize` tokens; only every Nth layer keeps the full context.
 */
export function computeKvCacheBytes(
  model: ModelSpec,
  options: KvCacheOptions,
): KvCacheBreakdown {
  const ctx = Math.max(1, Math.floor(options.ctx));
  const batch = Math.max(1, Math.floor(options.batch ?? 1));
  const bytesPerElement = getKvQuant(options.kvQuant ?? "f16").bytesPerElement;

  let elementsPerTokenPerLayer: number;
  if (model.attention === "mla") {
    const mla = model.mla;
    if (!mla) throw new Error(`Model ${model.id} declares MLA but has no mla block`);
    elementsPerTokenPerLayer = mla.kvLoraRank + mla.qkRopeHeadDim;
  } else {
    elementsPerTokenPerLayer = 2 * model.nKvHeads * model.headDim;
  }

  const perTokenPerLayerBytes = elementsPerTokenPerLayer * bytesPerElement * batch;

  const window = model.attentionWindow;
  const bytesPerLayer: number[] = [];
  let windowedLayers = 0;

  for (let layer = 0; layer < model.nLayers; layer++) {
    let layerCtx = ctx;
    if (window && window.fullAttentionEvery > 1) {
      // Convention: the last layer of every group of `fullAttentionEvery` is
      // the full-attention one (matching Gemma 3's 5 local : 1 global pattern
      // and gpt-oss's strict alternation).
      const isFullAttention = (layer + 1) % window.fullAttentionEvery === 0;
      if (!isFullAttention) {
        layerCtx = Math.min(ctx, window.windowSize);
        windowedLayers++;
      }
    }
    bytesPerLayer.push(perTokenPerLayerBytes * layerCtx);
  }

  const totalBytes = bytesPerLayer.reduce((sum, b) => sum + b, 0);

  return {
    totalBytes,
    bytesPerLayer,
    bytesPerToken: totalBytes / ctx,
    windowedLayers,
    bytesPerElement,
  };
}

/* -------------------------------------------------------------------------- */
/* 4. Overheads                                                                */
/* -------------------------------------------------------------------------- */

/**
 * Per-device runtime overhead: memory that disappears before a single weight
 * is loaded. On CUDA this is the primary context (~300-500 MiB), the loaded
 * kernel image, and cuBLAS/cuBLASLt workspaces; users routinely see 0.6-0.9
 * GiB gone on `nvidia-smi` for an idle llama.cpp process. Datacenter parts
 * carry more driver-side bookkeeping. Metal has no comparable context, but
 * the compiled pipeline library and command buffers still cost a few hundred
 * MiB. These are EMPIRICAL numbers, not derived, and are the least precise
 * term in the model -- assume +-0.3 GiB.
 *
 * This is charged once per physical device, which is why an 8x setup loses
 * ~5.6 GiB to overhead before anything useful happens.
 */
export const RUNTIME_CONTEXT_BYTES: Readonly<Record<DeviceFamily, number>> = {
  "cuda-consumer": 0.7 * GIB,
  "cuda-datacenter": 1.0 * GIB,
  rocm: 0.9 * GIB,
  metal: 0.5 * GIB,
  cpu: 0.2 * GIB,
};

export interface ActivationOptions {
  ctx: number;
  batch?: number;
  /**
   * llama.cpp's `--ubatch-size`: the number of tokens evaluated in one graph
   * pass. This, not the context length, sets the width of the compute buffer.
   */
  physicalBatch?: number;
  /** Flash attention avoids materialising the ctx x ctx score matrix. */
  flashAttention?: boolean;
  /** Activations are computed in the KV dtype's precision, 2 bytes by default. */
  bytesPerActivation?: number;
}

/**
 * Residual-width tensors kept alive simultaneously inside one layer's graph
 * (input, normed input, q, k, v, attention output, ...). Fitted so that the
 * result lands inside the range llama.cpp reports as "compute buffer size"
 * across the 1B-70B range; it is not derived from first principles.
 */
const RESIDENT_HIDDEN_TENSORS = 18;
/** Same idea for intermediate-width tensors (gate, up, activated product). */
const RESIDENT_FFN_TENSORS = 6;
/** Backends never allocate a trivially small graph buffer. */
const MIN_ACTIVATION_BYTES = 64 * MIB;

/**
 * Compute / activation buffer.
 *
 * Scales with the *physical* batch (ubatch), not the context -- except for the
 * attention score matrix, which is O(ubatch * ctx * n_heads) and only exists
 * when flash attention is off. That term is why disabling flash attention at
 * 32k context can cost more VRAM than the KV cache itself.
 *
 * For a mixture of experts the FFN intermediate is materialised for the top-k
 * experts each token routes to, so the intermediate width scales with
 * `expertsPerToken`.
 *
 * EMPIRICAL: assume +-50% on this term. It is typically 1-3% of the total for
 * any model large enough to be worth checking, and dominates only for tiny
 * models at huge context with flash attention disabled.
 */
export function computeActivationBytes(
  model: ModelSpec,
  options: ActivationOptions,
): number {
  const ctx = Math.max(1, Math.floor(options.ctx));
  const batch = Math.max(1, Math.floor(options.batch ?? 1));
  const physicalBatch = Math.max(1, Math.floor(options.physicalBatch ?? 512));
  const flashAttention = options.flashAttention ?? true;
  const bpa = options.bytesPerActivation ?? 2;

  // `n_ubatch` is the width of one graph pass in tokens, counted across every
  // sequence sharing the batch -- not per sequence. Serving 32 sequences at
  // ubatch 512 still evaluates 512 tokens per pass, so the graph does not grow
  // with `--batch`; only the FP32 logit buffer below does.
  const tokensInFlight = Math.min(ctx * batch, physicalBatch);

  const ffnWidth = model.moe
    ? model.moe.expertFfnHidden * (model.moe.expertsPerToken + model.moe.nSharedExperts)
    : model.ffnHidden;

  const graphBytes =
    tokensInFlight *
    (model.hiddenSize * RESIDENT_HIDDEN_TENSORS + ffnWidth * RESIDENT_FFN_TENSORS) *
    bpa;

  // Logits are FP32 and only produced for the tokens actually being scored.
  const logitBytes = batch * model.vocabSize * 4;

  // Without flash attention the full score matrix is materialised in FP32.
  const scoreBytes = flashAttention ? 0 : tokensInFlight * ctx * model.nHeads * 4;

  return Math.max(MIN_ACTIVATION_BYTES, graphBytes + logitBytes + scoreBytes);
}

/* -------------------------------------------------------------------------- */
/* Composition                                                                 */
/* -------------------------------------------------------------------------- */

export interface MemoryFootprint {
  weights: WeightBreakdown;
  kv: KvCacheBreakdown;
  runtimeContextBytes: number;
  activationBytes: number;
  totalBytes: number;
}

export interface FootprintOptions extends KvCacheOptions, ActivationOptions {
  family: DeviceFamily;
  /** Physical devices; runtime context is charged once per device. */
  deviceCount?: number;
}

/** Full memory footprint of a single-process, fully-resident deployment. */
export function computeFootprint(
  model: ModelSpec,
  quant: QuantSpec,
  options: FootprintOptions,
): MemoryFootprint {
  const weights = computeWeightBytes(model, quant);
  const kv = computeKvCacheBytes(model, options);
  const deviceCount = Math.max(1, Math.floor(options.deviceCount ?? 1));
  const runtimeContextBytes = RUNTIME_CONTEXT_BYTES[options.family] * deviceCount;
  const activationBytes = computeActivationBytes(model, options) * deviceCount;

  return {
    weights,
    kv,
    runtimeContextBytes,
    activationBytes,
    totalBytes: weights.totalBytes + kv.totalBytes + runtimeContextBytes + activationBytes,
  };
}
