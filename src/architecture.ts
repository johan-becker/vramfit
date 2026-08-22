import type { ModelSpec } from "./types.js";

/**
 * Parameter accounting derived from a model's architecture.
 *
 * Two independent things need this:
 *
 *  1. Memory. Weight bytes depend on how many parameters sit in the token
 *     embedding / output head (usually kept at higher precision) versus the
 *     repeating transformer blocks.
 *  2. Speed. Decode is bandwidth-bound, so what matters is how many bytes are
 *     *read* per token. That excludes the input embedding table -- a token
 *     embedding is one row lookup, not a matmul -- and, for a mixture of
 *     experts, excludes every expert the router did not pick.
 *
 * Deriving these from architecture rather than trusting a single published
 * number also gives the test suite something real to check: the derived total
 * has to land within a couple of percent of the published parameter count, or
 * one of the architecture fields in models.json is wrong.
 */

/**
 * Gated FFNs (SwiGLU / GeGLU) have three weight matrices per block:
 * gate_proj and up_proj (hidden -> intermediate) and down_proj
 * (intermediate -> hidden). Every model in the bundled database is gated;
 * a classic 2-matrix GPT-2 style MLP would need this to be 2.
 */
const GATED_FFN_MATRICES = 3;

export interface Architecture {
  /** Token embedding table: vocab x hidden. */
  inputEmbedParams: number;
  /** Output projection (lm_head). Zero when embeddings are tied, because the
   *  same matrix is reused -- but it is still read during decode. */
  outputHeadParams: number;
  /** Matrix actually read to produce logits, tied or not. */
  logitMatrixParams: number;
  /** Attention weights in one layer. */
  attnParamsPerLayer: number;
  /** All feed-forward weights across every layer (all experts). */
  ffnParamsAllLayers: number;
  /** Every parameter in the repeating blocks, all layers. */
  blockParams: number;
  /** Block parameters read to decode one token (top-k experts only). */
  activeBlockParams: number;
  /** Sum of the above plus embeddings: cross-check against ModelSpec.totalParams. */
  derivedTotalParams: number;
  /** Parameters multiplied per decoded token: blocks + logit matrix. */
  activeMatmulParams: number;
}

function attentionParamsPerLayer(model: ModelSpec): number {
  const h = model.hiddenSize;

  if (model.attention === "mla") {
    const mla = model.mla;
    if (!mla) {
      throw new Error(`Model ${model.id} declares attention "mla" but has no mla block`);
    }
    const qHeadWidth = model.nHeads * (mla.qkNopeHeadDim + mla.qkRopeHeadDim);
    // Queries are optionally low-rank factorised (V2/V3 do it, V2-Lite does not).
    const qParams =
      mla.qLoraRank === null ? h * qHeadWidth : h * mla.qLoraRank + mla.qLoraRank * qHeadWidth;
    // Down-projection to the cached latent plus the decoupled RoPE key.
    const kvDownParams = h * (mla.kvLoraRank + mla.qkRopeHeadDim);
    // Up-projection reconstructing per-head K (nope part) and V from the latent.
    const kvUpParams = mla.kvLoraRank * model.nHeads * (mla.qkNopeHeadDim + mla.vHeadDim);
    const outParams = model.nHeads * mla.vHeadDim * h;
    return qParams + kvDownParams + kvUpParams + outParams;
  }

  // Grouped-query attention. Note the asymmetry: q and o are sized by the
  // number of *query* heads, k and v by the number of *KV* heads.
  const qWidth = model.nHeads * model.headDim;
  const kvWidth = model.nKvHeads * model.headDim;
  return h * qWidth + h * kvWidth * 2 + qWidth * h;
}

function denseFfnParams(model: ModelSpec, intermediate: number): number {
  return GATED_FFN_MATRICES * model.hiddenSize * intermediate;
}

export function deriveArchitecture(model: ModelSpec): Architecture {
  const inputEmbedParams = model.vocabSize * model.hiddenSize;
  const outputHeadParams = model.tiedEmbeddings ? 0 : inputEmbedParams;
  const logitMatrixParams = inputEmbedParams; // tied or not, one vocab x hidden matmul

  const attnParamsPerLayer = attentionParamsPerLayer(model);

  let ffnParamsAllLayers = 0;
  let activeFfnParamsPerLayerSum = 0;

  const moe = model.moe;
  if (moe) {
    const denseLayerCount = Math.min(moe.denseLayers, model.nLayers);
    const moeLayerCount = model.nLayers - denseLayerCount;
    const denseWidth = moe.denseFfnHidden ?? model.ffnHidden;

    const denseTotal = denseFfnParams(model, denseWidth) * denseLayerCount;
    ffnParamsAllLayers += denseTotal;
    activeFfnParamsPerLayerSum += denseTotal;

    const oneExpert = denseFfnParams(model, moe.expertFfnHidden);
    const routerParams = model.hiddenSize * moe.nExperts;
    // Shared experts run for every token, routed experts only when selected.
    const alwaysOnExperts = moe.nSharedExperts;
    const perMoeLayerTotal = oneExpert * (moe.nExperts + alwaysOnExperts) + routerParams;
    const perMoeLayerActive =
      oneExpert * (moe.expertsPerToken + alwaysOnExperts) + routerParams;

    ffnParamsAllLayers += perMoeLayerTotal * moeLayerCount;
    activeFfnParamsPerLayerSum += perMoeLayerActive * moeLayerCount;
  } else {
    const total = denseFfnParams(model, model.ffnHidden) * model.nLayers;
    ffnParamsAllLayers = total;
    activeFfnParamsPerLayerSum = total;
  }

  const attnAllLayers = attnParamsPerLayer * model.nLayers;
  const blockParams = attnAllLayers + ffnParamsAllLayers;
  const activeBlockParams = attnAllLayers + activeFfnParamsPerLayerSum;

  return {
    inputEmbedParams,
    outputHeadParams,
    logitMatrixParams,
    attnParamsPerLayer,
    ffnParamsAllLayers,
    blockParams,
    activeBlockParams,
    derivedTotalParams: inputEmbedParams + outputHeadParams + blockParams,
    activeMatmulParams: activeBlockParams + logitMatrixParams,
  };
}

/**
 * Fraction of block weights read per decoded token. 1.0 for dense models,
 * and roughly (top_k + shared) / n_experts for a mixture of experts once the
 * always-dense attention weights are folded in.
 */
export function activeBlockFraction(model: ModelSpec): number {
  const arch = deriveArchitecture(model);
  if (arch.blockParams === 0) return 1;
  return arch.activeBlockParams / arch.blockParams;
}
