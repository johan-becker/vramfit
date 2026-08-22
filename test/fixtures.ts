import type { ModelSpec } from "../src/types.js";

/**
 * Hand-built model specs for the arithmetic tests.
 *
 * These are deliberately independent of `src/data/models.json`: a test that
 * reads the same file the code reads cannot catch a bad formula, only a bad
 * lookup. Every number below comes from the model's published config.json and
 * is written out so the expected values in the tests can be recomputed by hand
 * from this file alone.
 */

const BASE: Omit<
  ModelSpec,
  | "id"
  | "name"
  | "totalParams"
  | "activeParams"
  | "nLayers"
  | "hiddenSize"
  | "nHeads"
  | "nKvHeads"
  | "headDim"
  | "ffnHidden"
  | "vocabSize"
> = {
  aliases: [],
  tiedEmbeddings: false,
  attention: "gqa",
  mla: null,
  moe: null,
  attentionWindow: null,
  maxCtx: 8192,
  defaultCtx: 8192,
  source: "test fixture",
};

/** Llama 3.1 8B: 32 query heads, 8 KV heads (GQA group size 4). */
export const LLAMA_3_1_8B: ModelSpec = {
  ...BASE,
  id: "fixture-llama-3.1-8b",
  name: "Llama 3.1 8B",
  totalParams: 8_030_000_000,
  activeParams: 8_030_000_000,
  nLayers: 32,
  hiddenSize: 4096,
  nHeads: 32,
  nKvHeads: 8,
  headDim: 128,
  ffnHidden: 14336,
  vocabSize: 128_256,
  maxCtx: 131_072,
  defaultCtx: 8192,
};

/**
 * Llama 2 70B: 64 query heads and 64 KV heads -- plain multi-head attention.
 * Paired with LLAMA_3_1_70B below, which is architecturally identical except
 * for having 8 KV heads, to pin the exact 8x GQA saving.
 */
export const LLAMA_2_70B_MHA: ModelSpec = {
  ...BASE,
  id: "fixture-llama-2-70b-mha",
  name: "Llama 2 70B (MHA)",
  totalParams: 68_980_000_000,
  activeParams: 68_980_000_000,
  nLayers: 80,
  hiddenSize: 8192,
  nHeads: 64,
  nKvHeads: 64,
  headDim: 128,
  ffnHidden: 28_672,
  vocabSize: 32_000,
  maxCtx: 4096,
  defaultCtx: 4096,
};

export const LLAMA_3_1_70B: ModelSpec = {
  ...LLAMA_2_70B_MHA,
  id: "fixture-llama-3.1-70b",
  name: "Llama 3.1 70B (GQA)",
  totalParams: 70_554_000_000,
  activeParams: 70_554_000_000,
  nKvHeads: 8,
  vocabSize: 128_256,
  maxCtx: 131_072,
};

/** Multi-query attention: one KV head shared by every query head. */
export const MQA_7B: ModelSpec = {
  ...BASE,
  id: "fixture-mqa-7b",
  name: "MQA 7B",
  totalParams: 7_000_000_000,
  activeParams: 7_000_000_000,
  nLayers: 32,
  hiddenSize: 4096,
  nHeads: 32,
  nKvHeads: 1,
  headDim: 128,
  ffnHidden: 11_008,
  vocabSize: 32_000,
};

/** Mixtral 8x7B: 46.7B total parameters, 12.9B active per token. */
export const MIXTRAL_8X7B: ModelSpec = {
  ...BASE,
  id: "fixture-mixtral-8x7b",
  name: "Mixtral 8x7B",
  totalParams: 46_703_000_000,
  activeParams: 12_879_000_000,
  nLayers: 32,
  hiddenSize: 4096,
  nHeads: 32,
  nKvHeads: 8,
  headDim: 128,
  ffnHidden: 14_336,
  vocabSize: 32_000,
  maxCtx: 32_768,
  defaultCtx: 32_768,
  moe: {
    nExperts: 8,
    expertsPerToken: 2,
    expertFfnHidden: 14_336,
    nSharedExperts: 0,
    denseLayers: 0,
    denseFfnHidden: null,
  },
};

/** DeepSeek-V2-Lite: MLA plus a fine-grained MoE with two shared experts. */
export const DEEPSEEK_V2_LITE: ModelSpec = {
  ...BASE,
  id: "fixture-deepseek-v2-lite",
  name: "DeepSeek-V2-Lite",
  totalParams: 15_706_000_000,
  activeParams: 2_400_000_000,
  nLayers: 27,
  hiddenSize: 2048,
  nHeads: 16,
  nKvHeads: 16,
  headDim: 128,
  ffnHidden: 10_944,
  vocabSize: 102_400,
  maxCtx: 163_840,
  defaultCtx: 32_768,
  attention: "mla",
  mla: {
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    qkNopeHeadDim: 128,
    vHeadDim: 128,
    qLoraRank: null,
  },
  moe: {
    nExperts: 64,
    expertsPerToken: 6,
    expertFfnHidden: 1408,
    nSharedExperts: 2,
    denseLayers: 1,
    denseFfnHidden: 10_944,
  },
};

/**
 * A GQA model with interleaved sliding-window attention: 5 local layers per
 * global layer, 1024-token window. Same shape as Gemma 3 12B.
 */
export const SWA_12B: ModelSpec = {
  ...BASE,
  id: "fixture-swa-12b",
  name: "SWA 12B",
  totalParams: 11_766_000_000,
  activeParams: 11_766_000_000,
  nLayers: 48,
  hiddenSize: 3840,
  nHeads: 16,
  nKvHeads: 8,
  headDim: 256,
  ffnHidden: 15_360,
  vocabSize: 262_208,
  tiedEmbeddings: true,
  maxCtx: 131_072,
  defaultCtx: 8192,
  attentionWindow: { windowSize: 1024, fullAttentionEvery: 6 },
};
