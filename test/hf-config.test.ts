import { describe, expect, it } from "vitest";
import { SpecValidationError } from "../src/db/validate.js";
import { HfConfigError, modelFromHfConfig } from "../src/hf/config.js";
import {
  SafetensorsError,
  parseSafetensorsIndex,
  readSafetensorsHeader,
} from "../src/hf/safetensors.js";
import { computeKvCacheBytes } from "../src/memory.js";

/**
 * Reading a HuggingFace `config.json` from a local path.
 *
 * The configs below are the published ones, trimmed to the fields that change
 * an answer. They are here as literals rather than as downloads for the usual
 * reason -- the suite has to pass offline -- and because the interesting cases
 * are the inconsistencies between them, which is exactly what a fixture set
 * can hold and a single downloaded file cannot.
 */

const LLAMA_31_8B = {
  _name_or_path: "meta-llama/Meta-Llama-3.1-8B-Instruct",
  architectures: ["LlamaForCausalLM"],
  model_type: "llama",
  hidden_size: 4096,
  intermediate_size: 14_336,
  max_position_embeddings: 131_072,
  num_attention_heads: 32,
  num_hidden_layers: 32,
  num_key_value_heads: 8,
  tie_word_embeddings: false,
  torch_dtype: "bfloat16",
  vocab_size: 128_256,
};

const MIXTRAL_8X7B = {
  _name_or_path: "mistralai/Mixtral-8x7B-Instruct-v0.1",
  model_type: "mixtral",
  hidden_size: 4096,
  intermediate_size: 14_336,
  max_position_embeddings: 32_768,
  num_attention_heads: 32,
  num_hidden_layers: 32,
  num_key_value_heads: 8,
  num_local_experts: 8,
  num_experts_per_tok: 2,
  tie_word_embeddings: false,
  torch_dtype: "bfloat16",
  vocab_size: 32_000,
};

const QWEN_MOE = {
  _name_or_path: "Qwen/Qwen1.5-MoE-A2.7B",
  model_type: "qwen2_moe",
  hidden_size: 2048,
  intermediate_size: 5632,
  moe_intermediate_size: 1408,
  shared_expert_intermediate_size: 5632,
  num_experts: 60,
  num_experts_per_tok: 4,
  max_position_embeddings: 32_768,
  num_attention_heads: 16,
  num_hidden_layers: 24,
  num_key_value_heads: 16,
  tie_word_embeddings: false,
  torch_dtype: "bfloat16",
  vocab_size: 151_936,
};

const DEEPSEEK_V2_LITE = {
  _name_or_path: "deepseek-ai/DeepSeek-V2-Lite",
  model_type: "deepseek_v2",
  hidden_size: 2048,
  intermediate_size: 10_944,
  moe_intermediate_size: 1408,
  n_routed_experts: 64,
  n_shared_experts: 2,
  num_experts_per_tok: 6,
  first_k_dense_replace: 1,
  kv_lora_rank: 512,
  q_lora_rank: null,
  qk_nope_head_dim: 128,
  qk_rope_head_dim: 64,
  v_head_dim: 128,
  max_position_embeddings: 163_840,
  num_attention_heads: 16,
  num_hidden_layers: 27,
  num_key_value_heads: 16,
  tie_word_embeddings: false,
  torch_dtype: "bfloat16",
  vocab_size: 102_400,
};

const GEMMA_3_27B = {
  _name_or_path: "google/gemma-3-27b-it",
  architectures: ["Gemma3ForConditionalGeneration"],
  model_type: "gemma3",
  torch_dtype: "bfloat16",
  text_config: {
    hidden_size: 5376,
    intermediate_size: 21_504,
    head_dim: 128,
    max_position_embeddings: 131_072,
    num_attention_heads: 32,
    num_hidden_layers: 62,
    num_key_value_heads: 16,
    sliding_window: 1024,
    sliding_window_pattern: 6,
    tie_word_embeddings: true,
    vocab_size: 262_208,
  },
  vision_config: { hidden_size: 1152, num_hidden_layers: 27 },
};

describe("modelFromHfConfig", () => {
  it("maps the fields a Llama config actually carries", () => {
    const { model, paramSource } = modelFromHfConfig(LLAMA_31_8B, { origin: "./config.json" });

    expect(model.name).toBe("Meta-Llama-3.1-8B-Instruct");
    expect(model.id).toBe("meta-llama-3.1-8b-instruct");
    expect(model.nLayers).toBe(32);
    expect(model.nHeads).toBe(32);
    expect(model.nKvHeads).toBe(8);
    // head_dim is absent from this config: hidden_size / num_attention_heads.
    expect(model.headDim).toBe(128);
    expect(model.ffnHidden).toBe(14_336);
    expect(model.vocabSize).toBe(128_256);
    expect(model.maxCtx).toBe(131_072);
    expect(model.attention).toBe("gqa");
    expect(model.source).toBe("HuggingFace config.json at ./config.json");
    // No weight files were offered, so the architecture's own sum is used.
    expect(paramSource).toBe("architecture");
    expect(model.totalParams).toBe(8_029_995_008);
  });

  it("reads an absent num_key_value_heads as multi-head, not multi-query", () => {
    // The dangerous direction: reading it as 1 would undersize a pre-GQA
    // model's cache by the head count.
    const { model } = modelFromHfConfig({
      ...LLAMA_31_8B,
      num_key_value_heads: undefined,
      num_hidden_layers: 32,
    });
    expect(model.nKvHeads).toBe(32);
  });

  it("prefers an explicit head_dim over the division", () => {
    // Llama 3.2 1B: 2048 hidden, 32 heads, but head_dim 64 -- the division
    // would give 64 here as well, so use a config where they differ.
    const { model } = modelFromHfConfig({ ...LLAMA_31_8B, head_dim: 256 });
    expect(model.headDim).toBe(256);
  });

  it("reads Mixtral's expert block, and the active parameters it implies", () => {
    const { model } = modelFromHfConfig(MIXTRAL_8X7B);
    expect(model.moe).toEqual({
      nExperts: 8,
      expertsPerToken: 2,
      // No moe_intermediate_size: the experts are intermediate_size wide.
      expertFfnHidden: 14_336,
      nSharedExperts: 0,
      denseLayers: 0,
      denseFfnHidden: null,
    });
    expect(model.totalParams / 1e9).toBeCloseTo(46.7, 1);
    // Memory of a 47B, speed of a 13B. Mistral's published 12.88B counts the
    // embedding table too; this is what a decode step actually multiplies.
    expect(model.activeParams / 1e9).toBeCloseTo(12.75, 1);
    expect(model.activeParams / model.totalParams).toBeCloseTo(0.273, 3);
  });

  it("recovers Qwen's shared expert count from its combined width", () => {
    const { model } = modelFromHfConfig(QWEN_MOE);
    expect(model.moe?.nExperts).toBe(60);
    expect(model.moe?.expertFfnHidden).toBe(1408);
    // shared_expert_intermediate_size 5632 / moe_intermediate_size 1408.
    expect(model.moe?.nSharedExperts).toBe(4);
  });

  it("reads DeepSeek's latent attention, shared experts and dense prefix", () => {
    const { model } = modelFromHfConfig(DEEPSEEK_V2_LITE);
    expect(model.attention).toBe("mla");
    expect(model.mla).toEqual({
      kvLoraRank: 512,
      qkNopeHeadDim: 128,
      qkRopeHeadDim: 64,
      vHeadDim: 128,
      // q_lora_rank is null in this config: the queries are not compressed.
      qLoraRank: null,
    });
    expect(model.moe?.nSharedExperts).toBe(2);
    expect(model.moe?.denseLayers).toBe(1);

    // 27 layers x (512 + 64) elements per token, not 27 x 2 x 16 x 128.
    const kv = computeKvCacheBytes(model, { ctx: 4096 });
    expect(kv.totalBytes).toBe(27 * (512 + 64) * 4096 * 2);
  });

  it("descends into text_config, and reads the sliding-window pattern", () => {
    const { model } = modelFromHfConfig(GEMMA_3_27B);
    expect(model.nLayers).toBe(62);
    expect(model.hiddenSize).toBe(5376);
    expect(model.tiedEmbeddings).toBe(true);
    expect(model.attentionWindow).toEqual({ windowSize: 1024, fullAttentionEvery: 6 });

    // 52 of 62 layers capped at 1024 tokens rather than the full context.
    const kv = computeKvCacheBytes(model, { ctx: 32_768 });
    expect(kv.windowedLayers).toBe(52);
    expect(kv.totalBytes).toBeLessThan(32_768 * 62 * 2 * 16 * 128 * 2 * 0.2);
  });

  it("ignores a sliding window the config has switched off", () => {
    // Qwen2.5 ships sliding_window with use_sliding_window false. Honouring
    // it would undersize the cache by the ratio of window to context.
    const { model } = modelFromHfConfig({
      ...LLAMA_31_8B,
      sliding_window: 4096,
      use_sliding_window: false,
    });
    expect(model.attentionWindow).toBeNull();
  });

  it("reads the period out of a layer_types list", () => {
    const layerTypes = Array.from({ length: 32 }, (_, index) =>
      (index + 1) % 4 === 0 ? "full_attention" : "sliding_attention",
    );
    const { model } = modelFromHfConfig({
      ...LLAMA_31_8B,
      sliding_window: 512,
      layer_types: layerTypes,
    });
    expect(model.attentionWindow).toEqual({ windowSize: 512, fullAttentionEvery: 4 });
  });

  it("reads an absent tie_word_embeddings as tied, the way transformers does", () => {
    // Every other fixture here sets the key, which is exactly the case
    // save_pretrained does not produce: PretrainedConfig defaults it to true
    // and to_diff_dict omits whatever equals the default, so the checkpoints
    // that tie -- Gemma, Phi-3, several Qwen releases -- ship without it.
    const GEMMA_2_2B = {
      _name_or_path: "google/gemma-2-2b-it",
      model_type: "gemma2",
      hidden_size: 2304,
      intermediate_size: 9216,
      head_dim: 256,
      num_attention_heads: 8,
      num_hidden_layers: 26,
      num_key_value_heads: 4,
      max_position_embeddings: 8192,
      torch_dtype: "bfloat16",
      vocab_size: 256_000,
    };

    const { model, notes } = modelFromHfConfig(GEMMA_2_2B);
    expect(model.tiedEmbeddings).toBe(true);
    expect(notes[0]).toMatch(/does not set tie_word_embeddings/);

    // Reading it as untied invents a second 256000 x 2304 matrix: +589.8M
    // parameters, 22.6% of the model, on a key that is absent by design.
    const untied = modelFromHfConfig({ ...GEMMA_2_2B, tie_word_embeddings: false }).model;
    expect(untied.totalParams - model.totalParams).toBe(256_000 * 2304);
  });

  it("charges a tied embedding table once", () => {
    const untied = modelFromHfConfig(LLAMA_31_8B).model;
    const tied = modelFromHfConfig({ ...LLAMA_31_8B, tie_word_embeddings: true }).model;
    expect(untied.totalParams - tied.totalParams).toBe(128_256 * 4096);
  });

  it("names the field it cannot do without", () => {
    expect(() => modelFromHfConfig({ hidden_size: 4096 })).toThrow(HfConfigError);
    expect(() => modelFromHfConfig({ hidden_size: 4096 })).toThrow(
      /no layer count: none of "num_hidden_layers", "n_layer", "num_layers", "n_layers" is a number/,
    );
    expect(() => modelFromHfConfig({ ...LLAMA_31_8B, vocab_size: undefined })).toThrow(
      /no vocabulary size/,
    );
    expect(() => modelFromHfConfig("not an object")).toThrow(/is not a JSON object/);
  });

  it("refuses a half-specified latent attention block", () => {
    expect(() =>
      modelFromHfConfig({ ...DEEPSEEK_V2_LITE, v_head_dim: undefined }),
    ).toThrow(/sets kv_lora_rank but not all of qk_rope_head_dim/);
  });

  it("still validates the result like any other spec", () => {
    // 7 KV heads do not divide 32 query heads into groups.
    expect(() => modelFromHfConfig({ ...LLAMA_31_8B, num_key_value_heads: 7 })).toThrow(
      SpecValidationError,
    );
  });
});

describe("modelFromHfConfig with a weight count", () => {
  it("prefers the weight files, which include the norms and biases", () => {
    // meta-llama/Meta-Llama-3.1-8B-Instruct: 16,060,530,688 bytes of bf16.
    const { model, paramSource, notes } = modelFromHfConfig(LLAMA_31_8B, {
      weights: { totalParams: 16_060_530_688 / 2, source: "index" },
    });
    expect(paramSource).toBe("index");
    expect(model.totalParams).toBe(8_030_265_344);
    expect(notes).toEqual([]);
  });

  it("falls back to the decoder when the weight files describe more than it", () => {
    // A multimodal repository's shards hold the vision tower as well.
    const { model, paramSource, notes } = modelFromHfConfig(LLAMA_31_8B, {
      weights: { totalParams: 11_000_000_000, source: "index" },
    });
    expect(paramSource).toBe("architecture");
    expect(model.totalParams).toBe(8_029_995_008);
    expect(notes[0]).toMatch(/weight files hold 11000000000 parameters but the text decoder/);
    expect(notes[0]).toMatch(/vision tower/);
  });
});

/* -------------------------------------------------------------------------- */
/* safetensors                                                                 */
/* -------------------------------------------------------------------------- */

function safetensorsBytes(header: Record<string, unknown>, truncateTo?: number): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  return truncateTo === undefined ? out : out.slice(0, truncateTo);
}

function memorySource(bytes: Uint8Array) {
  return {
    size: bytes.length,
    read: (offset: number, length: number): Uint8Array =>
      bytes.subarray(offset, offset + Math.max(0, length)),
  };
}

describe("readSafetensorsHeader", () => {
  it("counts parameters exactly, from shapes rather than from bytes", () => {
    const header = readSafetensorsHeader(
      memorySource(
        safetensorsBytes({
          __metadata__: { format: "pt" },
          "model.embed_tokens.weight": {
            dtype: "BF16",
            shape: [128_256, 4096],
            data_offsets: [0, 1_050_673_152],
          },
          "model.norm.weight": { dtype: "F32", shape: [4096], data_offsets: [0, 16_384] },
        }),
      ),
    );

    expect(header.totalParams).toBe(128_256 * 4096 + 4096);
    // Mixed dtypes: the byte total is not the parameter count times two.
    expect(header.totalBytes).toBe(128_256 * 4096 * 2 + 4096 * 4);
    expect(header.tensors).toHaveLength(2);
  });

  it("refuses a header that is not one", () => {
    const bytes = safetensorsBytes({ t: { dtype: "BF16", shape: [4] } });
    expect(() => readSafetensorsHeader(memorySource(bytes.slice(0, 4)))).toThrow(
      /too short to hold a header length/,
    );
    expect(() => readSafetensorsHeader(memorySource(safetensorsBytes({}, undefined)))).toThrow(
      /declares no tensors/,
    );
    expect(() =>
      readSafetensorsHeader(memorySource(safetensorsBytes({ t: { dtype: "Q4_K", shape: [4] } }))),
    ).toThrow(/dtype "Q4_K", whose width is unknown/);
    expect(() => readSafetensorsHeader(memorySource(bytes.slice(0, 12)))).toThrow(SafetensorsError);
  });
});

describe("parseSafetensorsIndex", () => {
  it("reads the byte total and the shard count", () => {
    const index = parseSafetensorsIndex(
      {
        metadata: { total_size: 16_060_530_688 },
        weight_map: {
          "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
          "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
          "lm_head.weight": "model-00004-of-00004.safetensors",
        },
      },
      "./model.safetensors.index.json",
    );
    expect(index.totalSizeBytes).toBe(16_060_530_688);
    expect(index.shards).toBe(2);
    expect(index.tensors).toBe(3);
  });

  it("refuses an index with no usable total", () => {
    expect(() => parseSafetensorsIndex({ weight_map: {} }, "index.json")).toThrow(
      /has no metadata block/,
    );
    expect(() => parseSafetensorsIndex({ metadata: {} }, "index.json")).toThrow(
      /no positive metadata\.total_size/,
    );
  });
});
