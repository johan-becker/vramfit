import { deriveArchitecture } from "../architecture.js";
import { parseModelSpec } from "../db/validate.js";
import type {
  AttentionWindowSpec,
  MlaSpec,
  ModelSpec,
  MoeSpec,
} from "../types.js";

/**
 * Mapping a HuggingFace `config.json` onto a `ModelSpec`.
 *
 * The file is read from a local path and nothing else happens: no hub lookup,
 * no tokeniser download, no `transformers` import. What makes this more than a
 * rename is that the fields people most need are the ones the ecosystem is
 * least consistent about. `num_key_value_heads` is absent on multi-head models
 * and means "same as the query heads"; `head_dim` is absent on most and means
 * `hidden_size / num_attention_heads`, except on the models where it does not;
 * the expert count is `num_local_experts` on Mixtral, `num_experts` on
 * Qwen3-MoE and `n_routed_experts` on DeepSeek; `sliding_window` is set but
 * inert on Qwen2 unless `use_sliding_window` is true; and `tie_word_embeddings`
 * is absent from every config that ties, because `PretrainedConfig` defaults it
 * to true and `to_diff_dict` writes only what differs from the default.
 *
 * Getting any of those wrong is a wrong VRAM number rather than a crash, which
 * is why each one is spelled out below with the alternatives it accepts.
 */

/** Thrown when a config.json cannot be turned into a model. */
export class HfConfigError extends Error {
  override readonly name = "HfConfigError";
}

type Json = Record<string, unknown>;

function asObject(value: unknown, path: string): Json {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new HfConfigError(`${path} is not a JSON object`);
  }
  return value as Json;
}

/**
 * The block that describes the language model.
 *
 * Multimodal releases nest it under `text_config` and keep the vision tower
 * beside it -- Gemma 3 and Llama 3.2 Vision both do. Descending is right: the
 * decoder is the part whose KV cache and weights this tool sizes.
 */
export function textConfigOf(config: Json): Json {
  if (config["num_hidden_layers"] !== undefined || config["n_layer"] !== undefined) return config;
  const nested = config["text_config"];
  if (typeof nested === "object" && nested !== null && !Array.isArray(nested)) {
    return nested as Json;
  }
  return config;
}

function pick(config: Json, keys: readonly string[]): unknown {
  for (const key of keys) {
    const value = config[key];
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

function pickNumber(config: Json, keys: readonly string[]): number | undefined {
  const value = pick(config, keys);
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  return value;
}

function pickString(config: Json, keys: readonly string[]): string | undefined {
  const value = pick(config, keys);
  return typeof value === "string" && value.trim() !== "" ? value : undefined;
}

function pickBoolean(config: Json, keys: readonly string[]): boolean | undefined {
  const value = pick(config, keys);
  return typeof value === "boolean" ? value : undefined;
}

function requireNumber(config: Json, keys: readonly string[], what: string): number {
  const value = pickNumber(config, keys);
  if (value === undefined) {
    throw new HfConfigError(
      `the config has no ${what}: none of ${keys.map((key) => `"${key}"`).join(", ")} is a number`,
    );
  }
  return value;
}

/* -------------------------------------------------------------------------- */
/* Key aliases, one list per field                                             */
/* -------------------------------------------------------------------------- */

const LAYERS = ["num_hidden_layers", "n_layer", "num_layers", "n_layers"] as const;
const HIDDEN = ["hidden_size", "n_embd", "d_model", "model_dim"] as const;
const HEADS = ["num_attention_heads", "n_head", "num_heads"] as const;
const KV_HEADS = ["num_key_value_heads", "num_kv_heads", "n_head_kv"] as const;
const HEAD_DIM = ["head_dim", "attention_head_dim", "v_head_dim"] as const;
const FFN = ["intermediate_size", "ffn_dim", "n_inner", "ffn_hidden_size"] as const;
const VOCAB = ["vocab_size", "n_vocab", "padded_vocab_size"] as const;
const MAX_CTX = ["max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"] as const;
const TIED = ["tie_word_embeddings", "tie_weights"] as const;
const EXPERTS = ["num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"] as const;
const EXPERTS_PER_TOKEN = ["num_experts_per_tok", "moe_topk", "top_k", "n_group_experts"] as const;
const EXPERT_FFN = ["moe_intermediate_size", "expert_intermediate_size", "ffn_dim"] as const;
const SHARED_EXPERTS = ["n_shared_experts", "num_shared_experts", "moe_num_shared_experts"] as const;
const SHARED_EXPERT_FFN = ["shared_expert_intermediate_size"] as const;
const DENSE_LAYERS = ["first_k_dense_replace", "moe_layer_start_index", "num_dense_layers"] as const;

/* -------------------------------------------------------------------------- */
/* Sub-blocks                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Multi-head latent attention.
 *
 * `kv_lora_rank` is the field that decides it: the model caches one compressed
 * latent per token per layer rather than per-head keys and values, and the KV
 * formula has to change entirely rather than be scaled.
 */
function mlaFrom(config: Json): MlaSpec | null {
  const kvLoraRank = pickNumber(config, ["kv_lora_rank"]);
  if (kvLoraRank === undefined || kvLoraRank <= 0) return null;

  const qkRopeHeadDim = pickNumber(config, ["qk_rope_head_dim"]);
  const qkNopeHeadDim = pickNumber(config, ["qk_nope_head_dim"]);
  const vHeadDim = pickNumber(config, ["v_head_dim"]);
  if (qkRopeHeadDim === undefined || qkNopeHeadDim === undefined || vHeadDim === undefined) {
    throw new HfConfigError(
      "the config sets kv_lora_rank but not all of qk_rope_head_dim, qk_nope_head_dim and v_head_dim, which multi-head latent attention needs",
    );
  }
  const qLoraRank = pickNumber(config, ["q_lora_rank"]) ?? 0;
  return {
    kvLoraRank,
    qkRopeHeadDim,
    qkNopeHeadDim,
    vHeadDim,
    qLoraRank: qLoraRank > 0 ? qLoraRank : null,
  };
}

function moeFrom(config: Json, nLayers: number, ffnHidden: number): MoeSpec | null {
  const nExperts = pickNumber(config, EXPERTS);
  if (nExperts === undefined || nExperts <= 1) return null;

  const expertFfnHidden = pickNumber(config, EXPERT_FFN) ?? ffnHidden;
  // Qwen2-MoE publishes the shared experts' combined width rather than their
  // count, the way GGUF does; DeepSeek publishes the count directly.
  const sharedWidth = pickNumber(config, SHARED_EXPERT_FFN);
  const nSharedExperts =
    pickNumber(config, SHARED_EXPERTS) ??
    (sharedWidth !== undefined && expertFfnHidden > 0
      ? Math.round(sharedWidth / expertFfnHidden)
      : 0);

  const denseLayers = Math.min(nLayers, Math.max(0, pickNumber(config, DENSE_LAYERS) ?? 0));

  return {
    nExperts,
    expertsPerToken: pickNumber(config, EXPERTS_PER_TOKEN) ?? 1,
    expertFfnHidden,
    nSharedExperts,
    denseLayers,
    denseFfnHidden: denseLayers > 0 ? ffnHidden : null,
  };
}

/**
 * Interleaved sliding-window attention.
 *
 * Two traps. Qwen2 writes a `sliding_window` that is inert unless
 * `use_sliding_window` is true, so honouring it there would undersize the
 * cache by the ratio of window to context -- the dangerous direction. And
 * Gemma 3 states its 5-local-to-1-global pattern in `sliding_window_pattern`,
 * while Gemma 2 states nothing and simply alternates.
 */
function attentionWindowFrom(config: Json): AttentionWindowSpec | null {
  const windowSize = pickNumber(config, ["sliding_window", "attention_window_size"]);
  if (windowSize === undefined || windowSize <= 0) return null;
  if (pickBoolean(config, ["use_sliding_window"]) === false) return null;

  const declared = pickNumber(config, ["sliding_window_pattern", "layer_types_period"]);
  const pattern = declared ?? patternFromLayerTypes(config) ?? 2;
  if (pattern < 2) return null;
  return { windowSize, fullAttentionEvery: pattern };
}

/**
 * Newer configs list the flavour of every layer in `layer_types` rather than
 * stating a period. The period is the distance between full-attention layers,
 * which is what the KV formula needs; a list that is not periodic is refused
 * rather than averaged, since vramfit models one repeating pattern.
 */
function patternFromLayerTypes(config: Json): number | undefined {
  const raw = config["layer_types"];
  if (!Array.isArray(raw) || raw.length === 0) return undefined;
  const fullAt: number[] = [];
  raw.forEach((entry, index) => {
    if (typeof entry === "string" && entry.includes("full")) fullAt.push(index);
  });
  if (fullAt.length === 0) return undefined;
  const first = fullAt[0] as number;
  if (fullAt.length === 1) return raw.length;
  const period = (fullAt[1] as number) - first;
  const periodic = fullAt.every((position, index) => position === first + index * period);
  return periodic && period >= 2 ? period : undefined;
}

/* -------------------------------------------------------------------------- */
/* The model                                                                   */
/* -------------------------------------------------------------------------- */

/** How a parameter count was arrived at, weakest last. */
export type ParamSource = "safetensors" | "index" | "architecture";

export interface HfModelOptions {
  /** What to record in `ModelSpec.source`; normally the path it came from. */
  origin?: string;
  /**
   * Parameter count read from the weight files, when they were there to read.
   * Preferred over the architecture's own sum, which leaves out biases, layer
   * norms and rotary tables.
   */
  weights?: { totalParams: number; source: Exclude<ParamSource, "architecture"> };
  /** Context to check by default. Clamped to the architecture's maximum. */
  defaultCtx?: number;
}

export interface HfModel {
  model: ModelSpec;
  /** Where `model.totalParams` came from. */
  paramSource: ParamSource;
  /** Anything the reader had to decide, in the report's own words. */
  notes: string[];
}

const DEFAULT_CHECK_CONTEXT = 8192;

/**
 * How far a weight-file count may sit from the architecture's own before it
 * is treated as describing something else -- the same 5% the spec validator
 * allows, since that is the check the result has to pass anyway.
 *
 * The case this catches is real: a multimodal repository's weight files hold
 * the vision tower as well as the decoder, and charging its parameters to the
 * decoder would report a model that does not exist. The decoder's own count
 * is used instead, and the difference is reported rather than swallowed.
 */
const WEIGHT_COUNT_TOLERANCE = 0.05;

function slug(value: string, fallback: string): string {
  const cleaned = value
    .trim()
    .toLowerCase()
    // A repo id is "org/model"; the model half is the name people use.
    .replace(/^.*\//, "")
    .replaceAll(/[^a-z0-9.]+/g, "-")
    .replaceAll(/^-+|-+$/g, "");
  return cleaned === "" ? fallback : cleaned;
}

function displayName(config: Json, architecture: string): string {
  const repo = pickString(config, ["_name_or_path", "name_or_path"]);
  if (repo !== undefined) return repo.replace(/^.*\//, "");
  const architectures = config["architectures"];
  if (Array.isArray(architectures) && typeof architectures[0] === "string") {
    return architectures[0];
  }
  return `${architecture} model`;
}

/** Turn a parsed `config.json` into a validated `ModelSpec`. */
export function modelFromHfConfig(value: unknown, options: HfModelOptions = {}): HfModel {
  const root = asObject(value, options.origin ?? "config.json");
  const config = textConfigOf(root);

  const nLayers = requireNumber(config, LAYERS, "layer count");
  const hiddenSize = requireNumber(config, HIDDEN, "hidden size");
  const nHeads = requireNumber(config, HEADS, "attention head count");
  // Absent means multi-head attention, i.e. one KV head per query head. The
  // opposite assumption -- absent means 1 -- would undersize the cache by the
  // head count on every pre-GQA model.
  const nKvHeads = pickNumber(config, KV_HEADS) ?? nHeads;
  const headDim = pickNumber(config, HEAD_DIM) ?? Math.floor(hiddenSize / nHeads);
  const ffnHidden = requireNumber(config, FFN, "feed-forward width");
  const vocabSize = requireNumber(config, VOCAB, "vocabulary size");
  const maxCtx = pickNumber(config, MAX_CTX) ?? DEFAULT_CHECK_CONTEXT;

  const architecture = pickString(config, ["model_type"]) ?? pickString(root, ["model_type"]) ?? "unknown";
  const name = displayName(root, architecture);
  const mla = mlaFrom(config);
  // Absent means tied. `PretrainedConfig.__init__` defaults the field to true
  // and `to_diff_dict` omits any value equal to the default, so a checkpoint
  // that ties -- Gemma, Phi-3, several Qwen and StableLM releases -- ships a
  // config.json without the key at all. Reading that as "untied" charges a
  // second vocab x hidden matrix that is not on disk: +22.6% of parameters on
  // gemma-2-2b, which then fails the safetensors cross-check as well.
  const tiedDeclared = pickBoolean(config, TIED) ?? pickBoolean(root, TIED);

  const draft: ModelSpec = {
    id: slug(name, `hf-${architecture}`),
    name,
    aliases: [],
    totalParams: 1,
    activeParams: 1,
    nLayers,
    hiddenSize,
    nHeads,
    nKvHeads,
    // An MLA model's "head dim" is the reconstructed key width; the cache is
    // sized from the mla block, not from this.
    headDim: mla === null ? headDim : mla.qkNopeHeadDim + mla.qkRopeHeadDim,
    ffnHidden,
    vocabSize,
    tiedEmbeddings: tiedDeclared ?? true,
    attention: mla === null ? "gqa" : "mla",
    mla,
    moe: moeFrom(config, nLayers, ffnHidden),
    attentionWindow: attentionWindowFrom(config),
    maxCtx,
    defaultCtx: Math.min(maxCtx, Math.max(1, Math.floor(options.defaultCtx ?? DEFAULT_CHECK_CONTEXT))),
    source:
      options.origin === undefined
        ? "HuggingFace config.json"
        : `HuggingFace config.json at ${options.origin}`,
  };

  // The architecture is what the config actually describes, so it is the
  // fallback; a count read from the weight files is preferred because it
  // includes the norms and biases the architecture sum leaves out.
  const derived = deriveArchitecture(draft);
  const notes: string[] = [];
  if (tiedDeclared === undefined) {
    notes.push(
      `This config does not set tie_word_embeddings, which transformers defaults to true and writes out only when it is false. The output projection is therefore charged as the embedding table rather than as a second ${vocabSize} x ${hiddenSize} matrix; pass a spec with --model-json if the checkpoint really does carry both.`,
    );
  }
  let paramSource: ParamSource = "architecture";
  let totalParams = derived.derivedTotalParams;

  const weights = options.weights;
  if (weights !== undefined) {
    const drift = Math.abs(weights.totalParams - derived.derivedTotalParams) / derived.derivedTotalParams;
    if (drift <= WEIGHT_COUNT_TOLERANCE) {
      totalParams = weights.totalParams;
      paramSource = weights.source;
    } else {
      notes.push(
        `The weight files hold ${Math.round(weights.totalParams)} parameters but the text decoder in this config accounts for ${Math.round(derived.derivedTotalParams)} -- ${(drift * 100).toFixed(1)}% apart, which usually means the checkpoint carries a vision tower or another head as well. Sized from the decoder; the rest would occupy memory too if you loaded it.`,
      );
    }
  }

  draft.totalParams = totalParams;
  draft.activeParams = draft.moe === null ? totalParams : derived.activeMatmulParams;

  return { model: parseModelSpec(draft, options.origin ?? "config.json"), paramSource, notes };
}
