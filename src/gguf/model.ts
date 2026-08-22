import { deriveArchitecture } from "../architecture.js";
import { parseModelSpec } from "../db/validate.js";
import { QUANTS, findQuant } from "../quant.js";
import type {
  AttentionWindowSpec,
  MlaSpec,
  ModelSpec,
  MoeSpec,
  QuantFamily,
  QuantSpec,
} from "../types.js";
import { GGUF_FILE_TYPES, isGgufArray, type GgufValue } from "./format.js";
import { GgufError, type GgufHeader } from "./reader.js";

/**
 * Turning a GGUF header into the two things the arithmetic needs: a
 * `ModelSpec` and a `QuantSpec`.
 *
 * Both are *measured*, not looked up. The shape table gives the exact
 * parameter count -- every tensor, including the norms and biases a published
 * "8B" rounds away -- and the per-tensor ggml types give the exact bits per
 * weight the file actually stores, separately for the transformer blocks, the
 * token embedding table and the output head. That is strictly better than the
 * quantization table in `quant.ts`, which has to average over the mixes
 * llama.cpp might have used: here there is no averaging, because the file in
 * front of us has already made every one of those choices.
 *
 * The result goes through the same `parseModelSpec` validator as the bundled
 * database and as `--model-json`, so a file whose metadata contradicts its own
 * tensor table is refused with the field named rather than turned into a
 * confident wrong answer.
 */

/** The metadata keys are `{architecture}.{field}`, e.g. `llama.block_count`. */
function key(architecture: string, field: string): string {
  return `${architecture}.${field}`;
}

/**
 * A metadata number.
 *
 * A few architectures publish per-layer arrays where most publish a scalar --
 * `feed_forward_length` and `attention.head_count` both appear in both forms.
 * The leading element is used for those, because vramfit models one uniform
 * repeating block; a file whose layers genuinely differ is out of scope and
 * says so through the parameter-count reconciliation rather than silently.
 */
function optionalNumber(header: GgufHeader, name: string): number | undefined {
  const raw: GgufValue | undefined = header.metadata.get(name);
  if (raw === undefined) return undefined;
  if (typeof raw === "number") return raw;
  if (isGgufArray(raw)) {
    const first = raw.values[0];
    if (typeof first === "number") return first;
  }
  return undefined;
}

function requireNumber(header: GgufHeader, name: string): number {
  const value = optionalNumber(header, name);
  if (value === undefined) {
    throw new GgufError(`the file has no numeric "${name}" in its metadata`);
  }
  return value;
}

function optionalString(header: GgufHeader, name: string): string | undefined {
  const raw = header.metadata.get(name);
  return typeof raw === "string" ? raw : undefined;
}

function arrayLength(header: GgufHeader, name: string): number | undefined {
  const raw = header.metadata.get(name);
  return isGgufArray(raw) ? raw.length : undefined;
}

/** The architecture prefix every model-specific key is namespaced under. */
export function ggufArchitecture(header: GgufHeader): string {
  const architecture = optionalString(header, "general.architecture");
  if (architecture === undefined || architecture.trim() === "") {
    throw new GgufError('the file has no "general.architecture" string in its metadata');
  }
  return architecture;
}

/** The `general.file_type` mix, when it names one vramfit knows. */
export function ggufFileType(header: GgufHeader): string | undefined {
  const code = optionalNumber(header, "general.file_type");
  return code === undefined ? undefined : GGUF_FILE_TYPES[code];
}

/** A printable id: lower case, hyphens, nothing a terminal would interpret. */
function slug(value: string, fallback: string): string {
  const cleaned = value
    .trim()
    .toLowerCase()
    .replaceAll(/[^a-z0-9.]+/g, "-")
    .replaceAll(/^-+|-+$/g, "");
  return cleaned === "" ? fallback : cleaned;
}

/* -------------------------------------------------------------------------- */
/* Quantization                                                                */
/* -------------------------------------------------------------------------- */

const TOKEN_EMBED_TENSOR = "token_embd.weight";
const OUTPUT_HEAD_TENSOR = "output.weight";

interface TensorGroup {
  elements: number;
  bytes: number;
  /** ggml type names present, most bytes first. */
  types: string[];
}

interface TensorTally {
  elements: number;
  bytes: number;
  /** Bytes contributed by each ggml type, so the dominant one can be named. */
  byType: Map<string, number>;
}

function emptyTally(): TensorTally {
  return { elements: 0, bytes: 0, byType: new Map<string, number>() };
}

function finishTally(tally: TensorTally): TensorGroup {
  return {
    elements: tally.elements,
    bytes: tally.bytes,
    types: [...tally.byType.entries()].toSorted((a, b) => b[1] - a[1]).map(([name]) => name),
  };
}

function groupTensors(header: GgufHeader): {
  blocks: TensorGroup;
  tokenEmbed: TensorGroup;
  outputHead: TensorGroup;
} {
  const blocks = emptyTally();
  const tokenEmbed = emptyTally();
  const outputHead = emptyTally();

  for (const tensor of header.tensors) {
    const group =
      tensor.name === TOKEN_EMBED_TENSOR
        ? tokenEmbed
        : tensor.name === OUTPUT_HEAD_TENSOR
          ? outputHead
          : blocks;
    group.elements += tensor.elements;
    group.bytes += tensor.bytes;
    group.byType.set(tensor.type.name, (group.byType.get(tensor.type.name) ?? 0) + tensor.bytes);
  }

  return {
    blocks: finishTally(blocks),
    tokenEmbed: finishTally(tokenEmbed),
    outputHead: finishTally(outputHead),
  };
}

function quantFamilyOf(typeName: string): QuantFamily {
  if (typeName === "F32" || typeName === "F16" || typeName === "BF16" || typeName === "F64") {
    return "float";
  }
  if (typeName === "MXFP4") return "mxfp4";
  if (typeName.endsWith("_K") || typeName.startsWith("IQ") || typeName.startsWith("TQ")) {
    return "gguf-k";
  }
  return "gguf-legacy";
}

/**
 * Quality rank for a mix that is not in the table, interpolated in bits
 * between the two table entries it falls between. The table's ranks are not
 * linear in bits -- a k-quant outranks a legacy quant of the same width -- so
 * interpolating between neighbours keeps a measured file ordered sensibly
 * against the named ones instead of inventing a scale of its own.
 */
function interpolateQualityRank(bitsPerWeight: number): number {
  const ladder = QUANTS.map((quant) => ({
    bits: quant.bitsPerWeight,
    rank: quant.qualityRank,
  })).toSorted((a, b) => a.bits - b.bits);

  const first = ladder[0];
  const last = ladder.at(-1);
  if (first === undefined || last === undefined) return 0;
  if (bitsPerWeight <= first.bits) return first.rank;
  if (bitsPerWeight >= last.bits) return last.rank;

  for (let index = 1; index < ladder.length; index++) {
    const low = ladder[index - 1] as { bits: number; rank: number };
    const high = ladder[index] as { bits: number; rank: number };
    if (bitsPerWeight <= high.bits) {
      const span = high.bits - low.bits;
      const share = span === 0 ? 0 : (bitsPerWeight - low.bits) / span;
      return low.rank + share * (high.rank - low.rank);
    }
  }
  return last.rank;
}

/**
 * The quantization the file is actually in.
 *
 * Three separate rates, measured from the three groups of tensors that are
 * quantized differently, which is exactly the split `computeWeightBytes`
 * charges for:
 *
 *   bits_per_weight  = block bytes  * 8 / block weights
 *   tokenEmbedBits   = token_embd bytes * 8 / token_embd weights
 *   outputHeadBits   = output bytes * 8 / output weights, or token_embd's
 *                      rate when the model ties them
 *
 * Feed the result back into `computeWeightBytes` with the model this function's
 * neighbour derives and the total comes back to the byte -- which is the
 * point: the prediction and the file agree because the prediction was read
 * off the file.
 */
export function quantFromGguf(header: GgufHeader): QuantSpec {
  const { blocks, tokenEmbed, outputHead } = groupTensors(header);
  if (blocks.elements === 0) {
    throw new GgufError("the tensor table has no transformer-block tensors to measure");
  }

  const bitsPerWeight = (blocks.bytes * 8) / blocks.elements;
  const tokenEmbedBits =
    tokenEmbed.elements > 0 ? (tokenEmbed.bytes * 8) / tokenEmbed.elements : bitsPerWeight;
  const outputHeadBits =
    outputHead.elements > 0 ? (outputHead.bytes * 8) / outputHead.elements : tokenEmbedBits;

  const dominant = blocks.types[0] ?? "F16";
  const label = ggufFileType(header) ?? dominant;
  const named = findQuant(label);

  return {
    id: "file",
    label,
    family: quantFamilyOf(dominant),
    bitsPerWeight,
    tokenEmbedBits,
    outputHeadBits,
    qualityRank: named?.qualityRank ?? interpolateQualityRank(bitsPerWeight),
    notes: `Measured from the file itself: ${header.tensorCount} tensors, blocks at ${bitsPerWeight.toFixed(2)} bits/weight (${blocks.types.join(" + ")}), output head at ${outputHeadBits.toFixed(2)}.`,
  };
}

/* -------------------------------------------------------------------------- */
/* Model                                                                       */
/* -------------------------------------------------------------------------- */

export interface GgufModelOptions {
  /** What to record in `ModelSpec.source`; normally the path it came from. */
  origin?: string;
  /** Context to check by default. Clamped to the architecture's maximum. */
  defaultCtx?: number;
}

/** Context vramfit checks by default when the file does not suggest one. */
const DEFAULT_CHECK_CONTEXT = 8192;

function headDimFrom(header: GgufHeader, architecture: string, hidden: number, heads: number): number {
  const fallback = Math.floor(hidden / heads);
  const keyLength = optionalNumber(header, key(architecture, "attention.key_length")) ?? fallback;
  const valueLength = optionalNumber(header, key(architecture, "attention.value_length")) ?? keyLength;
  // K and V are cached together, so a model whose key and value widths differ
  // costs `key + value` elements per head per token -- which is what the
  // averaged width gives when the KV formula multiplies it back by two.
  const averaged = (keyLength + valueLength) / 2;
  if (!Number.isInteger(averaged)) {
    throw new GgufError(
      `key_length ${keyLength} and value_length ${valueLength} average to ${averaged}, which is not a whole number of elements`,
    );
  }
  return averaged;
}

function vocabSizeFrom(header: GgufHeader, architecture: string): number {
  const declared = optionalNumber(header, key(architecture, "vocab_size"));
  if (declared !== undefined) return declared;
  const counted = arrayLength(header, "tokenizer.ggml.tokens");
  if (counted !== undefined) return counted;
  const embedding = header.tensors.find((tensor) => tensor.name === TOKEN_EMBED_TENSOR);
  // token_embd is [hidden, vocab] in ggml's fastest-axis-first order.
  const fromTensor = embedding?.dims[1];
  if (fromTensor !== undefined) return fromTensor;
  throw new GgufError(
    `neither "${key(architecture, "vocab_size")}" nor "tokenizer.ggml.tokens" nor a ${TOKEN_EMBED_TENSOR} tensor is present, so the vocabulary size cannot be determined`,
  );
}

function moeFrom(
  header: GgufHeader,
  architecture: string,
  nLayers: number,
  ffnHidden: number,
): MoeSpec | null {
  const nExperts = optionalNumber(header, key(architecture, "expert_count")) ?? 0;
  if (nExperts <= 0) return null;

  const expertFfnHidden =
    optionalNumber(header, key(architecture, "expert_feed_forward_length")) ?? ffnHidden;
  // GGUF stores the shared experts' combined width rather than their count,
  // because llama.cpp fuses them into one wider FFN. Dividing recovers the
  // count DeepSeek and Qwen3 publish.
  const sharedWidth =
    optionalNumber(header, key(architecture, "expert_shared_feed_forward_length")) ?? 0;
  const nSharedExperts =
    expertFfnHidden > 0 ? Math.round(sharedWidth / expertFfnHidden) : 0;
  const denseLayers = Math.min(
    nLayers,
    optionalNumber(header, key(architecture, "leading_dense_block_count")) ?? 0,
  );

  return {
    nExperts,
    expertsPerToken: optionalNumber(header, key(architecture, "expert_used_count")) ?? 1,
    expertFfnHidden,
    nSharedExperts,
    denseLayers,
    denseFfnHidden: denseLayers > 0 ? ffnHidden : null,
  };
}

/**
 * Sliding-window attention.
 *
 * The interleaving is only modelled when the file states its period. A window
 * on its own means every layer is windowed, which is what the field means
 * everywhere it appears without a pattern beside it; guessing Gemma 2's
 * alternation for the rest halves the cache of a model that has no full
 * layers at all.
 */
function attentionWindowFrom(
  header: GgufHeader,
  architecture: string,
): AttentionWindowSpec | null {
  const windowSize = optionalNumber(header, key(architecture, "attention.sliding_window"));
  if (windowSize === undefined || windowSize <= 0) return null;
  const pattern = optionalNumber(header, key(architecture, "attention.sliding_window_pattern"));
  // A pattern of 1 means every layer is a full-attention layer, which is the
  // same thing as no windowing at all.
  if (pattern !== undefined && pattern < 2) return null;
  return { windowSize, fullAttentionEvery: pattern ?? null };
}

function mlaFrom(header: GgufHeader, architecture: string): MlaSpec | null {
  const kvLoraRank = optionalNumber(header, key(architecture, "attention.kv_lora_rank"));
  if (kvLoraRank === undefined || kvLoraRank <= 0) return null;

  const qkRopeHeadDim = optionalNumber(header, key(architecture, "rope.dimension_count"));
  const keyLength = optionalNumber(header, key(architecture, "attention.key_length"));
  const valueLength = optionalNumber(header, key(architecture, "attention.value_length"));
  if (qkRopeHeadDim === undefined || keyLength === undefined || valueLength === undefined) {
    throw new GgufError(
      `${architecture} declares attention.kv_lora_rank but not the rope.dimension_count, attention.key_length and attention.value_length that multi-head latent attention needs`,
    );
  }
  const qLoraRank = optionalNumber(header, key(architecture, "attention.q_lora_rank")) ?? 0;

  return {
    kvLoraRank,
    // key_length is the full head: the non-positional part plus the decoupled
    // RoPE key that is cached beside the latent.
    qkNopeHeadDim: keyLength - qkRopeHeadDim,
    qkRopeHeadDim,
    vHeadDim: valueLength,
    qLoraRank: qLoraRank > 0 ? qLoraRank : null,
  };
}

/**
 * The model the file describes.
 *
 * `totalParams` is summed from the shape table rather than read from a
 * metadata field, because no metadata field carries it: `general.parameter
 * _count` is optional, frequently absent and frequently the rounded headline
 * figure. Summing the tensors is exact and needs nothing but the header.
 */
export function modelFromGguf(header: GgufHeader, options: GgufModelOptions = {}): ModelSpec {
  const architecture = ggufArchitecture(header);

  const nLayers = requireNumber(header, key(architecture, "block_count"));
  const hiddenSize = requireNumber(header, key(architecture, "embedding_length"));
  const nHeads = requireNumber(header, key(architecture, "attention.head_count"));
  const nKvHeads = optionalNumber(header, key(architecture, "attention.head_count_kv")) ?? nHeads;
  const ffnHidden =
    optionalNumber(header, key(architecture, "feed_forward_length")) ?? hiddenSize * 4;
  const maxCtx =
    optionalNumber(header, key(architecture, "context_length")) ?? DEFAULT_CHECK_CONTEXT;

  const mla = mlaFrom(header, architecture);
  const headDim =
    mla === null
      ? headDimFrom(header, architecture, hiddenSize, nHeads)
      : (optionalNumber(header, key(architecture, "attention.key_length")) as number);

  const totalParams = header.tensors.reduce((sum, tensor) => sum + tensor.elements, 0);
  if (totalParams <= 0) {
    throw new GgufError("the tensor table is empty, so the file describes no weights");
  }

  const name =
    optionalString(header, "general.name") ??
    optionalString(header, "general.basename") ??
    `${architecture} model`;

  // Built as a `ModelSpec` rather than as loose data so the active-parameter
  // count can be derived from it before it is validated: the validator checks
  // `activeParams` against the routing the architecture implies, which is a
  // number only the architecture knows.
  const draft: ModelSpec = {
    id: slug(name, `gguf-${architecture}`),
    name,
    aliases: [],
    totalParams,
    activeParams: totalParams,
    nLayers,
    hiddenSize,
    nHeads,
    nKvHeads,
    headDim,
    ffnHidden,
    vocabSize: vocabSizeFrom(header, architecture),
    // llama.cpp writes no output projection when the model reuses the
    // embedding table for it, so the tensor's absence is the tie flag.
    tiedEmbeddings: !header.tensors.some((tensor) => tensor.name === OUTPUT_HEAD_TENSOR),
    attention: mla === null ? "gqa" : "mla",
    mla,
    moe: moeFrom(header, architecture, nLayers, ffnHidden),
    attentionWindow: attentionWindowFrom(header, architecture),
    maxCtx,
    defaultCtx: Math.min(maxCtx, Math.max(1, Math.floor(options.defaultCtx ?? DEFAULT_CHECK_CONTEXT))),
    source: options.origin === undefined ? "GGUF header" : `GGUF header of ${options.origin}`,
  };

  if (draft.moe !== null) {
    // Only some of the weights are read per token, and the router decides
    // which. The architecture knows the ratio; no metadata field states it.
    draft.activeParams = deriveArchitecture(draft).activeMatmulParams;
  }

  return parseModelSpec(draft, "gguf");
}

export interface GgufModel {
  header: GgufHeader;
  model: ModelSpec;
  /** The quantization measured from the file's own tensor table. */
  quant: QuantSpec;
  architecture: string;
  /** `general.file_type`'s name, when the file declares a known one. */
  fileType: string | undefined;
  /** Anything the reader had to decide, in the report's own words. */
  notes: string[];
}

/** Anything about the file that the report should say out loud. */
function ggufNotes(model: ModelSpec): string[] {
  const notes: string[] = [];
  const window = model.attentionWindow;
  if (window !== null && window.fullAttentionEvery === null) {
    notes.push(
      `This file declares attention.sliding_window ${window.windowSize} and no attention.sliding_window_pattern, so every one of the ${model.nLayers} layers is sized as windowed. llama.cpp does not implement sliding-window attention for every architecture and may allocate the full context on all of them instead.`,
    );
  }
  return notes;
}

/** Everything vramfit reads out of one GGUF file, in one call. */
export function describeGguf(header: GgufHeader, options: GgufModelOptions = {}): GgufModel {
  const model = modelFromGguf(header, options);
  return {
    header,
    model,
    quant: quantFromGguf(header),
    architecture: ggufArchitecture(header),
    fileType: ggufFileType(header),
    notes: ggufNotes(model),
  };
}
