import type { KvQuantSpec, QuantSpec } from "./types.js";

/**
 * Quantization tables.
 *
 * The single most common source of wrong VRAM estimates is assuming a quant is
 * as wide as its name. "Q4_K_M" is not 4 bits per weight. GGUF k-quants store
 * super-blocks of 256 weights with two levels of scales and minimums, and the
 * "_M" and "_S" mixes deliberately promote some tensor types (attention V,
 * FFN down, output) to a wider quant. The published llama.cpp figures for a
 * 7B Llama are the ones used here:
 *
 *   Q2_K    2.63 GiB for 6.74B params  -> 3.35 bpw
 *   Q3_K_M  3.07 GiB                   -> 3.91 bpw
 *   Q4_0    3.56 GiB                   -> 4.55 bpw
 *   Q4_K_S  3.59 GiB                   -> 4.57 bpw
 *   Q4_K_M  3.80 GiB                   -> 4.83 bpw
 *   Q5_K_S  4.33 GiB                   -> 5.52 bpw
 *   Q5_K_M  4.45 GiB                   -> 5.67 bpw
 *   Q6_K    5.15 GiB                   -> 6.56 bpw
 *   Q8_0    6.67 GiB                   -> 8.50 bpw
 *
 * (Source: the size/perplexity table in llama.cpp's quantization docs and
 * examples/quantize/README.md. Using the nominal "4 bits" for Q4_K_M
 * underestimates a 70B model by about 7 GiB, which is exactly the margin
 * between "fits on a 48 GB card" and "does not".)
 *
 * The vocabulary tensors are the second half of the story, and they need two
 * separate rates:
 *
 *   - `tokenEmbedBits` -- the `token_embd` lookup table. GGUF k-quant mixes
 *     leave it at the base rate. GPTQ, AWQ and MXFP4 quantize only the linear
 *     layers and keep it in FP16/BF16.
 *   - `outputHeadBits`  -- the matrix that produces logits (`output.weight`,
 *     or `token_embd` again when embeddings are tied). llama.cpp promotes it
 *     to Q6_K in every k-quant mix, because quantizing the vocabulary
 *     projection costs far more quality than quantizing a block.
 *
 * For a 32k-vocab Mistral this is a rounding error. For a 262k-vocab Gemma 3
 * or a 201k-vocab gpt-oss the vocabulary tensors are billions of parameters
 * and collapsing the two rates into one is a multi-GiB mistake. Modelling them
 * separately is what brings the predictions in `test/gguf-sizes.test.ts` to
 * within ~1% of the real published file sizes.
 */
const QUANT_LIST: readonly QuantSpec[] = [
  {
    id: "f16",
    label: "F16",
    family: "float",
    bitsPerWeight: 16,
    tokenEmbedBits: 16,
    outputHeadBits: 16,
    qualityRank: 100,
    notes: "Unquantized half precision. The reference point.",
  },
  {
    id: "bf16",
    label: "BF16",
    family: "float",
    bitsPerWeight: 16,
    tokenEmbedBits: 16,
    outputHeadBits: 16,
    qualityRank: 99,
    notes: "Same size as F16; wider exponent, so it is the native training dtype for most modern models.",
  },
  {
    id: "q8_0",
    label: "Q8_0",
    family: "gguf-legacy",
    bitsPerWeight: 8.5,
    tokenEmbedBits: 8.5,
    outputHeadBits: 8.5,
    qualityRank: 90,
    notes: "Blocks of 32 int8 weights plus one FP16 scale: 34 bytes per 32 weights. Perplexity loss is under 0.01%.",
  },
  {
    id: "q6_k",
    label: "Q6_K",
    family: "gguf-k",
    bitsPerWeight: 6.56,
    tokenEmbedBits: 6.56,
    outputHeadBits: 6.56,
    qualityRank: 80,
    notes: "Practically lossless and the usual stopping point before F16.",
  },
  {
    id: "q5_k_m",
    label: "Q5_K_M",
    family: "gguf-k",
    bitsPerWeight: 5.67,
    tokenEmbedBits: 5.67,
    outputHeadBits: 6.56,
    qualityRank: 70,
    notes: "Q5_K body with attention V and FFN down promoted to Q6_K; output head at Q6_K.",
  },
  {
    id: "q5_k_s",
    label: "Q5_K_S",
    family: "gguf-k",
    bitsPerWeight: 5.52,
    tokenEmbedBits: 5.52,
    outputHeadBits: 6.56,
    qualityRank: 65,
    notes: "Uniform Q5_K body, output head at Q6_K.",
  },
  {
    id: "q4_k_m",
    label: "Q4_K_M",
    family: "gguf-k",
    bitsPerWeight: 4.83,
    tokenEmbedBits: 4.83,
    outputHeadBits: 6.56,
    qualityRank: 60,
    notes: "The default recommendation: best quality-per-byte in the GGUF lineup for most models.",
  },
  {
    id: "q4_k_s",
    label: "Q4_K_S",
    family: "gguf-k",
    bitsPerWeight: 4.57,
    tokenEmbedBits: 4.57,
    outputHeadBits: 6.56,
    qualityRank: 55,
    notes: "Uniform Q4_K body. About 5% smaller than Q4_K_M for a measurable quality drop.",
  },
  {
    id: "q4_0",
    label: "Q4_0",
    family: "gguf-legacy",
    bitsPerWeight: 4.55,
    tokenEmbedBits: 4.55,
    outputHeadBits: 6.56,
    qualityRank: 45,
    notes: "Legacy format: 32 weights in 18 bytes. Same size as Q4_K_S and measurably worse; kept for old hardware paths.",
  },
  {
    id: "mxfp4",
    label: "MXFP4",
    family: "mxfp4",
    bitsPerWeight: 4.25,
    tokenEmbedBits: 16,
    outputHeadBits: 16,
    qualityRank: 50,
    notes: "OCP microscaling FP4: 32 E2M1 values sharing one E8M0 scale = 4.25 bpw. Native format of gpt-oss.",
  },
  {
    id: "awq-4bit",
    label: "AWQ-4bit",
    family: "awq",
    bitsPerWeight: 4.25,
    tokenEmbedBits: 16,
    outputHeadBits: 16,
    qualityRank: 52,
    notes: "4-bit weights with group size 128 plus FP16 scales and zero points. Embeddings and lm_head stay FP16.",
  },
  {
    id: "gptq-4bit",
    label: "GPTQ-4bit",
    family: "gptq",
    bitsPerWeight: 4.25,
    tokenEmbedBits: 16,
    outputHeadBits: 16,
    qualityRank: 48,
    notes: "4-bit weights with group size 128 plus FP16 scales and int32 zero points. Embeddings and lm_head stay FP16.",
  },
  {
    id: "q3_k_m",
    label: "Q3_K_M",
    family: "gguf-k",
    bitsPerWeight: 3.91,
    tokenEmbedBits: 3.91,
    outputHeadBits: 5.67,
    qualityRank: 30,
    notes: "Noticeably degraded. Worth it only when the alternative is offloading to system RAM.",
  },
  {
    id: "q2_k",
    label: "Q2_K",
    family: "gguf-k",
    bitsPerWeight: 3.35,
    tokenEmbedBits: 3.35,
    outputHeadBits: 4.83,
    qualityRank: 10,
    notes: "Last resort. Small models fall apart; very large models survive it surprisingly well.",
  },
] as const;

/**
 * KV cache element types supported by llama.cpp (`--cache-type-k/-v`).
 * Every figure below is the exact block layout, not an approximation:
 *
 *   f16   2 bytes per element
 *   q8_0  32 int8 + 1 FP16 scale               = 34/32 = 1.0625
 *   q5_1  16B nibbles + 4B high bits + d + m   = 24/32 = 0.75
 *   q5_0  16B nibbles + 4B high bits + d       = 22/32 = 0.6875
 *   q4_1  16B nibbles + FP16 d + FP16 m        = 20/32 = 0.625
 *   q4_0  16B nibbles + FP16 d                 = 18/32 = 0.5625
 */
const KV_QUANT_LIST: readonly KvQuantSpec[] = [
  { id: "f16", bytesPerElement: 2, notes: "Default. No quality loss." },
  { id: "bf16", bytesPerElement: 2, notes: "Same size as f16." },
  {
    id: "q8_0",
    bytesPerElement: 1.0625,
    notes: "Halves the cache for a quality loss usually below measurement noise. The safe first move at long context.",
  },
  { id: "q5_1", bytesPerElement: 0.75, notes: "Middle ground; needs flash attention off on some backends." },
  { id: "q5_0", bytesPerElement: 0.6875, notes: "Slightly smaller than q5_1, slightly worse." },
  { id: "q4_1", bytesPerElement: 0.625, notes: "Aggressive. Degrades long-range recall." },
  {
    id: "q4_0",
    bytesPerElement: 0.5625,
    notes: "Most aggressive. Quantizing K hurts much more than quantizing V; prefer q8_0 for K and q4_0 for V.",
  },
] as const;

export const QUANTS: readonly QuantSpec[] = QUANT_LIST;
export const KV_QUANTS: readonly KvQuantSpec[] = KV_QUANT_LIST;

/** GGUF-loadable quants only, best quality first. Used by `vramfit best`. */
export const GGUF_QUANT_FAMILIES: ReadonlySet<string> = new Set(["float", "gguf-k", "gguf-legacy"]);

function normalizeQuantId(id: string): string {
  return id.trim().toLowerCase().replaceAll("-", "_").replaceAll(".", "_");
}

const QUANT_INDEX = new Map<string, QuantSpec>();
for (const q of QUANT_LIST) {
  QUANT_INDEX.set(normalizeQuantId(q.id), q);
  QUANT_INDEX.set(normalizeQuantId(q.label), q);
}
// A few spellings people actually type.
QUANT_INDEX.set("fp16", QUANT_INDEX.get("f16") as QuantSpec);
QUANT_INDEX.set("q4km", QUANT_INDEX.get("q4_k_m") as QuantSpec);
QUANT_INDEX.set("q5km", QUANT_INDEX.get("q5_k_m") as QuantSpec);
QUANT_INDEX.set("q4", QUANT_INDEX.get("q4_k_m") as QuantSpec);
QUANT_INDEX.set("q8", QUANT_INDEX.get("q8_0") as QuantSpec);
QUANT_INDEX.set("q6", QUANT_INDEX.get("q6_k") as QuantSpec);
QUANT_INDEX.set("int4", QUANT_INDEX.get("gptq_4bit") as QuantSpec);

const KV_QUANT_INDEX = new Map<string, KvQuantSpec>();
for (const q of KV_QUANT_LIST) {
  KV_QUANT_INDEX.set(normalizeQuantId(q.id), q);
}
KV_QUANT_INDEX.set("fp16", KV_QUANT_INDEX.get("f16") as KvQuantSpec);
KV_QUANT_INDEX.set("q8", KV_QUANT_INDEX.get("q8_0") as KvQuantSpec);
KV_QUANT_INDEX.set("q4", KV_QUANT_INDEX.get("q4_0") as KvQuantSpec);

/** Look up a weight quantization by id, label or common alias. */
export function findQuant(id: string): QuantSpec | undefined {
  return QUANT_INDEX.get(normalizeQuantId(id));
}

/** Look up a weight quantization, throwing a helpful error when unknown. */
export function getQuant(id: string): QuantSpec {
  const found = findQuant(id);
  if (!found) {
    throw new Error(
      `Unknown quantization "${id}". Known: ${QUANT_LIST.map((q) => q.label).join(", ")}`,
    );
  }
  return found;
}

export function findKvQuant(id: string): KvQuantSpec | undefined {
  return KV_QUANT_INDEX.get(normalizeQuantId(id));
}

export function getKvQuant(id: string): KvQuantSpec {
  const found = findKvQuant(id);
  if (!found) {
    throw new Error(
      `Unknown KV cache type "${id}". Known: ${KV_QUANT_LIST.map((q) => q.id).join(", ")}`,
    );
  }
  return found;
}

/** Quants sorted best-quality-first. */
export function quantsByQuality(family?: ReadonlySet<string>): QuantSpec[] {
  return QUANT_LIST.filter((q) => !family || family.has(q.family)).toSorted(
    (a, b) => b.qualityRank - a.qualityRank || b.bitsPerWeight - a.bitsPerWeight,
  );
}

/** bytes = params * bits / 8. The whole of rule 1. */
export function quantizedBytes(params: number, bitsPerWeight: number): number {
  return (params * bitsPerWeight) / 8;
}
