/**
 * The GGUF container format, as far as vramfit needs to know it.
 *
 * A GGUF file is a header followed by tensor data:
 *
 *   magic "GGUF" | version u32 | tensor_count u64 | metadata_count u64
 *   metadata_count x (key string, value_type u32, value)
 *   tensor_count   x (name string, n_dims u32, dims u64[n_dims], type u32, offset u64)
 *   padding to `general.alignment`
 *   tensor data
 *
 * Everything this package wants -- layer count, head counts, head widths,
 * vocabulary size, expert geometry, the quantization each tensor is stored in
 * and the exact parameter count implied by the shape table -- lives in the
 * header. The data section, which is the multi-gigabyte part, is never read.
 *
 * The layout above is GGUF v2 and v3. v1 wrote 32-bit lengths and counts
 * everywhere v2 writes 64-bit ones; no converter has emitted it since 2023 and
 * reading it with the v2 rules would silently produce nonsense, so it is
 * refused by name rather than guessed at.
 */

/** "GGUF" as the four bytes a little-endian writer puts on disk. */
export const GGUF_MAGIC = "GGUF";
/** The same four bytes seen through a big-endian writer. */
export const GGUF_MAGIC_BYTESWAPPED = "FUGG";

/** Versions this reader understands. */
export const SUPPORTED_GGUF_VERSIONS: readonly number[] = [2, 3];

/** Tensor data starts at a multiple of this unless `general.alignment` says otherwise. */
export const DEFAULT_GGUF_ALIGNMENT = 32;

/**
 * Metadata value types, indexed by the number the specification gives them.
 * The order is the specification's and must not be rearranged.
 */
export const GGUF_TYPE_NAMES = [
  "uint8",
  "int8",
  "uint16",
  "int16",
  "uint32",
  "int32",
  "float32",
  "bool",
  "string",
  "array",
  "uint64",
  "int64",
  "float64",
] as const;

export type GgufTypeName = (typeof GGUF_TYPE_NAMES)[number];

/** Width of every fixed-size metadata type; `string` and `array` have none. */
export const GGUF_FIXED_WIDTHS: Readonly<Partial<Record<GgufTypeName, number>>> = {
  uint8: 1,
  int8: 1,
  uint16: 2,
  int16: 2,
  uint32: 4,
  int32: 4,
  float32: 4,
  bool: 1,
  uint64: 8,
  int64: 8,
  float64: 8,
};

export function ggufTypeName(code: number): GgufTypeName | undefined {
  return GGUF_TYPE_NAMES[code];
}

/**
 * An array metadata value.
 *
 * `values` is capped: `tokenizer.ggml.tokens` is a 128k-entry array of strings
 * in every Llama 3 file and nothing here needs its contents -- only its
 * length, which is the vocabulary size. The elements past the cap are walked
 * and skipped rather than decoded, so the length is always exact.
 */
export interface GgufArray {
  readonly elementType: GgufTypeName;
  /** Elements the file declares. Exact even when `values` is truncated. */
  readonly length: number;
  /** The leading elements, up to the reader's cap. */
  readonly values: readonly GgufValue[];
  /** True when `values.length < length`. */
  readonly truncated: boolean;
}

export type GgufValue = number | boolean | string | GgufArray;

export function isGgufArray(value: GgufValue | undefined): value is GgufArray {
  return typeof value === "object" && value !== null;
}

/**
 * A ggml tensor type: how many weights share one block, and how many bytes
 * that block occupies. `typeSize * 8 / blockSize` is the format's true bits
 * per weight -- 144 * 8 / 256 = 4.5 for Q4_K, not the "4" in the name.
 *
 * The table is ggml's own `type_traits`. Numbering has holes where formats
 * were removed (Q4_2, Q4_3 and the repacked Q4_0_N_M variants); an id that is
 * not listed is reported by number rather than guessed at, because guessing a
 * block size is a wrong file size rather than a missing one.
 */
export interface GgmlTypeSpec {
  readonly id: number;
  readonly name: string;
  readonly blockSize: number;
  readonly typeSize: number;
}

const GGML_TYPE_LIST: readonly GgmlTypeSpec[] = [
  { id: 0, name: "F32", blockSize: 1, typeSize: 4 },
  { id: 1, name: "F16", blockSize: 1, typeSize: 2 },
  { id: 2, name: "Q4_0", blockSize: 32, typeSize: 18 },
  { id: 3, name: "Q4_1", blockSize: 32, typeSize: 20 },
  { id: 6, name: "Q5_0", blockSize: 32, typeSize: 22 },
  { id: 7, name: "Q5_1", blockSize: 32, typeSize: 24 },
  { id: 8, name: "Q8_0", blockSize: 32, typeSize: 34 },
  { id: 9, name: "Q8_1", blockSize: 32, typeSize: 36 },
  { id: 10, name: "Q2_K", blockSize: 256, typeSize: 84 },
  { id: 11, name: "Q3_K", blockSize: 256, typeSize: 110 },
  { id: 12, name: "Q4_K", blockSize: 256, typeSize: 144 },
  { id: 13, name: "Q5_K", blockSize: 256, typeSize: 176 },
  { id: 14, name: "Q6_K", blockSize: 256, typeSize: 210 },
  { id: 15, name: "Q8_K", blockSize: 256, typeSize: 292 },
  { id: 16, name: "IQ2_XXS", blockSize: 256, typeSize: 66 },
  { id: 17, name: "IQ2_XS", blockSize: 256, typeSize: 74 },
  { id: 18, name: "IQ3_XXS", blockSize: 256, typeSize: 98 },
  { id: 19, name: "IQ1_S", blockSize: 256, typeSize: 50 },
  { id: 20, name: "IQ4_NL", blockSize: 32, typeSize: 18 },
  { id: 21, name: "IQ3_S", blockSize: 256, typeSize: 110 },
  { id: 22, name: "IQ2_S", blockSize: 256, typeSize: 82 },
  { id: 23, name: "IQ4_XS", blockSize: 256, typeSize: 136 },
  { id: 24, name: "I8", blockSize: 1, typeSize: 1 },
  { id: 25, name: "I16", blockSize: 1, typeSize: 2 },
  { id: 26, name: "I32", blockSize: 1, typeSize: 4 },
  { id: 27, name: "I64", blockSize: 1, typeSize: 8 },
  { id: 28, name: "F64", blockSize: 1, typeSize: 8 },
  { id: 29, name: "IQ1_M", blockSize: 256, typeSize: 56 },
  { id: 30, name: "BF16", blockSize: 1, typeSize: 2 },
  { id: 34, name: "TQ1_0", blockSize: 256, typeSize: 54 },
  { id: 35, name: "TQ2_0", blockSize: 256, typeSize: 66 },
  { id: 39, name: "MXFP4", blockSize: 32, typeSize: 17 },
] as const;

export const GGML_TYPES: readonly GgmlTypeSpec[] = GGML_TYPE_LIST;

const GGML_TYPE_INDEX = new Map<number, GgmlTypeSpec>(
  GGML_TYPE_LIST.map((type) => [type.id, type]),
);

export function findGgmlType(id: number): GgmlTypeSpec | undefined {
  return GGML_TYPE_INDEX.get(id);
}

/** Bits per weight a ggml type stores, scales and minimums included. */
export function ggmlBitsPerWeight(type: GgmlTypeSpec): number {
  return (type.typeSize * 8) / type.blockSize;
}

/** Bytes a tensor of `elements` weights occupies, rounded up to whole blocks. */
export function ggmlTensorBytes(type: GgmlTypeSpec, elements: number): number {
  return Math.ceil(elements / type.blockSize) * type.typeSize;
}

/**
 * `general.file_type`, llama.cpp's `llama_ftype` enum: the mix a file was
 * quantized with. Only the mixes that name a quantization vramfit models are
 * listed; anything else falls back to measuring the tensor table, which is
 * more reliable anyway and is what the reported bits per weight come from.
 */
export const GGUF_FILE_TYPES: Readonly<Record<number, string>> = {
  0: "F32",
  1: "F16",
  2: "Q4_0",
  3: "Q4_1",
  7: "Q8_0",
  8: "Q5_0",
  9: "Q5_1",
  10: "Q2_K",
  11: "Q3_K_S",
  12: "Q3_K_M",
  13: "Q3_K_L",
  14: "Q4_K_S",
  15: "Q4_K_M",
  16: "Q5_K_S",
  17: "Q5_K_M",
  18: "Q6_K",
  32: "BF16",
  38: "MXFP4",
};
