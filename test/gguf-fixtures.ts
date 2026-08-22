import { GGUF_TYPE_NAMES, type GgufTypeName } from "../src/gguf/format.js";
import type { ByteSource } from "../src/gguf/reader.js";

/**
 * GGUF fixtures built byte by byte.
 *
 * Nothing here downloads a model, and nothing here uses the reader to write
 * what the reader then reads back -- the writer below is an independent
 * implementation of the container layout, so a test that round-trips through
 * it is checking the format, not checking the reader against itself.
 *
 * `llamaGgufBytes` builds a header with the real tensor names, shapes and
 * quantization mix of a Llama-3.1-8B-shaped Q4_K_M conversion, which is 292
 * tensor entries and an exact 8,030,261,312 parameters. Only the header is
 * built; the multi-gigabyte data section is never materialised, which is the
 * same trick the reader relies on.
 */

class ByteWriter {
  private buffer = new Uint8Array(4096);
  private view = new DataView(this.buffer.buffer);
  private length = 0;

  private ensure(extra: number): number {
    if (this.length + extra > this.buffer.length) {
      let capacity = this.buffer.length * 2;
      while (capacity < this.length + extra) capacity *= 2;
      const grown = new Uint8Array(capacity);
      grown.set(this.buffer.subarray(0, this.length));
      this.buffer = grown;
      this.view = new DataView(grown.buffer);
    }
    const at = this.length;
    this.length += extra;
    return at;
  }

  // `ensure` may replace `this.view`, so the offset is taken first and the
  // view read afterwards -- writing `this.view.setUint8(this.ensure(1), v)`
  // would resolve the view before the buffer grew.
  u8(value: number): this {
    const at = this.ensure(1);
    this.view.setUint8(at, value);
    return this;
  }

  i8(value: number): this {
    const at = this.ensure(1);
    this.view.setInt8(at, value);
    return this;
  }

  u16(value: number): this {
    const at = this.ensure(2);
    this.view.setUint16(at, value, true);
    return this;
  }

  i16(value: number): this {
    const at = this.ensure(2);
    this.view.setInt16(at, value, true);
    return this;
  }

  u32(value: number): this {
    const at = this.ensure(4);
    this.view.setUint32(at, value, true);
    return this;
  }

  i32(value: number): this {
    const at = this.ensure(4);
    this.view.setInt32(at, value, true);
    return this;
  }

  f32(value: number): this {
    const at = this.ensure(4);
    this.view.setFloat32(at, value, true);
    return this;
  }

  f64(value: number): this {
    const at = this.ensure(8);
    this.view.setFloat64(at, value, true);
    return this;
  }

  u64(value: number | bigint): this {
    const at = this.ensure(8);
    this.view.setBigUint64(at, BigInt(value), true);
    return this;
  }

  i64(value: number | bigint): this {
    const at = this.ensure(8);
    this.view.setBigInt64(at, BigInt(value), true);
    return this;
  }

  raw(bytes: Uint8Array): this {
    const at = this.ensure(bytes.length);
    this.buffer.set(bytes, at);
    return this;
  }

  /** A length-prefixed UTF-8 string, the way GGUF v2/v3 writes one. */
  text(value: string): this {
    const encoded = new TextEncoder().encode(value);
    return this.u64(encoded.length).raw(encoded);
  }

  pad(to: number): this {
    const padding = (to - (this.length % to)) % to;
    if (padding > 0) this.ensure(padding);
    return this;
  }

  get size(): number {
    return this.length;
  }

  finish(): Uint8Array {
    return this.buffer.slice(0, this.length);
  }
}

export type GgufFixtureValue =
  | { type: Exclude<GgufTypeName, "array">; value: number | boolean | string }
  | { type: "array"; elementType: GgufTypeName; values: GgufFixtureValue[]; declaredLength?: number };

export const u8 = (value: number): GgufFixtureValue => ({ type: "uint8", value });
export const i8 = (value: number): GgufFixtureValue => ({ type: "int8", value });
export const u16 = (value: number): GgufFixtureValue => ({ type: "uint16", value });
export const i16 = (value: number): GgufFixtureValue => ({ type: "int16", value });
export const u32 = (value: number): GgufFixtureValue => ({ type: "uint32", value });
export const i32 = (value: number): GgufFixtureValue => ({ type: "int32", value });
export const f32 = (value: number): GgufFixtureValue => ({ type: "float32", value });
export const f64 = (value: number): GgufFixtureValue => ({ type: "float64", value });
export const u64 = (value: number): GgufFixtureValue => ({ type: "uint64", value });
export const i64 = (value: number): GgufFixtureValue => ({ type: "int64", value });
export const bool = (value: boolean): GgufFixtureValue => ({ type: "bool", value });
export const str = (value: string): GgufFixtureValue => ({ type: "string", value });
export const arr = (
  elementType: GgufTypeName,
  values: GgufFixtureValue[],
): GgufFixtureValue => ({ type: "array", elementType, values });

function writeValue(writer: ByteWriter, entry: GgufFixtureValue): void {
  if (entry.type === "array") {
    writer.u32(GGUF_TYPE_NAMES.indexOf(entry.elementType));
    writer.u64(entry.declaredLength ?? entry.values.length);
    for (const value of entry.values) writeValue(writer, value);
    return;
  }
  const value = entry.value;
  switch (entry.type) {
    case "uint8":
      writer.u8(value as number);
      return;
    case "int8":
      writer.i8(value as number);
      return;
    case "uint16":
      writer.u16(value as number);
      return;
    case "int16":
      writer.i16(value as number);
      return;
    case "uint32":
      writer.u32(value as number);
      return;
    case "int32":
      writer.i32(value as number);
      return;
    case "float32":
      writer.f32(value as number);
      return;
    case "float64":
      writer.f64(value as number);
      return;
    case "uint64":
      writer.u64(value as number);
      return;
    case "int64":
      writer.i64(value as number);
      return;
    case "bool":
      writer.u8(value === true ? 1 : 0);
      return;
    case "string":
      writer.text(value as string);
      return;
  }
}

export interface FixtureTensor {
  name: string;
  dims: number[];
  /** ggml type id: 0 F32, 1 F16, 8 Q8_0, 12 Q4_K, 14 Q6_K, 39 MXFP4. */
  type: number;
}

export interface GgufBuildOptions {
  /** Overrides the four magic bytes, for the "not a GGUF file" cases. */
  magic?: string;
  version?: number;
  /** Padding boundary for the data section. */
  alignment?: number;
  /** Written instead of the real counts, to build a deliberately wrong file. */
  declaredTensorCount?: number;
  declaredMetadataCount?: number;
  /** Cut the result to this many bytes, to build a truncated file. */
  truncateTo?: number;
}

/** A GGUF header assembled from metadata entries and a tensor shape table. */
export class GgufBuilder {
  private readonly entries: [string, GgufFixtureValue][] = [];
  private readonly tensors: FixtureTensor[] = [];

  kv(key: string, value: GgufFixtureValue): this {
    this.entries.push([key, value]);
    return this;
  }

  tensor(name: string, dims: number[], type: number): this {
    this.tensors.push({ name, dims, type });
    return this;
  }

  /** The header bytes, padded to the alignment the data section would start at. */
  build(options: GgufBuildOptions = {}): Uint8Array {
    const writer = new ByteWriter();
    const encoder = new TextEncoder();
    writer.raw(encoder.encode(options.magic ?? "GGUF"));
    writer.u32(options.version ?? 3);
    writer.u64(options.declaredTensorCount ?? this.tensors.length);
    writer.u64(options.declaredMetadataCount ?? this.entries.length);

    for (const [key, value] of this.entries) {
      writer.text(key);
      writer.u32(GGUF_TYPE_NAMES.indexOf(value.type));
      writeValue(writer, value);
    }

    for (const tensor of this.tensors) {
      writer.text(tensor.name);
      writer.u32(tensor.dims.length);
      for (const dim of tensor.dims) writer.u64(dim);
      writer.u32(tensor.type);
      writer.u64(0);
    }

    writer.pad(options.alignment ?? 32);
    const bytes = writer.finish();
    return options.truncateTo === undefined ? bytes : bytes.slice(0, options.truncateTo);
  }
}

/**
 * A `ByteSource` over a buffer that records what was asked for.
 *
 * `size` can be set far beyond the buffer, which is how a 5 GB checkpoint is
 * simulated without allocating one: reads past the buffer come back as zeros,
 * and a reader that stayed inside the header never asks for them.
 */
export class RecordingSource implements ByteSource {
  readonly reads: { offset: number; length: number }[] = [];
  bytesRead = 0;
  highWaterMark = 0;

  constructor(
    private readonly bytes: Uint8Array,
    readonly size: number = bytes.length,
  ) {}

  read(offset: number, length: number): Uint8Array {
    const available = Math.max(0, Math.min(length, this.size - offset));
    this.reads.push({ offset, length: available });
    this.bytesRead += available;
    this.highWaterMark = Math.max(this.highWaterMark, offset + available);
    const out = new Uint8Array(available);
    const fromBuffer = Math.max(0, Math.min(available, this.bytes.length - offset));
    if (fromBuffer > 0) out.set(this.bytes.subarray(offset, offset + fromBuffer));
    return out;
  }
}

/* -------------------------------------------------------------------------- */
/* A realistic Llama 3.1 8B header                                             */
/* -------------------------------------------------------------------------- */

const F32 = 0;
const Q4_K = 12;
const Q6_K = 14;

export const LLAMA_GGUF_SHAPE = {
  nLayers: 32,
  hiddenSize: 4096,
  nHeads: 32,
  nKvHeads: 8,
  headDim: 128,
  ffnHidden: 14_336,
  vocabSize: 128_256,
  maxCtx: 131_072,
} as const;

export interface LlamaGgufOptions {
  /** Leave `llama.vocab_size` out, so the reader has to count the tokens. */
  omitVocabSize?: boolean;
  /** Write this many tokenizer entries, to exercise the array skip path. */
  tokenCount?: number;
  /** Drop `output.weight`, i.e. tie the embedding table to the output head. */
  tiedEmbeddings?: boolean;
  version?: number;
}

/**
 * A Q4_K_M-style Llama 3.1 8B header: the same tensor names and shapes
 * llama.cpp writes, with attention V and the FFN down projection promoted to
 * Q6_K the way the `_M` mix promotes them. The promotion is applied to every
 * layer rather than to the subset llama.cpp picks, so the fixture is a little
 * wider than a real Q4_K_M -- it is a self-consistent file to measure, not a
 * reproduction of one particular conversion.
 */
export function llamaGgufBuilder(options: LlamaGgufOptions = {}): GgufBuilder {
  const shape = LLAMA_GGUF_SHAPE;
  const builder = new GgufBuilder();

  builder
    .kv("general.architecture", str("llama"))
    .kv("general.name", str("Meta Llama 3.1 8B Instruct"))
    .kv("general.file_type", u32(15))
    .kv("llama.block_count", u32(shape.nLayers))
    .kv("llama.context_length", u32(shape.maxCtx))
    .kv("llama.embedding_length", u32(shape.hiddenSize))
    .kv("llama.feed_forward_length", u32(shape.ffnHidden))
    .kv("llama.attention.head_count", u32(shape.nHeads))
    .kv("llama.attention.head_count_kv", u32(shape.nKvHeads))
    .kv("llama.attention.key_length", u32(shape.headDim))
    .kv("llama.attention.value_length", u32(shape.headDim))
    .kv("llama.attention.layer_norm_rms_epsilon", f32(1e-5))
    .kv("llama.rope.freq_base", f32(500_000))
    .kv("tokenizer.ggml.model", str("gpt2"));

  if (options.omitVocabSize !== true) {
    builder.kv("llama.vocab_size", u32(shape.vocabSize));
  }

  const tokenCount = options.tokenCount ?? 0;
  if (tokenCount > 0) {
    builder.kv(
      "tokenizer.ggml.tokens",
      arr(
        "string",
        Array.from({ length: tokenCount }, (_, index) => str(`t${index}`)),
      ),
    );
  }

  builder.tensor("token_embd.weight", [shape.hiddenSize, shape.vocabSize], Q4_K);
  for (let layer = 0; layer < shape.nLayers; layer++) {
    const kvWidth = shape.nKvHeads * shape.headDim;
    builder
      .tensor(`blk.${layer}.attn_norm.weight`, [shape.hiddenSize], F32)
      .tensor(`blk.${layer}.attn_q.weight`, [shape.hiddenSize, shape.hiddenSize], Q4_K)
      .tensor(`blk.${layer}.attn_k.weight`, [shape.hiddenSize, kvWidth], Q4_K)
      .tensor(`blk.${layer}.attn_v.weight`, [shape.hiddenSize, kvWidth], Q6_K)
      .tensor(`blk.${layer}.attn_output.weight`, [shape.hiddenSize, shape.hiddenSize], Q4_K)
      .tensor(`blk.${layer}.ffn_norm.weight`, [shape.hiddenSize], F32)
      .tensor(`blk.${layer}.ffn_gate.weight`, [shape.hiddenSize, shape.ffnHidden], Q4_K)
      .tensor(`blk.${layer}.ffn_up.weight`, [shape.hiddenSize, shape.ffnHidden], Q4_K)
      .tensor(`blk.${layer}.ffn_down.weight`, [shape.ffnHidden, shape.hiddenSize], Q6_K);
  }
  builder.tensor("output_norm.weight", [shape.hiddenSize], F32);
  if (options.tiedEmbeddings !== true) {
    builder.tensor("output.weight", [shape.hiddenSize, shape.vocabSize], Q6_K);
  }
  builder.tensor("rope_freqs.weight", [shape.headDim / 2], F32);

  return builder;
}

export function llamaGgufBytes(options: LlamaGgufOptions = {}): Uint8Array {
  return llamaGgufBuilder(options).build(
    options.version === undefined ? {} : { version: options.version },
  );
}
