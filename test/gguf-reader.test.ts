import { describe, expect, it } from "vitest";
import { ggmlBitsPerWeight, findGgmlType, isGgufArray } from "../src/gguf/format.js";
import { GgufError, readGgufHeader } from "../src/gguf/reader.js";
import {
  GgufBuilder,
  LLAMA_GGUF_SHAPE,
  RecordingSource,
  arr,
  bool,
  f32,
  f64,
  i16,
  i32,
  i64,
  i8,
  llamaGgufBytes,
  str,
  u16,
  u32,
  u64,
  u8,
  type GgufFixtureValue,
} from "./gguf-fixtures.js";

/**
 * The reader, exercised against headers assembled byte by byte. No file in
 * this suite came off the network, and none of them has a data section: the
 * whole point of the reader is that it never asks for one.
 */

function source(bytes: Uint8Array, size?: number): RecordingSource {
  return new RecordingSource(bytes, size);
}

describe("readGgufHeader", () => {
  it("reads the magic, version and counts", () => {
    const bytes = new GgufBuilder()
      .kv("general.architecture", str("llama"))
      .tensor("token_embd.weight", [8, 16], 0)
      .build();

    const header = readGgufHeader(source(bytes));
    expect(header.version).toBe(3);
    expect(header.tensorCount).toBe(1);
    expect(header.metadataCount).toBe(1);
    expect(header.metadata.get("general.architecture")).toBe("llama");
  });

  it("reads v2 as well as v3 -- the layout is identical", () => {
    const bytes = new GgufBuilder()
      .kv("general.architecture", str("llama"))
      .tensor("token_embd.weight", [8, 16], 0)
      .build({ version: 2 });
    expect(readGgufHeader(source(bytes)).version).toBe(2);
  });

  it("round-trips every scalar metadata type", () => {
    const bytes = new GgufBuilder()
      .kv("a.u8", u8(200))
      .kv("a.i8", i8(-100))
      .kv("a.u16", u16(60_000))
      .kv("a.i16", i16(-30_000))
      .kv("a.u32", u32(4_000_000_000))
      .kv("a.i32", i32(-2_000_000_000))
      .kv("a.u64", u64(9_007_199_254_740_991))
      .kv("a.i64", i64(-9_007_199_254_740_991))
      .kv("a.f32", f32(0.5))
      .kv("a.f64", f64(0.1))
      .kv("a.bool", bool(true))
      .kv("a.string", str("hello"))
      .tensor("t", [32], 0)
      .build();

    const { metadata } = readGgufHeader(source(bytes));
    expect(metadata.get("a.u8")).toBe(200);
    expect(metadata.get("a.i8")).toBe(-100);
    expect(metadata.get("a.u16")).toBe(60_000);
    expect(metadata.get("a.i16")).toBe(-30_000);
    expect(metadata.get("a.u32")).toBe(4_000_000_000);
    expect(metadata.get("a.i32")).toBe(-2_000_000_000);
    expect(metadata.get("a.u64")).toBe(9_007_199_254_740_991);
    expect(metadata.get("a.i64")).toBe(-9_007_199_254_740_991);
    expect(metadata.get("a.f32")).toBe(0.5);
    expect(metadata.get("a.f64")).toBeCloseTo(0.1, 12);
    expect(metadata.get("a.bool")).toBe(true);
    expect(metadata.get("a.string")).toBe("hello");
  });

  it("reads arrays, including an array of arrays", () => {
    const bytes = new GgufBuilder()
      .kv("a.numbers", arr("uint32", [u32(1), u32(2), u32(3)]))
      .kv("a.words", arr("string", [str("alpha"), str("beta")]))
      .kv("a.nested", arr("array", [arr("uint32", [u32(7), u32(8)]), arr("uint32", [u32(9)])]))
      .tensor("t", [32], 0)
      .build();

    const { metadata } = readGgufHeader(source(bytes));
    const numbers = metadata.get("a.numbers");
    expect(isGgufArray(numbers) && numbers.values).toEqual([1, 2, 3]);
    const words = metadata.get("a.words");
    expect(isGgufArray(words) && words.values).toEqual(["alpha", "beta"]);

    const nested = metadata.get("a.nested");
    if (!isGgufArray(nested)) throw new Error("expected an array");
    expect(nested.length).toBe(2);
    const inner = nested.values[0];
    expect(isGgufArray(inner) && inner.values).toEqual([7, 8]);
  });

  it("counts a long array exactly without decoding it", () => {
    // A tokenizer's token list is the reason this matters: 128k strings that
    // nothing needs, whose length is the vocabulary size.
    const tokens = Array.from({ length: 5000 }, (_, index) => str(`token-${index}`));
    const bytes = new GgufBuilder()
      .kv("tokenizer.ggml.tokens", arr("string", tokens))
      .kv("after.the.array", u32(42))
      .tensor("t", [32], 0)
      .build();

    const { metadata } = readGgufHeader(source(bytes), { maxArrayValues: 4 });
    const list = metadata.get("tokenizer.ggml.tokens");
    if (!isGgufArray(list)) throw new Error("expected an array");
    expect(list.length).toBe(5000);
    expect(list.values).toEqual(["token-0", "token-1", "token-2", "token-3"]);
    expect(list.truncated).toBe(true);
    // The skip has to land exactly on the next key, or everything after a
    // tokenizer would be misread rather than missing.
    expect(metadata.get("after.the.array")).toBe(42);
  });

  it("skips a fixed-width array in one step and keeps its length", () => {
    const bytes = new GgufBuilder()
      .kv("tokenizer.ggml.token_type", arr("int32", Array.from({ length: 1000 }, () => i32(1))))
      .kv("after.the.array", str("still here"))
      .tensor("t", [32], 0)
      .build();

    const { metadata } = readGgufHeader(source(bytes), { maxArrayValues: 0 });
    const list = metadata.get("tokenizer.ggml.token_type");
    expect(isGgufArray(list) && list.length).toBe(1000);
    expect(metadata.get("after.the.array")).toBe("still here");
  });

  it("records the shape table with the byte cost of each tensor", () => {
    const bytes = new GgufBuilder()
      .kv("general.architecture", str("llama"))
      .tensor("blk.0.ffn_down.weight", [14_336, 4096], 14)
      .build();

    const header = readGgufHeader(source(bytes));
    const tensor = header.tensors[0];
    if (tensor === undefined) throw new Error("expected one tensor");
    expect(tensor.dims).toEqual([14_336, 4096]);
    expect(tensor.type.name).toBe("Q6_K");
    expect(tensor.elements).toBe(58_720_256);
    // 58720256 / 256 blocks x 210 bytes
    expect(tensor.bytes).toBe((58_720_256 / 256) * 210);
    expect(header.tensorBytes).toBe(tensor.bytes);
  });

  it("pads the data offset up to the declared alignment", () => {
    const aligned = readGgufHeader(
      source(new GgufBuilder().kv("general.alignment", u32(64)).tensor("t", [32], 0).build({ alignment: 64 })),
    );
    expect(aligned.alignment).toBe(64);
    expect(aligned.dataOffset % 64).toBe(0);
    expect(aligned.dataOffset).toBeGreaterThanOrEqual(aligned.headerBytes);

    const standard = readGgufHeader(source(new GgufBuilder().tensor("t", [32], 0).build()));
    expect(standard.alignment).toBe(32);
    expect(standard.dataOffset % 32).toBe(0);
  });
});

describe("readGgufHeader diagnostics", () => {
  it("names a file that is not GGUF at all", () => {
    const bytes = new TextEncoder().encode("This is a README, not a checkpoint.");
    expect(() => readGgufHeader(source(bytes))).toThrow(GgufError);
    expect(() => readGgufHeader(source(bytes))).toThrow(
      /not a GGUF file: it starts with "This" rather than "GGUF"/,
    );
  });

  it("recognises a big-endian file rather than reading it as garbage", () => {
    const bytes = new GgufBuilder().tensor("t", [32], 0).build({ magic: "FUGG" });
    expect(() => readGgufHeader(source(bytes))).toThrow(/big-endian GGUF file/);
  });

  it("refuses v1, whose lengths are 32-bit", () => {
    const bytes = new GgufBuilder().tensor("t", [32], 0).build({ version: 1 });
    expect(() => readGgufHeader(source(bytes))).toThrow(/GGUF v1 uses 32-bit lengths/);
  });

  it("refuses a version it has never seen", () => {
    const bytes = new GgufBuilder().tensor("t", [32], 0).build({ version: 9 });
    expect(() => readGgufHeader(source(bytes))).toThrow(/GGUF version 9 is not supported/);
  });

  it("reports a truncated file with the byte it ran out at", () => {
    const full = llamaGgufBytes();
    const cut = full.slice(0, 2048);
    expect(() => readGgufHeader(source(cut))).toThrow(GgufError);
    expect(() => readGgufHeader(source(cut))).toThrow(/truncated/);
  });

  it("refuses a shape table that claims more tensors than it carries", () => {
    // The count is a promise the rest of the file has to keep. Overstating it
    // walks the reader into the alignment padding, where the next "tensor" is
    // a zero-length name with no dimensions -- refused by name rather than
    // accepted as a tensor of no weights.
    const bytes = new GgufBuilder()
      .kv("general.architecture", str("llama"))
      .tensor("t", [32], 0)
      .build({ declaredTensorCount: 5 });
    expect(() => readGgufHeader(source(bytes))).toThrow(GgufError);
    expect(() => readGgufHeader(source(bytes))).toThrow(
      /declares 0 dimensions; ggml allows 1 to 4/,
    );
  });

  it("names a ggml type whose block layout it does not know", () => {
    const bytes = new GgufBuilder().tensor("t", [32], 250).build();
    expect(() => readGgufHeader(source(bytes))).toThrow(
      /tensor "t" has ggml type 250, which this build does not know the block layout of/,
    );
  });

  it("names a metadata value type that is not in the specification", () => {
    // Type 13 is one past float64, the last type the specification defines.
    const bytes = new GgufBuilder().kv("a.key", u32(1)).tensor("t", [32], 0).build();
    // The value type sits directly after the key; rewrite it in place.
    const patched = Uint8Array.from(bytes);
    const keyAt = bytes.indexOf(0x61); // "a" of "a.key"
    patched[keyAt + 5] = 13;
    expect(() => readGgufHeader(source(patched))).toThrow(/which is not a GGUF type/);
  });

  it("refuses a metadata string long enough to be an allocation attack", () => {
    const bytes = new GgufBuilder().kv("a.key", str("x".repeat(64))).tensor("t", [32], 0).build();
    expect(() => readGgufHeader(source(bytes), { maxStringBytes: 8 })).toThrow(
      /declares 64 bytes, past the 8-byte limit/,
    );
  });

  it("refuses a tensor count past its limit before allocating for it", () => {
    const bytes = new GgufBuilder().tensor("t", [32], 0).build({ declaredTensorCount: 9_000_000 });
    expect(() => readGgufHeader(source(bytes), { maxTensors: 1024 })).toThrow(
      /declares 9000000 tensors, past the 1024 limit/,
    );
  });
});

describe("readGgufHeader on a multi-gigabyte file", () => {
  it("reads the header and not the model", () => {
    const bytes = llamaGgufBytes();
    // What the real Q4_K_M conversion of this shape occupies on disk.
    const fileBytes = 5_172_420_864 + bytes.length;
    const recording = source(bytes, fileBytes);

    const header = readGgufHeader(recording, { chunkBytes: 4096 });

    expect(header.tensorCount).toBe(LLAMA_GGUF_SHAPE.nLayers * 9 + 4);
    expect(header.fileBytes).toBe(fileBytes);
    // Everything the reader touched is header, plus at most the tail of the
    // window it was in when the header ran out.
    expect(recording.highWaterMark).toBeLessThanOrEqual(header.dataOffset + 4096);
    expect(header.bytesRead).toBeLessThan(header.dataOffset + 4096);
    expect(header.bytesRead / fileBytes).toBeLessThan(0.0001);
  });

  it("refuses an array nested deeper than the limit, rather than overflowing", () => {
    // Every other bound in this reader is enforced; nesting was the one that
    // was not. Each level costs 12 bytes in the file and one stack frame to
    // walk it, so a small crafted header reached the stack limit and threw a
    // RangeError with no file name, no offset and no help line -- exactly
    // what "fail with a clear diagnostic" is supposed to prevent.
    const nest = (levels: number): GgufFixtureValue =>
      levels === 0 ? arr("uint32", [u32(1)]) : arr("array", [nest(levels - 1)]);

    const build = (levels: number): Uint8Array =>
      new GgufBuilder()
        .kv("general.architecture", str("llama"))
        .kv("a.deep", nest(levels))
        .tensor("token_embd.weight", [8, 16], 0)
        .build();

    const deep = build(200);
    expect(() => readGgufHeader(source(deep))).toThrow(GgufError);
    expect(() => readGgufHeader(source(deep))).toThrow(
      /metadata "a\.deep".* nests arrays more than 64 deep/,
    );
    // The walk that skips undecoded elements recurses the same way.
    expect(() => readGgufHeader(source(deep), { maxArrayValues: 0 })).toThrow(
      /nests arrays more than 64 deep/,
    );

    // Two levels is as deep as a real file goes, and still reads.
    const shallow = build(1);
    expect(readGgufHeader(source(shallow)).metadata.has("a.deep")).toBe(true);
    expect(() => readGgufHeader(source(shallow), { maxArrayDepth: 1 })).toThrow(
      /nests arrays more than 1 deep/,
    );
  });

  it("stays bounded even with a 128k-entry tokenizer in the header", () => {
    const bytes = llamaGgufBytes({ tokenCount: 128_256, omitVocabSize: true });
    const recording = source(bytes, 5_172_420_864 + bytes.length);
    const header = readGgufHeader(recording, { chunkBytes: 64 * 1024 });

    const tokens = header.metadata.get("tokenizer.ggml.tokens");
    expect(isGgufArray(tokens) && tokens.length).toBe(128_256);
    expect(isGgufArray(tokens) && tokens.values.length).toBe(64);
    // The token list is walked, not decoded: reading it costs its own bytes
    // and nothing else.
    expect(header.bytesRead).toBeLessThan(header.dataOffset + 64 * 1024);
  });
});

describe("ggml type table", () => {
  it("states the true bits per weight, not the ones in the name", () => {
    const q4k = findGgmlType(12);
    const q6k = findGgmlType(14);
    const mxfp4 = findGgmlType(39);
    if (!q4k || !q6k || !mxfp4) throw new Error("expected Q4_K, Q6_K and MXFP4");

    // 144 bytes per 256 weights, not 4 bits per weight.
    expect(ggmlBitsPerWeight(q4k)).toBe(4.5);
    expect(ggmlBitsPerWeight(q6k)).toBe(6.5625);
    // 32 FP4 values sharing one E8M0 byte: 17 bytes per 32 weights.
    expect(ggmlBitsPerWeight(mxfp4)).toBe(4.25);
  });

  it("has no id claimed twice", () => {
    const ids = new Set<number>();
    for (const type of [12, 14, 8, 1, 0, 30, 39]) {
      expect(findGgmlType(type)).toBeDefined();
      expect(ids.has(type)).toBe(false);
      ids.add(type);
    }
  });
});
