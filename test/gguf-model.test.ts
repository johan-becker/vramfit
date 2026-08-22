import { describe, expect, it } from "vitest";
import { getDevice } from "../src/db/index.js";
import { checkFit } from "../src/fit.js";
import { describeGguf, modelFromGguf, quantFromGguf } from "../src/gguf/model.js";
import { GgufError, readGgufHeader } from "../src/gguf/reader.js";
import { computeKvCacheBytes, computeWeightBytes } from "../src/memory.js";
import { SpecValidationError } from "../src/db/validate.js";
import {
  GgufBuilder,
  LLAMA_GGUF_SHAPE,
  RecordingSource,
  llamaGgufBuilder,
  mlaGgufBuilder,
  moeGgufBuilder,
  str,
  u32,
} from "./gguf-fixtures.js";

/**
 * Mapping a header onto a `ModelSpec` and a `QuantSpec`.
 *
 * The expected numbers below are computed by hand from the fixture's own
 * shapes, not copied from what the code returns. The Llama fixture is exactly
 * the shape of a Llama 3.1 8B, so its parameter count is the real one:
 *
 *   token_embd  4096 x 128256                       =   525,336,576
 *   output      4096 x 128256                       =   525,336,576
 *   per layer   4096x4096 + 2 x 4096x1024
 *               + 4096x4096 + 3 x 4096x14336
 *               + 2 x 4096 (norms)                  =   218,112,000
 *   x 32 layers                                     = 6,979,584,000
 *   output_norm 4096, rope_freqs 64                 =         4,160
 *                                                     -------------
 *                                                     8,030,261,312
 */

const LLAMA_TOTAL_PARAMS = 8_030_261_312;
/** Q4_K blocks at 144/256 bytes, Q6_K at 210/256, F32 norms at 4. */
const LLAMA_TENSOR_BYTES = 5_172_420_864;

function header(bytes: Uint8Array) {
  return readGgufHeader(new RecordingSource(bytes));
}

describe("modelFromGguf", () => {
  it("derives the architecture from the metadata block", () => {
    const model = modelFromGguf(header(llamaGgufBuilder().build()), { origin: "./llama.gguf" });

    expect(model.name).toBe("Meta Llama 3.1 8B Instruct");
    expect(model.id).toBe("meta-llama-3.1-8b-instruct");
    expect(model.nLayers).toBe(LLAMA_GGUF_SHAPE.nLayers);
    expect(model.hiddenSize).toBe(LLAMA_GGUF_SHAPE.hiddenSize);
    expect(model.nHeads).toBe(LLAMA_GGUF_SHAPE.nHeads);
    // The mistake this whole package exists to prevent, read straight off
    // the file: 8 KV heads, not 32.
    expect(model.nKvHeads).toBe(LLAMA_GGUF_SHAPE.nKvHeads);
    expect(model.headDim).toBe(LLAMA_GGUF_SHAPE.headDim);
    expect(model.ffnHidden).toBe(LLAMA_GGUF_SHAPE.ffnHidden);
    expect(model.vocabSize).toBe(LLAMA_GGUF_SHAPE.vocabSize);
    expect(model.maxCtx).toBe(LLAMA_GGUF_SHAPE.maxCtx);
    expect(model.attention).toBe("gqa");
    expect(model.moe).toBeNull();
    expect(model.source).toBe("GGUF header of ./llama.gguf");
  });

  it("counts parameters from the shape table rather than a headline", () => {
    const model = modelFromGguf(header(llamaGgufBuilder().build()));
    expect(model.totalParams).toBe(LLAMA_TOTAL_PARAMS);
    expect(model.activeParams).toBe(LLAMA_TOTAL_PARAMS);
  });

  it("reads a tied embedding table from the absence of output.weight", () => {
    const untied = modelFromGguf(header(llamaGgufBuilder().build()));
    expect(untied.tiedEmbeddings).toBe(false);

    const tied = modelFromGguf(header(llamaGgufBuilder({ tiedEmbeddings: true }).build()));
    expect(tied.tiedEmbeddings).toBe(true);
    // One matrix fewer on disk: exactly 4096 x 128256 parameters.
    expect(LLAMA_TOTAL_PARAMS - tied.totalParams).toBe(4096 * 128_256);
  });

  it("counts the vocabulary from the tokenizer when no vocab_size is written", () => {
    const model = modelFromGguf(
      header(llamaGgufBuilder({ omitVocabSize: true, tokenCount: 128_256 }).build()),
    );
    expect(model.vocabSize).toBe(128_256);
  });

  it("falls back to the token_embd shape when neither is present", () => {
    const model = modelFromGguf(header(llamaGgufBuilder({ omitVocabSize: true }).build()));
    expect(model.vocabSize).toBe(128_256);
  });

  it("reads expert geometry, and derives the active parameters the router implies", () => {
    const model = modelFromGguf(header(moeGgufBuilder().build()));

    expect(model.moe).not.toBeNull();
    expect(model.moe?.nExperts).toBe(8);
    expect(model.moe?.expertsPerToken).toBe(2);
    expect(model.moe?.expertFfnHidden).toBe(1024);
    expect(model.totalParams).toBe(55_071_232);
    // 4 layers x (attention 655,360 + router 4,096 + 2 of 8 experts
    // 3,149,824) + the 1,048,576-parameter logit matrix.
    expect(model.activeParams).toBe(16_269_312);
    expect(model.activeParams / model.totalParams).toBeLessThan(0.35);
  });

  it("reads multi-head latent attention, shared experts and leading dense layers", () => {
    const model = modelFromGguf(header(mlaGgufBuilder().build()));

    expect(model.attention).toBe("mla");
    expect(model.mla).toEqual({
      kvLoraRank: 128,
      qkNopeHeadDim: 32,
      qkRopeHeadDim: 16,
      vHeadDim: 32,
      qLoraRank: null,
    });
    // GGUF stores the shared experts' combined width; two of 256 makes 512.
    expect(model.moe?.nSharedExperts).toBe(2);
    expect(model.moe?.denseLayers).toBe(1);
    expect(model.moe?.denseFfnHidden).toBe(1024);
    expect(model.totalParams).toBe(17_346_560);
    // 4 attention layers, one dense FFN, and 2 routed + 2 shared experts on
    // each of the other three, plus the logit matrix.
    expect(model.activeParams).toBe(9_220_096);
  });

  it("reads interleaved sliding-window attention, and ignores a degenerate pattern", () => {
    const windowed = modelFromGguf(
      header(
        llamaGgufBuilder()
          .kv("llama.attention.sliding_window", u32(1024))
          .kv("llama.attention.sliding_window_pattern", u32(6))
          .build(),
      ),
    );
    expect(windowed.attentionWindow).toEqual({ windowSize: 1024, fullAttentionEvery: 6 });

    // A pattern of 1 means every layer is a full-attention layer, which is
    // not windowing at all.
    const flat = modelFromGguf(
      header(
        llamaGgufBuilder()
          .kv("llama.attention.sliding_window", u32(1024))
          .kv("llama.attention.sliding_window_pattern", u32(1))
          .build(),
      ),
    );
    expect(flat.attentionWindow).toBeNull();
  });

  it("windows every layer when the file states a window and no pattern", () => {
    // Interleaving is what the pattern key says, not what the window key
    // implies: guessing Gemma 2's alternation for a file that declares no
    // pattern halves the cache of a model with no full-attention layer.
    const described = describeGguf(
      header(llamaGgufBuilder().kv("llama.attention.sliding_window", u32(4096)).build()),
    );
    expect(described.model.attentionWindow).toEqual({ windowSize: 4096, fullAttentionEvery: null });

    const kv = computeKvCacheBytes(described.model, { ctx: 32_768 });
    expect(kv.windowedLayers).toBe(32);
    expect(kv.totalBytes).toBe(2 * 32 * 8 * 128 * 4096 * 2);
    expect(described.notes.join(" ")).toMatch(/every one of the 32 layers is sized as windowed/);
  });

  it("refuses a file whose metadata contradicts its own shape table", () => {
    // Halving the declared layer count leaves the tensor table describing
    // twice the parameters the architecture accounts for.
    const bytes = llamaGgufBuilder().kv("llama.block_count", u32(16)).build();
    expect(() => modelFromGguf(header(bytes))).toThrow(SpecValidationError);
    expect(() => modelFromGguf(header(bytes))).toThrow(/gguf.totalParams is 8030261312/);
  });

  it("names the metadata it cannot do without", () => {
    const noArchitecture = new GgufBuilder().tensor("token_embd.weight", [8, 16], 0).build();
    expect(() => modelFromGguf(header(noArchitecture))).toThrow(
      /no "general.architecture" string/,
    );

    const noLayers = new GgufBuilder()
      .kv("general.architecture", str("llama"))
      .tensor("token_embd.weight", [8, 16], 0)
      .build();
    expect(() => modelFromGguf(header(noLayers))).toThrow(
      /no numeric "llama.block_count" in its metadata/,
    );
  });
});

describe("quantFromGguf", () => {
  it("measures the three rates the file actually stores", () => {
    const quant = quantFromGguf(header(llamaGgufBuilder().build()));

    // token_embd is Q4_K: 144 bytes per 256 weights.
    expect(quant.tokenEmbedBits).toBe(4.5);
    // output.weight is Q6_K: 210 bytes per 256 weights.
    expect(quant.outputHeadBits).toBe(6.5625);
    // The blocks are a mix of Q4_K, Q6_K and F32 norms.
    expect(quant.bitsPerWeight).toBeCloseTo(5.096, 3);
    expect(quant.label).toBe("Q4_K_M");
    expect(quant.family).toBe("gguf-k");
    expect(quant.id).toBe("file");
  });

  it("agrees with the file it was measured from, to the byte", () => {
    const parsed = header(llamaGgufBuilder().build());
    const weights = computeWeightBytes(modelFromGguf(parsed), quantFromGguf(parsed));

    // This is the point of measuring rather than looking up: the prediction
    // and the file cannot disagree, because the prediction was read off the
    // file's own shape and type tables.
    expect(parsed.tensorBytes).toBe(LLAMA_TENSOR_BYTES);
    expect(weights.totalBytes).toBeCloseTo(LLAMA_TENSOR_BYTES, 0);
    expect(weights.effectiveBitsPerWeight).toBeCloseTo(5.153, 3);
  });

  it("keeps a tied model's single matrix at the promoted rate", () => {
    const parsed = header(llamaGgufBuilder({ tiedEmbeddings: true }).build());
    const quant = quantFromGguf(parsed);
    // With no output.weight the embedding table is the logit matrix, so the
    // head rate is the embedding rate rather than an invented promotion.
    expect(quant.outputHeadBits).toBe(quant.tokenEmbedBits);

    const weights = computeWeightBytes(modelFromGguf(parsed), quant);
    expect(weights.tokenEmbedBytes).toBe(0);
    expect(weights.totalBytes).toBeCloseTo(parsed.tensorBytes, 0);
  });

  it("falls back to the dominant tensor type when the file declares no mix", () => {
    const parsed = header(moeGgufBuilder().build());
    const quant = quantFromGguf(parsed);
    expect(quant.label).toBe("F16");
    expect(quant.family).toBe("float");
    expect(quant.bitsPerWeight).toBeCloseTo(16, 1);
  });

  it("refuses a file with nothing but vocabulary tensors to measure", () => {
    const bytes = new GgufBuilder()
      .kv("general.architecture", str("llama"))
      .tensor("token_embd.weight", [8, 16], 0)
      .build();
    expect(() => quantFromGguf(header(bytes))).toThrow(GgufError);
    expect(() => quantFromGguf(header(bytes))).toThrow(/no transformer-block tensors/);
  });
});

describe("describeGguf", () => {
  it("hands the rest of the package a model and a quantization it can check", () => {
    const described = describeGguf(header(llamaGgufBuilder().build()), {
      origin: "./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    });
    expect(described.architecture).toBe("llama");
    expect(described.fileType).toBe("Q4_K_M");

    const fit = checkFit(described.model, described.quant, getDevice("4090"), { ctx: 8192 });
    expect(fit.fits).toBe(true);
    // 4.82 GiB of weights and a 1 GiB cache on a 24 GiB card.
    expect(fit.footprint.weights.totalBytes).toBeCloseTo(LLAMA_TENSOR_BYTES, 0);
    expect(fit.footprint.kv.totalBytes).toBe(2 * 32 * 8 * 128 * 8192 * 2);
    expect(fit.maxContext).toBe(131_072);
  });
});
