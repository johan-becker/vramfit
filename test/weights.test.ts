import { describe, expect, it } from "vitest";
import { deriveArchitecture } from "../src/architecture.js";
import { computeWeightBytes } from "../src/memory.js";
import { getQuant } from "../src/quant.js";
import { LLAMA_3_1_8B, MIXTRAL_8X7B, SWA_12B } from "./fixtures.js";

describe("weight bytes", () => {
  it("matches the hand-computed Llama 3.1 8B Q4_K_M breakdown", () => {
    const inputEmbed = 128_256 * 4096; // 525_336_576
    const blockParams = 8_030_000_000 - 2 * inputEmbed;

    const expectedBlock = (blockParams * 4.83) / 8; // 4_213_768_584
    const expectedEmbed = (inputEmbed * 4.83) / 8; //   317_171_958
    const expectedLogit = (inputEmbed * 6.56) / 8; //   430_775_992

    const w = computeWeightBytes(LLAMA_3_1_8B, getQuant("q4_k_m"));
    expect(w.blockBytes).toBeCloseTo(expectedBlock, 0);
    expect(w.embeddingBytes).toBeCloseTo(expectedEmbed + expectedLogit, 0);
    expect(w.totalBytes).toBeCloseTo(expectedBlock + expectedEmbed + expectedLogit, 0);
    expect(w.totalBytes / 1e9).toBeCloseTo(4.962, 2);
  });

  it("reports an effective bits-per-weight above the body rate", () => {
    const w = computeWeightBytes(LLAMA_3_1_8B, getQuant("q4_k_m"));
    // The Q6_K output head drags the file average above 4.83 bpw.
    expect(w.effectiveBitsPerWeight).toBeGreaterThan(4.83);
    expect(w.effectiveBitsPerWeight).toBeCloseTo(4.944, 2);
  });

  it("would be 18% low if the nominal '4 bits' were used", () => {
    const real = computeWeightBytes(LLAMA_3_1_8B, getQuant("q4_k_m")).totalBytes;
    const naive = (8_030_000_000 * 4) / 8;
    expect(naive / real).toBeLessThan(0.83);
  });

  it("scales exactly with bits per weight for the block portion", () => {
    const q8 = computeWeightBytes(LLAMA_3_1_8B, getQuant("q8_0"));
    const f16 = computeWeightBytes(LLAMA_3_1_8B, getQuant("f16"));
    expect(f16.blockBytes / q8.blockBytes).toBeCloseTo(16 / 8.5, 10);
    // F16 has no promotion anywhere, so the file is exactly 2 bytes per param.
    expect(f16.totalBytes).toBeCloseTo(8_030_000_000 * 2, 0);
  });

  it("stores a tied embedding matrix once, at the promoted rate", () => {
    const w = computeWeightBytes(SWA_12B, getQuant("q4_k_m"));
    const arch = deriveArchitecture(SWA_12B);
    expect(w.embeddingBytes).toBeCloseTo((arch.inputEmbedParams * 6.56) / 8, 0);

    // The same model with untied embeddings carries a second matrix.
    const untied = computeWeightBytes({ ...SWA_12B, tiedEmbeddings: false }, getQuant("q4_k_m"));
    expect(untied.embeddingBytes).toBeGreaterThan(w.embeddingBytes * 1.7);
  });

  it("keeps vocabulary tensors in FP16 for MXFP4, which dominates gpt-oss-like models", () => {
    const bigVocab = { ...LLAMA_3_1_8B, vocabSize: 201_088, totalParams: 20_910_000_000 };
    const mxfp4 = computeWeightBytes(bigVocab, getQuant("mxfp4"));
    const asIfEverythingWere4Bit = (20_910_000_000 * 4.25) / 8;
    // The two BF16 vocabulary matrices add over 2 GB on their own.
    expect(mxfp4.totalBytes - asIfEverythingWere4Bit).toBeGreaterThan(2e9);
  });

  it("divides block bytes evenly across layers for the offload split", () => {
    const w = computeWeightBytes(LLAMA_3_1_8B, getQuant("q4_k_m"));
    expect(w.bytesPerLayer * 32).toBeCloseTo(w.blockBytes, 0);
  });
});

describe("active weight bytes (what decode actually reads)", () => {
  it("equals blocks plus the logit matrix for a dense model", () => {
    const w = computeWeightBytes(LLAMA_3_1_8B, getQuant("q4_k_m"));
    const arch = deriveArchitecture(LLAMA_3_1_8B);
    expect(w.activeBytes).toBeCloseTo(
      w.blockBytes + (arch.logitMatrixParams * 6.56) / 8,
      0,
    );
    // It is below the file size: the embedding table is never streamed.
    expect(w.activeBytes).toBeLessThan(w.totalBytes);
  });

  it("reads only the routed experts for a mixture of experts", () => {
    const w = computeWeightBytes(MIXTRAL_8X7B, getQuant("q4_k_m"));
    // Mixtral: 46.7B in memory, ~12.7B multiplied per token. Memory follows
    // the total, speed follows the active share.
    expect(w.totalBytes / 1e9).toBeCloseTo(28.23, 1);
    expect(w.activeBytes / w.totalBytes).toBeCloseTo(0.2737, 3);
    expect(w.activeBytes).toBeLessThan(w.totalBytes / 3);
  });

  it("never lets active bytes exceed total bytes", () => {
    for (const model of [LLAMA_3_1_8B, MIXTRAL_8X7B, SWA_12B]) {
      for (const quant of ["f16", "q8_0", "q4_k_m", "q2_k"]) {
        const w = computeWeightBytes(model, getQuant(quant));
        expect(w.activeBytes).toBeLessThanOrEqual(w.totalBytes);
      }
    }
  });
});
