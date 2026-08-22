import { describe, expect, it } from "vitest";
import { getModel } from "../src/db/index.js";
import { computeWeightBytes } from "../src/memory.js";
import { getQuant } from "../src/quant.js";

/**
 * The end-to-end check on the weight arithmetic: predicted file size versus
 * the size of the real GGUF you would download.
 *
 * Every expected figure below is the file size shown on the Hugging Face repo
 * for that quantization (the bartowski / lmstudio-community conversions,
 * quoted in decimal GB as the file listing shows them). They are the only
 * ground truth available offline, and they exercise the three things the
 * weight model gets right that a naive `params * bits / 8` does not:
 *
 *   1. GGUF k-quants are wider than their names (Q4_K_M is 4.83 bpw, not 4).
 *   2. The logit matrix is promoted to Q6_K by every k-quant mix.
 *   3. Tied embeddings are stored once, not twice.
 *
 * The naive calculation is 18% low on Llama 3.1 8B Q4_K_M; this one is inside
 * 2% on every dense model in the table, which is the tolerance asserted here.
 */

interface PublishedGguf {
  model: string;
  quant: string;
  /** File size as published, in decimal GB (1e9 bytes). */
  publishedGB: number;
  repo: string;
}

const DENSE_FILES: readonly PublishedGguf[] = [
  {
    model: "llama-3.2-1b",
    quant: "q4_k_m",
    publishedGB: 0.81,
    repo: "bartowski/Llama-3.2-1B-Instruct-GGUF",
  },
  {
    model: "llama-3.2-3b",
    quant: "q4_k_m",
    publishedGB: 2.02,
    repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
  },
  {
    model: "llama-3.1-8b",
    quant: "q4_k_m",
    publishedGB: 4.92,
    repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
  },
  {
    model: "llama-3.1-8b",
    quant: "q5_k_m",
    publishedGB: 5.73,
    repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
  },
  {
    model: "llama-3.1-8b",
    quant: "q6_k",
    publishedGB: 6.6,
    repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
  },
  {
    model: "llama-3.1-8b",
    quant: "q8_0",
    publishedGB: 8.54,
    repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
  },
  {
    model: "llama-3.3-70b",
    quant: "q4_k_m",
    publishedGB: 42.52,
    repo: "bartowski/Llama-3.3-70B-Instruct-GGUF",
  },
  {
    model: "mistral-7b",
    quant: "q4_k_m",
    publishedGB: 4.37,
    repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
  },
  {
    model: "mistral-7b",
    quant: "q8_0",
    publishedGB: 7.7,
    repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
  },
  {
    model: "qwen2.5-7b",
    quant: "q4_k_m",
    publishedGB: 4.68,
    repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
  },
  {
    model: "qwen2.5-32b",
    quant: "q4_k_m",
    publishedGB: 19.85,
    repo: "bartowski/Qwen2.5-32B-Instruct-GGUF",
  },
  {
    model: "gemma-2-9b",
    quant: "q4_k_m",
    publishedGB: 5.76,
    repo: "bartowski/gemma-2-9b-it-GGUF",
  },
  {
    model: "phi-4",
    quant: "q4_k_m",
    publishedGB: 9.05,
    repo: "bartowski/phi-4-GGUF",
  },
];

function predictedGB(modelId: string, quantId: string): number {
  return computeWeightBytes(getModel(modelId), getQuant(quantId)).totalBytes / 1e9;
}

describe("predicted GGUF file sizes", () => {
  for (const file of DENSE_FILES) {
    it(`is within 2% of ${file.repo} ${file.quant.toUpperCase()} (${file.publishedGB} GB)`, () => {
      const predicted = predictedGB(file.model, file.quant);
      const error = Math.abs(predicted - file.publishedGB) / file.publishedGB;
      expect(
        error,
        `predicted ${predicted.toFixed(2)} GB vs published ${file.publishedGB} GB`,
      ).toBeLessThan(0.02);
    });
  }

  it("beats the naive params * bits / 8 on every file in the table", () => {
    for (const file of DENSE_FILES) {
      const model = getModel(file.model);
      const quant = getQuant(file.quant);
      const nominalBits = Number.parseFloat(quant.label.replace(/^[a-z]+/i, "")) || 16;
      const naive = (model.totalParams * nominalBits) / 8 / 1e9;
      const naiveError = Math.abs(naive - file.publishedGB) / file.publishedGB;
      const modelledError =
        Math.abs(predictedGB(file.model, file.quant) - file.publishedGB) / file.publishedGB;
      expect(modelledError, `${file.model} ${file.quant}`).toBeLessThan(naiveError);
    }
  });

  it("would be 18% low on Llama 3.1 8B Q4_K_M with the nominal 4 bits", () => {
    const naive = (getModel("llama-3.1-8b").totalParams * 4) / 8 / 1e9;
    expect(naive).toBeCloseTo(4.015, 2);
    expect((4.92 - naive) / 4.92).toBeCloseTo(0.184, 2);
    expect(predictedGB("llama-3.1-8b", "q4_k_m")).toBeCloseTo(4.96, 1);
  });

  it("stores a tied embedding table once, at the promoted rate", () => {
    // Llama 3.2 3B ties its 128256 x 3072 embedding with the output head, so
    // the file carries one vocabulary matrix, quantized at Q6_K rather than at
    // the Q4_K_M body rate.
    const model = getModel("llama-3.2-3b");
    expect(model.tiedEmbeddings).toBe(true);

    const w = computeWeightBytes(model, getQuant("q4_k_m"));
    const oneMatrixAtQ6K = (128_256 * 3072 * 6.56) / 8;
    expect(w.embeddingBytes).toBeCloseTo(oneMatrixAtQ6K, 0);

    // Charging a second copy at the body rate would add 0.24 GB: 12% of a
    // 2.02 GB file, and the difference between "fits in 2 GB" and "does not".
    const secondCopy = (128_256 * 3072 * 4.83) / 8;
    expect(secondCopy).toBeGreaterThan(0.23e9);
    expect(secondCopy / w.totalBytes).toBeGreaterThan(0.11);
  });
});

describe("known limitation: MoE k-quant mixes", () => {
  it("overestimates Mixtral 8x7B Q4_K_M, and says so", () => {
    // llama.cpp quantizes the expert tensors of an MoE at a narrower mix than
    // the dense "_M" recipe implies, so Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf
    // is 26.44 GB where the dense rate predicts ~28.2 GB. Roughly 7% high, and
    // high is the safe direction for a "will it fit" tool -- but it is a real
    // limitation, so it is pinned here rather than left to be discovered.
    const predicted = predictedGB("mixtral-8x7b", "q4_k_m");
    const published = 26.44;
    expect(predicted).toBeGreaterThan(published);
    expect((predicted - published) / published).toBeLessThan(0.1);
  });

  it("is accurate for an MoE shipped in its native format", () => {
    // gpt-oss ships as MXFP4 from OpenAI rather than as a llama.cpp requant,
    // so the 4.25 bpw expert body plus BF16 vocabulary tensors is the actual
    // file layout. The 20B release is a ~12.8 GB download.
    expect(predictedGB("gpt-oss-20b", "mxfp4")).toBeCloseTo(12.8, 0);

    // And the headline claim for the 120B -- one 80 GB H100 holds it -- comes
    // out of the same arithmetic with room left for a KV cache.
    const oss120b = predictedGB("gpt-oss-120b", "mxfp4");
    expect(oss120b).toBeGreaterThan(58);
    expect(oss120b).toBeLessThan(70);
  });

  it("keeps the MXFP4 vocabulary tensors at BF16, which a flat rate would not", () => {
    // gpt-oss has a 201088-token vocabulary and untied embeddings: two
    // 201088 x 2880 BF16 matrices are 2.3 GB that a flat 4.25 bpw model
    // would price at 0.6 GB.
    const model = getModel("gpt-oss-20b");
    const w = computeWeightBytes(model, getQuant("mxfp4"));
    const flatRate = (model.totalParams * 4.25) / 8;
    expect(w.totalBytes - flatRate).toBeGreaterThan(1.7e9);
    expect(w.embeddingBytes).toBeCloseTo(2 * 201_088 * 2880 * 2, 0);
  });
});
