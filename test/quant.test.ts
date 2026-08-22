import { describe, expect, it } from "vitest";
import {
  GGUF_QUANT_FAMILIES,
  KV_QUANTS,
  QUANTS,
  findQuant,
  getKvQuant,
  getQuant,
  quantizedBytes,
  quantsByQuality,
} from "../src/quant.js";

describe("weight quantization table", () => {
  it("uses effective bits per weight, not the nominal name", () => {
    // The whole point: "Q4_K_M" is not four bits. These are llama.cpp's
    // published figures, and every one of them is above its nominal width.
    expect(getQuant("Q4_K_M").bitsPerWeight).toBe(4.83);
    expect(getQuant("Q4_K_S").bitsPerWeight).toBe(4.57);
    expect(getQuant("Q5_K_M").bitsPerWeight).toBe(5.67);
    expect(getQuant("Q5_K_S").bitsPerWeight).toBe(5.52);
    expect(getQuant("Q6_K").bitsPerWeight).toBe(6.56);
    expect(getQuant("Q8_0").bitsPerWeight).toBe(8.5);
    expect(getQuant("Q3_K_M").bitsPerWeight).toBe(3.91);
    expect(getQuant("Q2_K").bitsPerWeight).toBe(3.35);
    expect(getQuant("Q4_0").bitsPerWeight).toBe(4.55);
    expect(getQuant("F16").bitsPerWeight).toBe(16);
    expect(getQuant("BF16").bitsPerWeight).toBe(16);
    expect(getQuant("MXFP4").bitsPerWeight).toBe(4.25);
    expect(getQuant("AWQ-4bit").bitsPerWeight).toBe(4.25);
    expect(getQuant("GPTQ-4bit").bitsPerWeight).toBe(4.25);
  });

  it("ranks legacy Q4_0 below Q4_K_S at effectively the same size", () => {
    // Q4_0 and Q4_K_S land within 0.5% of each other (4.55 vs 4.57 bpw), but
    // the k-quant is measurably better, so Q4_0 should never be recommended.
    const legacy = getQuant("q4_0");
    const kquant = getQuant("q4_k_s");
    expect(Math.abs(legacy.bitsPerWeight - kquant.bitsPerWeight)).toBeLessThan(0.05);
    expect(legacy.qualityRank).toBeLessThan(kquant.qualityRank);
  });

  it("promotes the logit matrix but not the embedding table for k-quants", () => {
    const q4km = getQuant("q4_k_m");
    expect(q4km.tokenEmbedBits).toBe(4.83);
    expect(q4km.outputHeadBits).toBe(6.56);
  });

  it("keeps both vocabulary tensors in FP16 for GPTQ/AWQ/MXFP4", () => {
    for (const id of ["gptq-4bit", "awq-4bit", "mxfp4"]) {
      const q = getQuant(id);
      expect(q.bitsPerWeight).toBe(4.25);
      expect(q.tokenEmbedBits).toBe(16);
      expect(q.outputHeadBits).toBe(16);
    }
  });

  it("resolves aliases and is case- and separator-insensitive", () => {
    expect(findQuant("q4_k_m")).toBe(findQuant("Q4-K-M"));
    expect(findQuant("Q4.K.M")).toBe(findQuant("q4_k_m"));
    expect(findQuant("fp16")).toBe(findQuant("f16"));
    expect(findQuant("q4")?.id).toBe("q4_k_m");
    expect(findQuant("q8")?.id).toBe("q8_0");
    expect(findQuant("int4")?.id).toBe("gptq-4bit");
    expect(findQuant("  Q6_K  ")?.id).toBe("q6_k");
  });

  it("throws a listing error for an unknown quant", () => {
    expect(() => getQuant("q3_k_xxl")).toThrow(/Unknown quantization/);
    expect(() => getQuant("q3_k_xxl")).toThrow(/Q4_K_M/);
  });

  it("orders quants best-first and can restrict to the GGUF ecosystem", () => {
    const all = quantsByQuality();
    expect(all[0]?.id).toBe("f16");
    expect(all.at(-1)?.id).toBe("q2_k");
    for (let i = 1; i < all.length; i++) {
      expect(all[i - 1]!.qualityRank).toBeGreaterThanOrEqual(all[i]!.qualityRank);
    }

    const gguf = quantsByQuality(GGUF_QUANT_FAMILIES);
    expect(gguf.map((q) => q.id)).not.toContain("awq-4bit");
    expect(gguf.map((q) => q.id)).not.toContain("gptq-4bit");
    expect(gguf.map((q) => q.id)).not.toContain("mxfp4");
    expect(gguf.map((q) => q.id)).toContain("q4_k_m");
  });

  it("has unique ids and no duplicate quality ranks", () => {
    const ids = QUANTS.map((q) => q.id);
    expect(new Set(ids).size).toBe(ids.length);
    const ranks = QUANTS.map((q) => q.qualityRank);
    expect(new Set(ranks).size).toBe(ranks.length);
  });
});

describe("KV cache element types", () => {
  it("derives bytes per element from the real block layouts", () => {
    // q8_0: 32 int8 values + one FP16 scale = 34 bytes / 32 values
    expect(getKvQuant("q8_0").bytesPerElement).toBeCloseTo(34 / 32, 10);
    // q5_1: 16B low nibbles + 4B high bits + FP16 d + FP16 m = 24 bytes
    expect(getKvQuant("q5_1").bytesPerElement).toBeCloseTo(24 / 32, 10);
    // q5_0: same without the minimum = 22 bytes
    expect(getKvQuant("q5_0").bytesPerElement).toBeCloseTo(22 / 32, 10);
    // q4_1: 16B nibbles + FP16 d + FP16 m = 20 bytes
    expect(getKvQuant("q4_1").bytesPerElement).toBeCloseTo(20 / 32, 10);
    // q4_0: 16B nibbles + FP16 d = 18 bytes
    expect(getKvQuant("q4_0").bytesPerElement).toBeCloseTo(18 / 32, 10);
    expect(getKvQuant("f16").bytesPerElement).toBe(2);
  });

  it("matches the exact ratios people rely on", () => {
    // q8_0 is a hair over half of f16, not exactly half -- the FP16 scale costs
    // 6.25% and that is the difference between fitting and not at 128k.
    expect(getKvQuant("q8_0").bytesPerElement / 2).toBeCloseTo(0.53125, 10);
    expect(getKvQuant("q4_0").bytesPerElement / 2).toBeCloseTo(0.28125, 10);
  });

  it("rejects an unknown cache type", () => {
    expect(() => getKvQuant("q2_k")).toThrow(/Unknown KV cache type/);
  });

  it("has unique ids", () => {
    const ids = KV_QUANTS.map((q) => q.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("quantizedBytes", () => {
  it("is params * bits / 8", () => {
    expect(quantizedBytes(8_000_000_000, 16)).toBe(16_000_000_000);
    expect(quantizedBytes(8_000_000_000, 8)).toBe(8_000_000_000);
    // 7B at Q4_K_M: 7e9 * 4.83 / 8
    expect(quantizedBytes(7_000_000_000, 4.83)).toBeCloseTo(4_226_250_000, 0);
  });
});
