import { describe, expect, it } from "vitest";
import { activeBlockFraction, deriveArchitecture } from "../src/architecture.js";
import {
  DEEPSEEK_V2_LITE,
  LLAMA_3_1_8B,
  MIXTRAL_8X7B,
  SWA_12B,
} from "./fixtures.js";

/** Relative difference, for "within N% of the published figure" assertions. */
function relative(actual: number, expected: number): number {
  return Math.abs(actual - expected) / expected;
}

describe("deriveArchitecture: dense GQA model", () => {
  const arch = deriveArchitecture(LLAMA_3_1_8B);

  it("counts attention weights with query heads for q/o and KV heads for k/v", () => {
    // q: 4096 * (32 * 128) = 16_777_216
    // k: 4096 * ( 8 * 128) =  4_194_304
    // v: 4096 * ( 8 * 128) =  4_194_304
    // o: (32 * 128) * 4096 = 16_777_216
    expect(arch.attnParamsPerLayer).toBe(41_943_040);
  });

  it("counts three matrices per gated FFN", () => {
    expect(arch.ffnParamsAllLayers).toBe(3 * 4096 * 14_336 * 32);
  });

  it("reconstructs the published 8.03B parameter count", () => {
    expect(relative(arch.derivedTotalParams, LLAMA_3_1_8B.totalParams)).toBeLessThan(0.01);
  });

  it("excludes the embedding lookup from the per-token matmul count", () => {
    // Reading a token embedding is a row gather, not a matmul; only the
    // output projection is multiplied.
    expect(arch.activeMatmulParams).toBe(arch.blockParams + arch.logitMatrixParams);
    expect(arch.activeMatmulParams).toBeLessThan(LLAMA_3_1_8B.totalParams);
  });

  it("reports every block as active for a dense model", () => {
    expect(activeBlockFraction(LLAMA_3_1_8B)).toBe(1);
  });
});

describe("deriveArchitecture: tied embeddings", () => {
  const arch = deriveArchitecture(SWA_12B);

  it("stores the embedding matrix once but still reads it for logits", () => {
    expect(arch.outputHeadParams).toBe(0);
    expect(arch.logitMatrixParams).toBe(arch.inputEmbedParams);
    expect(arch.inputEmbedParams).toBe(262_208 * 3840);
  });

  it("reconstructs the published parameter count", () => {
    expect(relative(arch.derivedTotalParams, SWA_12B.totalParams)).toBeLessThan(0.01);
  });
});

describe("deriveArchitecture: mixture of experts", () => {
  const arch = deriveArchitecture(MIXTRAL_8X7B);

  it("reconstructs Mixtral's 46.7B total", () => {
    expect(relative(arch.derivedTotalParams, MIXTRAL_8X7B.totalParams)).toBeLessThan(0.01);
  });

  it("reconstructs Mixtral's 12.9B active count", () => {
    // Published "active parameters" for Mixtral include the embedding table;
    // the matmul count does not, so the derived figure sits one embedding
    // table (131M) below 12.879B.
    const published = MIXTRAL_8X7B.activeParams;
    expect(arch.activeMatmulParams).toBeLessThanOrEqual(published);
    expect(arch.activeMatmulParams + arch.inputEmbedParams).toBeGreaterThanOrEqual(
      published * 0.99,
    );
  });

  it("activates 2 of 8 experts, so blocks are ~27% active", () => {
    // (attention + 2/8 of the experts) / (attention + all experts).
    expect(activeBlockFraction(MIXTRAL_8X7B)).toBeCloseTo(0.2717, 3);
  });

  it("separates total from active: memory follows one, speed the other", () => {
    expect(arch.blockParams).toBeGreaterThan(arch.activeBlockParams * 3);
  });
});

describe("deriveArchitecture: MLA + fine-grained MoE with shared experts", () => {
  const arch = deriveArchitecture(DEEPSEEK_V2_LITE);

  it("sizes MLA projections from the latent ranks", () => {
    // q  : 2048 * 16 * (128 + 64)      = 6_291_456   (no query LoRA in V2-Lite)
    // kv_a: 2048 * (512 + 64)          = 1_179_648
    // kv_b: 512 * 16 * (128 + 128)     = 2_097_152
    // o  : 16 * 128 * 2048             = 4_194_304
    expect(arch.attnParamsPerLayer).toBe(13_762_560);
  });

  it("reconstructs the published 15.7B total across 1 dense + 26 MoE layers", () => {
    expect(relative(arch.derivedTotalParams, DEEPSEEK_V2_LITE.totalParams)).toBeLessThan(0.01);
  });

  it("reconstructs the published 2.4B active count", () => {
    expect(relative(arch.activeMatmulParams, DEEPSEEK_V2_LITE.activeParams)).toBeLessThan(0.03);
  });

  it("keeps shared experts always on", () => {
    const withoutShared = deriveArchitecture({
      ...DEEPSEEK_V2_LITE,
      moe: { ...DEEPSEEK_V2_LITE.moe!, nSharedExperts: 0 },
    });
    expect(withoutShared.activeBlockParams).toBeLessThan(arch.activeBlockParams);
  });
});
