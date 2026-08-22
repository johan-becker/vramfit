import { describe, expect, it } from "vitest";
import {
  RUNTIME_CONTEXT_BYTES,
  computeActivationBytes,
  computeFootprint,
} from "../src/memory.js";
import { getQuant } from "../src/quant.js";
import { GIB, MIB } from "../src/units.js";
import { LLAMA_3_1_8B, MIXTRAL_8X7B } from "./fixtures.js";

describe("runtime context overhead", () => {
  it("stays inside the documented 0.2-1.0 GiB band for every family", () => {
    for (const [family, bytes] of Object.entries(RUNTIME_CONTEXT_BYTES)) {
      expect(bytes, family).toBeGreaterThanOrEqual(0.2 * GIB);
      expect(bytes, family).toBeLessThanOrEqual(1.0 * GIB);
    }
  });

  it("charges CUDA more than Metal, and datacenter more than consumer", () => {
    expect(RUNTIME_CONTEXT_BYTES["cuda-consumer"]).toBeGreaterThan(
      RUNTIME_CONTEXT_BYTES.metal,
    );
    expect(RUNTIME_CONTEXT_BYTES["cuda-datacenter"]).toBeGreaterThan(
      RUNTIME_CONTEXT_BYTES["cuda-consumer"],
    );
  });

  it("is charged once per physical device", () => {
    const one = computeFootprint(LLAMA_3_1_8B, getQuant("q4_k_m"), {
      family: "cuda-consumer",
      ctx: 4096,
    });
    const four = computeFootprint(LLAMA_3_1_8B, getQuant("q4_k_m"), {
      family: "cuda-consumer",
      ctx: 4096,
      deviceCount: 4,
    });
    expect(four.runtimeContextBytes).toBe(one.runtimeContextBytes * 4);
    expect(four.weights.totalBytes).toBe(one.weights.totalBytes);
  });
});

describe("activation / compute buffer", () => {
  it("matches the hand-computed graph size for Llama 3.1 8B at 8K", () => {
    // ubatch 512 tokens; 18 residual-width + 6 intermediate-width tensors:
    //   512 * (4096 * 18 + 14336 * 6) * 2 bytes = 163_577_856
    //   + FP32 logits 128256 * 4                =     513_024
    const expected = 512 * (4096 * 18 + 14_336 * 6) * 2 + 128_256 * 4;
    expect(computeActivationBytes(LLAMA_3_1_8B, { ctx: 8192 })).toBe(expected);
    expect(expected).toBe(164_090_880);
  });

  it("depends on the physical batch, not the context, when flash attention is on", () => {
    const short = computeActivationBytes(LLAMA_3_1_8B, { ctx: 2048 });
    const long = computeActivationBytes(LLAMA_3_1_8B, { ctx: 131_072 });
    expect(short).toBe(long);
  });

  it("grows with context when flash attention is off", () => {
    // Without flash attention the ubatch x ctx x n_heads score matrix is
    // materialised in FP32: 512 * 8192 * 32 * 4 = 512 MiB on its own.
    const on = computeActivationBytes(LLAMA_3_1_8B, { ctx: 8192, flashAttention: true });
    const off = computeActivationBytes(LLAMA_3_1_8B, { ctx: 8192, flashAttention: false });
    expect(off - on).toBe(512 * 8192 * 32 * 4);
    expect(off - on).toBe(512 * MIB);
  });

  it("shrinks with a smaller ubatch", () => {
    const big = computeActivationBytes(LLAMA_3_1_8B, { ctx: 8192, physicalBatch: 512 });
    const small = computeActivationBytes(LLAMA_3_1_8B, { ctx: 8192, physicalBatch: 128 });
    expect(small).toBeLessThan(big);
  });

  it("never drops below the 64 MiB floor backends allocate anyway", () => {
    const tiny = computeActivationBytes(
      { ...LLAMA_3_1_8B, hiddenSize: 64, ffnHidden: 128, vocabSize: 1000 },
      { ctx: 8, physicalBatch: 8 },
    );
    expect(tiny).toBe(64 * MIB);
  });

  it("sizes MoE intermediates by top-k, not by expert count", () => {
    // Mixtral routes to 2 of 8 experts, so the materialised intermediate is
    // 2 * 14336 wide, not 8 * 14336.
    const moe = computeActivationBytes(MIXTRAL_8X7B, { ctx: 4096 });
    const dense = computeActivationBytes({ ...MIXTRAL_8X7B, moe: null }, { ctx: 4096 });
    const expectedRatioNumerator = 4096 * 18 + 14_336 * 2 * 6;
    const expectedRatioDenominator = 4096 * 18 + 14_336 * 6;
    expect((moe - 32_000 * 4) / (dense - 32_000 * 4)).toBeCloseTo(
      expectedRatioNumerator / expectedRatioDenominator,
      6,
    );
  });

  it("scales the logit buffer, and only the logit buffer, with the logical batch", () => {
    // The logit buffer is one FP32 vocabulary row per sequence being scored,
    // so four sequences cost three extra rows -- and nothing else moves.
    const b1 = computeActivationBytes(LLAMA_3_1_8B, { ctx: 8192, batch: 1 });
    const b4 = computeActivationBytes(LLAMA_3_1_8B, { ctx: 8192, batch: 4 });
    expect(b4 - b1).toBe(3 * 128_256 * 4);
  });

  it("keeps the graph one micro-batch wide however many sequences share it", () => {
    // ubatch is the total number of tokens evaluated in one graph pass across
    // every sequence, not a per-sequence figure. Multiplying it by the logical
    // batch inflated the buffer to 4.89 GiB at -b 32 and turned a 21.5 GiB
    // configuration into a 26.2 GiB "does not fit".
    const graph = 512 * (4096 * 18 + 14_336 * 6) * 2;
    expect(computeActivationBytes(LLAMA_3_1_8B, { ctx: 4096, batch: 32 })).toBe(
      graph + 32 * 128_256 * 4,
    );
  });

  it("never widens the graph beyond the tokens there are to evaluate", () => {
    // 64 tokens across 4 sequences is 256 tokens in flight, under the 512-wide
    // micro-batch, so the graph is sized for 256.
    const graph = (tokens: number): number =>
      tokens * (4096 * 18 + 14_336 * 6) * 2 + 4 * 128_256 * 4;
    expect(computeActivationBytes(LLAMA_3_1_8B, { ctx: 64, batch: 4 })).toBe(graph(256));
    expect(computeActivationBytes(LLAMA_3_1_8B, { ctx: 4096, batch: 4 })).toBe(graph(512));
  });
});

describe("composed footprint", () => {
  it("sums to exactly its four parts", () => {
    const f = computeFootprint(LLAMA_3_1_8B, getQuant("q4_k_m"), {
      family: "cuda-consumer",
      ctx: 8192,
    });
    expect(f.totalBytes).toBe(
      f.weights.totalBytes + f.kv.totalBytes + f.runtimeContextBytes + f.activationBytes,
    );
  });

  it("puts Llama 3.1 8B Q4_K_M at 8K around 6.6 GiB on a consumer GPU", () => {
    // 4.62 GiB weights + 1.00 GiB KV + 0.70 GiB context + 0.15 GiB compute
    const f = computeFootprint(LLAMA_3_1_8B, getQuant("q4_k_m"), {
      family: "cuda-consumer",
      ctx: 8192,
    });
    expect(f.totalBytes / GIB).toBeGreaterThan(6.3);
    expect(f.totalBytes / GIB).toBeLessThan(6.9);
  });
});
