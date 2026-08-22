import { describe, expect, it } from "vitest";
import { computeKvCacheBytes } from "../src/memory.js";
import { GIB, MIB } from "../src/units.js";
import {
  DEEPSEEK_V2_LITE,
  LLAMA_2_70B_MHA,
  LLAMA_3_1_70B,
  LLAMA_3_1_8B,
  MQA_7B,
  SWA_12B,
} from "./fixtures.js";

describe("KV cache: standard attention", () => {
  it("matches the hand-computed size for Llama 3.1 8B at 8K", () => {
    // 2 (K and V) * 32 layers * 8 KV heads * 128 head dim * 8192 tokens * 2 bytes
    //   = 1_073_741_824 bytes, which is exactly 1 GiB.
    const kv = computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 8192 });
    expect(kv.totalBytes).toBe(2 * 32 * 8 * 128 * 8192 * 2);
    expect(kv.totalBytes).toBe(GIB);
  });

  it("scales linearly in context and in batch", () => {
    const one = computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 8192 });
    const doubleCtx = computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 16_384 });
    const doubleBatch = computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 8192, batch: 2 });
    expect(doubleCtx.totalBytes).toBe(one.totalBytes * 2);
    expect(doubleBatch.totalBytes).toBe(one.totalBytes * 2);
    expect(one.bytesPerToken).toBe(one.totalBytes / 8192);
  });

  it("uses n_kv_heads, not n_heads: the exact 8x GQA saving", () => {
    // Llama 2 70B and Llama 3.1 70B have identical depth, head dim and head
    // count. The only difference is 64 KV heads vs 8. Using n_heads for the
    // cache overestimates the modern model by exactly the GQA group size.
    const mha = computeKvCacheBytes(LLAMA_2_70B_MHA, { ctx: 4096 });
    const gqa = computeKvCacheBytes(LLAMA_3_1_70B, { ctx: 4096 });
    expect(mha.totalBytes / gqa.totalBytes).toBe(8);

    // And the absolute values, hand-computed:
    //   MHA: 2 * 80 * 64 * 128 * 4096 * 2 = 10 GiB
    //   GQA: 2 * 80 *  8 * 128 * 4096 * 2 = 1.25 GiB
    expect(mha.totalBytes).toBe(10 * GIB);
    expect(gqa.totalBytes).toBe(1.25 * GIB);
  });

  it("gets Llama 3.1 70B at 32K right (10 GiB, not 80)", () => {
    const kv = computeKvCacheBytes(LLAMA_3_1_70B, { ctx: 32_768 });
    expect(kv.totalBytes).toBe(10 * GIB);
    const wrong = computeKvCacheBytes(LLAMA_2_70B_MHA, { ctx: 32_768 });
    expect(wrong.totalBytes).toBe(80 * GIB);
  });

  it("handles MQA as the degenerate case of GQA", () => {
    // 2 * 32 * 1 * 128 * 8192 * 2 = 134_217_728 bytes = 128 MiB
    const kv = computeKvCacheBytes(MQA_7B, { ctx: 8192 });
    expect(kv.totalBytes).toBe(128 * MIB);
  });

  it("reports one entry per layer", () => {
    const kv = computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 8192 });
    expect(kv.bytesPerLayer).toHaveLength(32);
    expect(kv.bytesPerLayer.reduce((a, b) => a + b, 0)).toBe(kv.totalBytes);
    expect(new Set(kv.bytesPerLayer).size).toBe(1);
  });
});

describe("KV cache: quantization", () => {
  it("applies the exact block-layout byte rates", () => {
    const f16 = computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 8192, kvQuant: "f16" });
    const q8 = computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 8192, kvQuant: "q8_0" });
    const q4 = computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 8192, kvQuant: "q4_0" });

    expect(q8.totalBytes).toBe(f16.totalBytes * (1.0625 / 2));
    expect(q4.totalBytes).toBe(f16.totalBytes * (0.5625 / 2));

    // 1 GiB at f16 -> 544 MiB at q8_0 -> 288 MiB at q4_0
    expect(q8.totalBytes).toBe(544 * MIB);
    expect(q4.totalBytes).toBe(288 * MIB);
  });

  it("carries the element size through to the breakdown", () => {
    expect(computeKvCacheBytes(LLAMA_3_1_8B, { ctx: 1024, kvQuant: "q8_0" }).bytesPerElement)
      .toBe(1.0625);
  });
});

describe("KV cache: multi-head latent attention", () => {
  it("caches one compressed latent plus the RoPE key, not per-head K and V", () => {
    // 27 layers * (512 kv_lora_rank + 64 qk_rope_head_dim) * 32768 * 2 bytes
    const ctx = 32_768;
    const expected = 27 * (512 + 64) * ctx * 2;
    const kv = computeKvCacheBytes(DEEPSEEK_V2_LITE, { ctx });
    expect(kv.totalBytes).toBe(expected);
    expect(expected).toBe(1_019_215_872);
  });

  it("is roughly 7x smaller than the same geometry with GQA/MHA", () => {
    const ctx = 32_768;
    const mla = computeKvCacheBytes(DEEPSEEK_V2_LITE, { ctx }).totalBytes;
    // What the naive formula would give for 16 KV heads of 128 dims:
    const naive = 2 * 27 * 16 * 128 * ctx * 2;
    expect(naive / mla).toBeCloseTo(7.11, 2);
    expect(mla).toBeLessThan(naive / 7);
  });

  it("still honours KV quantization and batch", () => {
    const a = computeKvCacheBytes(DEEPSEEK_V2_LITE, { ctx: 4096, kvQuant: "q8_0", batch: 3 });
    const b = computeKvCacheBytes(DEEPSEEK_V2_LITE, { ctx: 4096 });
    expect(a.totalBytes).toBe(b.totalBytes * 3 * (1.0625 / 2));
  });

  it("refuses an MLA model with no MLA geometry", () => {
    expect(() =>
      computeKvCacheBytes({ ...DEEPSEEK_V2_LITE, mla: null }, { ctx: 1024 }),
    ).toThrow(/MLA/);
  });
});

describe("KV cache: interleaved sliding-window attention", () => {
  it("caps local layers at the window and keeps every 6th layer global", () => {
    // 48 layers, 5 local : 1 global, 1024-token window, at 32K context.
    // Global layers: 8. Local layers: 40, each capped at 1024 tokens.
    // per token per layer = 2 * 8 KV heads * 256 head dim * 2 bytes = 8192 B
    const ctx = 32_768;
    const kv = computeKvCacheBytes(SWA_12B, { ctx });
    const perTokenPerLayer = 2 * 8 * 256 * 2;
    const expected = perTokenPerLayer * (8 * ctx + 40 * 1024);
    expect(kv.windowedLayers).toBe(40);
    expect(kv.totalBytes).toBe(expected);
    expect(kv.totalBytes).toBe(2_483_027_968);
  });

  it("is over 5x smaller than assuming every layer is global", () => {
    const ctx = 32_768;
    const withSwa = computeKvCacheBytes(SWA_12B, { ctx }).totalBytes;
    const withoutSwa = computeKvCacheBytes({ ...SWA_12B, attentionWindow: null }, { ctx })
      .totalBytes;
    expect(withoutSwa).toBe(12 * GIB);
    expect(withoutSwa / withSwa).toBeCloseTo(5.19, 2);
  });

  it("collapses to the plain formula when context is below the window", () => {
    const ctx = 512;
    const withSwa = computeKvCacheBytes(SWA_12B, { ctx }).totalBytes;
    const withoutSwa = computeKvCacheBytes({ ...SWA_12B, attentionWindow: null }, { ctx })
      .totalBytes;
    expect(withSwa).toBe(withoutSwa);
  });
});
