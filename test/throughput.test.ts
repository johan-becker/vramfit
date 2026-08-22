import { describe, expect, it } from "vitest";
import { deriveArchitecture } from "../src/architecture.js";
import { getDevice, getModel } from "../src/db/index.js";
import { computeKvCacheBytes, computeWeightBytes } from "../src/memory.js";
import { getQuant } from "../src/quant.js";
import {
  DECODE_ERROR_BAND,
  DEQUANT_EFFICIENCY,
  MEMORY_BANDWIDTH_EFFICIENCY,
  PREFILL_ERROR_BAND,
  PREFILL_MFU,
  bandwidthEfficiency,
  estimateDecodeFrom,
  estimatePrefillFrom,
  estimateThroughput,
  prefillFlopsPerToken,
} from "../src/throughput.js";
import type { DeviceFamily } from "../src/types.js";
import { GB_DECIMAL, GIB } from "../src/units.js";
import {
  DEEPSEEK_V2_LITE,
  LLAMA_3_1_8B,
  LLAMA_7B_MHA,
  MIXTRAL_8X7B,
  SWA_12B,
} from "./fixtures.js";

const FAMILIES: readonly DeviceFamily[] = [
  "cuda-consumer",
  "cuda-datacenter",
  "rocm",
  "metal",
  "cpu",
];

describe("bandwidth efficiency", () => {
  it("stays inside the 0.70-0.85 band at native precision", () => {
    for (const family of FAMILIES) {
      const efficiency = bandwidthEfficiency(family, 16);
      expect(efficiency, family).toBe(MEMORY_BANDWIDTH_EFFICIENCY[family]);
      expect(efficiency, family).toBeGreaterThanOrEqual(0.7);
      expect(efficiency, family).toBeLessThanOrEqual(0.85);
    }
  });

  it("applies the full dequantization penalty at 4 bits", () => {
    for (const family of FAMILIES) {
      expect(bandwidthEfficiency(family, 4), family).toBeCloseTo(
        MEMORY_BANDWIDTH_EFFICIENCY[family] * DEQUANT_EFFICIENCY[family],
        12,
      );
    }
  });

  it("interpolates linearly between the two anchors", () => {
    const at10 = bandwidthEfficiency("cuda-consumer", 10);
    const at16 = bandwidthEfficiency("cuda-consumer", 16);
    const at4 = bandwidthEfficiency("cuda-consumer", 4);
    expect(at10).toBeCloseTo((at16 + at4) / 2, 12);
  });

  it("clamps outside the anchors instead of extrapolating", () => {
    expect(bandwidthEfficiency("metal", 32)).toBe(bandwidthEfficiency("metal", 16));
    expect(bandwidthEfficiency("metal", 2)).toBeCloseTo(bandwidthEfficiency("metal", 4), 12);
  });

  it("penalises Metal's dequantization kernels more than CUDA's", () => {
    // Measured: an M2 Ultra loses 40% of its F16 MBU at Q4_0, a 4090 loses 21%.
    expect(DEQUANT_EFFICIENCY.metal).toBeLessThan(DEQUANT_EFFICIENCY["cuda-consumer"]);
    expect(DEQUANT_EFFICIENCY["cuda-consumer"]).toBeLessThan(
      DEQUANT_EFFICIENCY["cuda-datacenter"],
    );
  });
});

describe("decode: the arithmetic", () => {
  it("is bytes per step divided into effective bandwidth", () => {
    const decode = estimateDecodeFrom({
      weightBytesPerStep: 4e9,
      kvBytesPerStep: 1e9,
      peakBandwidthBytesPerSecond: 1000e9,
      efficiency: 0.5,
      batch: 1,
    });
    expect(decode.bytesPerStep).toBe(5e9);
    expect(decode.effectiveBandwidthBytesPerSecond).toBe(500e9);
    expect(decode.tokensPerSecond).toBe(100);
    expect(decode.aggregateTokensPerSecond).toBe(100);
  });

  it("reads the weights once for the whole batch", () => {
    // Four sequences share one weight read, so aggregate throughput is 4x the
    // per-sequence rate at the same bytes per step. (Their caches are four
    // times larger, which is charged through kvBytesPerStep by the caller.)
    const decode = estimateDecodeFrom({
      weightBytesPerStep: 4e9,
      kvBytesPerStep: 1e9,
      peakBandwidthBytesPerSecond: 1000e9,
      efficiency: 0.5,
      batch: 4,
    });
    expect(decode.tokensPerSecond).toBe(100);
    expect(decode.aggregateTokensPerSecond).toBe(400);
  });

  it("scales exactly with bandwidth", () => {
    const slow = estimateThroughput(LLAMA_3_1_8B, getQuant("q4_k_m"), getDevice("rtx-4090"), {
      ctx: 4096,
      peakBandwidthBytesPerSecond: 500e9,
    });
    const fast = estimateThroughput(LLAMA_3_1_8B, getQuant("q4_k_m"), getDevice("rtx-4090"), {
      ctx: 4096,
      peakBandwidthBytesPerSecond: 1000e9,
    });
    expect(fast.decode.tokensPerSecond / slow.decode.tokensPerSecond).toBeCloseTo(2, 10);
  });

  it("composes exactly from the weight and KV models", () => {
    const model = LLAMA_3_1_8B;
    const quant = getQuant("q4_k_m");
    const device = getDevice("rtx-4090");
    const ctx = 8192;

    const weights = computeWeightBytes(model, quant);
    const kv = computeKvCacheBytes(model, { ctx });
    const efficiency = bandwidthEfficiency(device.family, weights.effectiveBitsPerWeight);
    const expected =
      (device.bandwidthGBs * GB_DECIMAL * efficiency) / (weights.activeBytes + kv.totalBytes);

    const estimate = estimateThroughput(model, quant, device, { ctx });
    expect(estimate.decode.tokensPerSecond).toBeCloseTo(expected, 9);
    expect(estimate.decode.weightBytesPerStep).toBe(weights.activeBytes);
    expect(estimate.decode.kvBytesPerStep).toBe(kv.totalBytes);
  });
});

describe("decode: what the estimate is for", () => {
  it("falls off as the KV cache grows, which is why long chats feel slower", () => {
    const args = [LLAMA_3_1_8B, getQuant("q4_k_m"), getDevice("rtx-4090")] as const;
    const short = estimateThroughput(...args, { ctx: 128 });
    const long = estimateThroughput(...args, { ctx: 32_768 });

    // 4.64 GB of weights at 128 tokens; +4 GiB of cache at 32K.
    expect(long.decode.kvBytesPerStep).toBe(4 * GIB);
    expect(long.decode.tokensPerSecond).toBeLessThan(short.decode.tokensPerSecond * 0.6);
    expect(short.decode.kvBytesPerStep / short.decode.weightBytesPerStep).toBeLessThan(0.01);
  });

  it("makes a mixture of experts decode at its active size, not its total", () => {
    const quant = getQuant("q4_k_m");
    const device = getDevice("a100-80gb");
    const moe = estimateThroughput(MIXTRAL_8X7B, quant, device, { ctx: 4096 });
    // The same 46.7B of memory, but every expert read on every token.
    const asDense = estimateThroughput(
      { ...MIXTRAL_8X7B, moe: null, activeParams: MIXTRAL_8X7B.totalParams },
      quant,
      device,
      { ctx: 4096 },
    );
    expect(moe.decode.tokensPerSecond / asDense.decode.tokensPerSecond).toBeGreaterThan(3);
    expect(moe.decode.weightBytesPerStep).toBeLessThan(asDense.decode.weightBytesPerStep / 3);
  });

  it("prefers a narrower quant when bandwidth, not quality, is the constraint", () => {
    const device = getDevice("rtx-4090");
    const q8 = estimateThroughput(LLAMA_3_1_8B, getQuant("q8_0"), device, { ctx: 4096 });
    const q4 = estimateThroughput(LLAMA_3_1_8B, getQuant("q4_k_m"), device, { ctx: 4096 });
    expect(q4.decode.tokensPerSecond).toBeGreaterThan(q8.decode.tokensPerSecond);
    // But not by the full 1.76x the byte counts suggest: the dequantization
    // penalty eats part of the win.
    const byteRatio = q8.decode.bytesPerStep / q4.decode.bytesPerStep;
    const speedRatio = q4.decode.tokensPerSecond / q8.decode.tokensPerSecond;
    expect(speedRatio).toBeLessThan(byteRatio);
  });
});

/**
 * The external ground truth. Every measured figure is a published llama.cpp
 * `tg128` result for the original LLaMA 7B, the model those tables are quoted
 * for. Estimates are required to land inside the declared error band -- if a
 * change to the efficiency tables breaks reality, it breaks here.
 */
describe("decode: calibration against published benchmarks", () => {
  const CASES: ReadonlyArray<[device: string, quant: string, measured: number]> = [
    ["m2-ultra", "f16", 42.75],
    ["m2-ultra", "q4_0", 88.64],
    ["rtx-4090", "q4_0", 152.2],
  ];

  for (const [deviceId, quantId, measured] of CASES) {
    it(`is within ${Math.round(DECODE_ERROR_BAND * 100)}% of ${measured} tok/s on ${deviceId} at ${quantId}`, () => {
      const estimate = estimateThroughput(
        LLAMA_7B_MHA,
        getQuant(quantId),
        getDevice(deviceId),
        { ctx: 128 },
      );
      const error = Math.abs(estimate.decode.tokensPerSecond - measured) / measured;
      expect(
        error,
        `estimated ${estimate.decode.tokensPerSecond.toFixed(1)} tok/s vs measured ${measured}`,
      ).toBeLessThan(DECODE_ERROR_BAND);
    });
  }

  it("reproduces the F16-to-Q4_0 speedup Apple silicon actually shows", () => {
    // 88.64 / 42.75 = 2.07x, well short of the 3.5x the byte counts imply,
    // because Metal's Q4 dequantization is where the rest of the win goes.
    const f16 = estimateThroughput(LLAMA_7B_MHA, getQuant("f16"), getDevice("m2-ultra"), {
      ctx: 128,
    });
    const q4 = estimateThroughput(LLAMA_7B_MHA, getQuant("q4_0"), getDevice("m2-ultra"), {
      ctx: 128,
    });
    expect(q4.decode.tokensPerSecond / f16.decode.tokensPerSecond).toBeCloseTo(2.07, 0);
  });
});

describe("prefill", () => {
  it("counts 2 FLOPs per active parameter plus the attention matmuls", () => {
    const ctx = 2048;
    const arch = deriveArchitecture(LLAMA_3_1_8B);
    // 32 layers x 2048 attended tokens x (32 heads x 128 dims) x 2 matmuls
    // x 2 FLOPs, already halved for causal masking.
    const attention = 2 * (32 * ctx) * (32 * 128);
    expect(prefillFlopsPerToken(LLAMA_3_1_8B, ctx)).toBe(
      2 * arch.activeMatmulParams + attention,
    );
    expect(prefillFlopsPerToken(LLAMA_3_1_8B, ctx)).toBe(15_546_187_776);
  });

  it("is dominated by the weight matmuls at ordinary prompt lengths", () => {
    const estimate = estimatePrefillFrom({
      model: LLAMA_3_1_8B,
      ctx: 2048,
      peakFlopsPerSecond: 100e12,
      efficiency: 0.5,
    });
    expect(estimate.attentionFlopsPerToken / estimate.flopsPerToken).toBeLessThan(0.05);
    expect(estimate.tokensPerSecond).toBeCloseTo(50e12 / estimate.flopsPerToken, 6);
  });

  it("grows quadratically in total: attention overtakes the matmuls at 128K", () => {
    const short = estimatePrefillFrom({
      model: LLAMA_3_1_8B,
      ctx: 2048,
      peakFlopsPerSecond: 100e12,
      efficiency: 0.5,
    });
    const long = estimatePrefillFrom({
      model: LLAMA_3_1_8B,
      ctx: 131_072,
      peakFlopsPerSecond: 100e12,
      efficiency: 0.5,
    });
    expect(long.attentionFlopsPerToken).toBe(short.attentionFlopsPerToken * 64);
    expect(long.attentionFlopsPerToken).toBeGreaterThan(2 * 7.5e9);
    expect(long.tokensPerSecond).toBeLessThan(short.tokensPerSecond / 2);
  });

  it("charges sliding-window layers only for their window", () => {
    const ctx = 32_768;
    const windowed = estimatePrefillFrom({
      model: SWA_12B,
      ctx,
      peakFlopsPerSecond: 100e12,
      efficiency: 0.5,
    });
    const global = estimatePrefillFrom({
      model: { ...SWA_12B, attentionWindow: null },
      ctx,
      peakFlopsPerSecond: 100e12,
      efficiency: 0.5,
    });
    // 8 global layers at 32768 + 40 local layers at 1024, versus 48 x 32768.
    expect(global.attentionFlopsPerToken / windowed.attentionFlopsPerToken).toBeCloseTo(
      (48 * ctx) / (8 * ctx + 40 * 1024),
      6,
    );
  });

  it("uses the decompressed head width for latent attention", () => {
    // MLA caches 576 elements per token but attends at 16 x (128 + 64) = 3072.
    const ctx = 4096;
    const arch = deriveArchitecture(DEEPSEEK_V2_LITE);
    const expected = 2 * (27 * ctx) * (16 * (128 + 64));
    expect(prefillFlopsPerToken(DEEPSEEK_V2_LITE, ctx) - 2 * arch.activeMatmulParams).toBe(
      expected,
    );
  });

  it("puts an 8B prefill in the right ballpark on real hardware", () => {
    // Published pp512 for Llama-3-8B Q4_K_M: ~4.5k tok/s on a 4090, ~1.1k on
    // an M2 Ultra. Prefill varies more between backends than decode does, so
    // this is the wider band.
    const cases: ReadonlyArray<[string, number]> = [
      ["rtx-4090", 4500],
      ["m2-ultra", 1100],
    ];
    for (const [deviceId, measured] of cases) {
      const estimate = estimateThroughput(
        getModel("llama-3.1-8b"),
        getQuant("q4_k_m"),
        getDevice(deviceId),
        { ctx: 512 },
      );
      const error = Math.abs(estimate.prefill.tokensPerSecond - measured) / measured;
      expect(
        error,
        `${deviceId}: estimated ${estimate.prefill.tokensPerSecond.toFixed(0)} vs ${measured}`,
      ).toBeLessThan(PREFILL_ERROR_BAND);
    }
  });

  it("keeps every prefill MFU below the theoretical ceiling", () => {
    for (const family of FAMILIES) {
      expect(PREFILL_MFU[family], family).toBeGreaterThan(0.2);
      expect(PREFILL_MFU[family], family).toBeLessThan(0.6);
    }
  });
});

describe("composed estimate", () => {
  it("derives time to first token from the prefill rate", () => {
    const estimate = estimateThroughput(
      LLAMA_3_1_8B,
      getQuant("q4_k_m"),
      getDevice("rtx-4090"),
      { ctx: 8192, promptTokens: 4000 },
    );
    expect(estimate.timeToFirstTokenSeconds).toBeCloseTo(
      4000 / estimate.prefill.tokensPerSecond,
      9,
    );
    expect(estimate.timeToFirstTokenSeconds).toBeGreaterThan(0);
    expect(estimate.timeToFirstTokenSeconds).toBeLessThan(10);
  });

  it("defaults the prompt to the whole context", () => {
    const estimate = estimateThroughput(
      LLAMA_3_1_8B,
      getQuant("q4_k_m"),
      getDevice("rtx-4090"),
      { ctx: 2048 },
    );
    expect(estimate.timeToFirstTokenSeconds).toBeCloseTo(
      2048 / estimate.prefill.tokensPerSecond,
      9,
    );
  });

  it("honours explicit efficiency overrides for calibrated setups", () => {
    const estimate = estimateThroughput(
      LLAMA_3_1_8B,
      getQuant("q4_k_m"),
      getDevice("rtx-4090"),
      { ctx: 4096, efficiency: 1, prefillEfficiency: 1 },
    );
    expect(estimate.decode.efficiency).toBe(1);
    expect(estimate.decode.effectiveBandwidthBytesPerSecond).toBe(1008 * GB_DECIMAL);
    expect(estimate.prefill.effectiveFlopsPerSecond).toBeCloseTo(165.2e12, 3);
  });

  it("publishes the error bands alongside the numbers", () => {
    const estimate = estimateThroughput(
      LLAMA_3_1_8B,
      getQuant("q4_k_m"),
      getDevice("rtx-4090"),
      { ctx: 4096 },
    );
    expect(estimate.decodeErrorBand).toBe(DECODE_ERROR_BAND);
    expect(estimate.prefillErrorBand).toBe(PREFILL_ERROR_BAND);
    expect(estimate.prefillErrorBand).toBeGreaterThan(estimate.decodeErrorBand);
  });

  it("prefills far faster than it decodes, on every device", () => {
    for (const deviceId of ["rtx-4090", "h100", "m2-ultra", "cpu-ddr5-dual"]) {
      const estimate = estimateThroughput(
        getModel("llama-3.1-8b"),
        getQuant("q4_k_m"),
        getDevice(deviceId),
        { ctx: 2048 },
      );
      expect(
        estimate.prefill.tokensPerSecond,
        deviceId,
      ).toBeGreaterThan(estimate.decode.tokensPerSecond);
    }
  });
});
