import { describe, expect, it } from "vitest";
import { computeKvCacheBytes, computeWeightBytes } from "../src/memory.js";
import { blendBandwidth, blendCompute, planOffload } from "../src/offload.js";
import { getQuant } from "../src/quant.js";
import { GB_DECIMAL, GIB } from "../src/units.js";
import { LLAMA_3_1_8B, MIXTRAL_8X7B, SWA_12B } from "./fixtures.js";

const Q4 = getQuant("q4_k_m");

function parts(model = LLAMA_3_1_8B, ctx = 8192) {
  return {
    weights: computeWeightBytes(model, Q4),
    kv: computeKvCacheBytes(model, { ctx }),
  };
}

describe("layer placement", () => {
  it("places exactly as many layers as the budget holds", () => {
    const { weights, kv } = parts();
    // Uniform layers here: 32 identical blocks plus 32 identical cache slices.
    const perLayer = weights.bytesPerLayer + (kv.bytesPerLayer[0] as number);

    for (const wanted of [0, 1, 7, 31]) {
      // Half a layer short of the next one, so the fill has to stop.
      const plan = planOffload(LLAMA_3_1_8B, weights, kv, perLayer * wanted + perLayer / 2);
      expect(plan.gpuLayers, `budget for ${wanted}`).toBe(wanted);
      expect(plan.cpuLayers).toBe(32 - wanted);
      expect(plan.totalLayers).toBe(32);
    }
  });

  it("places nothing with no budget, and never goes negative", () => {
    const { weights, kv } = parts();
    for (const budget of [0, -1, -1e12]) {
      const plan = planOffload(LLAMA_3_1_8B, weights, kv, budget);
      expect(plan.gpuLayers).toBe(0);
      expect(plan.deviceResidentBytes).toBe(0);
      expect(plan.deviceReadBytesPerToken).toBe(0);
      expect(plan.hostResidentBytes).toBe(weights.totalBytes + kv.totalBytes);
    }
  });

  it("adds the vocabulary tensors only once every layer is placed", () => {
    const { weights, kv } = parts();
    const allLayers = weights.blockBytes + kv.totalBytes;

    const justShort = planOffload(LLAMA_3_1_8B, weights, kv, allLayers + weights.embeddingBytes / 2);
    expect(justShort.gpuLayers).toBe(32);
    expect(justShort.vocabOnDevice).toBe(false);
    expect(justShort.hostResidentBytes).toBeCloseTo(weights.embeddingBytes, 0);

    const enough = planOffload(LLAMA_3_1_8B, weights, kv, allLayers + weights.embeddingBytes);
    expect(enough.vocabOnDevice).toBe(true);
    expect(enough.hostResidentBytes).toBe(0);
    // Sub-microbyte: the two sides sum the same terms in a different order.
    expect(enough.hostReadBytesPerToken).toBeLessThan(1);
    expect(enough.deviceReadShare).toBeCloseTo(1, 12);
  });

  it("fills from the tail of the model, which is what -ngl does", () => {
    // A sliding-window model has cheap local layers and expensive global ones,
    // so which end you fill from is observable. Layers 6, 12, ... 48 are the
    // global ones (every 6th), and llama.cpp offloads the last layers first,
    // so a two-layer budget takes layers 47 and 48: one local, one global.
    const { weights, kv } = parts(SWA_12B, 32_768);
    const lastTwo = (kv.bytesPerLayer[47] as number) + (kv.bytesPerLayer[46] as number);
    const budget = 2 * weights.bytesPerLayer + lastTwo;

    const plan = planOffload(SWA_12B, weights, kv, budget);
    expect(plan.gpuLayers).toBe(2);
    // Filling from the front would have taken two cheap local layers and left
    // budget over; from the back it takes the expensive global layer 48.
    const firstTwo = (kv.bytesPerLayer[0] as number) + (kv.bytesPerLayer[1] as number);
    expect(lastTwo).toBeGreaterThan(firstTwo);
    expect(plan.deviceResidentBytes).toBeCloseTo(budget, 0);
  });

  it("stores every expert but only reads the routed ones", () => {
    const { weights, kv } = parts(MIXTRAL_8X7B, 4096);
    const perLayer = weights.bytesPerLayer + (kv.bytesPerLayer[0] as number);
    const plan = planOffload(MIXTRAL_8X7B, weights, kv, perLayer * 16);

    expect(plan.gpuLayers).toBe(16);
    // Half the layers resident, but well under half the per-token read, since
    // an offloaded expert is only read when a token routes to it.
    expect(plan.deviceResidentBytes / (weights.totalBytes + kv.totalBytes)).toBeCloseTo(0.5, 1);
    expect(plan.deviceReadBytesPerToken).toBeLessThan(plan.deviceResidentBytes);
    expect(plan.deviceReadBytesPerToken + plan.hostReadBytesPerToken).toBeCloseTo(
      weights.activeBytes + kv.totalBytes,
      0,
    );
  });

  it("accounts for every byte, resident and read", () => {
    const { weights, kv } = parts();
    for (const fraction of [0, 0.25, 0.5, 0.9, 1, 2]) {
      const plan = planOffload(
        LLAMA_3_1_8B,
        weights,
        kv,
        (weights.totalBytes + kv.totalBytes) * fraction,
      );
      expect(plan.deviceResidentBytes + plan.hostResidentBytes).toBeCloseTo(
        weights.totalBytes + kv.totalBytes,
        0,
      );
      expect(plan.deviceReadBytesPerToken + plan.hostReadBytesPerToken).toBeCloseTo(
        weights.activeBytes + kv.totalBytes,
        0,
      );
      expect(plan.deviceReadShare).toBeGreaterThanOrEqual(0);
      expect(plan.deviceReadShare).toBeLessThanOrEqual(1);
    }
  });
});

describe("blended bandwidth", () => {
  const pair = {
    devicePeakBytesPerSecond: 1000e9,
    deviceEfficiency: 0.6,
    hostPeakBytesPerSecond: 100e9,
    hostEfficiency: 0.5,
  };

  it("is the harmonic mean weighted by the bytes read from each side", () => {
    const plan = {
      totalLayers: 10,
      gpuLayers: 9,
      cpuLayers: 1,
      vocabOnDevice: false,
      deviceResidentBytes: 9e9,
      hostResidentBytes: 1e9,
      deviceReadBytesPerToken: 9e9,
      hostReadBytesPerToken: 1e9,
      deviceReadShare: 0.9,
    };
    const blended = blendBandwidth(plan, pair);

    // 1/B = 0.9/1000e9 + 0.1/100e9 -> B = 526.3 GB/s, not the arithmetic
    // mean's 910 GB/s. Ten percent of the bytes cost more time than the rest.
    expect(blended.peakBytesPerSecond).toBeCloseTo(1 / (0.9 / 1000e9 + 0.1 / 100e9), 0);
    expect(blended.peakBytesPerSecond / GB_DECIMAL).toBeCloseTo(526.3, 1);
    expect(blended.peakBytesPerSecond).toBeLessThan(0.6 * (0.9 * 1000e9 + 0.1 * 100e9));
  });

  it("applies each side's own kernel efficiency", () => {
    const plan = {
      totalLayers: 2,
      gpuLayers: 1,
      cpuLayers: 1,
      vocabOnDevice: false,
      deviceResidentBytes: 1e9,
      hostResidentBytes: 1e9,
      deviceReadBytesPerToken: 1e9,
      hostReadBytesPerToken: 1e9,
      deviceReadShare: 0.5,
    };
    const blended = blendBandwidth(plan, pair);
    const expectedSeconds = 1e9 / (1000e9 * 0.6) + 1e9 / (100e9 * 0.5);
    expect(blended.secondsPerToken).toBeCloseTo(expectedSeconds, 12);
    expect(blended.effectiveBytesPerSecond).toBeCloseTo(2e9 / expectedSeconds, 0);
    expect(blended.effectiveBytesPerSecond / blended.peakBytesPerSecond).toBeCloseTo(
      blended.efficiency,
      12,
    );
  });

  it("collapses to the device alone when nothing is offloaded", () => {
    const plan = {
      totalLayers: 4,
      gpuLayers: 4,
      cpuLayers: 0,
      vocabOnDevice: true,
      deviceResidentBytes: 4e9,
      hostResidentBytes: 0,
      deviceReadBytesPerToken: 4e9,
      hostReadBytesPerToken: 0,
      deviceReadShare: 1,
    };
    const blended = blendBandwidth(plan, pair);
    expect(blended.peakBytesPerSecond).toBeCloseTo(1000e9, 0);
    expect(blended.efficiency).toBeCloseTo(0.6, 12);
  });

  it("punishes a real 4090 for the first 10% of a model it cannot hold", () => {
    // The headline claim in the module docs, checked: 90% resident on a
    // 1008 GB/s card, 10% in 89.6 GB/s DDR5, lands near 498 GB/s -- less than
    // half the card's bandwidth for a tenth of the model displaced.
    const plan = {
      totalLayers: 10,
      gpuLayers: 9,
      cpuLayers: 1,
      vocabOnDevice: false,
      deviceResidentBytes: 9,
      hostResidentBytes: 1,
      deviceReadBytesPerToken: 9,
      hostReadBytesPerToken: 1,
      deviceReadShare: 0.9,
    };
    const blended = blendBandwidth(plan, {
      devicePeakBytesPerSecond: 1008 * GB_DECIMAL,
      deviceEfficiency: 1,
      hostPeakBytesPerSecond: 89.6 * GB_DECIMAL,
      hostEfficiency: 1,
    });
    expect(blended.peakBytesPerSecond / GB_DECIMAL).toBeCloseTo(498, 0);
    expect(blended.peakBytesPerSecond).toBeLessThan(0.5 * 1008 * GB_DECIMAL);
  });
});

describe("blended compute", () => {
  const plan = {
    totalLayers: 80,
    gpuLayers: 40,
    cpuLayers: 40,
    vocabOnDevice: false,
    deviceResidentBytes: 20 * GIB,
    hostResidentBytes: 20 * GIB,
    deviceReadBytesPerToken: 20 * GIB,
    hostReadBytesPerToken: 20 * GIB,
    deviceReadShare: 0.5,
  };

  it("weights by layers, because prefill is compute bound", () => {
    const blended = blendCompute(plan, {
      devicePeakFlopsPerSecond: 100e12,
      deviceEfficiency: 1,
      hostPeakFlopsPerSecond: 1e12,
      hostEfficiency: 1,
    });
    // Half the layers on a 100x slower processor: 1.98 TFLOP/s, not 50.5.
    expect(blended.peakFlopsPerSecond).toBeCloseTo(1 / (0.5 / 100e12 + 0.5 / 1e12), 0);
    expect(blended.peakFlopsPerSecond / 1e12).toBeCloseTo(1.98, 2);
  });

  it("collapses to the device when every layer is resident", () => {
    const blended = blendCompute(
      { ...plan, gpuLayers: 80, cpuLayers: 0 },
      {
        devicePeakFlopsPerSecond: 100e12,
        deviceEfficiency: 0.4,
        hostPeakFlopsPerSecond: 1e12,
        hostEfficiency: 0.4,
      },
    );
    expect(blended.peakFlopsPerSecond).toBeCloseTo(100e12, 0);
    expect(blended.efficiency).toBeCloseTo(0.4, 12);
  });
});
