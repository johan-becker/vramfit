import { describe, expect, it } from "vitest";
import { getDevice, getModel } from "../src/db/index.js";
import {
  DEFAULT_SYSTEM_RAM_GIB,
  checkFit,
  computeCapacity,
  evaluateQuants,
  maxContextFor,
  recommendQuant,
} from "../src/fit.js";
import { computeFootprint } from "../src/memory.js";
import { GGUF_QUANT_FAMILIES, getQuant } from "../src/quant.js";
import { GIB, bytesToGiB } from "../src/units.js";

const Q4 = getQuant("q4_k_m");

describe("capacity", () => {
  it("gives a discrete GPU its whole framebuffer", () => {
    const capacity = computeCapacity(getDevice("rtx-4090"));
    expect(capacity.perDeviceBytes).toBe(24 * GIB);
    expect(capacity.totalBytes).toBe(24 * GIB);
    expect(capacity.installedBytes).toBe(24 * GIB);
  });

  it("applies the macOS wired-memory limit to unified memory", () => {
    const capacity = computeCapacity(getDevice("m2-max"));
    expect(capacity.installedBytes).toBe(96 * GIB);
    expect(capacity.totalBytes).toBe(96 * GIB * 0.75);
    expect(bytesToGiB(capacity.totalBytes)).toBe(72);
  });

  it("multiplies by the device count and accepts a VRAM override", () => {
    expect(computeCapacity(getDevice("rtx-3090"), { gpus: 4 }).totalBytes).toBe(96 * GIB);
    expect(computeCapacity(getDevice("m4-max"), { vramGiB: 48 }).totalBytes).toBe(48 * GIB * 0.75);
  });

  it("never reads a device count below one", () => {
    expect(computeCapacity(getDevice("rtx-4090"), { gpus: 0 }).gpus).toBe(1);
    expect(computeCapacity(getDevice("rtx-4090"), { gpus: -3 }).gpus).toBe(1);
  });
});

describe("checkFit: the verdict", () => {
  it("puts an 8B Q4_K_M at 8K comfortably inside a 12 GB card", () => {
    const fit = checkFit(getModel("llama-3.1-8b"), Q4, getDevice("rtx-3060-12gb"), {
      ctx: 8192,
    });
    expect(fit.fits).toBe(true);
    expect(bytesToGiB(fit.usedBytes)).toBeGreaterThan(6);
    expect(bytesToGiB(fit.usedBytes)).toBeLessThan(7.5);
    expect(fit.headroomBytes).toBeGreaterThan(4 * GIB);
    expect(fit.utilization).toBeLessThan(0.65);
    expect(fit.offload).toBeNull();
  });

  it("refuses a 70B Q4_K_M on one 24 GB card and reports the shortfall", () => {
    const fit = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-4090"), { ctx: 8192 });
    expect(fit.fits).toBe(false);
    expect(fit.headroomBytes).toBeLessThan(0);
    expect(fit.utilization).toBeGreaterThan(1.5);
    expect(fit.offload).not.toBeNull();
  });

  it("keeps headroom, utilisation and the footprint mutually consistent", () => {
    for (const deviceId of ["rtx-3060-12gb", "rtx-4090", "m2-max", "h100"]) {
      const fit = checkFit(getModel("qwen2.5-32b"), Q4, getDevice(deviceId), { ctx: 16_384 });
      expect(fit.usedBytes, deviceId).toBe(fit.footprint.totalBytes);
      expect(fit.headroomBytes, deviceId).toBeCloseTo(
        fit.capacity.totalBytes - fit.usedBytes,
        6,
      );
      expect(fit.utilization, deviceId).toBeCloseTo(fit.usedBytes / fit.capacity.totalBytes, 12);
      expect(fit.fits, deviceId).toBe(fit.headroomBytes >= 0);
      expect(fit.fits, deviceId).toBe(fit.utilization <= 1);
    }
  });

  it("does not inflate the compute buffer with the number of sequences", () => {
    // A batched-serving configuration: 32 sequences of 4K on one 24 GiB card.
    // The compute buffer is one 512-token graph plus 32 FP32 logit rows, well
    // under a GiB; charging one graph per sequence made it 4.89 GiB and turned
    // this into a false "does not fit" with the exit code a deploy gate reads.
    const fit = checkFit(getModel("llama-3.1-8b"), Q4, getDevice("rtx-4090"), {
      ctx: 4096,
      batch: 32,
    });
    expect(bytesToGiB(fit.footprint.activationBytes)).toBeLessThan(0.5);
    expect(bytesToGiB(fit.usedBytes)).toBeLessThan(24);
    expect(fit.fits).toBe(true);
  });

  it("defaults the context to the model's own default", () => {
    const model = getModel("llama-3.1-8b");
    expect(checkFit(model, Q4, getDevice("rtx-4090")).ctx).toBe(model.defaultCtx);
  });

  it("charges the runtime context once per device, so 2x24 GB is not 48 GB of model", () => {
    const model = getModel("llama-3.3-70b");
    const one = checkFit(model, Q4, getDevice("rtx-3090"), { ctx: 8192 });
    const two = checkFit(model, Q4, getDevice("rtx-3090"), { ctx: 8192, gpus: 2 });

    expect(two.capacity.totalBytes).toBe(one.capacity.totalBytes * 2);
    expect(two.footprint.runtimeContextBytes).toBe(one.footprint.runtimeContextBytes * 2);
    expect(two.footprint.weights.totalBytes).toBe(one.footprint.weights.totalBytes);
    // A pair of 3090s does hold it -- but the second card's overheads eat
    // 1 GiB of the 24 it added.
    expect(one.fits).toBe(false);
    expect(two.fits).toBe(true);
    expect(two.usedBytes - one.usedBytes).toBeGreaterThan(0.9 * GIB);
  });
});

describe("checkFit: partial offload", () => {
  const fit = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-4090"), {
    ctx: 8192,
    systemRamGiB: 64,
  });

  it("reports how many layers fit rather than just refusing", () => {
    expect(fit.offload).not.toBeNull();
    const plan = fit.offload?.plan;
    expect(plan?.totalLayers).toBe(80);
    expect(plan?.gpuLayers).toBeGreaterThan(20);
    expect(plan?.gpuLayers).toBeLessThan(80);
    expect((plan?.gpuLayers ?? 0) + (plan?.cpuLayers ?? 0)).toBe(80);
  });

  it("keeps the placed layers inside the budget left after the fixed overheads", () => {
    const budget =
      fit.capacity.totalBytes -
      fit.footprint.runtimeContextBytes -
      fit.footprint.activationBytes;
    expect(fit.offload?.plan.deviceResidentBytes).toBeLessThanOrEqual(budget);
  });

  it("estimates the much lower blended throughput, not the card's", () => {
    const resident = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-4090"), {
      ctx: 8192,
      vramGiB: 80,
    });
    expect(resident.fits).toBe(true);
    expect(fit.throughput.decode.tokensPerSecond).toBeLessThan(
      resident.throughput.decode.tokensPerSecond / 4,
    );
    // Single-digit tokens per second is the honest answer for a 70B with
    // nearly half its layers in DDR5.
    expect(fit.throughput.decode.tokensPerSecond).toBeGreaterThan(0.5);
    expect(fit.throughput.decode.tokensPerSecond).toBeLessThan(6);
  });

  it("slows prefill down too, because the CPU layers do their own matmuls", () => {
    const resident = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-4090"), {
      ctx: 8192,
      vramGiB: 80,
    });
    expect(fit.throughput.prefill.tokensPerSecond).toBeLessThan(
      resident.throughput.prefill.tokensPerSecond / 10,
    );
    expect(fit.throughput.timeToFirstTokenSeconds).toBeGreaterThan(
      resident.throughput.timeToFirstTokenSeconds * 10,
    );
  });

  it("says so when even system RAM is not enough", () => {
    const cramped = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-4090"), {
      ctx: 8192,
      systemRamGiB: 8,
    });
    expect(cramped.offload?.feasible).toBe(false);
    expect(cramped.offload?.systemRamRequiredBytes).toBeGreaterThan(8 * GIB);
    expect(cramped.warnings.join(" ")).toMatch(/more RAM or a narrower quantization/);

    expect(fit.offload?.feasible).toBe(true);
    expect(fit.offload?.systemRamAvailableBytes).toBe(64 * GIB);
  });

  it("applies --efficiency to the device side only, never to system RAM", () => {
    // An efficiency figure is calibrated by measuring a GPU-resident run, so
    // raising it must not also speed up the DDR5 side of a split -- which the
    // harmonic blend is dominated by. Applying it to both turned 2.22 tok/s
    // into 4.02 tok/s, an 81% inflation from calibrating the fast path.
    const options = { ctx: 4096, systemRamGiB: 128 };
    const derived = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-4090"), options);
    const calibrated = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-4090"), {
      ...options,
      efficiency: 0.95,
    });

    expect(derived.offload).not.toBeNull();
    expect(calibrated.offload?.plan.cpuLayers).toBeGreaterThan(0);
    const ratio =
      calibrated.throughput.decode.tokensPerSecond / derived.throughput.decode.tokensPerSecond;
    expect(ratio).toBeGreaterThan(1);
    expect(ratio).toBeLessThan(1.15);
    // The blended efficiency stays pulled down by the host's derived figure.
    expect(calibrated.offload?.bandwidth.efficiency).toBeLessThan(0.7);
  });

  it("assumes a stated default amount of system RAM when not told", () => {
    const assumed = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-4090"), {
      ctx: 8192,
    });
    expect(assumed.offload?.systemRamAvailableBytes).toBe(DEFAULT_SYSTEM_RAM_GIB * GIB);
  });
});

describe("maxContextFor", () => {
  it("finds the exact boundary: one more token does not fit", () => {
    const model = getModel("llama-3.1-8b");
    const device = getDevice("rtx-3060-12gb");
    const max = maxContextFor(model, Q4, device, {});
    expect(max).toBeGreaterThan(1024);
    expect(max).toBeLessThan(model.maxCtx);

    const capacity = computeCapacity(device).totalBytes;
    const at = (ctx: number) =>
      computeFootprint(model, Q4, { family: device.family, ctx }).totalBytes;
    expect(at(max)).toBeLessThanOrEqual(capacity);
    expect(at(max + 1)).toBeGreaterThan(capacity);
  });

  it("never exceeds the architecture's own maximum", () => {
    const model = getModel("llama-3.1-8b");
    expect(maxContextFor(model, Q4, getDevice("h100"))).toBe(model.maxCtx);
    expect(maxContextFor(model, getQuant("q2_k"), getDevice("m3-ultra"))).toBe(model.maxCtx);
  });

  it("returns zero when not even one token fits", () => {
    expect(maxContextFor(getModel("llama-3.3-70b"), getQuant("f16"), getDevice("rtx-3060-12gb"))).toBe(0);
  });

  it("grows when the cache is quantized and shrinks with batch", () => {
    const model = getModel("llama-3.1-8b");
    const device = getDevice("rtx-4090");
    const f16 = maxContextFor(model, getQuant("q8_0"), device);
    const q8 = maxContextFor(model, getQuant("q8_0"), device, { kvQuant: "q8_0" });
    const batched = maxContextFor(model, getQuant("q8_0"), device, { batch: 4 });
    expect(q8).toBeGreaterThan(f16);
    expect(batched).toBeLessThan(f16);
  });

  it("matches the fit verdict at its own boundary", () => {
    const model = getModel("gemma-3-27b");
    const device = getDevice("rtx-4090");
    const max = maxContextFor(model, Q4, device);
    expect(checkFit(model, Q4, device, { ctx: max }).fits).toBe(true);
    expect(checkFit(model, Q4, device, { ctx: max + 1 }).fits).toBe(false);
  });
});

describe("quantization search", () => {
  it("recommends the highest-quality quant that still fits", () => {
    const model = getModel("llama-3.1-8b");
    const device = getDevice("rtx-3060-12gb");
    const best = recommendQuant(model, device, { ctx: 8192 });
    expect(best?.quant.id).toBe("q8_0");
    expect(best?.fit.fits).toBe(true);

    // Everything ranked above it must genuinely not fit.
    for (const option of evaluateQuants(model, device, { ctx: 8192 })) {
      if (option.quant.qualityRank > (best?.quant.qualityRank ?? 0)) {
        expect(option.fit.fits, option.quant.label).toBe(false);
      }
    }
  });

  it("returns candidates best-quality first", () => {
    const options = evaluateQuants(getModel("mistral-7b"), getDevice("rtx-4090"));
    const ranks = options.map((option) => option.quant.qualityRank);
    expect(ranks).toEqual([...ranks].toSorted((a, b) => b - a));
  });

  it("stays inside one ecosystem, because you cannot load AWQ in llama.cpp", () => {
    const options = evaluateQuants(getModel("mistral-7b"), getDevice("rtx-4090"));
    for (const option of options) {
      expect(GGUF_QUANT_FAMILIES.has(option.quant.family), option.quant.label).toBe(true);
    }
    const awq = evaluateQuants(getModel("mistral-7b"), getDevice("rtx-4090"), {
      families: new Set(["awq", "gptq"]),
    });
    expect(awq.map((option) => option.quant.id).toSorted()).toEqual(["awq-4bit", "gptq-4bit"]);
  });

  it("finds nothing when the model is far too large for the device", () => {
    expect(recommendQuant(getModel("qwen3-235b-a22b"), getDevice("rtx-3060-12gb"))).toBeUndefined();
  });

  it("trades quality for context: a narrower quant leaves room for more tokens", () => {
    const options = evaluateQuants(getModel("qwen2.5-14b"), getDevice("rtx-4090"), {
      ctx: 4096,
    });
    const q8 = options.find((option) => option.quant.id === "q8_0");
    const q4 = options.find((option) => option.quant.id === "q4_k_m");
    expect(q4?.maxContext).toBeGreaterThan(q8?.maxContext ?? 0);
    expect(q4?.decodeTokensPerSecond).toBeGreaterThan(q8?.decodeTokensPerSecond ?? 0);
  });
});

describe("warnings", () => {
  it("flags a context beyond what the model was trained for", () => {
    const model = getModel("mistral-7b");
    const fit = checkFit(model, Q4, getDevice("rtx-4090"), { ctx: model.maxCtx * 2 });
    expect(fit.warnings.join(" ")).toMatch(/exceeds Mistral 7B v0\.3's trained maximum/);
  });

  it("explains the macOS wired-memory limit on Apple silicon", () => {
    const fit = checkFit(getModel("llama-3.1-8b"), Q4, getDevice("m3-max"), { ctx: 8192 });
    expect(fit.warnings.join(" ")).toMatch(/iogpu\.wired_limit_mb/);
  });

  it("says that extra GPUs buy capacity, not speed", () => {
    const fit = checkFit(getModel("llama-3.3-70b"), Q4, getDevice("rtx-3090"), { gpus: 4 });
    expect(fit.warnings.join(" ")).toMatch(/buy capacity, not tokens per second/);
  });

  it("warns that AWQ and GPTQ are CUDA-only in practice", () => {
    const fit = checkFit(getModel("llama-3.1-8b"), getQuant("awq-4bit"), getDevice("m4-max"), {
      ctx: 4096,
    });
    expect(fit.warnings.join(" ")).toMatch(/AWQ-4bit kernels are CUDA-only/);
    expect(
      checkFit(getModel("llama-3.1-8b"), getQuant("awq-4bit"), getDevice("rtx-4090")).warnings,
    ).toEqual([]);
  });

  it("warns that offloading a mixture of experts is especially bad", () => {
    const fit = checkFit(getModel("qwen3-235b-a22b"), Q4, getDevice("rtx-4090"), {
      ctx: 4096,
      systemRamGiB: 256,
    });
    expect(fit.warnings.join(" ")).toMatch(/route to an expert in system RAM/);
  });

  it("stays quiet for an ordinary, comfortable configuration", () => {
    expect(checkFit(getModel("llama-3.1-8b"), Q4, getDevice("rtx-4090"), { ctx: 8192 }).warnings)
      .toEqual([]);
  });
});
