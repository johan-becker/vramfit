import { describe, expect, it } from "vitest";
import { deriveArchitecture } from "../src/architecture.js";
import {
  createRegistry,
  findDevice,
  findModel,
  getDevice,
  getModel,
  listDevices,
  listModels,
  normalizeKey,
  parseDatabase,
  parseDeviceSpec,
  parseModelSpec,
  SpecValidationError,
} from "../src/db/index.js";
import type { DeviceSpec, ModelSpec } from "../src/types.js";

/**
 * The bundled database is data, so these tests treat it as data: every entry
 * is re-derived from its own architecture fields and compared against the
 * published parameter count. A typo in `hiddenSize` or a query-head count
 * pasted into `nKvHeads` moves that reconstruction by percent, not by rounding.
 */

const MODELS = listModels();
const DEVICES = listDevices();

function relative(actual: number, expected: number): number {
  return Math.abs(actual - expected) / expected;
}

/** A minimal valid model, used as the base for the rejection cases. */
function validModel(): Record<string, unknown> {
  return {
    id: "unit-test-7b",
    name: "Unit Test 7B",
    aliases: ["ut7b"],
    totalParams: 7_000_000_000,
    activeParams: 7_000_000_000,
    nLayers: 32,
    hiddenSize: 4096,
    nHeads: 32,
    nKvHeads: 8,
    headDim: 128,
    // The architecture has to add up to totalParams below: 32 layers of
    // 41.9M attention plus 3 x 4096 x 13824 of FFN, plus two 32000 x 4096
    // vocabulary matrices, is 7.04B.
    ffnHidden: 13_824,
    vocabSize: 32_000,
    tiedEmbeddings: false,
    attention: "gqa",
    mla: null,
    moe: null,
    attentionWindow: null,
    maxCtx: 32_768,
    defaultCtx: 8192,
    source: "unit test",
  };
}

function validDevice(): Record<string, unknown> {
  return {
    id: "unit-test-gpu",
    name: "Unit Test GPU",
    aliases: [],
    family: "cuda-consumer",
    vramGiB: 24,
    bandwidthGBs: 1008,
    fp16Tflops: 165.2,
    unifiedMemory: false,
    usableFraction: 1,
    source: "unit test",
  };
}

describe("bundled model database", () => {
  it("covers the families the README promises", () => {
    const ids = new Set(MODELS.map((model) => model.id));
    for (const required of [
      "llama-3.2-1b",
      "llama-3.1-8b",
      "llama-3.3-70b",
      "qwen2.5-7b",
      "qwen3-30b-a3b",
      "mistral-7b",
      "mixtral-8x7b",
      "gemma-2-9b",
      "gemma-3-27b",
      "phi-4",
      "deepseek-v2-lite",
      "gpt-oss-20b",
      "gpt-oss-120b",
    ]) {
      expect(ids, `missing ${required}`).toContain(required);
    }
    expect(MODELS.length).toBeGreaterThanOrEqual(20);
  });

  it("declares the format the natively-quantized models ship in", () => {
    for (const model of MODELS) {
      const native = model.nativeQuant;
      if (model.id.startsWith("gpt-oss")) {
        expect(native, model.id).toBe("mxfp4");
      } else {
        expect(native, model.id).toBeUndefined();
      }
    }
  });

  it("reconstructs every published parameter count from the architecture", () => {
    for (const model of MODELS) {
      const arch = deriveArchitecture(model);
      // 0.5% is loose enough for a published count rounded to three
      // significant figures and tight enough to catch a wrong head count,
      // which moves the total by 3-10%.
      expect(
        relative(arch.derivedTotalParams, model.totalParams),
        `${model.id}: derived ${arch.derivedTotalParams} vs published ${model.totalParams}`,
      ).toBeLessThan(0.005);
    }
  });

  it("keeps activeParams equal to totalParams for dense models only", () => {
    for (const model of MODELS) {
      if (model.moe === null) {
        expect(model.activeParams, model.id).toBe(model.totalParams);
      } else {
        expect(model.activeParams, model.id).toBeLessThan(model.totalParams);
      }
    }
  });

  it("reproduces the published active-parameter count for every MoE model", () => {
    // Vendors disagree on whether "active parameters" includes the embedding
    // table, the lm_head, both or neither -- gpt-oss counts one vocabulary
    // matrix, Qwen counts two -- so this is a 10% band rather than an equality.
    // It still catches a wrong expertsPerToken, which moves it by 2x or more.
    for (const model of MODELS) {
      if (model.moe === null) continue;
      const arch = deriveArchitecture(model);
      expect(
        relative(arch.activeMatmulParams, model.activeParams),
        `${model.id}: derived ${arch.activeMatmulParams} vs published ${model.activeParams}`,
      ).toBeLessThan(0.1);
    }
  });

  it("never confuses query heads with KV heads", () => {
    for (const model of MODELS) {
      expect(model.nKvHeads, model.id).toBeLessThanOrEqual(model.nHeads);
      expect(model.nHeads % model.nKvHeads, model.id).toBe(0);
    }
    // Spot-check the values that make or break a 70B estimate.
    expect(getModel("llama-3.3-70b").nKvHeads).toBe(8);
    expect(getModel("llama-3.3-70b").nHeads).toBe(64);
    expect(getModel("qwen2.5-7b").nKvHeads).toBe(4);
  });

  it("carries the geometry each attention flavour needs", () => {
    for (const model of MODELS) {
      if (model.attention === "mla") {
        expect(model.mla, model.id).not.toBeNull();
      } else {
        expect(model.mla, model.id).toBeNull();
      }
      if (model.attentionWindow) {
        expect(model.attentionWindow.fullAttentionEvery, model.id).toBeGreaterThan(1);
        expect(model.attentionWindow.windowSize, model.id).toBeGreaterThan(0);
      }
    }
    expect(getModel("deepseek-v2-lite").mla?.kvLoraRank).toBe(512);
    expect(getModel("gemma-3-27b").attentionWindow?.fullAttentionEvery).toBe(6);
  });

  it("records where every entry came from", () => {
    for (const model of MODELS) {
      expect(model.source, model.id).toMatch(/config\.json/);
    }
  });

  it("keeps defaultCtx inside maxCtx", () => {
    for (const model of MODELS) {
      expect(model.defaultCtx, model.id).toBeLessThanOrEqual(model.maxCtx);
    }
  });
});

describe("bundled device database", () => {
  it("covers the devices the README promises", () => {
    const ids = new Set(DEVICES.map((device) => device.id));
    for (const required of [
      "rtx-3060-12gb",
      "rtx-3090",
      "rtx-4090",
      "rtx-5090",
      "a100-80gb",
      "h100-80gb",
      "l40s",
      "rx-7900-xtx",
      "m1-max",
      "m4-max",
      "cpu-ddr5-dual",
    ]) {
      expect(ids, `missing ${required}`).toContain(required);
    }
  });

  it("has the headline RTX 4090 numbers exactly right", () => {
    const gpu = getDevice("4090");
    expect(gpu.vramGiB).toBe(24);
    expect(gpu.bandwidthGBs).toBe(1008); // 384-bit GDDR6X at 21 Gbps
    expect(gpu.family).toBe("cuda-consumer");
    expect(gpu.usableFraction).toBe(1);
    expect(gpu.unifiedMemory).toBe(false);
  });

  it("marks Apple silicon as unified memory with the macOS wired limit applied", () => {
    const metal = DEVICES.filter((device) => device.family === "metal");
    expect(metal.length).toBeGreaterThanOrEqual(12);
    for (const device of metal) {
      expect(device.unifiedMemory, device.id).toBe(true);
      // macOS will not wire the whole of a unified pool for the GPU.
      expect(device.usableFraction, device.id).toBeLessThan(1);
      expect(device.usableFraction, device.id).toBeGreaterThanOrEqual(0.7);
      expect(device.notes ?? "", device.id).toMatch(/wired_limit_mb/);
    }
    expect(getDevice("m2-ultra").bandwidthGBs).toBe(800);
    expect(getDevice("m3-ultra").vramGiB).toBe(512);
  });

  it("gives discrete GPUs their whole framebuffer", () => {
    for (const device of DEVICES) {
      if (device.unifiedMemory) continue;
      expect(device.usableFraction, device.id).toBe(1);
    }
  });

  it("keeps bandwidth and compute in physically sane ranges", () => {
    for (const device of DEVICES) {
      // Dual-channel DDR4 at the bottom, HBM3 at the top.
      expect(device.bandwidthGBs, device.id).toBeGreaterThanOrEqual(50);
      expect(device.bandwidthGBs, device.id).toBeLessThanOrEqual(4000);
      expect(device.fp16Tflops, device.id).toBeGreaterThan(0);
      expect(device.fp16Tflops, device.id).toBeLessThanOrEqual(1000);
    }
  });
});

describe("lookup", () => {
  it("resolves ids, display names and aliases, ignoring case and separators", () => {
    const canonical = getModel("llama-3.1-8b");
    for (const spelling of [
      "llama-3.1-8b",
      "Llama-3.1-8B",
      "Llama 3.1 8B",
      "llama3.1:8b",
      "  llama_3.1_8b  ",
    ]) {
      expect(findModel(spelling), spelling).toBe(canonical);
    }
    expect(findDevice("RTX 4090")).toBe(getDevice("rtx-4090"));
    expect(findDevice("4090")).toBe(getDevice("rtx-4090"));
  });

  it("returns undefined rather than throwing from find()", () => {
    expect(findModel("gpt-5-turbo-max")).toBeUndefined();
    expect(findDevice("quantum-accelerator")).toBeUndefined();
  });

  it("suggests near matches when a name is unknown", () => {
    expect(() => getModel("llama-3.1-9b")).toThrow(/Unknown model "llama-3\.1-9b"/);
    expect(() => getModel("llama-3.1-9b")).toThrow(/Did you mean/);
    expect(() => getDevice("rtx-4091")).toThrow(/Did you mean: rtx-4090/);
  });

  it("finds every variant of a family by substring", () => {
    const gemma = listModels().filter((model) => model.id.startsWith("gemma"));
    expect(gemma.length).toBeGreaterThanOrEqual(5);
  });

  it("normalises the way the lookup keys assume", () => {
    expect(normalizeKey("  Llama 3.1 8B ")).toBe("llama-3.1-8b");
    expect(normalizeKey("RTX_4090")).toBe("rtx-4090");
  });

  it("rejects a database whose aliases collide", () => {
    const a = { id: "a", name: "A", aliases: ["shared"] };
    const b = { id: "b", name: "B", aliases: ["shared"] };
    expect(() => createRegistry([a, b], "widget")).toThrow(
      /Duplicate widget alias "shared".*"a".*"b"/,
    );
  });

  it("has no colliding ids, names or aliases in the bundled data", () => {
    // Building the registries above would already have thrown; this states it
    // as an expectation so the reason for the constraint is written down.
    expect(() => createRegistry(MODELS, "model")).not.toThrow();
    expect(() => createRegistry(DEVICES, "device")).not.toThrow();
  });
});

describe("validation", () => {
  it("accepts a well-formed model and device", () => {
    expect(parseModelSpec(validModel()).id).toBe("unit-test-7b");
    expect(parseDeviceSpec(validDevice()).vramGiB).toBe(24);
  });

  it("defaults a missing alias list to empty rather than failing", () => {
    const raw = validModel();
    delete raw["aliases"];
    expect(parseModelSpec(raw).aliases).toEqual([]);
  });

  const modelRejections: ReadonlyArray<[string, (raw: Record<string, unknown>) => void, RegExp]> = [
    ["a missing KV head count", (raw) => delete raw["nKvHeads"], /nKvHeads must be a finite number/],
    [
      "KV heads that do not divide the query heads",
      (raw) => {
        raw["nKvHeads"] = 7;
      },
      /must divide nHeads \(32\) evenly/,
    ],
    [
      "more KV heads than query heads",
      (raw) => {
        raw["nKvHeads"] = 64;
      },
      /cannot exceed nHeads/,
    ],
    [
      "active parameters above the total",
      (raw) => {
        raw["activeParams"] = 8e9;
      },
      /cannot exceed totalParams/,
    ],
    [
      "a dense model with fewer active than total parameters",
      (raw) => {
        raw["activeParams"] = 3e9;
      },
      /must equal totalParams for a dense model/,
    ],
    [
      "MLA declared with no MLA geometry",
      (raw) => {
        raw["attention"] = "mla";
      },
      /mla is required when attention is "mla"/,
    ],
    [
      "an unknown attention kind",
      (raw) => {
        raw["attention"] = "linear";
      },
      /attention must be one of gqa, mla/,
    ],
    [
      "a fractional layer count",
      (raw) => {
        raw["nLayers"] = 31.5;
      },
      /nLayers must be a whole number/,
    ],
    [
      "a default context beyond the maximum",
      (raw) => {
        raw["defaultCtx"] = 65_536;
      },
      /defaultCtx cannot exceed maxCtx/,
    ],
    [
      "an omitted moe block (rather than an explicit null)",
      (raw) => delete raw["moe"],
      /moe must be present, use null when it does not apply/,
    ],
    [
      "a layer count no transformer has",
      (raw) => {
        raw["nLayers"] = 20_000_000;
      },
      /nLayers must be at most 4096/,
    ],
    [
      "a vocabulary larger than any tokenizer",
      (raw) => {
        raw["vocabSize"] = 2 ** 30;
      },
      /vocabSize must be at most/,
    ],
    [
      "a name carrying terminal escape sequences",
      (raw) => {
        raw["name"] = "Unit\u001B[31m\u001B[2J\nFITS  -  0.00 GiB of 99.00 GiB used";
      },
      /name must not contain control characters/,
    ],
    [
      "a router that picks more experts than exist",
      (raw) => {
        raw["moe"] = {
          nExperts: 8,
          expertsPerToken: 9,
          expertFfnHidden: 14_336,
          nSharedExperts: 0,
          denseLayers: 0,
          denseFfnHidden: null,
        };
        raw["activeParams"] = 1e9;
      },
      /expertsPerToken cannot exceed nExperts/,
    ],
  ];

  for (const [label, mutate, pattern] of modelRejections) {
    it(`rejects ${label}`, () => {
      const raw = validModel();
      mutate(raw);
      expect(() => parseModelSpec(raw)).toThrow(pattern);
      expect(() => parseModelSpec(raw)).toThrow(SpecValidationError);
    });
  }

  it("rejects a parameter count the architecture cannot account for", () => {
    // A 1000x slip is the likeliest error in a hand-written spec, and it used
    // to pass: computeWeightBytes clamped the negative block count to zero and
    // the report printed "0.70 GiB, 8M params at 745.13 effective bits/weight"
    // with a cheerful FITS and exit 0.
    const raw = validModel();
    raw["totalParams"] = 7_000_000;
    raw["activeParams"] = 7_000_000;
    expect(() => parseModelSpec(raw, "my-model.json")).toThrow(
      /my-model\.json\.totalParams is 7000000, but the architecture describes/,
    );
    expect(() => parseModelSpec(raw)).toThrow(SpecValidationError);
  });

  it("still accepts a parameter count rounded the way model cards round them", () => {
    const raw = validModel();
    raw["totalParams"] = 7_000_000_000;
    raw["activeParams"] = 7_000_000_000;
    expect(parseModelSpec(raw).totalParams).toBe(7_000_000_000);
  });

  it("rejects an active-parameter count the router cannot produce", () => {
    const raw = JSON.parse(JSON.stringify(getModel("mixtral-8x7b"))) as Record<string, unknown>;
    // 2 of 8 experts is 12.88B active, whatever you count the vocabulary as;
    // 42B is the figure you get from mistaking active for total.
    raw["activeParams"] = 42_000_000_000;
    expect(() => parseModelSpec(raw, "moe.json")).toThrow(
      /moe\.json\.activeParams is 42000000000, but routing 2 of 8 experts/,
    );
    // The published figure itself still passes, vocabulary convention and all.
    expect(parseModelSpec(JSON.parse(JSON.stringify(getModel("mixtral-8x7b")))).activeParams).toBe(
      getModel("mixtral-8x7b").activeParams,
    );
  });

  it("rejects a device name carrying terminal escape sequences", () => {
    // A spec pasted from a gist could otherwise clear the screen and print a
    // forged verdict above the real one: every one of these strings is written
    // straight into the report.
    const raw = validDevice();
    raw["name"] = "Evil\u001B[2J\nFITS  -  0.00 GiB of 99.00 GiB used";
    expect(() => parseDeviceSpec(raw)).toThrow(/name must not contain control characters/);

    const aliased = validDevice();
    aliased["aliases"] = ["fine", "not\u0007fine"];
    expect(() => parseDeviceSpec(aliased)).toThrow(
      /aliases\[1\] must not contain control characters/,
    );
  });

  it("rejects a native quantization the tables do not know", () => {
    const raw = validModel();
    raw["nativeQuant"] = "fp3";
    expect(() => parseModelSpec(raw, "my-model.json")).toThrow(
      /my-model\.json\.nativeQuant must name a known quantization/,
    );
    raw["nativeQuant"] = "mxfp4";
    expect(parseModelSpec(raw).nativeQuant).toBe("mxfp4");
  });

  it("rejects a device that claims more usable memory than it has", () => {
    const raw = validDevice();
    raw["usableFraction"] = 1.2;
    expect(() => parseDeviceSpec(raw)).toThrow(/usableFraction must be at most 1/);
  });

  it("rejects an unknown device family", () => {
    const raw = validDevice();
    raw["family"] = "tpu";
    expect(() => parseDeviceSpec(raw)).toThrow(
      /family must be one of cuda-consumer, cuda-datacenter, rocm, metal, cpu/,
    );
  });

  it("names the exact path of a nested failure", () => {
    const raw = validModel();
    raw["attentionWindow"] = { windowSize: 0, fullAttentionEvery: 6 };
    expect(() => parseModelSpec(raw, "models.json.models[3]")).toThrow(
      /models\.json\.models\[3\]\.attentionWindow\.windowSize must be greater than 0/,
    );
  });

  it("refuses a database written for a different schema version", () => {
    const payload = { schemaVersion: 2, models: [validModel()] };
    expect(() => parseDatabase(payload, "models", parseModelSpec, "models.json")).toThrow(
      /schemaVersion is 2, but this build of vramfit understands 1/,
    );
  });

  it("refuses an empty database", () => {
    const payload = { schemaVersion: 1, models: [] };
    expect(() => parseDatabase(payload, "models", parseModelSpec, "models.json")).toThrow(
      /models must not be empty/,
    );
  });

  it("validates a user-supplied spec exactly like a bundled one", () => {
    // This is the path `--model-json` takes, so it has to be the same code.
    const custom = { ...validModel(), id: "my-finetune", nKvHeads: 5 };
    expect(() => parseModelSpec(custom, "my-model.json")).toThrow(
      /my-model\.json\.nKvHeads must divide nHeads/,
    );
  });

  it("round-trips a bundled entry through the parser unchanged", () => {
    const model: ModelSpec = getModel("mixtral-8x7b");
    expect(parseModelSpec(JSON.parse(JSON.stringify(model)))).toEqual(model);
    const device: DeviceSpec = getDevice("h100");
    expect(parseDeviceSpec(JSON.parse(JSON.stringify(device)))).toEqual(device);
  });
});
