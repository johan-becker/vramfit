import { describe, expect, it } from "vitest";
import { EXIT_DOES_NOT_FIT, EXIT_OK, EXIT_USAGE, run } from "../src/cli/index.js";
import { getDevice } from "../src/db/index.js";
import { checkFit } from "../src/fit.js";
import {
  findLauncherRuntime,
  gpuLayersFor,
  gpuMemoryUtilizationFor,
  launcherPlan,
} from "../src/launcher.js";
import { getQuant } from "../src/quant.js";
import { FakeIo } from "./fake-io.js";
import { llamaGgufBytes } from "./gguf-fixtures.js";
import { LLAMA_3_1_8B, LLAMA_3_1_70B } from "./fixtures.js";

/**
 * The flags. Every value here is a translation of a `FitResult`, so what is
 * tested is that the translation is exact: the layer count is the one the
 * offload plan placed, the memory fraction is the one the footprint needs,
 * and the spelling is each runtime's own.
 */

function fitOf(model = LLAMA_3_1_8B, deviceId = "4090", options = {}) {
  return checkFit(model, getQuant("q4_k_m"), getDevice(deviceId), { ctx: 8192, ...options });
}

function invoke(argv: string[], binary: Record<string, Uint8Array> = {}) {
  const io = new FakeIo({ binary });
  return { code: run(argv, io), io };
}

describe("gpuLayersFor", () => {
  it("is n_layer + 1 when the whole model is resident", () => {
    // llama.cpp counts one extra offloadable layer for the output tensors,
    // which is what "-ngl 99" means, spelled exactly.
    expect(gpuLayersFor(fitOf())).toBe(LLAMA_3_1_8B.nLayers + 1);
  });

  it("is the number of layers the offload plan actually placed", () => {
    const fit = fitOf(LLAMA_3_1_70B, "4090", { ctx: 8192, systemRamGiB: 64 });
    expect(fit.fits).toBe(false);
    expect(fit.offload).not.toBeNull();
    expect(gpuLayersFor(fit)).toBe(fit.offload?.plan.gpuLayers);
    expect(gpuLayersFor(fit)).toBeLessThan(LLAMA_3_1_70B.nLayers);
  });
});

describe("gpuMemoryUtilizationFor", () => {
  it("is the footprint over the card's installed memory, rounded up", () => {
    const fit = fitOf();
    const raw = fit.usedBytes / fit.capacity.installedBytes;
    const value = gpuMemoryUtilizationFor(fit);
    // Rounding down would ask vLLM for less memory than the plan needs.
    expect(value).toBeGreaterThanOrEqual(raw);
    expect(value).toBeCloseTo(Math.ceil(raw * 100) / 100, 10);
  });

  it("is capped short of the top of the card", () => {
    // Above 95% there is no room for CUDA graph capture or allocator
    // fragmentation, and the server fails at startup rather than later.
    const fit = fitOf(LLAMA_3_1_70B, "a100-80gb", { ctx: 32_768 });
    expect(gpuMemoryUtilizationFor(fit)).toBeLessThanOrEqual(0.95);
  });
});

describe("launcherPlan", () => {
  it("spells one configuration three ways", () => {
    const plan = launcherPlan(fitOf(), { ggufPath: "./llama-3.1-8b-q4_k_m.gguf" });

    expect(plan.llamaCpp.command).toBe(
      "llama-server -m ./llama-3.1-8b-q4_k_m.gguf -ngl 33 -c 8192 -fa on",
    );
    expect(plan.ollama.modelfile).toEqual([
      "FROM ./llama-3.1-8b-q4_k_m.gguf",
      "PARAMETER num_gpu 33",
      "PARAMETER num_ctx 8192",
    ]);
    expect(plan.ollama.options).toEqual({ num_gpu: 33, num_ctx: 8192 });
    expect(plan.vllm.command).toMatch(/^vllm serve \.\/llama-3\.1-8b-q4_k_m\.gguf /);
    expect(plan.vllm.maxModelLen).toBe(8192);
    expect(plan.vllm.tensorParallelSize).toBe(1);
  });

  it("carries a quantized cache into each runtime's own spelling", () => {
    const plan = launcherPlan(fitOf(LLAMA_3_1_8B, "4090", { kvQuant: "q8_0" }));
    expect(plan.llamaCpp.args).toContain("--cache-type-k");
    expect(plan.llamaCpp.command).toMatch(/--cache-type-k q8_0 --cache-type-v q8_0/);
    // Ollama keeps it in the environment, not in the Modelfile.
    expect(plan.ollama.environment).toContain("OLLAMA_KV_CACHE_TYPE=q8_0");
    expect(plan.ollama.modelfile.join("\n")).not.toMatch(/q8_0/);
    // vLLM quantizes the cache to fp8 rather than to a block format.
    expect(plan.vllm.kvCacheDtype).toBe("fp8");
    expect(plan.notes.join(" ")).toMatch(/needs a build with flash attention available/);
  });

  it("says -fa off when the fit was computed without flash attention", () => {
    const plan = launcherPlan(fitOf(LLAMA_3_1_8B, "4090", { flashAttention: false }));
    expect(plan.llamaCpp.command).toMatch(/-fa off$/);
    expect(plan.ollama.environment).toContain("OLLAMA_FLASH_ATTENTION=0");
  });

  it("prints a non-default ubatch and a batch above one", () => {
    const plan = launcherPlan(fitOf(LLAMA_3_1_8B, "4090", { physicalBatch: 128, batch: 4 }));
    expect(plan.llamaCpp.command).toMatch(/-ub 128/);
    expect(plan.llamaCpp.command).toMatch(/--parallel 4/);
    expect(plan.ollama.modelfile).toContain("PARAMETER num_parallel 4");
    expect(plan.ollama.modelfile).toContain("PARAMETER num_batch 128");
    expect(plan.vllm.command).toMatch(/--max-num-seqs 4/);
  });

  it("names the tensor-parallel size, and what it does that a layer split does not", () => {
    const plan = launcherPlan(fitOf(LLAMA_3_1_70B, "3090", { gpus: 2 }));
    expect(plan.llamaCpp.command).toMatch(/--split-mode layer/);
    expect(plan.vllm.command).toMatch(/--tensor-parallel-size 2/);
    expect(plan.notes.join(" ")).toMatch(/buys capacity and not speed/);
  });

  it("warns that the ngl it printed is a ceiling, not a suggestion", () => {
    const plan = launcherPlan(fitOf(LLAMA_3_1_70B, "4090", { ctx: 8192, systemRamGiB: 64 }));
    expect(plan.notes.join(" ")).toMatch(
      /leaves \d+ of 80 layers on the CPU.*a higher number will load and then run out of memory/s,
    );
    expect(plan.notes.join(" ")).toMatch(/vLLM does not offload to system RAM/);
  });

  it("uses a placeholder rather than a path a runtime cannot open", () => {
    const fromDatabase = launcherPlan(fitOf(), { modelName: "llama-3.1-8b" });
    expect(fromDatabase.llamaCpp.command).toMatch(/-m <model\.gguf>/);
    expect(fromDatabase.ollama.modelfile[0]).toBe("FROM llama-3.1-8b");
    // vLLM wants a repository or a directory; a bundled id is neither, and a
    // command that looks right and fails is worse than a placeholder.
    expect(fromDatabase.vllm.command).toMatch(/^vllm serve <org\/model> /);

    const fromCheckpoint = launcherPlan(fitOf(), { checkpointPath: "./Qwen3-32B" });
    expect(fromCheckpoint.llamaCpp.command).toMatch(/-m <model\.gguf>/);
    expect(fromCheckpoint.vllm.command).toMatch(/^vllm serve \.\/Qwen3-32B /);
    expect(fromCheckpoint.notes.join(" ")).toMatch(/convert_hf_to_gguf\.py/);
  });
});

describe("findLauncherRuntime", () => {
  it("accepts the spellings people use", () => {
    expect(findLauncherRuntime("llama.cpp")).toBe("llama.cpp");
    expect(findLauncherRuntime("llama-cpp")).toBe("llama.cpp");
    expect(findLauncherRuntime("LlamaCpp")).toBe("llama.cpp");
    expect(findLauncherRuntime("vLLM")).toBe("vllm");
    expect(findLauncherRuntime("Ollama")).toBe("ollama");
    expect(findLauncherRuntime("all")).toBe("all");
    expect(findLauncherRuntime("tgi")).toBeUndefined();
  });
});

describe("vramfit check --launcher", () => {
  it("prints all three runtimes when no runtime is named", () => {
    const { code, io } = invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k", "--launcher"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/^Launch$/m);
    expect(io.output).toMatch(/^ {4}llama-server -m <model\.gguf> -ngl 33 -c 8192 -fa on$/m);
    expect(io.output).toMatch(/^ {4}PARAMETER num_gpu 33$/m);
    expect(io.output).toMatch(/^ {4}vllm serve <org\/model> --max-model-len 8192/m);
  });

  it("prints only the runtime asked for", () => {
    const { io } = invoke([
      "check",
      "llama-3.1-8b",
      "-d",
      "4090",
      "--ctx",
      "8k",
      "--launcher",
      "llama.cpp",
    ]);
    expect(io.output).toMatch(/llama-server/);
    expect(io.output).not.toMatch(/PARAMETER num_gpu/);
    expect(io.output).not.toMatch(/vllm serve/);
  });

  it("passes a GGUF path straight through to -m", () => {
    const { io } = invoke(
      ["check", "./m.gguf", "-d", "4090", "--ctx", "8k", "--launcher", "llama.cpp"],
      { "./m.gguf": llamaGgufBytes() },
    );
    expect(io.output).toMatch(/llama-server -m \.\/m\.gguf -ngl 33 -c 8192 -fa on/);
  });

  it("adds a launcher block to the JSON only when it was asked for", () => {
    const plain = JSON.parse(
      invoke(["check", "llama-3.1-8b", "-d", "4090", "--json"]).io.output,
    ) as Record<string, unknown>;
    expect(plain["launcher"]).toBeUndefined();

    const withFlags = JSON.parse(
      invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k", "--json", "--launcher"]).io
        .output,
    ) as { launcher: { llamaCpp: { nGpuLayers: number; args: string[] }; vllm: Record<string, unknown> } };
    expect(withFlags.launcher.llamaCpp.nGpuLayers).toBe(33);
    expect(withFlags.launcher.llamaCpp.args).toContain("-ngl");
    expect(withFlags.launcher.vllm["maxModelLen"]).toBe(8192);
  });

  it("refuses a runtime it has no flags for", () => {
    const { code, io } = invoke(["check", "llama-3.1-8b", "-d", "4090", "--launcher", "tgi"]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/--launcher expects one of llama\.cpp, ollama, vllm, all, got "tgi"/);
  });
});

describe("vramfit check --ngl", () => {
  it("prints the number and nothing else", () => {
    const { code, io } = invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k", "--ngl"]);
    expect(code).toBe(EXIT_OK);
    expect(io.stdout).toEqual(["33"]);
  });

  it("prints the offloaded count, and still exits 1", () => {
    const { code, io } = invoke([
      "check",
      "llama-3.3-70b",
      "-d",
      "4090",
      "--ctx",
      "8k",
      "--ram",
      "64",
      "--ngl",
    ]);
    expect(code).toBe(EXIT_DOES_NOT_FIT);
    expect(io.stdout).toHaveLength(1);
    const layers = Number(io.stdout[0]);
    expect(Number.isInteger(layers)).toBe(true);
    expect(layers).toBeGreaterThan(0);
    expect(layers).toBeLessThan(80);
  });
});
