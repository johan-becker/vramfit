import type { FitResult } from "./fit.js";
import { GIB } from "./units.js";

/**
 * The flags to actually type.
 *
 * Everything else in this package answers "will it fit". This answers "then
 * what do I run", which is the step where the arithmetic usually gets thrown
 * away and replaced by `-ngl 99` and a shrug. The numbers are already known by
 * the time a `FitResult` exists -- how many layers fit, what context they fit
 * at, how much of the card the whole thing occupies -- so producing the
 * command line is a translation, not a new estimate.
 *
 * Three runtimes, because they spell the same three decisions differently:
 *
 *   llama.cpp  `-ngl` is a layer count, `-c` is the context, and the KV cache
 *              type is a pair of flags.
 *   Ollama     the same numbers as `num_gpu` and `num_ctx`, set per model in a
 *              Modelfile or per request in the API, with flash attention and
 *              the cache type as environment variables rather than flags.
 *   vLLM       a memory *fraction* rather than a layer count, because it
 *              pre-allocates the KV cache to fill whatever it is given.
 *
 * Every value below is derived from the fit and none of them is a default
 * copied from a README.
 */

/**
 * Where the model is, told apart by what each runtime can actually open.
 *
 * They are not interchangeable, and pretending they are produces a command
 * line that looks right and fails: llama.cpp's `-m` takes a GGUF file and
 * nothing else, vLLM takes a HuggingFace repository or a directory of
 * safetensors, and Ollama's `FROM` takes either of those or a registry name.
 * A placeholder is printed wherever there is nothing of the right kind to
 * give, which is the honest output for a model looked up in the database.
 */
export interface LauncherOptions {
  /** A `.gguf` file on disk. */
  ggufPath?: string;
  /** A HuggingFace checkpoint directory on disk. */
  checkpointPath?: string;
  /** A registry name, for the runtimes that pull rather than load. */
  modelName?: string;
}

export interface LlamaCppPlan {
  /** `-ngl`: transformer layers placed in device memory. */
  nGpuLayers: number;
  /** `-c`. */
  ctx: number;
  /** `-ub`, omitted from the command line when it is llama.cpp's own default. */
  ubatch: number;
  /** `--parallel`, when more than one sequence is served at once. */
  parallel: number;
  flashAttention: boolean;
  /** `--cache-type-k` / `--cache-type-v`, omitted when the cache is f16. */
  cacheType: string | null;
  /** The full argument list, in the order a person would type it. */
  args: string[];
  /** `llama-server ...`, ready to paste. */
  command: string;
}

export interface OllamaPlan {
  numGpu: number;
  numCtx: number;
  numBatch: number;
  /** `PARAMETER` lines for a Modelfile. */
  modelfile: string[];
  /** The same settings as the `options` object of an API request. */
  options: Record<string, number>;
  /** Environment variables, which is where Ollama keeps the rest. */
  environment: string[];
}

export interface VllmPlan {
  /** `--gpu-memory-utilization`: the share of each card vLLM may occupy. */
  gpuMemoryUtilization: number;
  /** `--max-model-len`. */
  maxModelLen: number;
  /** `--tensor-parallel-size`. */
  tensorParallelSize: number;
  /** `--max-num-seqs`. */
  maxNumSeqs: number;
  /** `--kv-cache-dtype`, when the plan quantizes the cache. */
  kvCacheDtype: string | null;
  /** `--quantization`, when the weights are in a format vLLM names. */
  quantization: string | null;
  args: string[];
  command: string;
}

export interface LauncherPlan {
  llamaCpp: LlamaCppPlan;
  ollama: OllamaPlan;
  vllm: VllmPlan;
  /** Anything about these flags that would otherwise surprise you. */
  notes: string[];
}

/** llama.cpp's own default `--ubatch-size`; printing it would be noise. */
const DEFAULT_UBATCH = 512;
/**
 * The most of a card vLLM should be told it may have. Above this there is no
 * room for CUDA graph capture and allocator fragmentation, and the server
 * fails at startup rather than at the first long request.
 */
const MAX_GPU_MEMORY_UTILIZATION = 0.95;
/** Below this vLLM cannot allocate a usable number of KV blocks. */
const MIN_GPU_MEMORY_UTILIZATION = 0.1;

/**
 * `-ngl`.
 *
 * llama.cpp counts `n_layer + 1` offloadable layers: the repeating blocks plus
 * one for the output tensors. A model that fits gets all of them -- which is
 * what `-ngl 99` means, spelled exactly -- and one that does not gets the
 * number of blocks the offload plan actually placed.
 */
export function gpuLayersFor(fit: FitResult): number {
  if (fit.offload === null) return fit.model.nLayers + 1;
  return fit.offload.plan.vocabOnDevice
    ? fit.offload.plan.gpuLayers + 1
    : fit.offload.plan.gpuLayers;
}

/**
 * `--gpu-memory-utilization`.
 *
 * vLLM does not take a layer count. It takes the fraction of each device it
 * may occupy in total -- weights, cache, activations and its own overhead --
 * and fills whatever is left after the weights with KV blocks. So the number
 * to give it is this deployment's total footprint over the card's *installed*
 * memory, not over the usable figure, because vLLM measures the same way
 * `nvidia-smi` does.
 *
 * Rounded up to the nearest hundredth, since rounding down would ask for less
 * memory than the plan needs, and capped short of the top of the card.
 */
export function gpuMemoryUtilizationFor(fit: FitResult): number {
  const installed = fit.capacity.installedBytes;
  if (installed <= 0) return MAX_GPU_MEMORY_UTILIZATION;
  const raw = fit.usedBytes / installed;
  const rounded = Math.ceil(raw * 100) / 100;
  return Math.min(MAX_GPU_MEMORY_UTILIZATION, Math.max(MIN_GPU_MEMORY_UTILIZATION, rounded));
}

/** vLLM's `--kv-cache-dtype`, which has fewer options than llama.cpp's. */
function vllmCacheDtype(kvQuantId: string): string | null {
  if (kvQuantId === "f16" || kvQuantId === "bf16") return null;
  // vLLM quantizes the cache to 8-bit floating point rather than to a
  // block-quantized integer format; it is the same halving, by another route.
  return "fp8";
}

/** vLLM's `--quantization`, for the formats it loads by name. */
function vllmQuantization(family: string): string | null {
  if (family === "awq") return "awq";
  if (family === "gptq") return "gptq";
  if (family === "gguf-k" || family === "gguf-legacy") return "gguf";
  return null;
}

function llamaCppPlan(fit: FitResult, options: LauncherOptions): LlamaCppPlan {
  const nGpuLayers = gpuLayersFor(fit);
  const ubatch = fit.physicalBatch;
  const cacheType = fit.kvQuant.id === "f16" || fit.kvQuant.id === "bf16" ? null : fit.kvQuant.id;

  const args = [
    "-m",
    options.ggufPath ?? "<model.gguf>",
    "-ngl",
    String(nGpuLayers),
    "-c",
    String(fit.ctx),
  ];
  if (cacheType !== null) {
    args.push("--cache-type-k", cacheType, "--cache-type-v", cacheType);
  }
  if (ubatch !== DEFAULT_UBATCH) args.push("-ub", String(ubatch));
  // llama.cpp takes on/off/auto here; naming it explicitly is the point of
  // printing a command line at all.
  args.push("-fa", fit.flashAttention ? "on" : "off");
  if (fit.batch > 1) args.push("--parallel", String(fit.batch));
  if (fit.gpus > 1) args.push("--split-mode", "layer");

  return {
    nGpuLayers,
    ctx: fit.ctx,
    ubatch,
    parallel: fit.batch,
    flashAttention: fit.flashAttention,
    cacheType,
    args,
    command: `llama-server ${args.join(" ")}`,
  };
}

function ollamaPlan(fit: FitResult, options: LauncherOptions): OllamaPlan {
  const numGpu = gpuLayersFor(fit);
  const modelfile = [
    `FROM ${options.ggufPath ?? options.checkpointPath ?? options.modelName ?? "<model>"}`,
    `PARAMETER num_gpu ${numGpu}`,
    `PARAMETER num_ctx ${fit.ctx}`,
  ];
  if (fit.batch > 1) modelfile.push(`PARAMETER num_parallel ${fit.batch}`);
  if (fit.physicalBatch !== DEFAULT_UBATCH) {
    modelfile.push(`PARAMETER num_batch ${fit.physicalBatch}`);
  }

  // Ollama keeps these two out of the Modelfile: they are process-wide.
  const environment = [`OLLAMA_FLASH_ATTENTION=${fit.flashAttention ? 1 : 0}`];
  if (fit.kvQuant.id !== "f16" && fit.kvQuant.id !== "bf16") {
    environment.push(`OLLAMA_KV_CACHE_TYPE=${fit.kvQuant.id}`);
  }

  const numericOptions: Record<string, number> = { num_gpu: numGpu, num_ctx: fit.ctx };
  if (fit.batch > 1) numericOptions["num_parallel"] = fit.batch;

  return {
    numGpu,
    numCtx: fit.ctx,
    numBatch: fit.physicalBatch,
    modelfile,
    options: numericOptions,
    environment,
  };
}

function vllmPlan(fit: FitResult, options: LauncherOptions): VllmPlan {
  const gpuMemoryUtilization = gpuMemoryUtilizationFor(fit);
  const kvCacheDtype = vllmCacheDtype(fit.kvQuant.id);
  const quantization = vllmQuantization(fit.quant.family);

  const args = [
    options.checkpointPath ?? options.ggufPath ?? "<org/model>",
    "--max-model-len",
    String(fit.ctx),
    "--gpu-memory-utilization",
    gpuMemoryUtilization.toFixed(2),
  ];
  if (fit.gpus > 1) args.push("--tensor-parallel-size", String(fit.gpus));
  if (fit.batch > 1) args.push("--max-num-seqs", String(fit.batch));
  if (kvCacheDtype !== null) args.push("--kv-cache-dtype", kvCacheDtype);
  if (quantization !== null) args.push("--quantization", quantization);

  return {
    gpuMemoryUtilization,
    maxModelLen: fit.ctx,
    tensorParallelSize: fit.gpus,
    maxNumSeqs: fit.batch,
    kvCacheDtype,
    quantization,
    args,
    command: `vllm serve ${args.join(" ")}`,
  };
}

function collectNotes(
  fit: FitResult,
  plan: { vllm: VllmPlan; llamaCpp: LlamaCppPlan },
  options: LauncherOptions,
): string[] {
  const notes: string[] = [];

  if (options.ggufPath === undefined && options.checkpointPath !== undefined) {
    notes.push(
      "llama.cpp loads GGUF only, so -m is a placeholder: convert the checkpoint first with convert_hf_to_gguf.py and quantize it to the format checked above.",
    );
  }

  if (fit.offload !== null) {
    notes.push(
      `-ngl ${plan.llamaCpp.nGpuLayers} leaves ${fit.offload.plan.cpuLayers} of ${fit.model.nLayers} layers on the CPU. That is the most that fits; a higher number will load and then run out of memory.`,
    );
    if (!fit.offload.feasible) {
      notes.push(
        `The offloaded remainder needs ${(fit.offload.systemRamRequiredBytes / GIB).toFixed(2)} GiB of system RAM and only ${(fit.offload.systemRamAvailableBytes / GIB).toFixed(2)} GiB was assumed. Pass --ram to say what the machine really has.`,
      );
    }
    notes.push(
      "vLLM does not offload to system RAM: these flags describe the memory budget, not a configuration it can serve.",
    );
  }

  if (plan.vllm.gpuMemoryUtilization >= MAX_GPU_MEMORY_UTILIZATION) {
    notes.push(
      `--gpu-memory-utilization is capped at ${MAX_GPU_MEMORY_UTILIZATION}: this deployment wants ${((fit.usedBytes / fit.capacity.installedBytes) * 100).toFixed(0)}% of the card, and above 95% there is no room left for CUDA graph capture and allocator fragmentation.`,
    );
  }

  if (plan.vllm.quantization === "gguf") {
    notes.push(
      "vLLM's GGUF loader is experimental and single-file only; the usual path is to serve the safetensors checkpoint and let --quantization pick the kernel.",
    );
  }

  if (fit.gpus > 1) {
    notes.push(
      `llama.cpp splits by layer across the ${fit.gpus} devices, which buys capacity and not speed. vLLM's --tensor-parallel-size ${fit.gpus} does raise decode throughput; the estimate above does not model that.`,
    );
  }

  if (plan.llamaCpp.cacheType !== null) {
    notes.push(
      `A ${plan.llamaCpp.cacheType} cache needs a build with flash attention available; llama.cpp refuses the K cache type otherwise.`,
    );
  }

  return notes;
}

/** Every flag implied by one fit, for the three runtimes people use. */
export function launcherPlan(fit: FitResult, options: LauncherOptions = {}): LauncherPlan {
  const llamaCpp = llamaCppPlan(fit, options);
  const ollama = ollamaPlan(fit, options);
  const vllm = vllmPlan(fit, options);
  return { llamaCpp, ollama, vllm, notes: collectNotes(fit, { llamaCpp, vllm }, options) };
}

export const LAUNCHER_RUNTIMES = ["llama.cpp", "ollama", "vllm", "all"] as const;
export type LauncherRuntime = (typeof LAUNCHER_RUNTIMES)[number];

/** Resolve a `--launcher` value, accepting the spellings people use. */
export function findLauncherRuntime(value: string): LauncherRuntime | undefined {
  const normalized = value.trim().toLowerCase().replaceAll(/[\s_]+/g, "-");
  if (normalized === "llama.cpp" || normalized === "llama-cpp" || normalized === "llamacpp") {
    return "llama.cpp";
  }
  if (normalized === "ollama") return "ollama";
  if (normalized === "vllm") return "vllm";
  if (normalized === "all") return "all";
  return undefined;
}
