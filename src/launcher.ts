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
  /**
   * `FROM` and `PARAMETER` lines for a Modelfile, and nothing else: pasting
   * anything Ollama does not take there makes `ollama create` reject the file.
   */
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

/** A runtime a note can be about. `all` is a selection, never a subject. */
export type NoteRuntime = Exclude<LauncherRuntime, "all">;

export interface LauncherNote {
  text: string;
  /** The runtimes this note is about, so a narrowed report can filter it. */
  runtimes: readonly NoteRuntime[];
}

export interface LauncherPlan {
  llamaCpp: LlamaCppPlan;
  ollama: OllamaPlan;
  vllm: VllmPlan;
  /** Anything about these flags that would otherwise surprise you. */
  notes: string[];
  /**
   * The same notes, each tagged with the runtimes it is about. `notes` is this
   * list's text in the same order, so the two cannot drift; the JSON payload
   * keeps the untagged form, and a report narrowed with `--launcher vllm`
   * filters on the tags.
   */
  runtimeNotes: readonly LauncherNote[];
}

/** The quant families that only exist inside a GGUF file. */
const GGUF_FAMILIES: ReadonlySet<string> = new Set(["gguf-k", "gguf-legacy"]);

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
 * The quantized cache types Ollama's `OLLAMA_KV_CACHE_TYPE` accepts. llama.cpp
 * takes q5_1, q5_0 and q4_1 as well; Ollama does not, and printing one of them
 * puts a value in the environment that its daemon rejects.
 */
const OLLAMA_CACHE_TYPES: ReadonlySet<string> = new Set(["q8_0", "q4_0"]);
/** Cache types that need no flag anywhere, because they are the default. */
const UNQUANTIZED_CACHE_TYPES: ReadonlySet<string> = new Set(["f16", "bf16"]);

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

/**
 * vLLM's `--quantization`, for the formats it loads by name.
 *
 * `gguf` describes the *file*, not the weights, so it is only right when the
 * path being served is a GGUF one. A safetensors checkpoint or a bare
 * `<org/model>` carrying a GGUF-family quant would be told to load a format
 * it is not in -- a command that looks right and fails at load, which is the
 * outcome the placeholder rules above exist to avoid.
 */
function vllmQuantization(family: string, servesGguf: boolean): string | null {
  if (family === "awq") return "awq";
  if (family === "gptq") return "gptq";
  if (family === "gguf-k" || family === "gguf-legacy") return servesGguf ? "gguf" : null;
  return null;
}

function llamaCppPlan(fit: FitResult, options: LauncherOptions): LlamaCppPlan {
  const nGpuLayers = gpuLayersFor(fit);
  const ubatch = fit.physicalBatch;
  const cacheType = UNQUANTIZED_CACHE_TYPES.has(fit.kvQuant.id) ? null : fit.kvQuant.id;

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
  if (fit.physicalBatch !== DEFAULT_UBATCH) {
    modelfile.push(`PARAMETER num_batch ${fit.physicalBatch}`);
  }

  // Everything below is process-wide, so Ollama takes it from the environment
  // and not from a Modelfile -- concurrency included: `num_parallel` is not a
  // Modelfile parameter, and `ollama create` refuses a file that carries one.
  const environment = [`OLLAMA_FLASH_ATTENTION=${fit.flashAttention ? 1 : 0}`];
  if (fit.batch > 1) environment.push(`OLLAMA_NUM_PARALLEL=${fit.batch}`);
  if (OLLAMA_CACHE_TYPES.has(fit.kvQuant.id)) {
    environment.push(`OLLAMA_KV_CACHE_TYPE=${fit.kvQuant.id}`);
  }

  const numericOptions: Record<string, number> = { num_gpu: numGpu, num_ctx: fit.ctx };

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
  const served = options.checkpointPath ?? options.ggufPath ?? "<org/model>";
  const quantization = vllmQuantization(
    fit.quant.family,
    options.checkpointPath === undefined && options.ggufPath !== undefined,
  );

  const args = [
    served,
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

/**
 * The notes, each about the runtimes it actually concerns.
 *
 * `--launcher vllm` says the reader wants one runtime's answer, and three
 * quarters of the commentary being about the other two -- naming flags that
 * were not printed -- makes the narrowed form less useful than the default.
 */
function collectNotes(
  fit: FitResult,
  plan: { vllm: VllmPlan; llamaCpp: LlamaCppPlan },
  options: LauncherOptions,
): LauncherNote[] {
  const notes: LauncherNote[] = [];
  const note = (runtimes: readonly NoteRuntime[], text: string): void => {
    notes.push({ text, runtimes });
  };

  if (options.ggufPath === undefined && options.checkpointPath !== undefined) {
    note(
      ["llama.cpp", "ollama"],
      "llama.cpp loads GGUF only, so -m is a placeholder: convert the checkpoint first with convert_hf_to_gguf.py and quantize it to the format checked above.",
    );
  }

  if (fit.offload !== null) {
    note(
      ["llama.cpp", "ollama"],
      `-ngl ${plan.llamaCpp.nGpuLayers} (num_gpu to Ollama) leaves ${fit.offload.plan.cpuLayers} of ${fit.model.nLayers} layers on the CPU. That is the most that fits; a higher number will load and then run out of memory.`,
    );
    if (!fit.offload.feasible) {
      note(
        ["llama.cpp", "ollama"],
        `The offloaded remainder needs ${(fit.offload.systemRamRequiredBytes / GIB).toFixed(2)} GiB of system RAM and only ${(fit.offload.systemRamAvailableBytes / GIB).toFixed(2)} GiB was assumed. Pass --ram to say what the machine really has.`,
      );
    }
    note(
      ["vllm"],
      "vLLM does not offload to system RAM: these flags describe the memory budget, not a configuration it can serve.",
    );
  }

  if (plan.vllm.gpuMemoryUtilization >= MAX_GPU_MEMORY_UTILIZATION) {
    note(
      ["vllm"],
      `--gpu-memory-utilization is capped at ${MAX_GPU_MEMORY_UTILIZATION}: this deployment wants ${((fit.usedBytes / fit.capacity.installedBytes) * 100).toFixed(0)}% of the card, and above 95% there is no room left for CUDA graph capture and allocator fragmentation.`,
    );
  }

  if (plan.vllm.quantization === "gguf") {
    note(
      ["vllm"],
      "vLLM's GGUF loader is experimental and single-file only; the usual path is to serve the safetensors checkpoint and let --quantization pick the kernel.",
    );
  } else if (GGUF_FAMILIES.has(fit.quant.family)) {
    note(
      ["vllm"],
      `vLLM cannot load a llama.cpp ${fit.quant.label} mix out of ${options.checkpointPath === undefined ? "a repository" : "safetensors"}, so no --quantization is given: the figures above describe the memory budget of the equivalent GGUF deployment rather than a format vLLM names.`,
    );
  }

  if (fit.gpus > 1) {
    note(
      ["llama.cpp", "vllm"],
      `llama.cpp splits by layer across the ${fit.gpus} devices, which buys capacity and not speed. vLLM's --tensor-parallel-size ${fit.gpus} does raise decode throughput; the estimate above does not model that.`,
    );
  }

  if (plan.llamaCpp.cacheType !== null) {
    note(
      ["llama.cpp"],
      `A ${plan.llamaCpp.cacheType} cache needs a build with flash attention available; llama.cpp refuses the K cache type otherwise.`,
    );
    if (!OLLAMA_CACHE_TYPES.has(plan.llamaCpp.cacheType)) {
      note(
        ["ollama"],
        `OLLAMA_KV_CACHE_TYPE takes f16, q8_0 and q4_0 only, so there is no Ollama spelling for a ${plan.llamaCpp.cacheType} cache and none is printed. Use q8_0 there, or serve this one with llama.cpp.`,
      );
    }
  }

  return notes;
}

/** Every flag implied by one fit, for the three runtimes people use. */
export function launcherPlan(fit: FitResult, options: LauncherOptions = {}): LauncherPlan {
  const llamaCpp = llamaCppPlan(fit, options);
  const ollama = ollamaPlan(fit, options);
  const vllm = vllmPlan(fit, options);
  const runtimeNotes = collectNotes(fit, { llamaCpp, vllm }, options);
  return { llamaCpp, ollama, vllm, notes: runtimeNotes.map((note) => note.text), runtimeNotes };
}

/** The notes about one runtime, or all of them in order for `all`. */
export function notesFor(plan: LauncherPlan, runtime: LauncherRuntime): string[] {
  return plan.runtimeNotes
    .filter((note) => runtime === "all" || note.runtimes.includes(runtime))
    .map((note) => note.text);
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
