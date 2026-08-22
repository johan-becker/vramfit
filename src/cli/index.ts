import { readFileSync, statSync } from "node:fs";
import { getDevice, getModel, listDevices, listModels, parseDeviceSpec, parseModelSpec } from "../db/index.js";
import { compareDevices, parseDeviceList, type DeviceCandidate } from "../compare.js";
import { checkFit, evaluateQuants, recommendQuant, type FitOptions } from "../fit.js";
import {
  parseFleetConfig,
  planFleet,
  type FleetEntry,
  type FleetMachine,
} from "../fleet.js";
import { openByteSource } from "../gguf/index.js";
import {
  LAUNCHER_RUNTIMES,
  findLauncherRuntime,
  gpuLayersFor,
  launcherPlan,
  type LauncherPlan,
  type LauncherRuntime,
} from "../launcher.js";
import { getQuant } from "../quant.js";
import {
  USE_CASE_IDS,
  findUseCase,
  recommendModels,
  type RecommendOptions,
} from "../recommend.js";
import type { DeviceSpec, QuantSpec } from "../types.js";
import { Args, UsageError } from "./args.js";
import {
  directoryOf,
  isDirectoryPath,
  looksLikePath,
  looksLikeVramfitSpec,
  readJsonFile,
  resolveGgufPath,
  resolveHfCheckpoint,
  type PathKind,
  type ResolvedModel,
  type SourceIo,
} from "./source.js";
import {
  bestJson,
  checkJson,
  compareJson,
  fleetJson,
  recommendJson,
  renderBest,
  renderCheck,
  renderCompare,
  renderDevices,
  renderFleet,
  renderModels,
  renderRecommend,
} from "./report.js";

/**
 * Command dispatch.
 *
 * `run` is a pure function of its arguments and its IO object: it returns an
 * exit code and never touches `process`. That is what makes the CLI testable
 * without spawning anything, and it is why the whole surface -- exit codes,
 * error text, JSON payloads -- is covered by ordinary unit tests.
 */

/**
 * Everything the CLI is allowed to touch.
 *
 * `openBytes` and `pathKind` are optional so that an embedder who only wants
 * report text -- and the report tests, which have no filesystem at all -- does
 * not have to implement them. A command that needs one says so by name when it
 * is missing, rather than reaching around the object to the real disk.
 */
export interface Io extends SourceIo {
  out(text: string): void;
  err(text: string): void;
}

export const EXIT_OK = 0;
/** The model does not fit. Deliberately non-zero so it can gate a deploy. */
export const EXIT_DOES_NOT_FIT = 1;
/** Bad arguments, unknown model or device, malformed user JSON. */
export const EXIT_USAGE = 2;

export const defaultIo: Io = {
  out(text) {
    process.stdout.write(`${text}\n`);
  },
  err(text) {
    process.stderr.write(`${text}\n`);
  },
  readFile(path) {
    return readFileSync(path, "utf8");
  },
  openBytes(path) {
    return openByteSource(path);
  },
  pathKind(path): PathKind {
    const stat = statSync(path, { throwIfNoEntry: false });
    if (stat === undefined) return "missing";
    return stat.isDirectory() ? "directory" : "file";
  },
};

/** Read from package.json so `--version` cannot drift from the published one. */
export function readVersion(): string {
  try {
    const raw = readFileSync(new URL("../../package.json", import.meta.url), "utf8");
    const parsed = JSON.parse(raw) as { version?: unknown };
    return typeof parsed.version === "string" ? parsed.version : "0.0.0";
  } catch {
    return "0.0.0";
  }
}

const HELP = `vramfit -- will this model run on my machine, and how fast?

USAGE
  vramfit check <model> --device <device> [options]
  vramfit best  <model> --device <device> [options]
  vramfit compare <model> --devices <list> [options]
  vramfit recommend --device <device> [options]
  vramfit fleet --config <fleet.json> [options]
  vramfit devices [--json]
  vramfit models  [--json]

COMMANDS
  check     Full report for one model, quantization, context and device.
            Exits 1 when it does not fit, so it can gate a deploy script.
  best      Every quantization ranked by quality, with the largest context
            and the decode speed each one leaves room for.
  compare   One model across several devices: fits, headroom, largest
            context and decode speed, best first.
  recommend What to run on the hardware you have: every bundled model that
            fits, ranked, each with the trade-off it asks of you.
  fleet     Which of several machines can serve which of several models,
            from a JSON description of the hardware you have.
  devices   List the bundled devices.
  models    List the bundled models.

MODEL AND DEVICE
  <model>                  Bundled id, name or alias, e.g. llama-3.1-8b,
                           "Llama 3.1 8B", llama3.1:8b -- see "vramfit models"
                           -- or a path: a GGUF file read from its own header
                           (./Llama-3.1-8B-Q4_K_M.gguf), or a HuggingFace
                           checkpoint directory (./Llama-3.1-8B/).
  -d, --device <id>        Bundled device id, name or alias, e.g. 4090,
                           "RTX 4090", m3-max. See "vramfit devices".
      --devices <list>     Comma-separated devices for "compare", each with
                           an optional count: 4090,3090x2,m4-max.
      --gguf <path>        Read the model from a GGUF file whatever it is
                           named. Only the header is read, never the weights.
      --hf-config <path>   Read the model from a HuggingFace config.json.
                           A safetensors index beside it, if there is one,
                           gives the true parameter count.
      --model-json <path>  Use a model spec from a JSON file instead.
      --device-json <path> Use a device spec from a JSON file instead.

CONFIGURATION
  -q, --quant <id>         Weight quantization. Defaults to the format the
                           model ships in, else q4_k_m. f16, q8_0, q6_k,
                           q5_k_m, q4_k_m, q4_k_s, q3_k_m, q2_k, mxfp4,
                           awq-4bit, gptq-4bit and friends.
  -c, --ctx <n>            Context length; accepts 32768 or 32k. Defaults to
                           the model's own default.
  -b, --batch <n>          Concurrent sequences (default 1).
      --kv-quant <id>      KV cache type: f16, q8_0, q5_1, q5_0, q4_1, q4_0.
  -g, --gpus <n>           Identical devices sharing the model (default 1).
      --vram <GiB>         Usable memory per device, used as given. Not
                           scaled by the device's usable fraction.
      --ubatch <n>         Physical batch, llama.cpp --ubatch-size (default 512).
      --no-flash-attn      Model the compute buffer without flash attention.
      --prompt <n>         Prompt length for time-to-first-token.

OFFLOAD (used when the model does not fit)
      --ram <GiB>          System RAM available for offloaded layers.
      --ram-bandwidth <GB/s>   System RAM bandwidth (default 89.6, DDR5-5600).
      --cpu-tflops <n>     CPU dense FP16 throughput, for offloaded prefill.

CALIBRATION (device side only; offloaded layers keep their derived figures)
      --efficiency <0-1>       Override the memory-bandwidth efficiency.
      --prefill-efficiency <0-1>  Override the prefill MFU.

FLEET
      --config <path>      JSON: { machines: [{ name, device, gpus, vram,
                           ram }], models: [...], ctx, quant }. A model entry
                           is a name, a path, or { model, quant, ctx }.

RECOMMEND
      --use-case <id>      chat, code or long-context. Sets the context to
                           check at and the decode speed to clear (default
                           chat: 8K and 15 tok/s).
      --limit <n>          Show only the top n models.

LAUNCHER
      --launcher [runtime] Print the exact flags this fit implies, for
                           llama.cpp, ollama, vllm, or all three (default).
      --ngl                Print only the llama.cpp -ngl value and exit, for
                           use in a shell substitution.

OUTPUT
      --json               Machine-readable output.
  -h, --help               This text.
  -v, --version            Print the version.

EXAMPLES
  vramfit check llama-3.1-8b --device 4090 --ctx 32k
  vramfit check llama-3.3-70b -d 3090 -g 2 -q q4_k_m --ctx 8k
  vramfit best qwen2.5-32b --device m3-max
  vramfit compare llama-3.3-70b --devices 4090,4090x2,a100-80,m3-ultra
  vramfit recommend -d m4-max --use-case code
  vramfit fleet --config ./fleet.json
  vramfit check llama-3.3-70b -d 4090 --ctx 8k --launcher llama.cpp
  vramfit check gemma-3-27b -d 3060 --ram 64 --json
  vramfit check ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -d 4090 --ctx 32k
  vramfit check ./Qwen3-32B/ -d m4-max --ctx 32k

EXIT CODES
  0  fits    1  does not fit    2  bad usage`;

const SHARED_FLAGS = [
  "device",
  "device-json",
  "model-json",
  "gguf",
  "hf-config",
  "ctx",
  "batch",
  "kv-quant",
  "gpus",
  "vram",
  "ram",
  "ram-bandwidth",
  "cpu-tflops",
  "ubatch",
  "flash-attn",
  "prompt",
  "efficiency",
  "prefill-efficiency",
  "json",
  "help",
] as const;

function loadJsonFile<T>(
  io: Io,
  path: string,
  parse: (value: unknown, label: string) => T,
): T {
  return parse(readJsonFile(io, path), path);
}

/**
 * A path the user typed as the `<model>` argument.
 *
 * The extension decides how to read it: GGUF files are parsed from their own
 * header. Anything else is refused by name rather than sniffed, because a
 * wrong guess about a file format is a wrong answer about a deployment.
 */
function resolvePath(io: Io, path: string): ResolvedModel {
  const lower = path.toLowerCase();
  if (lower.endsWith(".gguf")) return resolveGgufPath(io, path);
  if (isDirectoryPath(io, path)) return resolveHfCheckpoint(io, path);
  if (lower.endsWith(".json")) {
    // A .json here is a HuggingFace config unless it is one of vramfit's own
    // specs, which is decided by looking rather than by asking: `nLayers` is
    // vramfit's spelling and `num_hidden_layers` is HuggingFace's, so the two
    // can never be confused for one another.
    const parsed = readJsonFile(io, path);
    if (looksLikeVramfitSpec(parsed)) {
      return { model: parseModelSpec(parsed, path), origin: "json", from: path };
    }
    return resolveHfCheckpoint(io, path, parsed);
  }
  throw new UsageError(
    `${path} is not a format vramfit reads. Pass a .gguf file, a HuggingFace checkpoint directory or its config.json, a bundled model name (see "vramfit models"), or a spec with --model-json.`,
  );
}

/** Resolve the `<model>` argument, from the database or from a file. */
function resolveModelSource(args: Args, io: Io): ResolvedModel {
  const ggufPath = args.string("gguf");
  if (ggufPath !== undefined) return resolveGgufPath(io, ggufPath);

  const hfPath = args.string("hf-config");
  if (hfPath !== undefined) return resolveHfCheckpoint(io, hfPath);

  const specPath = args.string("model-json");
  if (specPath !== undefined) {
    return { model: loadJsonFile(io, specPath, parseModelSpec), origin: "json", from: specPath };
  }

  const name = args.positionals[1];
  if (name === undefined) {
    throw new UsageError("A model is required. Try \"vramfit models\" for the bundled list.");
  }
  if (looksLikePath(name)) return resolvePath(io, name);
  return { model: getModel(name), origin: "database", from: name };
}

/**
 * The quantization to check in.
 *
 * `-q` wins. Otherwise a source that is already in a format -- a GGUF file,
 * measured from its own tensor table -- dictates it, then a model released in
 * one, then the default recommendation.
 */
function resolveQuant(args: Args, source: ResolvedModel): QuantSpec {
  const requested = args.string("quant");
  if (requested !== undefined) return getQuant(requested);
  if (source.quant !== undefined) return source.quant;
  return getQuant(source.model.nativeQuant ?? "q4_k_m");
}

function resolveDevice(args: Args, io: Io): DeviceSpec {
  const path = args.string("device-json");
  if (path !== undefined) return loadJsonFile(io, path, parseDeviceSpec);

  const name = args.string("device");
  if (name === undefined) {
    throw new UsageError("--device is required. Try \"vramfit devices\" for the bundled list.");
  }
  return getDevice(name);
}

function fitOptions(args: Args): FitOptions {
  const options: FitOptions = {};
  const ctx = args.tokens("ctx");
  if (ctx !== undefined) options.ctx = ctx;
  const batch = args.number("batch", { integer: true, min: 1 });
  if (batch !== undefined) options.batch = batch;
  const kvQuant = args.string("kv-quant");
  if (kvQuant !== undefined) options.kvQuant = kvQuant;
  const gpus = args.number("gpus", { integer: true, min: 1, max: 64 });
  if (gpus !== undefined) options.gpus = gpus;
  const vram = args.number("vram", { min: 0.25 });
  if (vram !== undefined) options.vramGiB = vram;
  const ram = args.number("ram", { min: 0.25 });
  if (ram !== undefined) options.systemRamGiB = ram;
  const ramBandwidth = args.number("ram-bandwidth", { min: 1 });
  if (ramBandwidth !== undefined) options.systemRamBandwidthGBs = ramBandwidth;
  const cpuTflops = args.number("cpu-tflops", { min: 0.01 });
  if (cpuTflops !== undefined) options.systemComputeTflops = cpuTflops;
  const ubatch = args.number("ubatch", { integer: true, min: 1 });
  if (ubatch !== undefined) options.physicalBatch = ubatch;
  const flashAttention = args.boolean("flash-attn");
  if (flashAttention !== undefined) options.flashAttention = flashAttention;
  const prompt = args.tokens("prompt");
  if (prompt !== undefined) options.promptTokens = prompt;
  const efficiency = args.number("efficiency", { min: 0.01, max: 1 });
  if (efficiency !== undefined) options.efficiency = efficiency;
  const prefillEfficiency = args.number("prefill-efficiency", { min: 0.01, max: 1 });
  if (prefillEfficiency !== undefined) options.prefillEfficiency = prefillEfficiency;
  return options;
}

/**
 * Refuse a positional the command has no use for.
 *
 * The parser gives value-less flags no separate value, so `--flash-attn false`
 * leaves "false" standing on its own. Ignoring it would silently answer a
 * different question from the one that was typed, which is the failure mode
 * the whole parser is written to avoid.
 */
function assertNoExtraArguments(args: Args, command: string, allowed: number): void {
  const extra = args.positionals[allowed];
  if (extra === undefined) return;
  const takes = allowed > 1 ? "one model name" : "no arguments";
  throw new UsageError(
    `Unexpected argument "${extra}". "${command}" takes ${takes}; a flag that carries a value needs it as --flag=value or --flag value.`,
  );
}

function emit(io: Io, lines: readonly string[]): void {
  io.out(lines.join("\n"));
}

function emitJson(io: Io, payload: unknown): void {
  io.out(JSON.stringify(payload, null, 2));
}

/**
 * Resolve `--launcher`.
 *
 * The value is optional -- `--launcher` on its own means all three runtimes,
 * which is what most people want -- so the raw flag is read rather than
 * `string()`, which would refuse the bare form as a missing value.
 */
function resolveLauncherRuntime(args: Args): LauncherRuntime | undefined {
  const raw = args.flag("launcher");
  if (raw === undefined) return undefined;
  if (raw === true) return "all";
  if (raw === false) return undefined;
  const runtime = findLauncherRuntime(raw);
  if (runtime === undefined) {
    throw new UsageError(
      `--launcher expects one of ${LAUNCHER_RUNTIMES.join(", ")}, got "${raw}"`,
    );
  }
  return runtime;
}

function runCheck(args: Args, io: Io, version: string): number {
  args.assertKnown([...SHARED_FLAGS, "quant", "launcher", "ngl"]);
  assertNoExtraArguments(args, "check", 2);

  const source = resolveModelSource(args, io);
  const model = source.model;
  const device = resolveDevice(args, io);
  // A model released in its own quantization -- or read out of a file that is
  // already in one -- is checked in that format unless the user asks for
  // another: modelling gpt-oss at Q4_K_M describes a file nobody publishes.
  const quant = resolveQuant(args, source);
  const options = fitOptions(args);

  const fit = checkFit(model, quant, device, options);

  // `--ngl` is the one-number form, for `-ngl $(vramfit check ... --ngl)`.
  // It prints the layer count and nothing else, so the substitution is usable
  // straight away; the exit code still says whether the model fits.
  if (args.boolean("ngl") === true) {
    io.out(String(gpuLayersFor(fit)));
    return fit.fits ? EXIT_OK : EXIT_DOES_NOT_FIT;
  }

  const recommendation = recommendQuant(model, device, options);
  const runtime = resolveLauncherRuntime(args);
  const launcher: LauncherPlan | undefined =
    runtime === undefined
      ? undefined
      : launcherPlan(fit, {
          // A GGUF path is passed as it stands; a HuggingFace checkpoint is
          // named by its directory, which is what vLLM and Ollama take, rather
          // than by the config.json inside it.
          ...(source.origin === "gguf"
            ? { ggufPath: source.from }
            : source.origin === "huggingface"
              ? { checkpointPath: directoryOf(source.from) }
              : {}),
          modelName: model.id,
        });

  if (args.boolean("json") === true) {
    emitJson(io, checkJson(fit, recommendation, version, source, launcher));
  } else {
    emit(
      io,
      renderCheck(fit, recommendation, {
        source,
        ...(launcher === undefined || runtime === undefined
          ? {}
          : { launcher: { plan: launcher, runtime } }),
      }),
    );
  }
  return fit.fits ? EXIT_OK : EXIT_DOES_NOT_FIT;
}

/** Resolve the `--devices 4090,3090x2,m4-max` list `compare` works from. */
function resolveDeviceList(args: Args): DeviceCandidate[] {
  const raw = args.string("devices");
  if (raw === undefined) {
    throw new UsageError(
      '--devices is required, as a comma-separated list: --devices 4090,3090x2,m4-max. Try "vramfit devices" for the bundled list.',
    );
  }
  return parseDeviceList(raw).map((entry) => ({
    device: getDevice(entry.query),
    gpus: entry.gpus,
  }));
}

function runCompare(args: Args, io: Io, version: string): number {
  args.assertKnown([...SHARED_FLAGS, "quant", "devices"]);
  assertNoExtraArguments(args, "compare", 2);

  const source = resolveModelSource(args, io);
  const model = source.model;
  const quant = resolveQuant(args, source);
  const candidates = resolveDeviceList(args);
  const options = fitOptions(args);
  const ctx = options.ctx ?? model.defaultCtx;

  const rows = compareDevices(model, quant, candidates, options);

  if (args.boolean("json") === true) {
    emitJson(io, compareJson(model, quant, rows, ctx, version));
  } else {
    emit(io, renderCompare(model, quant, rows, ctx, { source }));
  }
  return rows.some((row) => row.fit.fits) ? EXIT_OK : EXIT_DOES_NOT_FIT;
}

function runRecommend(args: Args, io: Io, version: string): number {
  args.assertKnown([...SHARED_FLAGS, "use-case", "limit"]);
  assertNoExtraArguments(args, "recommend", 1);

  const device = resolveDevice(args, io);
  const requested = args.string("use-case") ?? "chat";
  const profile = findUseCase(requested);
  if (profile === undefined) {
    throw new UsageError(
      `Unknown use case "${requested}". Expected one of ${USE_CASE_IDS.join(", ")}.`,
    );
  }

  const base = fitOptions(args);
  const options: RecommendOptions = { ...base, useCase: profile.id };
  const limit = args.number("limit", { integer: true, min: 1 });
  if (limit !== undefined) options.limit = limit;

  const ctx = base.ctx ?? profile.defaultContext;
  const gpus = base.gpus ?? 1;
  const rows = recommendModels(device, options);

  if (args.boolean("json") === true) {
    emitJson(io, recommendJson(device, profile, rows, ctx, gpus, version));
  } else {
    emit(io, renderRecommend(device, profile, rows, ctx, gpus));
  }
  return rows.length > 0 ? EXIT_OK : EXIT_DOES_NOT_FIT;
}

/**
 * Resolve one model entry of a fleet file.
 *
 * The entry is whatever `check` would accept in the same position -- a bundled
 * name or a path -- so a fleet file can mix the database with the checkpoints
 * actually sitting on the machines it describes.
 */
function resolveFleetEntry(
  io: Io,
  entry: { model: string; quant: string | undefined; ctx: number | undefined },
  defaults: { quant: string | undefined; ctx: number | undefined },
): FleetEntry {
  const source = looksLikePath(entry.model)
    ? resolvePath(io, entry.model)
    : { model: getModel(entry.model), origin: "database" as const, from: entry.model };

  const requested = entry.quant ?? defaults.quant;
  const quant =
    requested !== undefined
      ? getQuant(requested)
      : (source.quant ?? getQuant(source.model.nativeQuant ?? "q4_k_m"));

  return {
    label: source.origin === "database" ? source.model.name : entry.model,
    model: source.model,
    quant,
    ctx: entry.ctx ?? defaults.ctx ?? source.model.defaultCtx,
  };
}

function runFleet(args: Args, io: Io, version: string): number {
  args.assertKnown(["config", "json", "help"]);
  assertNoExtraArguments(args, "fleet", 1);

  const path = args.string("config");
  if (path === undefined) {
    throw new UsageError(
      "--config is required: a JSON file describing the machines and the models to place on them.",
    );
  }
  const config = loadJsonFile(io, path, parseFleetConfig);

  const machines: FleetMachine[] = config.machines.map((machine) => {
    const options: FitOptions = { gpus: machine.gpus };
    if (machine.vramGiB !== undefined) options.vramGiB = machine.vramGiB;
    if (machine.systemRamGiB !== undefined) options.systemRamGiB = machine.systemRamGiB;
    if (config.kvQuant !== undefined) options.kvQuant = config.kvQuant;
    if (config.batch !== undefined) options.batch = config.batch;
    return { name: machine.name, device: getDevice(machine.device), gpus: machine.gpus, options };
  });

  const entries = config.models.map((entry) =>
    resolveFleetEntry(io, entry, { quant: config.quant, ctx: config.ctx }),
  );
  const report = planFleet(machines, entries);

  if (args.boolean("json") === true) {
    emitJson(io, fleetJson(report, version));
  } else {
    emit(io, renderFleet(report));
  }
  // Every model placed somewhere is the green case; anything homeless is the
  // one a deploy script wants to hear about.
  return report.unserved.length === 0 ? EXIT_OK : EXIT_DOES_NOT_FIT;
}

function runBest(args: Args, io: Io, version: string): number {
  args.assertKnown(SHARED_FLAGS);
  assertNoExtraArguments(args, "best", 2);

  const source = resolveModelSource(args, io);
  const model = source.model;
  const device = resolveDevice(args, io);
  const options = fitOptions(args);
  const evaluated = evaluateQuants(model, device, options);
  const ctx = options.ctx ?? model.defaultCtx;

  if (args.boolean("json") === true) {
    emitJson(io, bestJson(model, device, evaluated, ctx, version));
  } else {
    emit(io, renderBest(model, device, evaluated, ctx, options.gpus ?? 1, { source }));
  }
  return evaluated.some((option) => option.fit.fits) ? EXIT_OK : EXIT_DOES_NOT_FIT;
}

function runList(args: Args, io: Io, kind: "devices" | "models"): number {
  args.assertKnown(["json", "help"]);
  assertNoExtraArguments(args, kind, 1);
  const asJson = args.boolean("json") === true;

  if (kind === "devices") {
    if (asJson) emitJson(io, listDevices());
    else emit(io, renderDevices(listDevices()));
  } else if (asJson) {
    emitJson(io, listModels());
  } else {
    emit(io, renderModels(listModels()));
  }
  return EXIT_OK;
}

/**
 * Parse, dispatch, and turn any failure into a usage message and exit code.
 *
 * Everything after `readVersion` is inside the one try/catch, including the
 * --version and --help reads: `args.boolean` throws on a value it cannot make
 * sense of, and an escaped throw would leave Node to print a stack trace and
 * exit 1 -- the code that means "does not fit" to a deploy gate.
 */
export function run(argv: readonly string[], io: Io = defaultIo): number {
  const version = readVersion();

  try {
    const args = Args.parse(argv);
    const command = args.positionals[0];

    if (args.boolean("version") === true && command === undefined) {
      io.out(version);
      return EXIT_OK;
    }
    if (command === undefined || command === "help" || args.boolean("help") === true) {
      io.out(HELP);
      return command === undefined && args.boolean("help") !== true ? EXIT_USAGE : EXIT_OK;
    }

    switch (command) {
      case "check":
        return runCheck(args, io, version);
      case "best":
        return runBest(args, io, version);
      case "compare":
        return runCompare(args, io, version);
      case "recommend":
        return runRecommend(args, io, version);
      case "fleet":
        return runFleet(args, io, version);
      case "devices":
        return runList(args, io, "devices");
      case "models":
        return runList(args, io, "models");
      case "version":
        io.out(version);
        return EXIT_OK;
      default:
        throw new UsageError(
          `Unknown command "${command}". Expected check, best, compare, recommend, fleet, devices, models or help.`,
        );
    }
  } catch (error) {
    io.err(`vramfit: ${(error as Error).message}`);
    if (error instanceof UsageError) io.err('Try "vramfit --help".');
    return EXIT_USAGE;
  }
}
