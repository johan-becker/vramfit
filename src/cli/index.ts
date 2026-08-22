import { readFileSync } from "node:fs";
import { getDevice, getModel, listDevices, listModels, parseDeviceSpec, parseModelSpec } from "../db/index.js";
import { checkFit, evaluateQuants, recommendQuant, type FitOptions } from "../fit.js";
import { getQuant } from "../quant.js";
import type { DeviceSpec, ModelSpec } from "../types.js";
import { Args, UsageError } from "./args.js";
import {
  bestJson,
  checkJson,
  renderBest,
  renderCheck,
  renderDevices,
  renderModels,
} from "./report.js";

/**
 * Command dispatch.
 *
 * `run` is a pure function of its arguments and its IO object: it returns an
 * exit code and never touches `process`. That is what makes the CLI testable
 * without spawning anything, and it is why the whole surface -- exit codes,
 * error text, JSON payloads -- is covered by ordinary unit tests.
 */

export interface Io {
  out(text: string): void;
  err(text: string): void;
  readFile(path: string): string;
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
  vramfit devices [--json]
  vramfit models  [--json]

COMMANDS
  check     Full report for one model, quantization, context and device.
            Exits 1 when it does not fit, so it can gate a deploy script.
  best      Every quantization ranked by quality, with the largest context
            and the decode speed each one leaves room for.
  devices   List the bundled devices.
  models    List the bundled models.

MODEL AND DEVICE
  <model>                  Bundled id, name or alias, e.g. llama-3.1-8b,
                           "Llama 3.1 8B", llama3.1:8b. See "vramfit models".
  -d, --device <id>        Bundled device id, name or alias, e.g. 4090,
                           "RTX 4090", m3-max. See "vramfit devices".
      --model-json <path>  Use a model spec from a JSON file instead.
      --device-json <path> Use a device spec from a JSON file instead.

CONFIGURATION
  -q, --quant <id>         Weight quantization (default q4_k_m). f16, q8_0,
                           q6_k, q5_k_m, q4_k_m, q4_k_s, q3_k_m, q2_k, mxfp4,
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

OUTPUT
      --json               Machine-readable output.
  -h, --help               This text.
  -v, --version            Print the version.

EXAMPLES
  vramfit check llama-3.1-8b --device 4090 --ctx 32k
  vramfit check llama-3.3-70b -d 3090 -g 2 -q q4_k_m --ctx 8k
  vramfit best qwen2.5-32b --device m3-max
  vramfit check gemma-3-27b -d 3060 --ram 64 --json

EXIT CODES
  0  fits    1  does not fit    2  bad usage`;

const SHARED_FLAGS = [
  "device",
  "device-json",
  "model-json",
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
  let text: string;
  try {
    text = io.readFile(path);
  } catch {
    throw new UsageError(`Cannot read ${path}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text) as unknown;
  } catch (cause) {
    throw new UsageError(`${path} is not valid JSON: ${(cause as Error).message}`);
  }
  return parse(parsed, path);
}

function resolveModel(args: Args, io: Io): ModelSpec {
  const path = args.string("model-json");
  if (path !== undefined) return loadJsonFile(io, path, parseModelSpec);

  const name = args.positionals[1];
  if (name === undefined) {
    throw new UsageError("A model is required. Try \"vramfit models\" for the bundled list.");
  }
  return getModel(name);
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

function runCheck(args: Args, io: Io, version: string): number {
  args.assertKnown([...SHARED_FLAGS, "quant"]);
  assertNoExtraArguments(args, "check", 2);

  const model = resolveModel(args, io);
  const device = resolveDevice(args, io);
  const quant = getQuant(args.string("quant") ?? "q4_k_m");
  const options = fitOptions(args);

  const fit = checkFit(model, quant, device, options);
  const recommendation = recommendQuant(model, device, options);

  if (args.boolean("json") === true) {
    emitJson(io, checkJson(fit, recommendation, version));
  } else {
    emit(io, renderCheck(fit, recommendation));
  }
  return fit.fits ? EXIT_OK : EXIT_DOES_NOT_FIT;
}

function runBest(args: Args, io: Io, version: string): number {
  args.assertKnown(SHARED_FLAGS);
  assertNoExtraArguments(args, "best", 2);

  const model = resolveModel(args, io);
  const device = resolveDevice(args, io);
  const options = fitOptions(args);
  const evaluated = evaluateQuants(model, device, options);
  const ctx = options.ctx ?? model.defaultCtx;

  if (args.boolean("json") === true) {
    emitJson(io, bestJson(model, device, evaluated, ctx, version));
  } else {
    emit(io, renderBest(model, device, evaluated, ctx, options.gpus ?? 1));
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
      case "devices":
        return runList(args, io, "devices");
      case "models":
        return runList(args, io, "models");
      case "version":
        io.out(version);
        return EXIT_OK;
      default:
        throw new UsageError(
          `Unknown command "${command}". Expected check, best, devices, models or help.`,
        );
    }
  } catch (error) {
    io.err(`vramfit: ${(error as Error).message}`);
    if (error instanceof UsageError) io.err('Try "vramfit --help".');
    return EXIT_USAGE;
  }
}
