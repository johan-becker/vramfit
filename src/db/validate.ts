import { deriveArchitecture } from "../architecture.js";
import { findQuant } from "../quant.js";
import type {
  AttentionKind,
  AttentionWindowSpec,
  DeviceFamily,
  DeviceSpec,
  MlaSpec,
  ModelSpec,
  MoeSpec,
} from "../types.js";

/**
 * Runtime validation for everything that enters the program as data.
 *
 * Two things go through here: the bundled `src/data/*.json` files, and any
 * model or device a user supplies with `--model-json` / `--device-json`. Both
 * are `unknown` as far as TypeScript is concerned -- a `.json` import is a
 * compile-time fiction, and a user file is genuinely arbitrary -- so both are
 * parsed by the same functions and reach the arithmetic already typed.
 *
 * Errors are deliberately verbose. A silently-missing `nKvHeads` would not
 * crash; it would produce a plausible-looking VRAM figure that is wrong by the
 * GQA group size, which is exactly the class of bug this package exists to
 * prevent. Every failure names the field, the expected shape and what was
 * actually there.
 */

/** Bumped when the on-disk shape changes incompatibly. */
export const SCHEMA_VERSION = 1;

/**
 * How far a declared `totalParams` may sit from the count its own architecture
 * implies. Published figures are rounded -- to three significant figures, or
 * to "8B" -- and the derived count ignores biases, norms and rotary tables, so
 * some drift is expected; the worst of the 24 bundled models is 0.040%. Five
 * percent leaves room for a hand-rounded spec while still catching the errors
 * that matter, which move the count by tens of percent or by factors of 1000.
 */
const TOTAL_PARAM_TOLERANCE = 0.05;

/**
 * Vendors disagree on whether "active parameters" counts the embedding table,
 * the output head, both or neither -- gpt-oss counts one vocabulary matrix,
 * Qwen counts two -- so the declared figure is checked against the range those
 * conventions span rather than against one number, with 10% of slack on each
 * end. That is still tight enough to catch a wrong `expertsPerToken`, which
 * moves the active count by a factor of two or more.
 */
const ACTIVE_PARAM_SLACK = 0.1;

const DEVICE_FAMILIES = [
  "cuda-consumer",
  "cuda-datacenter",
  "rocm",
  "metal",
  "cpu",
] as const satisfies readonly DeviceFamily[];

const ATTENTION_KINDS = ["gqa", "mla"] as const satisfies readonly AttentionKind[];

/**
 * Upper bounds on the shape fields.
 *
 * Every one of these is orders of magnitude above anything published -- 126
 * layers in Llama 3.1 405B, a 262144-token Gemma 3 vocabulary, 256 experts in
 * DeepSeek V3 -- so they never get in the way of a real spec. They are here
 * because the KV cache is built one entry per layer and the fit search
 * re-derives the footprint dozens of times, so an extra keystroke in
 * `nLayers` turned a hand-written spec into 3.5 GB of RSS and ten seconds of
 * CPU, or into "Invalid array length" naming neither the field nor the file.
 */
const MAX_LAYERS = 4096;
const MAX_HEADS = 4096;
/** Widest single tensor dimension: hidden size, head dim, FFN intermediate. */
const MAX_WIDTH = 262_144;
const MAX_VOCAB = 4_194_304;
const MAX_CTX = 134_217_728;
const MAX_EXPERTS = 8192;

/**
 * Control and format characters, which have no business in a name that is
 * printed to a terminal. A spec is exactly the kind of file people paste from
 * a gist, and every string here is written into the report unescaped: an ANSI
 * sequence in a device name can clear the screen and print a forged "FITS"
 * line above the real verdict, and a bidi override can reorder one.
 */
const CONTROL_CHARACTERS = /[\p{Cc}\p{Cf}]/u;

/** Thrown for every validation failure, so callers can tell bad data from bugs. */
export class SpecValidationError extends Error {
  override readonly name = "SpecValidationError";
  readonly path: string;

  constructor(path: string, message: string) {
    super(`${path} ${message}`);
    this.path = path;
  }
}

function fail(path: string, message: string): never {
  throw new SpecValidationError(path, message);
}

function describe(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "nothing";
  if (Array.isArray(value)) return `an array of ${value.length}`;
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "object") return "an object";
  return `${typeof value} ${String(value)}`;
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(path, `must be an object, got ${describe(value)}`);
  }
  return value as Record<string, unknown>;
}

interface NumberRules {
  /** Inclusive lower bound. */
  min?: number;
  /** Inclusive upper bound. */
  max?: number;
  /** Exclusive lower bound; use `{ above: 0 }` for "strictly positive". */
  above?: number;
  integer?: boolean;
}

function num(
  source: Record<string, unknown>,
  key: string,
  path: string,
  rules: NumberRules = {},
): number {
  const at = `${path}.${key}`;
  const raw = source[key];
  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    fail(at, `must be a finite number, got ${describe(raw)}`);
  }
  if (rules.integer === true && !Number.isInteger(raw)) {
    fail(at, `must be a whole number, got ${raw}`);
  }
  if (rules.above !== undefined && raw <= rules.above) {
    fail(at, `must be greater than ${rules.above}, got ${raw}`);
  }
  if (rules.min !== undefined && raw < rules.min) {
    fail(at, `must be at least ${rules.min}, got ${raw}`);
  }
  if (rules.max !== undefined && raw > rules.max) {
    fail(at, `must be at most ${rules.max}, got ${raw}`);
  }
  return raw;
}

function str(source: Record<string, unknown>, key: string, path: string): string {
  const at = `${path}.${key}`;
  const raw = source[key];
  if (typeof raw !== "string" || raw.trim() === "") {
    fail(at, `must be a non-empty string, got ${describe(raw)}`);
  }
  return text(raw, at);
}

/** A string that is safe to print. Shared by `str` and `strArray`. */
function text(raw: string, at: string): string {
  if (CONTROL_CHARACTERS.test(raw)) {
    fail(at, "must not contain control characters");
  }
  return raw;
}

function optionalStr(
  source: Record<string, unknown>,
  key: string,
  path: string,
): string | undefined {
  if (source[key] === undefined || source[key] === null) return undefined;
  return str(source, key, path);
}

function bool(source: Record<string, unknown>, key: string, path: string): boolean {
  const at = `${path}.${key}`;
  const raw = source[key];
  if (typeof raw !== "boolean") fail(at, `must be true or false, got ${describe(raw)}`);
  return raw;
}

function strArray(source: Record<string, unknown>, key: string, path: string): string[] {
  const at = `${path}.${key}`;
  const raw = source[key];
  if (raw === undefined) return [];
  if (!Array.isArray(raw)) fail(at, `must be an array of strings, got ${describe(raw)}`);
  return raw.map((entry, index) => {
    if (typeof entry !== "string" || entry.trim() === "") {
      fail(`${at}[${index}]`, `must be a non-empty string, got ${describe(entry)}`);
    }
    return text(entry, `${at}[${index}]`);
  });
}

function oneOf<T extends string>(
  source: Record<string, unknown>,
  key: string,
  path: string,
  allowed: readonly T[],
): T {
  const at = `${path}.${key}`;
  const raw = source[key];
  if (typeof raw !== "string" || !(allowed as readonly string[]).includes(raw)) {
    fail(at, `must be one of ${allowed.join(", ")}, got ${describe(raw)}`);
  }
  return raw as T;
}

/** A nested block that is either `null` or an object. `undefined` is a typo, not a value. */
function nullableObject(
  source: Record<string, unknown>,
  key: string,
  path: string,
): Record<string, unknown> | null {
  const at = `${path}.${key}`;
  const raw = source[key];
  if (raw === null) return null;
  if (raw === undefined) fail(at, "must be present, use null when it does not apply");
  return object(raw, at);
}

function parseMla(raw: Record<string, unknown>, path: string): MlaSpec {
  const qLoraRaw = raw["qLoraRank"];
  return {
    kvLoraRank: num(raw, "kvLoraRank", path, { integer: true, above: 0, max: MAX_WIDTH }),
    qkRopeHeadDim: num(raw, "qkRopeHeadDim", path, { integer: true, above: 0, max: MAX_WIDTH }),
    qkNopeHeadDim: num(raw, "qkNopeHeadDim", path, { integer: true, above: 0, max: MAX_WIDTH }),
    vHeadDim: num(raw, "vHeadDim", path, { integer: true, above: 0, max: MAX_WIDTH }),
    qLoraRank:
      qLoraRaw === null
        ? null
        : num(raw, "qLoraRank", path, { integer: true, above: 0, max: MAX_WIDTH }),
  };
}

function parseMoe(raw: Record<string, unknown>, path: string, nLayers: number): MoeSpec {
  const nExperts = num(raw, "nExperts", path, { integer: true, above: 0, max: MAX_EXPERTS });
  const expertsPerToken = num(raw, "expertsPerToken", path, { integer: true, above: 0 });
  if (expertsPerToken > nExperts) {
    fail(
      `${path}.expertsPerToken`,
      `cannot exceed nExperts (${nExperts}), got ${expertsPerToken}`,
    );
  }
  const denseLayers = num(raw, "denseLayers", path, { integer: true, min: 0 });
  if (denseLayers > nLayers) {
    fail(`${path}.denseLayers`, `cannot exceed nLayers (${nLayers}), got ${denseLayers}`);
  }
  const denseFfnRaw = raw["denseFfnHidden"];
  if (denseLayers > 0 && (denseFfnRaw === null || denseFfnRaw === undefined)) {
    fail(`${path}.denseFfnHidden`, `is required when denseLayers is ${denseLayers}`);
  }
  return {
    nExperts,
    expertsPerToken,
    expertFfnHidden: num(raw, "expertFfnHidden", path, { integer: true, above: 0, max: MAX_WIDTH }),
    nSharedExperts: num(raw, "nSharedExperts", path, { integer: true, min: 0, max: MAX_EXPERTS }),
    denseLayers,
    denseFfnHidden:
      denseFfnRaw === null || denseFfnRaw === undefined
        ? null
        : num(raw, "denseFfnHidden", path, { integer: true, above: 0, max: MAX_WIDTH }),
  };
}

function parseAttentionWindow(
  raw: Record<string, unknown>,
  path: string,
): AttentionWindowSpec {
  return {
    windowSize: num(raw, "windowSize", path, { integer: true, above: 0, max: MAX_CTX }),
    // 1 would mean "every layer is a full-attention layer", i.e. no windowing;
    // express that as attentionWindow: null instead of a degenerate block.
    fullAttentionEvery: num(raw, "fullAttentionEvery", path, {
      integer: true,
      min: 2,
      max: MAX_LAYERS,
    }),
  };
}

/**
 * Validate one model entry.
 *
 * Beyond per-field checks this enforces the cross-field invariants that keep
 * the arithmetic meaningful: query heads must divide evenly into KV head
 * groups, an MLA model must carry MLA geometry (and a GQA model must not), and
 * active parameters can never exceed total parameters.
 */
export function parseModelSpec(value: unknown, path = "model"): ModelSpec {
  const raw = object(value, path);

  const nLayers = num(raw, "nLayers", path, { integer: true, above: 0, max: MAX_LAYERS });
  const nHeads = num(raw, "nHeads", path, { integer: true, above: 0, max: MAX_HEADS });
  const nKvHeads = num(raw, "nKvHeads", path, { integer: true, above: 0, max: MAX_HEADS });
  if (nKvHeads > nHeads) {
    fail(`${path}.nKvHeads`, `cannot exceed nHeads (${nHeads}), got ${nKvHeads}`);
  }
  if (nHeads % nKvHeads !== 0) {
    fail(
      `${path}.nKvHeads`,
      `must divide nHeads (${nHeads}) evenly -- grouped-query attention shares one KV head per group -- got ${nKvHeads}`,
    );
  }

  const totalParams = num(raw, "totalParams", path, { above: 0 });
  const activeParams = num(raw, "activeParams", path, { above: 0 });
  if (activeParams > totalParams) {
    fail(
      `${path}.activeParams`,
      `cannot exceed totalParams (${totalParams}), got ${activeParams}`,
    );
  }

  const maxCtx = num(raw, "maxCtx", path, { integer: true, above: 0, max: MAX_CTX });
  const defaultCtx = num(raw, "defaultCtx", path, { integer: true, above: 0, max: MAX_CTX });
  if (defaultCtx > maxCtx) {
    fail(`${path}.defaultCtx`, `cannot exceed maxCtx (${maxCtx}), got ${defaultCtx}`);
  }

  const attention = oneOf(raw, "attention", path, ATTENTION_KINDS);
  const mlaRaw = nullableObject(raw, "mla", path);
  if (attention === "mla" && mlaRaw === null) {
    fail(`${path}.mla`, 'is required when attention is "mla"');
  }
  if (attention === "gqa" && mlaRaw !== null) {
    fail(`${path}.mla`, 'must be null unless attention is "mla"');
  }

  const moeRaw = nullableObject(raw, "moe", path);
  const windowRaw = nullableObject(raw, "attentionWindow", path);

  const notes = optionalStr(raw, "notes", path);
  const nativeQuant = optionalStr(raw, "nativeQuant", path);
  if (nativeQuant !== undefined && findQuant(nativeQuant) === undefined) {
    fail(`${path}.nativeQuant`, `must name a known quantization, got ${describe(nativeQuant)}`);
  }
  const spec: ModelSpec = {
    id: str(raw, "id", path),
    name: str(raw, "name", path),
    aliases: strArray(raw, "aliases", path),
    totalParams,
    activeParams,
    nLayers,
    hiddenSize: num(raw, "hiddenSize", path, { integer: true, above: 0, max: MAX_WIDTH }),
    nHeads,
    nKvHeads,
    headDim: num(raw, "headDim", path, { integer: true, above: 0, max: MAX_WIDTH }),
    ffnHidden: num(raw, "ffnHidden", path, { integer: true, above: 0, max: MAX_WIDTH }),
    vocabSize: num(raw, "vocabSize", path, { integer: true, above: 0, max: MAX_VOCAB }),
    tiedEmbeddings: bool(raw, "tiedEmbeddings", path),
    attention,
    mla: mlaRaw === null ? null : parseMla(mlaRaw, `${path}.mla`),
    moe: moeRaw === null ? null : parseMoe(moeRaw, `${path}.moe`, nLayers),
    attentionWindow:
      windowRaw === null ? null : parseAttentionWindow(windowRaw, `${path}.attentionWindow`),
    maxCtx,
    defaultCtx,
    source: str(raw, "source", path),
  };
  if (nativeQuant !== undefined) spec.nativeQuant = nativeQuant;
  if (notes !== undefined) spec.notes = notes;

  if (spec.moe === null && activeParams !== totalParams) {
    fail(
      `${path}.activeParams`,
      `must equal totalParams for a dense model; declare a moe block if only some parameters are active`,
    );
  }

  // The invariant that keeps the arithmetic honest: the shape fields and the
  // headline parameter count have to describe the same model. Without it a
  // 1000x typo in totalParams is schema-valid, and computeWeightBytes turns
  // the resulting negative block count into a plausible-looking file size.
  const arch = deriveArchitecture(spec);
  const drift = Math.abs(arch.derivedTotalParams - totalParams) / totalParams;
  if (drift > TOTAL_PARAM_TOLERANCE) {
    fail(
      `${path}.totalParams`,
      `is ${totalParams}, but the architecture describes ${Math.round(arch.derivedTotalParams)} parameters -- ${(drift * 100).toFixed(1)}% apart, above the ${TOTAL_PARAM_TOLERANCE * 100}% tolerance. One of the two is wrong.`,
    );
  }

  if (spec.moe !== null) {
    const vocabParams = arch.inputEmbedParams + arch.outputHeadParams;
    const low = arch.activeBlockParams * (1 - ACTIVE_PARAM_SLACK);
    const high = (arch.activeBlockParams + vocabParams) * (1 + ACTIVE_PARAM_SLACK);
    if (activeParams < low || activeParams > high) {
      fail(
        `${path}.activeParams`,
        `is ${activeParams}, but routing ${spec.moe.expertsPerToken} of ${spec.moe.nExperts} experts activates ${Math.round(arch.activeBlockParams)} parameters per token, ${Math.round(arch.activeBlockParams + vocabParams)} counting the vocabulary tensors.`,
      );
    }
  }

  return spec;
}

/** Validate one device entry, including the bounds that make the fit maths safe. */
export function parseDeviceSpec(value: unknown, path = "device"): DeviceSpec {
  const raw = object(value, path);
  const notes = optionalStr(raw, "notes", path);

  const spec: DeviceSpec = {
    id: str(raw, "id", path),
    name: str(raw, "name", path),
    aliases: strArray(raw, "aliases", path),
    family: oneOf(raw, "family", path, DEVICE_FAMILIES),
    vramGiB: num(raw, "vramGiB", path, { above: 0 }),
    bandwidthGBs: num(raw, "bandwidthGBs", path, { above: 0 }),
    fp16Tflops: num(raw, "fp16Tflops", path, { above: 0 }),
    unifiedMemory: bool(raw, "unifiedMemory", path),
    // A usableFraction of 0 would make every model "not fit" with a
    // divide-by-zero utilisation; above 1 would invent memory that is not there.
    usableFraction: num(raw, "usableFraction", path, { above: 0, max: 1 }),
    source: str(raw, "source", path),
  };
  if (notes !== undefined) spec.notes = notes;
  return spec;
}

/**
 * Unwrap and validate a `{ schemaVersion, note, <key>: [...] }` database file.
 * The version check is what turns "the JSON moved on and the code did not"
 * from a wrong number into a loud error.
 */
export function parseDatabase<T>(
  value: unknown,
  key: string,
  parseEntry: (entry: unknown, path: string) => T,
  path: string,
): T[] {
  const raw = object(value, path);
  const version = num(raw, "schemaVersion", path, { integer: true, above: 0 });
  if (version !== SCHEMA_VERSION) {
    fail(
      `${path}.schemaVersion`,
      `is ${version}, but this build of vramfit understands ${SCHEMA_VERSION}`,
    );
  }
  const entries = raw[key];
  if (!Array.isArray(entries)) {
    fail(`${path}.${key}`, `must be an array, got ${describe(entries)}`);
  }
  if (entries.length === 0) fail(`${path}.${key}`, "must not be empty");
  return entries.map((entry, index) => parseEntry(entry, `${path}.${key}[${index}]`));
}
