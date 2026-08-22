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

const DEVICE_FAMILIES = [
  "cuda-consumer",
  "cuda-datacenter",
  "rocm",
  "metal",
  "cpu",
] as const satisfies readonly DeviceFamily[];

const ATTENTION_KINDS = ["gqa", "mla"] as const satisfies readonly AttentionKind[];

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
    return entry;
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
    kvLoraRank: num(raw, "kvLoraRank", path, { integer: true, above: 0 }),
    qkRopeHeadDim: num(raw, "qkRopeHeadDim", path, { integer: true, above: 0 }),
    qkNopeHeadDim: num(raw, "qkNopeHeadDim", path, { integer: true, above: 0 }),
    vHeadDim: num(raw, "vHeadDim", path, { integer: true, above: 0 }),
    qLoraRank:
      qLoraRaw === null ? null : num(raw, "qLoraRank", path, { integer: true, above: 0 }),
  };
}

function parseMoe(raw: Record<string, unknown>, path: string, nLayers: number): MoeSpec {
  const nExperts = num(raw, "nExperts", path, { integer: true, above: 0 });
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
    expertFfnHidden: num(raw, "expertFfnHidden", path, { integer: true, above: 0 }),
    nSharedExperts: num(raw, "nSharedExperts", path, { integer: true, min: 0 }),
    denseLayers,
    denseFfnHidden:
      denseFfnRaw === null || denseFfnRaw === undefined
        ? null
        : num(raw, "denseFfnHidden", path, { integer: true, above: 0 }),
  };
}

function parseAttentionWindow(
  raw: Record<string, unknown>,
  path: string,
): AttentionWindowSpec {
  return {
    windowSize: num(raw, "windowSize", path, { integer: true, above: 0 }),
    // 1 would mean "every layer is a full-attention layer", i.e. no windowing;
    // express that as attentionWindow: null instead of a degenerate block.
    fullAttentionEvery: num(raw, "fullAttentionEvery", path, { integer: true, min: 2 }),
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

  const nLayers = num(raw, "nLayers", path, { integer: true, above: 0 });
  const nHeads = num(raw, "nHeads", path, { integer: true, above: 0 });
  const nKvHeads = num(raw, "nKvHeads", path, { integer: true, above: 0 });
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

  const maxCtx = num(raw, "maxCtx", path, { integer: true, above: 0 });
  const defaultCtx = num(raw, "defaultCtx", path, { integer: true, above: 0 });
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
  const spec: ModelSpec = {
    id: str(raw, "id", path),
    name: str(raw, "name", path),
    aliases: strArray(raw, "aliases", path),
    totalParams,
    activeParams,
    nLayers,
    hiddenSize: num(raw, "hiddenSize", path, { integer: true, above: 0 }),
    nHeads,
    nKvHeads,
    headDim: num(raw, "headDim", path, { integer: true, above: 0 }),
    ffnHidden: num(raw, "ffnHidden", path, { integer: true, above: 0 }),
    vocabSize: num(raw, "vocabSize", path, { integer: true, above: 0 }),
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
  if (notes !== undefined) spec.notes = notes;

  if (spec.moe === null && activeParams !== totalParams) {
    fail(
      `${path}.activeParams`,
      `must equal totalParams for a dense model; declare a moe block if only some parameters are active`,
    );
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
