import { SpecValidationError } from "./db/validate.js";
import { checkFit, type FitOptions, type FitResult } from "./fit.js";
import type { DeviceSpec, ModelSpec, QuantSpec } from "./types.js";

/**
 * Capacity planning across machines that are not the same.
 *
 * `check` answers one model on one device. A team has a workstation with two
 * 4090s, somebody's M3 Max, and an A100 in a rack, and the question is which
 * of them can serve which of the models on the list -- and where a model has
 * to go because only one machine can hold it.
 *
 * There is no new arithmetic here: every cell is a `checkFit`. What this adds
 * is the shape of the answer, and one deliberate restriction -- a machine is
 * described once and applies to every model, so a fleet file is a description
 * of hardware you have rather than of a configuration you invented per row.
 */

/* -------------------------------------------------------------------------- */
/* The file                                                                    */
/* -------------------------------------------------------------------------- */

export interface FleetMachineSpec {
  /** How the machine is named in the table. */
  name: string;
  /** Bundled device id, name or alias. */
  device: string;
  /** Identical devices in that machine. */
  gpus: number;
  /** Usable memory per device in GiB, overriding the database. */
  vramGiB: number | undefined;
  /** System RAM available for offloaded layers, GiB. */
  systemRamGiB: number | undefined;
}

export interface FleetModelSpec {
  /** Bundled id, name, alias, or a path the caller knows how to open. */
  model: string;
  /** Weight quantization for this row, overriding the file's default. */
  quant: string | undefined;
  /** Context for this row, overriding the file's default. */
  ctx: number | undefined;
}

export interface FleetConfig {
  machines: FleetMachineSpec[];
  models: FleetModelSpec[];
  /** Context every row is checked at unless it says otherwise. */
  ctx: number | undefined;
  quant: string | undefined;
  kvQuant: string | undefined;
  batch: number | undefined;
}

/** Same bounds as the model validator, for the same reason: a typo here
 *  multiplies into one `checkFit` per machine per model. */
const MAX_MACHINES = 512;
const MAX_MODELS = 512;
const MAX_CTX = 134_217_728;

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

function requiredString(source: Record<string, unknown>, key: string, path: string): string {
  const raw = source[key];
  if (typeof raw !== "string" || raw.trim() === "") {
    fail(`${path}.${key}`, `must be a non-empty string, got ${describe(raw)}`);
  }
  // Same rule as the spec validator: every one of these is printed into a
  // terminal report unescaped, and an escape sequence in a machine name can
  // forge a line above the real one.
  if (/[\p{Cc}\p{Cf}]/u.test(raw)) fail(`${path}.${key}`, "must not contain control characters");
  return raw;
}

function optionalString(
  source: Record<string, unknown>,
  key: string,
  path: string,
): string | undefined {
  const raw = source[key];
  if (raw === undefined || raw === null) return undefined;
  return requiredString(source, key, path);
}

function optionalNumber(
  source: Record<string, unknown>,
  key: string,
  path: string,
  rules: { min?: number; max?: number; integer?: boolean } = {},
): number | undefined {
  const raw = source[key];
  if (raw === undefined || raw === null) return undefined;
  const at = `${path}.${key}`;
  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    fail(at, `must be a finite number, got ${describe(raw)}`);
  }
  if (rules.integer === true && !Number.isInteger(raw)) {
    fail(at, `must be a whole number, got ${raw}`);
  }
  if (rules.min !== undefined && raw < rules.min) fail(at, `must be at least ${rules.min}, got ${raw}`);
  if (rules.max !== undefined && raw > rules.max) fail(at, `must be at most ${rules.max}, got ${raw}`);
  return raw;
}

function parseMachine(value: unknown, path: string): FleetMachineSpec {
  const raw = object(value, path);
  const device = requiredString(raw, "device", path);
  return {
    // A machine that does not name itself is named after what is in it, which
    // is what people mean when they leave it out for a one-machine file.
    name: optionalString(raw, "name", path) ?? device,
    device,
    gpus: optionalNumber(raw, "gpus", path, { integer: true, min: 1, max: 64 }) ?? 1,
    vramGiB: optionalNumber(raw, "vram", path, { min: 0.25 }),
    systemRamGiB: optionalNumber(raw, "ram", path, { min: 0.25 }),
  };
}

function parseModelEntry(value: unknown, path: string): FleetModelSpec {
  if (typeof value === "string") {
    if (value.trim() === "") fail(path, "must be a non-empty model name or path");
    return { model: value, quant: undefined, ctx: undefined };
  }
  const raw = object(value, path);
  return {
    model: requiredString(raw, "model", path),
    quant: optionalString(raw, "quant", path),
    ctx: optionalNumber(raw, "ctx", path, { integer: true, min: 1, max: MAX_CTX }),
  };
}

function array(value: unknown, key: string, path: string, limit: number): unknown[] {
  const raw = object(value, path)[key];
  if (!Array.isArray(raw)) fail(`${path}.${key}`, `must be an array, got ${describe(raw)}`);
  if (raw.length === 0) fail(`${path}.${key}`, "must not be empty");
  if (raw.length > limit) {
    fail(`${path}.${key}`, `has ${raw.length} entries, past the ${limit} limit`);
  }
  return raw;
}

/**
 * Validate a fleet file.
 *
 * Duplicate machine names are a hard error rather than a warning: the table
 * has one column per machine, and two columns with the same heading is a
 * report nobody can act on.
 */
export function parseFleetConfig(value: unknown, path = "fleet"): FleetConfig {
  const raw = object(value, path);
  const machines = array(raw, "machines", path, MAX_MACHINES).map((entry, index) =>
    parseMachine(entry, `${path}.machines[${index}]`),
  );
  const models = array(raw, "models", path, MAX_MODELS).map((entry, index) =>
    parseModelEntry(entry, `${path}.models[${index}]`),
  );

  const seen = new Set<string>();
  machines.forEach((machine, index) => {
    if (seen.has(machine.name)) {
      fail(`${path}.machines[${index}].name`, `is "${machine.name}", which is already taken`);
    }
    seen.add(machine.name);
  });

  return {
    machines,
    models,
    ctx: optionalNumber(raw, "ctx", path, { integer: true, min: 1, max: MAX_CTX }),
    quant: optionalString(raw, "quant", path),
    kvQuant: optionalString(raw, "kvQuant", path),
    batch: optionalNumber(raw, "batch", path, { integer: true, min: 1 }),
  };
}

/* -------------------------------------------------------------------------- */
/* The plan                                                                    */
/* -------------------------------------------------------------------------- */

export interface FleetMachine {
  name: string;
  device: DeviceSpec;
  gpus: number;
  /** Everything the machine contributes to a `checkFit` call. */
  options: FitOptions;
}

export interface FleetEntry {
  /** How the row is labelled: the model's name, or the path it came from. */
  label: string;
  model: ModelSpec;
  quant: QuantSpec;
  ctx: number;
}

export interface FleetCell {
  machine: FleetMachine;
  fit: FitResult;
}

export interface FleetRow {
  entry: FleetEntry;
  cells: FleetCell[];
  /** Machines that hold this model entirely in device memory. */
  servedBy: number;
}

export interface FleetReport {
  machines: FleetMachine[];
  rows: FleetRow[];
  /** Rows no machine can serve without offloading. */
  unserved: FleetRow[];
  /** Machines that serve nothing on the list. */
  idle: FleetMachine[];
}

/** Evaluate every model on every machine. */
export function planFleet(
  machines: readonly FleetMachine[],
  entries: readonly FleetEntry[],
): FleetReport {
  const rows = entries.map((entry): FleetRow => {
    const cells = machines.map((machine) => ({
      machine,
      fit: checkFit(entry.model, entry.quant, machine.device, {
        ...machine.options,
        ctx: entry.ctx,
      }),
    }));
    return { entry, cells, servedBy: cells.filter((cell) => cell.fit.fits).length };
  });

  return {
    machines: [...machines],
    rows,
    unserved: rows.filter((row) => row.servedBy === 0),
    idle: machines.filter(
      (machine, index) => !rows.some((row) => row.cells[index]?.fit.fits === true),
    ),
  };
}
