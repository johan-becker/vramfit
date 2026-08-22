import { readFileSync } from "node:fs";
import type { DeviceSpec, ModelSpec } from "../types.js";
import { createRegistry, type Registry } from "./registry.js";
import { parseDatabase, parseDeviceSpec, parseModelSpec } from "./validate.js";

/**
 * The bundled device and model database.
 *
 * The JSON lives next to the compiled code (`scripts/finish-build.mjs` copies
 * `src/data` to `dist/data`) and is resolved relative to `import.meta.url`, so
 * the same path works when vitest runs the TypeScript sources and when node
 * runs the published build. It is read with `fs` rather than imported, because
 * JSON module imports are still flagged experimental under NodeNext and would
 * put a warning on stderr of every CLI invocation.
 *
 * Loading is lazy and memoised: `vramfit devices` should not pay to parse and
 * validate two dozen model architectures, and a library consumer that only
 * wants the arithmetic should not touch the filesystem at all.
 */

const DATA_DIR = new URL("../data/", import.meta.url);

function readDatabaseFile(fileName: string): unknown {
  const url = new URL(fileName, DATA_DIR);
  let text: string;
  try {
    text = readFileSync(url, "utf8");
  } catch (cause) {
    throw new Error(`Cannot read bundled database ${fileName} at ${url.pathname}`, { cause });
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (cause) {
    throw new Error(`Bundled database ${fileName} is not valid JSON`, { cause });
  }
}

let modelRegistry: Registry<ModelSpec> | undefined;
let deviceRegistry: Registry<DeviceSpec> | undefined;

/** The bundled model registry, parsed and validated on first use. */
export function models(): Registry<ModelSpec> {
  modelRegistry ??= createRegistry(
    parseDatabase(readDatabaseFile("models.json"), "models", parseModelSpec, "models.json"),
    "model",
  );
  return modelRegistry;
}

/** The bundled device registry, parsed and validated on first use. */
export function devices(): Registry<DeviceSpec> {
  deviceRegistry ??= createRegistry(
    parseDatabase(readDatabaseFile("devices.json"), "devices", parseDeviceSpec, "devices.json"),
    "device",
  );
  return deviceRegistry;
}

/** Every bundled model, in database order (roughly by family then size). */
export function listModels(): readonly ModelSpec[] {
  return models().all();
}

/** Every bundled device, in database order. */
export function listDevices(): readonly DeviceSpec[] {
  return devices().all();
}

/** Resolve a model by id, display name or alias. Undefined when unknown. */
export function findModel(query: string): ModelSpec | undefined {
  return models().find(query);
}

/** Resolve a model, throwing an error that suggests near matches. */
export function getModel(query: string): ModelSpec {
  return models().get(query);
}

/** Resolve a device by id, display name or alias. Undefined when unknown. */
export function findDevice(query: string): DeviceSpec | undefined {
  return devices().find(query);
}

/** Resolve a device, throwing an error that suggests near matches. */
export function getDevice(query: string): DeviceSpec {
  return devices().get(query);
}

export { createRegistry, normalizeKey, type Named, type Registry } from "./registry.js";
export {
  SCHEMA_VERSION,
  SpecValidationError,
  parseDatabase,
  parseDeviceSpec,
  parseModelSpec,
} from "./validate.js";
