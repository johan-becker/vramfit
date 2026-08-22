import { describeGguf, type GgufModel } from "../gguf/index.js";
import { GgufError, readGgufHeader, type ByteSource } from "../gguf/reader.js";
import {
  HfConfigError,
  SafetensorsError,
  TORCH_DTYPE_BYTES,
  modelFromHfConfig,
  parseSafetensorsIndex,
  readSafetensorsHeader,
  textConfigOf,
  type HfModel,
  type HfModelOptions,
} from "../hf/index.js";
import type { ModelSpec, QuantSpec } from "../types.js";
import { UsageError } from "./args.js";

/**
 * Where a model comes from.
 *
 * Three answers: the bundled database, a GGUF file on disk, or a HuggingFace
 * checkpoint directory. Which one is meant is decided from the shape of the
 * name, before anything is opened -- `llama-3.1-8b` is a database key and
 * `./models/llama-3.1-8b.gguf` is a path, and no filesystem lookup is needed
 * to tell them apart. That matters: a tool that stats the working directory
 * before every database lookup answers a different question depending on what
 * happens to be sitting next to you.
 */

export type PathKind = "file" | "directory" | "missing";

/** A file opened for random access. Closed by whoever opened it. */
export interface ClosableByteSource extends ByteSource {
  close(): void;
}

/** The parts of the world the CLI is allowed to touch, all of them injectable. */
export interface SourceIo {
  readFile(path: string): string;
  openBytes?(path: string): ClosableByteSource;
  pathKind?(path: string): PathKind;
  /** True when stdout is a terminal. Absent means "assume it is not". */
  isTty?: boolean;
  /** Environment lookup, for NO_COLOR and friends. Absent means empty. */
  env?(name: string): string | undefined;
}

const PATH_PREFIXES = ["./", "../", ".\\", "..\\", "~/", "/"];
const MODEL_FILE_EXTENSIONS = [".gguf", ".json"];

/**
 * Does this argument name a file rather than a database entry?
 *
 * A directory separator, a leading `./`, a Windows drive letter or a known
 * extension. No bundled id, name or alias contains any of them, and a test
 * asserts that, so the two namespaces cannot start overlapping by accident.
 */
export function looksLikePath(value: string): boolean {
  const trimmed = value.trim();
  if (trimmed === "") return false;
  if (PATH_PREFIXES.some((prefix) => trimmed.startsWith(prefix))) return true;
  if (/^[a-z]:[\\/]/i.test(trimmed)) return true;
  if (trimmed.includes("/") || trimmed.includes("\\")) return true;
  const lower = trimmed.toLowerCase();
  return MODEL_FILE_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

/**
 * Join a directory and a file name with a forward slash.
 *
 * `node:path.join` would be the obvious thing, and is the wrong one: it emits
 * backslashes on Windows, which would make the paths this prints -- and the
 * keys a test's fake filesystem is written with -- platform-dependent. Windows
 * accepts forward slashes in every API this package uses.
 */
export function joinPath(directory: string, name: string): string {
  const base = directory.replace(/[\\/]+$/, "");
  return base === "" ? name : `${base}/${name}`;
}

/** True when the path names a directory, as far as the name and the IO know. */
export function isDirectoryPath(io: SourceIo, path: string): boolean {
  if (/[\\/]$/.test(path)) return true;
  return io.pathKind?.(path) === "directory";
}

function requireBytes(io: SourceIo, path: string): ClosableByteSource {
  if (io.openBytes === undefined) {
    throw new UsageError(`this build cannot open ${path} for reading`);
  }
  try {
    return io.openBytes(path);
  } catch {
    throw new UsageError(`Cannot read ${path}`);
  }
}

/**
 * Read one GGUF file's header and map it onto a model and a quantization.
 *
 * The file is opened, a few hundred kilobytes are read, and it is closed --
 * including when the parse throws, which is what the `finally` is for.
 */
export function loadGguf(io: SourceIo, path: string): GgufModel {
  const source = requireBytes(io, path);
  try {
    return describeGguf(readGgufHeader(source), { origin: path });
  } catch (error) {
    if (error instanceof GgufError) {
      throw new UsageError(`${path}: ${error.message}`);
    }
    throw error;
  } finally {
    source.close();
  }
}

/** Where a resolved model came from, as reported in the JSON payload. */
export type ModelOrigin = "database" | "gguf" | "huggingface" | "json";

export interface ResolvedModel {
  model: ModelSpec;
  /**
   * The quantization the source itself dictates -- a GGUF file is already in
   * one. Undefined when the source does not imply a format, in which case the
   * caller's default applies.
   */
  quant?: QuantSpec;
  origin: ModelOrigin;
  /** The path or name it was read from, for the report. */
  from: string;
  /** One line describing the source, printed above the report. */
  description?: string;
  /** Anything the reader had to decide, shown under the report's Notes. */
  notes?: string[];
}

/** Describe a GGUF file the way the report shows it above the verdict. */
export function describeGgufSource(path: string, gguf: GgufModel): string {
  const parts = [
    `GGUF v${gguf.header.version}`,
    `${gguf.header.tensorCount} tensors`,
    `${gguf.architecture} architecture`,
  ];
  if (gguf.fileType !== undefined) parts.push(gguf.fileType);
  return `Read from ${path}  (${parts.join(", ")})`;
}

/** Resolve a `<model>` argument that has already been recognised as a path. */
export function resolveGgufPath(io: SourceIo, path: string): ResolvedModel {
  const gguf = loadGguf(io, path);
  return {
    model: gguf.model,
    quant: gguf.quant,
    origin: "gguf",
    from: path,
    description: describeGgufSource(path, gguf),
  };
}

/* -------------------------------------------------------------------------- */
/* JSON on disk                                                                */
/* -------------------------------------------------------------------------- */

/** The directory a path sits in, with forward slashes and no trailing one. */
export function directoryOf(path: string): string {
  const at = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return at < 0 ? "." : path.slice(0, at);
}

/**
 * Read and parse a JSON file.
 *
 * V8's `JSON.parse` message quotes the first bytes of its input, so passing it
 * through would print the head of whatever file was named -- a mistyped path
 * in a CI step should not echo a secrets file into the build log. The position
 * it reports is kept; the excerpt is not.
 */
export function readJsonFile(io: SourceIo, path: string): unknown {
  let text: string;
  try {
    text = io.readFile(path);
  } catch {
    throw new UsageError(`Cannot read ${path}`);
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (cause) {
    const at = /in JSON at (position \d+(?: \(line \d+ column \d+\))?)/.exec(
      (cause as Error).message,
    );
    throw new UsageError(`${path} is not valid JSON${at ? ` (${at[1]})` : ""}`);
  }
}

/** True when a parsed object is a vramfit `ModelSpec` rather than an HF config. */
export function looksLikeVramfitSpec(value: unknown): boolean {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const object = value as Record<string, unknown>;
  return object["nLayers"] !== undefined || object["nKvHeads"] !== undefined;
}

/* -------------------------------------------------------------------------- */
/* HuggingFace checkpoints                                                     */
/* -------------------------------------------------------------------------- */

const CONFIG_FILE = "config.json";
const SAFETENSORS_INDEX_FILE = "model.safetensors.index.json";
const SAFETENSORS_FILE = "model.safetensors";

/** Bytes per parameter implied by `torch_dtype`, defaulting to 16-bit. */
function dtypeBytes(config: unknown): { bytes: number; declared: boolean } {
  if (typeof config !== "object" || config === null) return { bytes: 2, declared: false };
  const root = config as Record<string, unknown>;
  const raw = root["torch_dtype"] ?? root["dtype"] ?? textConfigOf(root)["torch_dtype"];
  if (typeof raw !== "string") return { bytes: 2, declared: false };
  const bytes = TORCH_DTYPE_BYTES[raw.toLowerCase()];
  return bytes === undefined ? { bytes: 2, declared: false } : { bytes, declared: true };
}

interface WeightCount {
  totalParams: number;
  source: "safetensors" | "index";
  /** The file it came from, for the report. */
  from: string;
  note?: string;
}

/**
 * The parameter count the weight files state, if any of them are there.
 *
 * A sharded repository is the common case for anything above 7B and carries an
 * index whose `metadata.total_size` is the byte total; dividing by the dtype
 * width recovers the count. A single-file repository is exact without any
 * division, because its header lists every tensor's shape.
 */
function weightCount(io: SourceIo, directory: string, config: unknown): WeightCount | undefined {
  const indexPath = joinPath(directory, SAFETENSORS_INDEX_FILE);
  if (io.pathKind?.(indexPath) === "file") {
    const index = parseSafetensorsIndex(readJsonFile(io, indexPath), indexPath);
    const dtype = dtypeBytes(config);
    const count: WeightCount = {
      totalParams: index.totalSizeBytes / dtype.bytes,
      source: "index",
      from: `${SAFETENSORS_INDEX_FILE} (${index.shards} shards)`,
    };
    if (!dtype.declared) {
      count.note = `${indexPath} gives bytes, not parameters, and the config declares no torch_dtype; 16-bit weights are assumed.`;
    }
    return count;
  }

  const singlePath = joinPath(directory, SAFETENSORS_FILE);
  if (io.pathKind?.(singlePath) === "file" && io.openBytes !== undefined) {
    const source = io.openBytes(singlePath);
    try {
      const header = readSafetensorsHeader(source);
      return { totalParams: header.totalParams, source: "safetensors", from: SAFETENSORS_FILE };
    } finally {
      source.close();
    }
  }
  return undefined;
}

/** Describe a HuggingFace checkpoint the way the report shows it. */
export function describeHfSource(path: string, hf: HfModel, from: string): string {
  const parts = [`${hf.model.nLayers} layers`, `${hf.model.nKvHeads} KV heads`];
  parts.push(
    hf.paramSource === "architecture"
      ? "parameters derived from the architecture"
      : `parameters from ${from}`,
  );
  return `Read from ${path}  (${parts.join(", ")})`;
}

/**
 * Resolve a HuggingFace checkpoint: a directory, or the `config.json` inside
 * one. The weight files beside it are consulted for the parameter count and
 * for nothing else -- only their headers are read.
 */
export function resolveHfCheckpoint(
  io: SourceIo,
  path: string,
  /** The already-parsed contents of `path`, when the caller has read it to
   *  decide what kind of file it was. Ignored when `path` is a directory. */
  parsedConfig?: unknown,
): ResolvedModel {
  const directory = isDirectoryPath(io, path) ? path : directoryOf(path);
  const configPath = isDirectoryPath(io, path) ? joinPath(path, CONFIG_FILE) : path;
  const config =
    parsedConfig !== undefined && configPath === path ? parsedConfig : readJsonFile(io, configPath);

  try {
    const counted = weightCount(io, directory, config);
    const options: HfModelOptions = { origin: configPath };
    if (counted !== undefined) {
      options.weights = { totalParams: counted.totalParams, source: counted.source };
    }
    const hf = modelFromHfConfig(config, options);
    const notes = [...hf.notes];
    if (counted?.note !== undefined && hf.paramSource !== "architecture") notes.push(counted.note);

    return {
      model: hf.model,
      origin: "huggingface",
      from: configPath,
      description: describeHfSource(configPath, hf, counted?.from ?? CONFIG_FILE),
      notes,
    };
  } catch (error) {
    if (error instanceof HfConfigError || error instanceof SafetensorsError) {
      throw new UsageError(`${configPath}: ${error.message}`);
    }
    throw error;
  }
}
