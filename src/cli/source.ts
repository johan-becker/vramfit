import { describeGguf, type GgufModel } from "../gguf/index.js";
import { GgufError, readGgufHeader, type ByteSource } from "../gguf/reader.js";
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
