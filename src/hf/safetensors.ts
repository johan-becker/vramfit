import type { ByteSource } from "../gguf/reader.js";

/**
 * Safetensors, for one thing only: the true parameter count.
 *
 * A HuggingFace `config.json` describes an architecture, not a checkpoint. The
 * parameter count it implies is close -- within a fraction of a percent -- but
 * it ignores biases, layer norms and rotary tables, and it cannot know whether
 * a repository ships extra heads. The weight files know exactly, and they say
 * so in their own headers without any of the weights being read:
 *
 *   - a sharded repository carries `model.safetensors.index.json`, whose
 *     `metadata.total_size` is the byte total across every shard;
 *   - a single-file repository carries `model.safetensors`, whose header is a
 *     JSON object of tensor shapes and dtypes behind an 8-byte length.
 *
 * The second is exact regardless of dtype. The first divides bytes by the
 * checkpoint's dtype width, which is exact for the one-dtype releases everyone
 * publishes and approximate for a mixed-precision one -- so it says which of
 * the two it used, and the report prints that.
 */

/** Thrown when a safetensors file or index is not what it claims to be. */
export class SafetensorsError extends Error {
  override readonly name = "SafetensorsError";
}

/** Bytes per element for the dtypes safetensors names. */
export const SAFETENSORS_DTYPE_BYTES: Readonly<Record<string, number>> = {
  F64: 8,
  I64: 8,
  U64: 8,
  F32: 4,
  I32: 4,
  U32: 4,
  F16: 2,
  BF16: 2,
  I16: 2,
  U16: 2,
  F8_E4M3: 1,
  F8_E5M2: 1,
  I8: 1,
  U8: 1,
  BOOL: 1,
};

/** Bytes per parameter for the `torch_dtype` strings a config.json carries. */
export const TORCH_DTYPE_BYTES: Readonly<Record<string, number>> = {
  float64: 8,
  double: 8,
  float32: 4,
  float: 4,
  float16: 2,
  half: 2,
  bfloat16: 2,
  float8_e4m3fn: 1,
  float8_e5m2: 1,
  int8: 1,
  uint8: 1,
};

/** A safetensors header is JSON; a gigabyte of it would not be. */
const MAX_HEADER_BYTES = 100 * 1024 * 1024;
/** The 8-byte little-endian header length in front of the JSON. */
const HEADER_LENGTH_BYTES = 8;

export interface SafetensorsTensor {
  name: string;
  dtype: string;
  shape: number[];
  elements: number;
  bytes: number;
}

export interface SafetensorsHeader {
  tensors: SafetensorsTensor[];
  /** Exact: summed from every tensor's own shape. */
  totalParams: number;
  /** Exact: summed from every tensor's own shape and dtype. */
  totalBytes: number;
  /** Bytes of header, which is all that was read. */
  headerBytes: number;
}

function shapeElements(shape: readonly unknown[], name: string): number {
  let elements = 1;
  for (const axis of shape) {
    if (typeof axis !== "number" || !Number.isInteger(axis) || axis < 0) {
      throw new SafetensorsError(`tensor "${name}" has a non-integer dimension ${String(axis)}`);
    }
    elements *= axis;
  }
  return elements;
}

/**
 * Read a `.safetensors` header and nothing else: eight bytes of length, then
 * that many bytes of JSON. The weights sit behind it and are never touched.
 */
export function readSafetensorsHeader(source: ByteSource): SafetensorsHeader {
  const prefix = source.read(0, HEADER_LENGTH_BYTES);
  if (prefix.length < HEADER_LENGTH_BYTES) {
    throw new SafetensorsError("truncated: the file is too short to hold a header length");
  }
  const length = Number(
    new DataView(prefix.buffer, prefix.byteOffset, prefix.byteLength).getBigUint64(0, true),
  );
  if (!Number.isSafeInteger(length) || length <= 0 || length > MAX_HEADER_BYTES) {
    throw new SafetensorsError(
      `the header claims ${length} bytes, which is not a plausible safetensors header`,
    );
  }

  const body = source.read(HEADER_LENGTH_BYTES, length);
  if (body.length < length) {
    throw new SafetensorsError(
      `truncated: the header claims ${length} bytes and the file has ${body.length}`,
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(body)) as unknown;
  } catch {
    throw new SafetensorsError("the header is not valid JSON");
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new SafetensorsError("the header is not a JSON object of tensors");
  }

  const tensors: SafetensorsTensor[] = [];
  for (const [name, raw] of Object.entries(parsed as Record<string, unknown>)) {
    // The one reserved key: free-form strings, not a tensor.
    if (name === "__metadata__") continue;
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      throw new SafetensorsError(`tensor "${name}" is not an object`);
    }
    const entry = raw as Record<string, unknown>;
    const dtype = entry["dtype"];
    const shape = entry["shape"];
    if (typeof dtype !== "string") {
      throw new SafetensorsError(`tensor "${name}" has no dtype`);
    }
    if (!Array.isArray(shape)) {
      throw new SafetensorsError(`tensor "${name}" has no shape`);
    }
    const width = SAFETENSORS_DTYPE_BYTES[dtype];
    if (width === undefined) {
      throw new SafetensorsError(`tensor "${name}" has dtype "${dtype}", whose width is unknown`);
    }
    const elements = shapeElements(shape, name);
    tensors.push({ name, dtype, shape: shape as number[], elements, bytes: elements * width });
  }

  if (tensors.length === 0) {
    throw new SafetensorsError("the header declares no tensors");
  }

  return {
    tensors,
    totalParams: tensors.reduce((sum, tensor) => sum + tensor.elements, 0),
    totalBytes: tensors.reduce((sum, tensor) => sum + tensor.bytes, 0),
    headerBytes: HEADER_LENGTH_BYTES + length,
  };
}

export interface SafetensorsIndex {
  /** `metadata.total_size`: bytes across every shard. */
  totalSizeBytes: number;
  /** Distinct shard files named by the weight map. */
  shards: number;
  /** Tensors the map names, i.e. how many weights are spread across them. */
  tensors: number;
}

/**
 * Parse `model.safetensors.index.json`.
 *
 * Only two things matter here: the byte total, and the fact that it is a real
 * index rather than some other JSON file that happened to be named one.
 */
export function parseSafetensorsIndex(value: unknown, path: string): SafetensorsIndex {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new SafetensorsError(`${path} is not a JSON object`);
  }
  const index = value as Record<string, unknown>;
  const metadata = index["metadata"];
  if (typeof metadata !== "object" || metadata === null || Array.isArray(metadata)) {
    throw new SafetensorsError(`${path} has no metadata block`);
  }
  const totalSize = (metadata as Record<string, unknown>)["total_size"];
  if (typeof totalSize !== "number" || !Number.isFinite(totalSize) || totalSize <= 0) {
    throw new SafetensorsError(
      `${path} has no positive metadata.total_size, so it cannot give a parameter count`,
    );
  }

  const weightMap = index["weight_map"];
  const entries =
    typeof weightMap === "object" && weightMap !== null && !Array.isArray(weightMap)
      ? Object.entries(weightMap as Record<string, unknown>)
      : [];

  return {
    totalSizeBytes: totalSize,
    shards: new Set(entries.map(([, shard]) => String(shard))).size,
    tensors: entries.length,
  };
}
