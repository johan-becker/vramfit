import {
  DEFAULT_GGUF_ALIGNMENT,
  GGUF_FIXED_WIDTHS,
  GGUF_MAGIC,
  GGUF_MAGIC_BYTESWAPPED,
  SUPPORTED_GGUF_VERSIONS,
  findGgmlType,
  ggmlTensorBytes,
  ggufTypeName,
  type GgmlTypeSpec,
  type GgufArray,
  type GgufTypeName,
  type GgufValue,
} from "./format.js";

/**
 * A streaming GGUF header reader.
 *
 * The point of the exercise is that a 40 GB checkpoint has to be readable on a
 * laptop, so nothing here ever holds more than one window of the file. Reading
 * is pull-based over a `ByteSource`: the reader asks for a window when it runs
 * out, and *skips* -- moves the cursor without asking for bytes -- over the
 * parts it does not need, which is most of a tokenizer's 128k-entry token
 * list and all of the tensor data.
 *
 * The bounds below are not paranoia for its own sake. A GGUF header is a
 * length-prefixed format read from a file the user did not write: a corrupted
 * or hostile 8-byte length would otherwise turn into a multi-gigabyte
 * allocation before anything had a chance to notice.
 */

/** Thrown for anything wrong with the file, as opposed to a bug in the reader. */
export class GgufError extends Error {
  override readonly name = "GgufError";
}

/**
 * Random access to a sequence of bytes. The file-backed implementation is
 * `openGgufFile`; the tests use an in-memory one, which is how a byte-level
 * fixture can exercise the whole reader without a filesystem.
 */
export interface ByteSource {
  /** Total length in bytes. */
  readonly size: number;
  /** Bytes from `offset`. May return fewer than asked for at the end. */
  read(offset: number, length: number): Uint8Array;
}

export interface GgufReadOptions {
  /** Bytes pulled from the source per refill. */
  chunkBytes?: number;
  /** Most array elements decoded per metadata entry; the rest are skipped. */
  maxArrayValues?: number;
  /** Longest metadata string accepted, in bytes. */
  maxStringBytes?: number;
  /** Deepest nesting accepted in an array of arrays. */
  maxArrayDepth?: number;
  /** Most tensors accepted in the shape table. */
  maxTensors?: number;
  /** Most metadata entries accepted. */
  maxMetadataEntries?: number;
}

const DEFAULT_CHUNK_BYTES = 256 * 1024;
const DEFAULT_MAX_ARRAY_VALUES = 64;
/** Metadata strings are names, templates and licences. A MiB is generous. */
const DEFAULT_MAX_STRING_BYTES = 1024 * 1024;
/**
 * Deepest array-of-arrays accepted. Real files reach 2; each level costs 12
 * bytes in the file and one JavaScript stack frame to walk, so without a
 * bound a 1.2 MB header nested 100,000 deep overflows the stack and leaves a
 * `RangeError` where the reader promises a diagnostic naming the file.
 */
const DEFAULT_MAX_ARRAY_DEPTH = 64;
/** DeepSeek V3 at 61 layers x 256 experts is about 5000 tensors. */
const DEFAULT_MAX_TENSORS = 1_048_576;
const DEFAULT_MAX_METADATA_ENTRIES = 65_536;
/** ggml's GGML_MAX_DIMS. */
const MAX_TENSOR_DIMS = 4;

export interface GgufTensorInfo {
  name: string;
  /** Shape in ggml order: `dims[0]` is the fastest-moving axis. */
  dims: number[];
  type: GgmlTypeSpec;
  /** Byte offset within the tensor-data section. */
  offset: number;
  /** product(dims) -- the number of weights. */
  elements: number;
  /** What those weights occupy on disk, at this tensor's own type. */
  bytes: number;
}

export interface GgufHeader {
  version: number;
  tensorCount: number;
  metadataCount: number;
  /** Every metadata entry, keyed as the file spells it. */
  metadata: Map<string, GgufValue>;
  tensors: GgufTensorInfo[];
  alignment: number;
  /** First byte of the tensor-data section. */
  dataOffset: number;
  /** Bytes of tensor data implied by the shape table. */
  tensorBytes: number;
  /** Bytes of header, from the magic to the last tensor entry. */
  headerBytes: number;
  /** The source's total size, for the "is this file complete" check. */
  fileBytes: number;
  /** Bytes actually pulled from the source. Far less than `fileBytes`. */
  bytesRead: number;
}

/**
 * A cursor over a `ByteSource` holding one window at a time.
 *
 * `take` reads; `skip` does not. That asymmetry is the whole memory story: a
 * skip past the end of the window simply moves the cursor, and the next read
 * refills from wherever it landed.
 */
class Cursor {
  private buffer: Uint8Array = new Uint8Array(0);
  private view: DataView = new DataView(new ArrayBuffer(0));
  private windowStart = 0;
  private at = 0;
  private consumed = 0;
  private readonly decoder = new TextDecoder("utf8", { fatal: false });

  constructor(
    private readonly source: ByteSource,
    private readonly chunkBytes: number,
  ) {}

  get position(): number {
    return this.at;
  }

  /** Bytes pulled from the source so far. */
  get bytesRead(): number {
    return this.consumed;
  }

  private window(length: number): number {
    const offset = this.at - this.windowStart;
    if (offset >= 0 && offset + length <= this.buffer.length) return offset;

    const chunk = this.source.read(this.at, Math.max(length, this.chunkBytes));
    this.consumed += chunk.length;
    if (chunk.length < length) {
      throw new GgufError(
        `truncated at byte ${this.at}: needed ${length} more bytes, the source has ${chunk.length}`,
      );
    }
    this.buffer = chunk;
    this.windowStart = this.at;
    this.view = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
    return 0;
  }

  private take(length: number): number {
    const offset = this.window(length);
    this.at += length;
    return offset;
  }

  skip(length: number): void {
    if (length < 0) throw new GgufError(`negative skip of ${length} bytes`);
    const next = this.at + length;
    if (next > this.source.size) {
      throw new GgufError(
        `truncated: an entry at byte ${this.at} runs ${next - this.source.size} bytes past the end of a ${this.source.size}-byte source`,
      );
    }
    this.at = next;
  }

  // Every accessor takes its offset first and reads `this.view` afterwards.
  // `this.view.getUint8(this.take(1))` would evaluate the view before take()
  // had refilled the window, and so read from the previous window.
  u8(): number {
    const at = this.take(1);
    return this.view.getUint8(at);
  }

  i8(): number {
    const at = this.take(1);
    return this.view.getInt8(at);
  }

  u16(): number {
    const at = this.take(2);
    return this.view.getUint16(at, true);
  }

  i16(): number {
    const at = this.take(2);
    return this.view.getInt16(at, true);
  }

  u32(): number {
    const at = this.take(4);
    return this.view.getUint32(at, true);
  }

  i32(): number {
    const at = this.take(4);
    return this.view.getInt32(at, true);
  }

  f32(): number {
    const at = this.take(4);
    return this.view.getFloat32(at, true);
  }

  f64(): number {
    const at = this.take(8);
    return this.view.getFloat64(at, true);
  }

  /**
   * A 64-bit count as a JavaScript number. Anything past 2^53 - 1 is a
   * corrupt length rather than a real one, and silently rounding it would
   * turn a bad file into a wrong answer.
   */
  u64(what: string): number {
    const at = this.take(8);
    const raw = this.view.getBigUint64(at, true);
    if (raw > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new GgufError(`${what} is ${raw}, past the largest exactly representable integer`);
    }
    return Number(raw);
  }

  i64(what: string): number {
    const at = this.take(8);
    const raw = this.view.getBigInt64(at, true);
    if (raw > BigInt(Number.MAX_SAFE_INTEGER) || raw < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new GgufError(`${what} is ${raw}, past the largest exactly representable integer`);
    }
    return Number(raw);
  }

  ascii(length: number): string {
    const offset = this.take(length);
    return this.decoder.decode(this.buffer.subarray(offset, offset + length));
  }

  string(what: string, maxBytes: number): string {
    const length = this.u64(`${what} length`);
    if (length > maxBytes) {
      throw new GgufError(`${what} declares ${length} bytes, past the ${maxBytes}-byte limit`);
    }
    return this.ascii(length);
  }

  /** Walk a string without decoding it, for the parts nothing needs. */
  skipString(what: string, maxBytes: number): void {
    const length = this.u64(`${what} length`);
    if (length > maxBytes) {
      throw new GgufError(`${what} declares ${length} bytes, past the ${maxBytes}-byte limit`);
    }
    this.skip(length);
  }
}

interface Limits {
  maxArrayValues: number;
  maxStringBytes: number;
  maxArrayDepth: number;
}

/** `depth` is the nesting level of the array being walked; 0 outside one. */
function readScalar(
  cursor: Cursor,
  type: GgufTypeName,
  what: string,
  limits: Limits,
  depth: number,
): GgufValue {
  switch (type) {
    case "uint8":
      return cursor.u8();
    case "int8":
      return cursor.i8();
    case "uint16":
      return cursor.u16();
    case "int16":
      return cursor.i16();
    case "uint32":
      return cursor.u32();
    case "int32":
      return cursor.i32();
    case "float32":
      return cursor.f32();
    case "bool":
      return cursor.u8() !== 0;
    case "string":
      return cursor.string(what, limits.maxStringBytes);
    case "uint64":
      return cursor.u64(what);
    case "int64":
      return cursor.i64(what);
    case "float64":
      return cursor.f64();
    case "array":
      return readArray(cursor, what, limits, depth + 1);
  }
}

function skipValue(
  cursor: Cursor,
  type: GgufTypeName,
  what: string,
  limits: Limits,
  depth: number,
): void {
  const width = GGUF_FIXED_WIDTHS[type];
  if (width !== undefined) {
    cursor.skip(width);
    return;
  }
  if (type === "string") {
    cursor.skipString(what, limits.maxStringBytes);
    return;
  }
  readArray(cursor, what, { ...limits, maxArrayValues: 0 }, depth + 1);
}

/**
 * An array value.
 *
 * Only the first `maxArrayValues` elements are decoded. The rest are walked --
 * one bulk skip for a fixed-width element type, one length-prefix read per
 * element for strings and nested arrays -- so `length` is always the number
 * the file declares and no vocabulary is ever materialised. `vocab_size` is
 * frequently absent from Llama files and has to be counted from
 * `tokenizer.ggml.tokens`, which is why the count matters and the contents
 * do not.
 */
function readArray(cursor: Cursor, what: string, limits: Limits, depth: number): GgufArray {
  // Nesting is walked by mutual recursion and costs 12 bytes a level in the
  // file, so it is bounded like every other length here: an unbounded walk
  // turns a small crafted header into a stack overflow rather than into a
  // diagnostic naming the file.
  if (depth > limits.maxArrayDepth) {
    throw new GgufError(`${what} nests arrays more than ${limits.maxArrayDepth} deep`);
  }
  const typeCode = cursor.u32();
  const elementType = ggufTypeName(typeCode);
  if (elementType === undefined) {
    throw new GgufError(`${what} has element type ${typeCode}, which is not a GGUF type`);
  }
  const length = cursor.u64(`${what} length`);

  const width = GGUF_FIXED_WIDTHS[elementType];
  const decoded = Math.min(length, limits.maxArrayValues);
  const values: GgufValue[] = [];
  for (let index = 0; index < decoded; index++) {
    values.push(readScalar(cursor, elementType, `${what}[${index}]`, limits, depth));
  }

  const remaining = length - decoded;
  if (remaining > 0) {
    if (width !== undefined) {
      cursor.skip(width * remaining);
    } else {
      for (let index = 0; index < remaining; index++) {
        skipValue(cursor, elementType, `${what}[${decoded + index}]`, limits, depth);
      }
    }
  }

  return { elementType, length, values, truncated: remaining > 0 };
}

function readMagic(cursor: Cursor): void {
  const magic = cursor.ascii(4);
  if (magic === GGUF_MAGIC) return;
  if (magic === GGUF_MAGIC_BYTESWAPPED) {
    throw new GgufError(
      "this is a big-endian GGUF file; vramfit reads little-endian files, which is what every published converter writes",
    );
  }
  throw new GgufError(
    `not a GGUF file: it starts with ${JSON.stringify(magic)} rather than "GGUF"`,
  );
}

function readVersion(cursor: Cursor): number {
  const version = cursor.u32();
  if (SUPPORTED_GGUF_VERSIONS.includes(version)) return version;
  if (version === 1) {
    throw new GgufError(
      "GGUF v1 uses 32-bit lengths throughout and is not readable with the v2 layout; re-convert the model",
    );
  }
  throw new GgufError(
    `GGUF version ${version} is not supported; this build reads v${SUPPORTED_GGUF_VERSIONS.join(" and v")}`,
  );
}

function alignmentFrom(metadata: Map<string, GgufValue>): number {
  const raw = metadata.get("general.alignment");
  if (raw === undefined) return DEFAULT_GGUF_ALIGNMENT;
  if (typeof raw !== "number" || !Number.isInteger(raw) || raw <= 0 || (raw & (raw - 1)) !== 0) {
    throw new GgufError(`general.alignment must be a positive power of two, got ${String(raw)}`);
  }
  return raw;
}

/**
 * Read the header of a GGUF file and nothing else.
 *
 * Returns the metadata block, the tensor shape table and where the tensor data
 * would start. `bytesRead` on the result is the number of bytes actually
 * pulled from the source, which is what makes "this does not load the model"
 * a testable claim rather than a promise.
 */
export function readGgufHeader(source: ByteSource, options: GgufReadOptions = {}): GgufHeader {
  const limits: Limits = {
    maxArrayValues: Math.max(0, Math.floor(options.maxArrayValues ?? DEFAULT_MAX_ARRAY_VALUES)),
    maxStringBytes: Math.max(1, Math.floor(options.maxStringBytes ?? DEFAULT_MAX_STRING_BYTES)),
    maxArrayDepth: Math.max(1, Math.floor(options.maxArrayDepth ?? DEFAULT_MAX_ARRAY_DEPTH)),
  };
  const maxTensors = Math.max(0, Math.floor(options.maxTensors ?? DEFAULT_MAX_TENSORS));
  const maxMetadata = Math.max(
    0,
    Math.floor(options.maxMetadataEntries ?? DEFAULT_MAX_METADATA_ENTRIES),
  );
  const cursor = new Cursor(source, Math.max(64, Math.floor(options.chunkBytes ?? DEFAULT_CHUNK_BYTES)));

  readMagic(cursor);
  const version = readVersion(cursor);
  const tensorCount = cursor.u64("tensor count");
  const metadataCount = cursor.u64("metadata count");

  if (tensorCount > maxTensors) {
    throw new GgufError(`the file declares ${tensorCount} tensors, past the ${maxTensors} limit`);
  }
  if (metadataCount > maxMetadata) {
    throw new GgufError(
      `the file declares ${metadataCount} metadata entries, past the ${maxMetadata} limit`,
    );
  }

  const metadata = new Map<string, GgufValue>();
  for (let index = 0; index < metadataCount; index++) {
    const key = cursor.string(`metadata key ${index}`, limits.maxStringBytes);
    const typeCode = cursor.u32();
    const type = ggufTypeName(typeCode);
    if (type === undefined) {
      throw new GgufError(`metadata "${key}" has value type ${typeCode}, which is not a GGUF type`);
    }
    // Later entries win, matching llama.cpp, but a duplicated key is worth
    // knowing about rather than silently resolving.
    metadata.set(key, readScalar(cursor, type, `metadata "${key}"`, limits, 0));
  }

  const tensors: GgufTensorInfo[] = [];
  for (let index = 0; index < tensorCount; index++) {
    const name = cursor.string(`tensor name ${index}`, limits.maxStringBytes);
    const dimCount = cursor.u32();
    if (dimCount === 0 || dimCount > MAX_TENSOR_DIMS) {
      throw new GgufError(
        `tensor "${name}" declares ${dimCount} dimensions; ggml allows 1 to ${MAX_TENSOR_DIMS}`,
      );
    }
    const dims: number[] = [];
    let elements = 1;
    for (let axis = 0; axis < dimCount; axis++) {
      const dim = cursor.u64(`tensor "${name}" dimension ${axis}`);
      if (dim <= 0) {
        throw new GgufError(`tensor "${name}" has dimension ${axis} of ${dim}`);
      }
      dims.push(dim);
      elements *= dim;
    }
    const typeCode = cursor.u32();
    const type = findGgmlType(typeCode);
    if (type === undefined) {
      throw new GgufError(
        `tensor "${name}" has ggml type ${typeCode}, which this build does not know the block layout of`,
      );
    }
    const offset = cursor.u64(`tensor "${name}" offset`);
    tensors.push({ name, dims, type, offset, elements, bytes: ggmlTensorBytes(type, elements) });
  }

  const headerBytes = cursor.position;
  const alignment = alignmentFrom(metadata);
  const dataOffset = Math.ceil(headerBytes / alignment) * alignment;
  const tensorBytes = tensors.reduce((sum, tensor) => sum + tensor.bytes, 0);

  if (Number.isFinite(source.size) && dataOffset > source.size) {
    throw new GgufError(
      `truncated: the header ends at byte ${dataOffset} but the source is ${source.size} bytes`,
    );
  }

  return {
    version,
    tensorCount,
    metadataCount,
    metadata,
    tensors,
    alignment,
    dataOffset,
    tensorBytes,
    headerBytes,
    fileBytes: source.size,
    bytesRead: cursor.bytesRead,
  };
}
