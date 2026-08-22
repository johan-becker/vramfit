/**
 * Reading a real GGUF checkpoint instead of trusting the bundled database.
 *
 * `readGgufHeader` is pure and works over any `ByteSource`; `readGgufFile`
 * is the thin filesystem wrapper. `describeGguf` turns the header into the
 * `ModelSpec` and `QuantSpec` the rest of the package consumes.
 */
export {
  DEFAULT_GGUF_ALIGNMENT,
  GGML_TYPES,
  GGUF_FILE_TYPES,
  GGUF_MAGIC,
  GGUF_TYPE_NAMES,
  SUPPORTED_GGUF_VERSIONS,
  findGgmlType,
  ggmlBitsPerWeight,
  ggmlTensorBytes,
  ggufTypeName,
  isGgufArray,
  type GgmlTypeSpec,
  type GgufArray,
  type GgufTypeName,
  type GgufValue,
} from "./format.js";
export {
  GgufError,
  readGgufHeader,
  type ByteSource,
  type GgufHeader,
  type GgufReadOptions,
  type GgufTensorInfo,
} from "./reader.js";
export { openByteSource, readGgufFile, type FileByteSource } from "./file.js";
export {
  describeGguf,
  ggufArchitecture,
  ggufFileType,
  modelFromGguf,
  quantFromGguf,
  type GgufModel,
  type GgufModelOptions,
} from "./model.js";
