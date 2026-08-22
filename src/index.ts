/**
 * vramfit -- will this model run on my machine, and how fast?
 *
 * Public library surface. The arithmetic is pure: plain data in, plain data
 * out, no network, no global state. The I/O in the package is local and
 * explicit: the lazy, memoised read of the bundled `data/*.json` database
 * exposed below, which only happens if you ask for a bundled model or device
 * by name, and the readers that open a GGUF or HuggingFace checkpoint you
 * point them at. Nothing here downloads anything, ever.
 */

export type {
  Architecture,
} from "./architecture.js";
export { activeBlockFraction, deriveArchitecture } from "./architecture.js";

export type { Named, Registry } from "./db/index.js";
export {
  SCHEMA_VERSION,
  SpecValidationError,
  createRegistry,
  devices,
  findDevice,
  findModel,
  getDevice,
  getModel,
  listDevices,
  listModels,
  models,
  normalizeKey,
  parseDatabase,
  parseDeviceSpec,
  parseModelSpec,
} from "./db/index.js";

export {
  GGML_TYPES,
  GGUF_FILE_TYPES,
  GGUF_MAGIC,
  GGUF_TYPE_NAMES,
  GgufError,
  SUPPORTED_GGUF_VERSIONS,
  describeGguf,
  findGgmlType,
  ggmlBitsPerWeight,
  ggmlTensorBytes,
  ggufArchitecture,
  ggufFileType,
  isGgufArray,
  modelFromGguf,
  openByteSource,
  quantFromGguf,
  readGgufFile,
  readGgufHeader,
} from "./gguf/index.js";
export type {
  ByteSource,
  FileByteSource,
  GgmlTypeSpec,
  GgufArray,
  GgufHeader,
  GgufModel,
  GgufModelOptions,
  GgufReadOptions,
  GgufTensorInfo,
  GgufTypeName,
  GgufValue,
} from "./gguf/index.js";

export {
  HfConfigError,
  SAFETENSORS_DTYPE_BYTES,
  SafetensorsError,
  TORCH_DTYPE_BYTES,
  modelFromHfConfig,
  parseSafetensorsIndex,
  readSafetensorsHeader,
  textConfigOf,
} from "./hf/index.js";
export type {
  HfModel,
  HfModelOptions,
  ParamSource,
  SafetensorsHeader,
  SafetensorsIndex,
  SafetensorsTensor,
} from "./hf/index.js";

export type {
  Capacity,
  FitOptions,
  FitResult,
  OffloadResult,
  QuantOption,
  QuantSearchOptions,
} from "./fit.js";
export {
  DEFAULT_SYSTEM_COMPUTE_TFLOPS,
  DEFAULT_SYSTEM_RAM_BANDWIDTH_GBS,
  DEFAULT_SYSTEM_RAM_GIB,
  checkFit,
  computeCapacity,
  evaluateQuants,
  maxContextFor,
  recommendQuant,
} from "./fit.js";

export type {
  BandwidthPair,
  BlendedBandwidth,
  BlendedCompute,
  ComputePair,
  OffloadPlan,
} from "./offload.js";
export { blendBandwidth, blendCompute, planOffload } from "./offload.js";

export type {
  ActivationOptions,
  FootprintOptions,
  KvCacheBreakdown,
  KvCacheOptions,
  MemoryFootprint,
  WeightBreakdown,
} from "./memory.js";
export {
  RUNTIME_CONTEXT_BYTES,
  computeActivationBytes,
  computeFootprint,
  computeKvCacheBytes,
  computeWeightBytes,
} from "./memory.js";

export {
  GGUF_QUANT_FAMILIES,
  KV_QUANTS,
  QUANTS,
  findKvQuant,
  findQuant,
  getKvQuant,
  getQuant,
  quantizedBytes,
  quantsByQuality,
} from "./quant.js";

export type {
  DecodeEstimate,
  DecodeInput,
  PrefillEstimate,
  PrefillInput,
  ThroughputEstimate,
  ThroughputOptions,
} from "./throughput.js";
export {
  DECODE_ERROR_BAND,
  DEQUANT_EFFICIENCY,
  MEMORY_BANDWIDTH_EFFICIENCY,
  PREFILL_ERROR_BAND,
  PREFILL_MFU,
  bandwidthEfficiency,
  estimateDecodeFrom,
  estimatePrefillFrom,
  estimateThroughput,
  prefillFlopsPerToken,
} from "./throughput.js";

export type {
  AttentionKind,
  AttentionWindowSpec,
  DeviceFamily,
  DeviceSpec,
  KvQuantSpec,
  MlaSpec,
  ModelSpec,
  MoeSpec,
  QuantFamily,
  QuantSpec,
} from "./types.js";

export {
  GB_DECIMAL,
  GIB,
  KIB,
  MIB,
  TFLOP,
  bytesToGiB,
  bytesToMiB,
  formatBytes,
  formatContext,
  formatParams,
  gbPerSecondToBytesPerSecond,
  gibToBytes,
} from "./units.js";
