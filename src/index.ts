/**
 * vramfit -- will this model run on my machine, and how fast?
 *
 * Public library surface. Everything here is pure arithmetic over plain data:
 * no I/O, no network, no global state.
 */

export type {
  Architecture,
} from "./architecture.js";
export { activeBlockFraction, deriveArchitecture } from "./architecture.js";

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
