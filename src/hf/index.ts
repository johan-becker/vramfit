/**
 * Reading a HuggingFace checkpoint from a local path.
 *
 * Parsing only: `config.json` in, `ModelSpec` out, plus the two safetensors
 * readers that give the parameter count the config cannot. Nothing here opens
 * a file or talks to the hub -- the caller supplies the bytes, which is what
 * lets the whole path be tested without a filesystem.
 */
export {
  HfConfigError,
  modelFromHfConfig,
  textConfigOf,
  type HfModel,
  type HfModelOptions,
  type ParamSource,
} from "./config.js";
export {
  SAFETENSORS_DTYPE_BYTES,
  SafetensorsError,
  TORCH_DTYPE_BYTES,
  parseSafetensorsIndex,
  readSafetensorsHeader,
  type SafetensorsHeader,
  type SafetensorsIndex,
  type SafetensorsTensor,
} from "./safetensors.js";
