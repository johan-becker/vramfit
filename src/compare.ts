import { checkFit, type FitOptions, type FitResult } from "./fit.js";
import type { DeviceSpec, ModelSpec, QuantSpec } from "./types.js";

/**
 * One model, several devices, one table.
 *
 * "Will it run on a 4090 or do I need two of them, and what does a Mac Studio
 * do instead?" is the question behind most of the forum threads this package
 * exists to replace, and answering it needs nothing new -- `checkFit` already
 * answers it once per device. What is worth getting right is the *ordering*,
 * because a list of eight devices with no ranking is as unhelpful as no list.
 *
 * The ranking is: what fits comes before what does not, then fastest first.
 * That is the honest order for a "which of these should I buy or use" table --
 * once a configuration fits, decode speed is what you actually feel, and a
 * device with more spare memory than another is only better if you intend to
 * use the context it buys, which the `Max ctx` column shows rather than the
 * ranking assuming.
 *
 * Rows that do not fit are ordered by how close they came, so the last few
 * lines read as "and here is what you would have to change".
 */

export interface DeviceCandidate {
  /** Bundled device id, name or alias. */
  device: DeviceSpec;
  /** Identical devices working together. */
  gpus: number;
}

export interface DeviceComparison {
  device: DeviceSpec;
  gpus: number;
  fit: FitResult;
  /** Decode speed as the deployment would actually run: offloaded if need be. */
  decodeTokensPerSecond: number;
  /** Free bytes when it fits, negative by the shortfall when it does not. */
  headroomBytes: number;
  /** True for the single row the table recommends. */
  best: boolean;
}

export interface DeviceListEntry {
  /** The device name, id or alias, with any count suffix removed. */
  query: string;
  /** Identical devices asked for by this entry. */
  gpus: number;
}

/** Split a `--devices` list into its entries, refusing an empty one. */
export function splitDeviceList(raw: string): string[] {
  const entries = raw
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry !== "");
  if (entries.length === 0) {
    throw new Error("no devices given; expected a list such as 4090,3090x2,m4-max");
  }
  return entries;
}

/**
 * Read one entry's optional `xN` count.
 *
 * The suffix is lexically ambiguous with a device name -- `rtx4090` and
 * `4090x2` are the same shape, and six bundled aliases end in `x` followed by
 * digits -- so the caller is expected to try the whole token as a name first
 * and only fall back to this when it is not one. `defaultGpus` is what an
 * entry with no suffix means, which is where `--gpus` enters.
 */
export function parseDeviceEntry(entry: string, defaultGpus = 1): DeviceListEntry {
  const match = /^(.*?)\s*[x*]\s*(\d+)$/i.exec(entry);
  if (match === null) return { query: entry, gpus: defaultGpus };
  const gpus = Number.parseInt(match[2] as string, 10);
  if (gpus < 1) {
    throw new Error(`"${entry}" asks for ${gpus} devices; the count must be at least 1`);
  }
  return { query: (match[1] as string).trim(), gpus };
}

/**
 * Parse a `--devices 4090,3090x2,m4-max` list.
 *
 * The `xN` suffix is the reason this is not a `split(",")`: comparing one
 * 4090 against two of them is the most common form of the question, and
 * spelling it as a separate `--gpus` flag would apply it to every row. An
 * entry that carries no suffix takes `defaultGpus`, so `--gpus 2` still means
 * something on a list that does not spell the count out per device.
 */
export function parseDeviceList(raw: string, defaultGpus = 1): DeviceListEntry[] {
  return splitDeviceList(raw).map((entry) => parseDeviceEntry(entry, defaultGpus));
}

function rank(a: DeviceComparison, b: DeviceComparison): number {
  if (a.fit.fits !== b.fit.fits) return a.fit.fits ? -1 : 1;
  if (a.fit.fits) {
    if (b.decodeTokensPerSecond !== a.decodeTokensPerSecond) {
      return b.decodeTokensPerSecond - a.decodeTokensPerSecond;
    }
    if (b.headroomBytes !== a.headroomBytes) return b.headroomBytes - a.headroomBytes;
  } else if (b.headroomBytes !== a.headroomBytes) {
    // Both short: the one that came closest goes first.
    return b.headroomBytes - a.headroomBytes;
  }
  // Two devices that are equal on every count still have to sort the same way
  // twice, or the table would shuffle between runs.
  return a.device.id.localeCompare(b.device.id);
}

/** Evaluate one model on each candidate device, best first. */
export function compareDevices(
  model: ModelSpec,
  quant: QuantSpec,
  candidates: readonly DeviceCandidate[],
  options: FitOptions = {},
): DeviceComparison[] {
  const rows = candidates.map((candidate): DeviceComparison => {
    const fit = checkFit(model, quant, candidate.device, { ...options, gpus: candidate.gpus });
    return {
      device: candidate.device,
      gpus: candidate.gpus,
      fit,
      decodeTokensPerSecond: fit.throughput.decode.tokensPerSecond,
      headroomBytes: fit.headroomBytes,
      best: false,
    };
  });

  const ordered = rows.toSorted(rank);
  const winner = ordered.find((row) => row.fit.fits);
  if (winner !== undefined) winner.best = true;
  return ordered;
}
