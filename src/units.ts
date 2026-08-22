/**
 * Unit conventions used throughout vramfit.
 *
 * Memory is binary. A "24 GB" RTX 3090 has 24576 MiB = 24 * 2^30 bytes, and an
 * Apple "64 GB" machine has 64 * 2^30 bytes. Every memory quantity in this
 * package is therefore expressed in bytes internally and reported in GiB
 * (2^30 bytes), and the device database field is named `vramGiB` rather than
 * `vramGB` so nobody has to guess.
 *
 * Bandwidth and FLOPs are decimal, because that is how vendors specify them:
 * the 3090's 936 GB/s means 936e9 bytes/s, not 936 * 2^30. Mixing the two up
 * is a silent 7.4% error, which is more than the spread between a lot of the
 * quantization levels this tool compares.
 */

export const KIB = 1024;
export const MIB = 1024 * 1024;
export const GIB = 1024 * 1024 * 1024;

/** Decimal giga-, for bandwidth (GB/s) and FLOPs (TFLOP/s). */
export const GB_DECIMAL = 1e9;
export const TFLOP = 1e12;

export function gibToBytes(gib: number): number {
  return gib * GIB;
}

export function bytesToGiB(bytes: number): number {
  return bytes / GIB;
}

export function bytesToMiB(bytes: number): number {
  return bytes / MIB;
}

/** Bandwidth given in decimal GB/s, returned as bytes per second. */
export function gbPerSecondToBytesPerSecond(gbs: number): number {
  return gbs * GB_DECIMAL;
}

/**
 * Format a byte count as a GiB string with a fixed number of decimals.
 * Small values fall back to MiB so a 40 MiB compute buffer does not print
 * as "0.04 GiB".
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (!Number.isFinite(bytes)) return "n/a";
  if (Math.abs(bytes) < GIB / 10) {
    return `${bytesToMiB(bytes).toFixed(bytes === 0 ? 0 : 1)} MiB`;
  }
  return `${bytesToGiB(bytes).toFixed(decimals)} GiB`;
}

/** Compact parameter counts: 8.03 B, 671 M, 1.24 B. */
export function formatParams(params: number): string {
  if (params >= 1e9) return `${(params / 1e9).toFixed(params / 1e9 >= 100 ? 0 : 2)}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(0)}M`;
  return `${params}`;
}

/** Context lengths read better as 8K / 128K than as 8192 / 131072. */
export function formatContext(tokens: number): string {
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
  return `${tokens}`;
}
