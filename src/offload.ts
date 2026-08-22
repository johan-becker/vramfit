import { activeBlockFraction } from "./architecture.js";
import type { KvCacheBreakdown, WeightBreakdown } from "./memory.js";
import type { ModelSpec } from "./types.js";

/**
 * Rule 6: partial offload.
 *
 * "It does not fit" is the least useful answer a capacity planner can give,
 * because it is almost never true in practice. llama.cpp's `--n-gpu-layers`
 * puts as many transformer blocks as will fit in device memory and evaluates
 * the rest on the CPU, out of system RAM. The model runs; it just runs at the
 * speed of the slowest memory any part of it lives in.
 *
 * The placement rules modelled here are llama.cpp's:
 *
 *   - Offloading is per repeating block, from the last layer backwards, and a
 *     layer takes its slice of the KV cache with it.
 *   - The non-repeating tensors (token_embd and the output projection) go to
 *     the device only once every layer is already there, which is why the last
 *     couple of hundred MB of a "nearly fits" model are so hard to place.
 *   - Fixed overheads -- the runtime context and the compute buffer -- are
 *     charged to the device before any layer is placed, because they have to
 *     be there for the device to do anything at all.
 */

export interface OffloadPlan {
  totalLayers: number;
  /** Layers resident in device memory. */
  gpuLayers: number;
  /** Layers left in system RAM and evaluated on the CPU. */
  cpuLayers: number;
  /** True when the vocabulary tensors also fit on the device. */
  vocabOnDevice: boolean;
  /** Weights plus KV cache placed in device memory. Excludes fixed overheads. */
  deviceResidentBytes: number;
  /** Weights plus KV cache left in system RAM. */
  hostResidentBytes: number;
  /** Bytes read from device memory to decode one token. */
  deviceReadBytesPerToken: number;
  /** Bytes read from system RAM to decode one token. */
  hostReadBytesPerToken: number;
  /** Share of the per-token read served by device memory, 0-1. */
  deviceReadShare: number;
}

/**
 * Fill `budgetBytes` of device memory with as many layers as it will hold.
 *
 * `budgetBytes` is what is left after the runtime context and compute buffer,
 * which the caller subtracts. Layers are not uniform when a model interleaves
 * sliding-window attention -- a local layer's cache is a fraction of a global
 * layer's -- so the fill walks the actual per-layer cache sizes rather than
 * dividing the total by the layer count.
 *
 * Storage and read cost differ for a mixture of experts: memory has to hold
 * every expert, but only the routed ones are read per token. Both are tracked,
 * because the first decides how many layers fit and the second decides how
 * slow the result is.
 */
export function planOffload(
  model: ModelSpec,
  weights: WeightBreakdown,
  kv: KvCacheBreakdown,
  budgetBytes: number,
): OffloadPlan {
  const activeShare = activeBlockFraction(model);
  const storagePerLayer = weights.bytesPerLayer;
  const readPerLayer = storagePerLayer * activeShare;

  let remaining = Math.max(0, budgetBytes);
  let gpuLayers = 0;
  let deviceResidentBytes = 0;
  let deviceReadBytesPerToken = 0;

  // llama.cpp offloads the tail of the model, so walk backwards.
  for (let layer = model.nLayers - 1; layer >= 0; layer--) {
    const layerKv = kv.bytesPerLayer[layer] ?? 0;
    const cost = storagePerLayer + layerKv;
    if (cost > remaining) break;
    remaining -= cost;
    gpuLayers++;
    deviceResidentBytes += cost;
    deviceReadBytesPerToken += readPerLayer + layerKv;
  }

  if (gpuLayers === model.nLayers) {
    // Re-derive from the exact totals rather than the running sum. Summing
    // `blockBytes / nLayers` eighty times drifts by a few kilobytes in double
    // precision, which is enough to decide the vocabulary placement below the
    // wrong way when a budget lands exactly on the boundary.
    deviceResidentBytes = weights.blockBytes + kv.totalBytes;
    deviceReadBytesPerToken = weights.blockBytes * activeShare + kv.totalBytes;
    remaining = Math.max(0, budgetBytes) - deviceResidentBytes;
  }

  const vocabOnDevice = gpuLayers === model.nLayers && weights.embeddingBytes <= remaining;
  if (vocabOnDevice) {
    deviceResidentBytes += weights.embeddingBytes;
    deviceReadBytesPerToken += weights.logitMatrixBytes;
  }

  const totalResident = weights.totalBytes + kv.totalBytes;
  const totalRead = weights.activeBytes + kv.totalBytes;

  return {
    totalLayers: model.nLayers,
    gpuLayers,
    cpuLayers: model.nLayers - gpuLayers,
    vocabOnDevice,
    deviceResidentBytes,
    hostResidentBytes: Math.max(0, totalResident - deviceResidentBytes),
    deviceReadBytesPerToken,
    hostReadBytesPerToken: Math.max(0, totalRead - deviceReadBytesPerToken),
    deviceReadShare: totalRead > 0 ? deviceReadBytesPerToken / totalRead : 1,
  };
}

export interface BandwidthPair {
  /** Device memory, peak bytes/second. */
  devicePeakBytesPerSecond: number;
  deviceEfficiency: number;
  /** System RAM, peak bytes/second. */
  hostPeakBytesPerSecond: number;
  hostEfficiency: number;
}

export interface BlendedBandwidth {
  /** Harmonic mean of the two peaks, weighted by the bytes read from each. */
  peakBytesPerSecond: number;
  /** The same blend with each side's kernel efficiency applied. */
  effectiveBytesPerSecond: number;
  /** effective / peak, so the two can be reported separately. */
  efficiency: number;
  /** Seconds of memory traffic per decoded token. */
  secondsPerToken: number;
}

/**
 * Blend two memory systems into the one figure decode speed depends on.
 *
 * A token's weights are read from wherever they live, one part after the
 * other, so the times add:
 *
 *     t = device_bytes / device_bw + host_bytes / host_bw
 *     effective_bw = (device_bytes + host_bytes) / t
 *
 * which is the harmonic mean of the two bandwidths weighted by their byte
 * shares. Harmonic, not arithmetic, is the whole point: it is dominated by the
 * slow side. Moving 10% of a model off a 1008 GB/s RTX 4090 into
 * 89.6 GB/s system RAM does not cost 10% of the speed -- the blended bandwidth
 * falls to 498 GB/s, less than half, because that last 10% of the bytes takes
 * longer to read than the other 90% put together. This is why "just offload a few layers"
 * disappoints, and why the number is worth computing rather than guessing.
 */
export function blendBandwidth(plan: OffloadPlan, bandwidths: BandwidthPair): BlendedBandwidth {
  const totalBytes = plan.deviceReadBytesPerToken + plan.hostReadBytesPerToken;
  if (totalBytes <= 0) {
    return {
      peakBytesPerSecond: bandwidths.devicePeakBytesPerSecond,
      effectiveBytesPerSecond:
        bandwidths.devicePeakBytesPerSecond * bandwidths.deviceEfficiency,
      efficiency: bandwidths.deviceEfficiency,
      secondsPerToken: 0,
    };
  }

  const peakSeconds =
    plan.deviceReadBytesPerToken / bandwidths.devicePeakBytesPerSecond +
    plan.hostReadBytesPerToken / bandwidths.hostPeakBytesPerSecond;
  const effectiveSeconds =
    plan.deviceReadBytesPerToken /
      (bandwidths.devicePeakBytesPerSecond * bandwidths.deviceEfficiency) +
    plan.hostReadBytesPerToken /
      (bandwidths.hostPeakBytesPerSecond * bandwidths.hostEfficiency);

  const peakBytesPerSecond = totalBytes / peakSeconds;
  const effectiveBytesPerSecond = totalBytes / effectiveSeconds;

  return {
    peakBytesPerSecond,
    effectiveBytesPerSecond,
    efficiency: effectiveBytesPerSecond / peakBytesPerSecond,
    secondsPerToken: effectiveSeconds,
  };
}

export interface ComputePair {
  devicePeakFlopsPerSecond: number;
  deviceEfficiency: number;
  hostPeakFlopsPerSecond: number;
  hostEfficiency: number;
}

export interface BlendedCompute {
  peakFlopsPerSecond: number;
  effectiveFlopsPerSecond: number;
  efficiency: number;
}

/**
 * The same blend for prefill, which is compute bound rather than bandwidth
 * bound and so is weighted by layers rather than by bytes: a layer left in
 * system RAM is evaluated by the CPU, at a couple of TFLOPs instead of a
 * couple of hundred. It is harmonic for the same reason -- the CPU layers
 * dominate the total time -- which is why a partly-offloaded model has a
 * time-to-first-token that feels out of proportion to the fraction offloaded.
 */
export function blendCompute(plan: OffloadPlan, compute: ComputePair): BlendedCompute {
  if (plan.totalLayers <= 0) {
    return {
      peakFlopsPerSecond: compute.devicePeakFlopsPerSecond,
      effectiveFlopsPerSecond: compute.devicePeakFlopsPerSecond * compute.deviceEfficiency,
      efficiency: compute.deviceEfficiency,
    };
  }

  const deviceShare = plan.gpuLayers / plan.totalLayers;
  const hostShare = 1 - deviceShare;

  const peakSeconds =
    deviceShare / compute.devicePeakFlopsPerSecond + hostShare / compute.hostPeakFlopsPerSecond;
  const effectiveSeconds =
    deviceShare / (compute.devicePeakFlopsPerSecond * compute.deviceEfficiency) +
    hostShare / (compute.hostPeakFlopsPerSecond * compute.hostEfficiency);

  const peakFlopsPerSecond = 1 / peakSeconds;
  const effectiveFlopsPerSecond = 1 / effectiveSeconds;
  return {
    peakFlopsPerSecond,
    effectiveFlopsPerSecond,
    efficiency: effectiveFlopsPerSecond / peakFlopsPerSecond,
  };
}
