import type { FitResult, QuantOption } from "../fit.js";
import type { DeviceSpec, ModelSpec } from "../types.js";
import { bytesToGiB, formatBytes, formatContext, formatParams } from "../units.js";
import {
  formatBandwidth,
  formatPercent,
  formatRate,
  formatSeconds,
  renderPairs,
  renderTable,
  wrap,
  type Pair,
} from "./format.js";

/**
 * Rendering. Everything here turns a `FitResult` into lines of text or into a
 * JSON payload; nothing here computes anything, so a wrong number in the
 * output is always a wrong number in the model.
 *
 * The report leads with the verdict, because that is the question. The
 * breakdown underneath exists so the verdict can be argued with: every row
 * carries the assumption behind it, and a reader who disagrees with one --
 * the runtime context, the compute buffer, the efficiency factor -- can see
 * exactly how much of the total it was responsible for.
 */

const WRAP_WIDTH = 78;

function describeKvGeometry(model: ModelSpec, fit: FitResult): string {
  const parts: string[] = [`${formatContext(fit.ctx)} tokens`];
  if (model.attention === "mla" && model.mla) {
    parts.push(
      `${model.nLayers} layers x ${model.mla.kvLoraRank + model.mla.qkRopeHeadDim}-wide latent (MLA)`,
    );
  } else {
    parts.push(`${model.nLayers} layers x ${model.nKvHeads} KV heads x ${model.headDim}`);
  }
  if (fit.footprint.kv.windowedLayers > 0) {
    parts.push(
      `${fit.footprint.kv.windowedLayers} windowed at ${model.attentionWindow?.windowSize ?? 0}`,
    );
  }
  if (fit.batch > 1) parts.push(`${fit.batch} sequences`);
  parts.push(fit.kvQuant.id);
  return parts.join(", ");
}

function memoryPairs(fit: FitResult): Pair[] {
  const { footprint, model, quant } = fit;
  const weightNote =
    model.moe === null
      ? `${formatParams(model.totalParams)} params at ${footprint.weights.effectiveBitsPerWeight.toFixed(2)} effective bits/weight`
      : `${formatParams(model.totalParams)} total / ${formatParams(model.activeParams)} active, ${footprint.weights.effectiveBitsPerWeight.toFixed(2)} effective bits/weight`;

  const pairs: Pair[] = [
    { label: "Weights", value: formatBytes(footprint.weights.totalBytes), note: weightNote },
    {
      label: "KV cache",
      value: formatBytes(footprint.kv.totalBytes),
      note: describeKvGeometry(model, fit),
    },
    {
      label: "Runtime context",
      value: formatBytes(footprint.runtimeContextBytes),
      note:
        fit.gpus > 1
          ? `${fit.device.family} driver and kernels, x${fit.gpus} devices`
          : `${fit.device.family} driver and kernels`,
    },
    {
      label: "Compute buffer",
      value: formatBytes(footprint.activationBytes),
      note: "activations, logits and graph scratch",
    },
    { label: "Total", value: formatBytes(footprint.totalBytes), note: `at ${quant.label}` },
    {
      label: "Available",
      value: formatBytes(fit.capacity.totalBytes),
      note:
        fit.gpus > 1
          ? `${fit.gpus} x ${fit.device.name}`
          : // A --vram override is already the usable figure, so there is no
            // fraction left to explain: capacity and installed are equal.
            fit.capacity.totalBytes < fit.capacity.installedBytes
            ? `${fit.device.name}, ${formatPercent(fit.device.usableFraction)} of ${bytesToGiB(fit.capacity.installedBytes).toFixed(0)} GiB wirable`
            : fit.device.name,
    },
  ];
  return pairs;
}

function verdictLine(fit: FitResult): string {
  const used = formatBytes(fit.usedBytes);
  const available = formatBytes(fit.capacity.totalBytes);
  if (fit.fits) {
    return `FITS  -  ${used} of ${available} used, ${formatBytes(fit.headroomBytes)} free (${formatPercent(fit.utilization)} utilised)`;
  }
  return `DOES NOT FIT  -  ${used} needed, ${available} available, ${formatBytes(-fit.headroomBytes)} short`;
}

function speedPairs(fit: FitResult): Pair[] {
  const decodeNote = fit.offload
    ? `blended across ${fit.offload.plan.gpuLayers} device and ${fit.offload.plan.cpuLayers} host layers`
    : `at ${formatContext(fit.ctx)} context, ${fit.batch === 1 ? "1 sequence" : `${fit.batch} sequences`}`;

  const pairs: Pair[] = [
    { label: "Decode", value: formatRate(fit.throughput.decode.tokensPerSecond), note: decodeNote },
  ];
  if (fit.batch > 1) {
    pairs.push({
      label: "Decode, aggregate",
      value: formatRate(fit.throughput.decode.aggregateTokensPerSecond),
      note: `${fit.batch} sequences in parallel`,
    });
  }
  pairs.push(
    {
      label: "Prefill",
      value: formatRate(fit.throughput.prefill.tokensPerSecond),
      note: `${(fit.throughput.prefill.flopsPerToken / 1e9).toFixed(1)} GFLOP per prompt token`,
    },
    {
      label: "Time to first token",
      value: formatSeconds(fit.throughput.timeToFirstTokenSeconds),
      note: `for a prompt of ${formatContext(fit.throughput.promptTokens)} tokens`,
    },
  );
  return pairs;
}

function offloadPairs(fit: FitResult): Pair[] {
  const offload = fit.offload;
  if (!offload) return [];
  const { plan, bandwidth } = offload;
  return [
    {
      label: "Layers in device memory",
      value: `${plan.gpuLayers} of ${plan.totalLayers}`,
      note: plan.vocabOnDevice ? "vocabulary tensors included" : "vocabulary tensors in system RAM",
    },
    {
      label: "Left in system RAM",
      value: formatBytes(offload.systemRamRequiredBytes),
      note: `of ${formatBytes(offload.systemRamAvailableBytes)} assumed available${offload.feasible ? "" : " -- NOT ENOUGH"}`,
    },
    {
      label: "Blended read bandwidth",
      value: formatBandwidth(bandwidth.peakBytesPerSecond),
      note: `harmonic mean; the device alone reads at ${formatBandwidth(fit.device.bandwidthGBs * 1e9)}`,
    },
  ];
}

/** The `vramfit check` report. */
export function renderCheck(
  fit: FitResult,
  recommendation: QuantOption | undefined,
): string[] {
  const lines: string[] = [];
  const heading = [
    fit.model.name,
    fit.quant.label,
    fit.gpus > 1 ? `${fit.gpus} x ${fit.device.name}` : fit.device.name,
  ].join("  |  ");

  lines.push(heading, "=".repeat(heading.length), "", verdictLine(fit), "");
  lines.push("Memory", ...renderPairs(memoryPairs(fit)), "");

  lines.push(
    "Capacity",
    ...renderPairs([
      {
        label: "Largest context that fits",
        value: fit.maxContext > 0 ? formatContext(fit.maxContext) : "none",
        note:
          fit.maxContext >= fit.model.maxCtx
            ? "the architecture's own maximum"
            : `${fit.quant.label}, batch ${fit.batch}, ${fit.kvQuant.id} cache`,
      },
      {
        label: `Best quant that fits at ${formatContext(fit.ctx)}`,
        value: recommendation ? recommendation.quant.label : "none",
        note: recommendation
          ? `${formatBytes(recommendation.fit.footprint.weights.totalBytes)} of weights, ${formatRate(recommendation.decodeTokensPerSecond)}`
          : "try a smaller model, a longer offload or more memory",
      },
    ]),
    "",
  );

  if (fit.offload) {
    lines.push("Partial offload", ...renderPairs(offloadPairs(fit)), "");
  }

  lines.push(
    `Speed (estimates: decode +/-${Math.round(fit.throughput.decodeErrorBand * 100)}%, prefill +/-${Math.round(fit.throughput.prefillErrorBand * 100)}%)`,
    ...renderPairs(speedPairs(fit)),
  );

  if (fit.warnings.length > 0) {
    lines.push("", "Notes");
    for (const warning of fit.warnings) {
      const wrapped = wrap(warning, WRAP_WIDTH, "    ");
      lines.push(`  - ${(wrapped[0] ?? "").trimStart()}`, ...wrapped.slice(1));
    }
  }

  return lines;
}

/** The `vramfit check --json` payload. Stable, documented in the README. */
export function checkJson(
  fit: FitResult,
  recommendation: QuantOption | undefined,
  version: string,
): unknown {
  return {
    vramfit: version,
    fits: fit.fits,
    model: {
      id: fit.model.id,
      name: fit.model.name,
      totalParams: fit.model.totalParams,
      activeParams: fit.model.activeParams,
      nLayers: fit.model.nLayers,
      nKvHeads: fit.model.nKvHeads,
      headDim: fit.model.headDim,
      attention: fit.model.attention,
      moe: fit.model.moe !== null,
    },
    device: {
      id: fit.device.id,
      name: fit.device.name,
      family: fit.device.family,
      vramGiB: fit.device.vramGiB,
      bandwidthGBs: fit.device.bandwidthGBs,
      usableFraction: fit.device.usableFraction,
      count: fit.gpus,
    },
    config: {
      quant: fit.quant.id,
      bitsPerWeight: fit.quant.bitsPerWeight,
      effectiveBitsPerWeight: fit.footprint.weights.effectiveBitsPerWeight,
      ctx: fit.ctx,
      batch: fit.batch,
      kvQuant: fit.kvQuant.id,
    },
    memory: {
      weightsBytes: fit.footprint.weights.totalBytes,
      kvCacheBytes: fit.footprint.kv.totalBytes,
      runtimeContextBytes: fit.footprint.runtimeContextBytes,
      activationBytes: fit.footprint.activationBytes,
      totalBytes: fit.footprint.totalBytes,
      capacityBytes: fit.capacity.totalBytes,
      headroomBytes: fit.headroomBytes,
      utilization: fit.utilization,
    },
    capacity: {
      maxContext: fit.maxContext,
      recommendedQuant: recommendation ? recommendation.quant.id : null,
    },
    throughput: {
      decodeTokensPerSecond: fit.throughput.decode.tokensPerSecond,
      aggregateDecodeTokensPerSecond: fit.throughput.decode.aggregateTokensPerSecond,
      prefillTokensPerSecond: fit.throughput.prefill.tokensPerSecond,
      promptTokens: fit.throughput.promptTokens,
      timeToFirstTokenSeconds: fit.throughput.timeToFirstTokenSeconds,
      decodeErrorBand: fit.throughput.decodeErrorBand,
      prefillErrorBand: fit.throughput.prefillErrorBand,
    },
    offload: fit.offload
      ? {
          gpuLayers: fit.offload.plan.gpuLayers,
          cpuLayers: fit.offload.plan.cpuLayers,
          vocabOnDevice: fit.offload.plan.vocabOnDevice,
          systemRamRequiredBytes: fit.offload.systemRamRequiredBytes,
          systemRamAvailableBytes: fit.offload.systemRamAvailableBytes,
          feasible: fit.offload.feasible,
          blendedBandwidthBytesPerSecond: fit.offload.bandwidth.peakBytesPerSecond,
        }
      : null,
    warnings: fit.warnings,
  };
}

/** The `vramfit best` table: every quantization, best quality first. */
export function renderBest(
  model: ModelSpec,
  device: DeviceSpec,
  options: readonly QuantOption[],
  ctx: number,
  gpus: number,
): string[] {
  const heading = `${model.name}  |  ${gpus > 1 ? `${gpus} x ${device.name}` : device.name}`;
  // The quantization notes are a paragraph each and would make every column
  // unreadable, so the table stays numeric and the note is shown once, for the
  // row that actually matters.
  const rows = options.map((option) => [
    option.quant.label,
    option.quant.bitsPerWeight.toFixed(2),
    formatBytes(option.fit.footprint.weights.totalBytes),
    formatBytes(option.fit.footprint.totalBytes),
    option.fit.fits ? "yes" : "no",
    option.maxContext > 0 ? formatContext(option.maxContext) : "-",
    formatRate(option.decodeTokensPerSecond),
  ]);

  const recommended = options.find((option) => option.fit.fits);

  return [
    heading,
    "=".repeat(heading.length),
    "",
    ...renderTable(
      [
        { header: "Quant" },
        { header: "bpw", align: "right" },
        { header: "Weights", align: "right" },
        { header: `Total at ${formatContext(ctx)}`, align: "right" },
        { header: "Fits", align: "right" },
        { header: "Max ctx", align: "right" },
        { header: "Decode", align: "right" },
      ],
      rows,
    ),
    "",
    ...(options.some((option) => !option.fit.fits)
      ? wrap(
          "Rows that do not fit show the decode speed with as many layers as possible offloaded to system RAM, which is what you would actually get.",
          WRAP_WIDTH,
        ).concat("")
      : []),
    ...(recommended
      ? [
          `Recommended: ${recommended.quant.label} -- highest quality that fits at ${formatContext(ctx)}, ${formatRate(recommended.decodeTokensPerSecond)}.`,
          ...wrap(recommended.quant.notes, WRAP_WIDTH, "  "),
        ]
      : [
          `Nothing in this range fits at ${formatContext(ctx)}. Reduce the context, offload layers, or pick a smaller model.`,
        ]),
  ];
}

export function bestJson(
  model: ModelSpec,
  device: DeviceSpec,
  options: readonly QuantOption[],
  ctx: number,
  version: string,
): unknown {
  return {
    vramfit: version,
    model: { id: model.id, name: model.name, totalParams: model.totalParams },
    device: { id: device.id, name: device.name, vramGiB: device.vramGiB },
    ctx,
    recommended: options.find((option) => option.fit.fits)?.quant.id ?? null,
    quants: options.map((option) => ({
      id: option.quant.id,
      label: option.quant.label,
      bitsPerWeight: option.quant.bitsPerWeight,
      qualityRank: option.quant.qualityRank,
      weightsBytes: option.fit.footprint.weights.totalBytes,
      totalBytes: option.fit.footprint.totalBytes,
      fits: option.fit.fits,
      maxContext: option.maxContext,
      decodeTokensPerSecond: option.decodeTokensPerSecond,
    })),
  };
}

/** `vramfit devices`. */
export function renderDevices(devices: readonly DeviceSpec[]): string[] {
  return renderTable(
    [
      { header: "ID" },
      { header: "Name" },
      { header: "Memory", align: "right" },
      { header: "Usable", align: "right" },
      { header: "Bandwidth", align: "right" },
      { header: "FP16", align: "right" },
      { header: "Family" },
    ],
    devices.map((device) => [
      device.id,
      device.name,
      `${device.vramGiB} GiB`,
      `${(device.vramGiB * device.usableFraction).toFixed(0)} GiB`,
      `${device.bandwidthGBs} GB/s`,
      `${device.fp16Tflops} TF`,
      device.family,
    ]),
  );
}

/** `vramfit models`. */
export function renderModels(models: readonly ModelSpec[]): string[] {
  return renderTable(
    [
      { header: "ID" },
      { header: "Name" },
      { header: "Params", align: "right" },
      { header: "Active", align: "right" },
      { header: "Layers", align: "right" },
      { header: "KV heads", align: "right" },
      { header: "Max ctx", align: "right" },
      { header: "Attention" },
    ],
    models.map((model) => [
      model.id,
      model.name,
      formatParams(model.totalParams),
      model.moe === null ? "dense" : formatParams(model.activeParams),
      String(model.nLayers),
      String(model.nKvHeads),
      formatContext(model.maxCtx),
      model.attention === "mla"
        ? "MLA"
        : model.attentionWindow
          ? `GQA + SWA/${model.attentionWindow.fullAttentionEvery}`
          : "GQA",
    ]),
  );
}
