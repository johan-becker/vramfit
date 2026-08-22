import type { DeviceComparison } from "../compare.js";
import type { FitResult, QuantOption } from "../fit.js";
import type { FleetMachine, FleetReport } from "../fleet.js";
import { notesFor, type LauncherPlan, type LauncherRuntime } from "../launcher.js";
import { findQuant } from "../quant.js";
import type { Recommendation, UseCaseProfile } from "../recommend.js";
import type { DeviceSpec, ModelSpec, QuantSpec } from "../types.js";
import { GIB, bytesToGiB, formatBytes, formatContext, formatParams } from "../units.js";
import { renderMemoryBar } from "./bar.js";
import { PLAIN_PALETTE, type Palette } from "./color.js";
import { renderExplain } from "./explain.js";
import type { ResolvedModel } from "./source.js";
import {
  formatBandwidth,
  formatPercent,
  formatRate,
  formatSeconds,
  renderPairs,
  renderTable,
  wrap,
  type Column,
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

/**
 * A table before it is drawn.
 *
 * The text and Markdown renderings differ only in how cells are separated, so
 * they share the cells: a column that gains a figure in one gains it in both,
 * and neither can quietly drift from the other.
 */
export interface TableShape {
  columns: Column[];
  rows: string[][];
}

/** Presentation choices a report takes from the command line. */
export interface ReportOptions {
  /** Where the model came from, when it did not come from the database. */
  source?: ResolvedModel;
  /** The launch flags to print, and which runtimes to print them for. */
  launcher?: { plan: LauncherPlan; runtime: LauncherRuntime };
  /** Colour, when the output is going somewhere that can show it. */
  palette?: Palette;
  /** Show every headline number with the arithmetic that produced it. */
  explain?: boolean;
}

/**
 * The line above the report naming the file the model was read from. Absent
 * for the bundled database, which needs no explanation, and for a spec file,
 * which is already the user's own words.
 */
function sourceLines(options: ReportOptions): string[] {
  const description = options.source?.description;
  return description === undefined ? [] : [description, ""];
}

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

function verdictLine(fit: FitResult, palette: Palette): string {
  const used = formatBytes(fit.usedBytes);
  const available = formatBytes(fit.capacity.totalBytes);
  if (fit.fits) {
    return `${palette.paint("good", "FITS")}  -  ${used} of ${available} used, ${formatBytes(fit.headroomBytes)} free (${formatPercent(fit.utilization)} utilised)`;
  }
  return `${palette.paint("bad", "DOES NOT FIT")}  -  ${used} needed, ${available} available, ${formatBytes(-fit.headroomBytes)} short`;
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
  options: ReportOptions = {},
): string[] {
  const lines: string[] = [];
  const heading = [
    fit.model.name,
    fit.quant.label,
    fit.gpus > 1 ? `${fit.gpus} x ${fit.device.name}` : fit.device.name,
  ].join("  |  ");

  const palette = options.palette ?? PLAIN_PALETTE;
  lines.push(
    heading,
    "=".repeat(heading.length),
    "",
    ...sourceLines(options),
    verdictLine(fit, palette),
    "",
  );
  lines.push(
    "Memory",
    ...renderPairs(memoryPairs(fit)),
    "",
    ...renderMemoryBar(fit, palette),
    "",
  );

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

  if (options.explain === true) {
    lines.push("", ...renderExplain(fit));
  }

  const launcher = options.launcher;
  if (launcher !== undefined) {
    lines.push("", ...renderLaunch(launcher.plan, launcher.runtime));
  }

  // Notes the source reader had to make -- a parameter count that did not
  // reconcile, an assumed dtype -- belong beside the ones the fit produced,
  // and come first, because they are about the inputs rather than the answer.
  const notes = [...(options.source?.notes ?? []), ...fit.warnings];
  if (notes.length > 0) {
    lines.push("", "Notes");
    for (const warning of notes) {
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
  source?: ResolvedModel,
  launcher?: LauncherPlan,
): unknown {
  return {
    vramfit: version,
    fits: fit.fits,
    model: {
      id: fit.model.id,
      name: fit.model.name,
      // Where the spec came from: the bundled database, a GGUF file read from
      // its own header, a HuggingFace config.json, or a --model-json spec.
      origin: source?.origin ?? "database",
      from: source?.from ?? fit.model.id,
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
    // Present only when --launcher was asked for: the documented payload has
    // a fixed set of keys, and a block of flags nobody requested does not
    // belong in it.
    ...(launcher === undefined ? {} : { launcher: launcherJson(launcher) }),
    warnings: fit.warnings,
  };
}

/** True when the row's layer split needs more host RAM than the caller has. */
function isOffloadInfeasible(option: QuantOption): boolean {
  return option.fit.offload !== null && !option.fit.offload.feasible;
}

/** Columns and cells shared by the text and Markdown renderings of `best`. */
export function bestTableShape(options: readonly QuantOption[], ctx: number): TableShape {
  // The quantization notes are a paragraph each and would make every column
  // unreadable, so the table stays numeric and the note is shown once, for the
  // row that actually matters.
  return {
    columns: [
      { header: "Quant" },
      { header: "bpw", align: "right" },
      { header: "Weights", align: "right" },
      { header: `Total at ${formatContext(ctx)}`, align: "right" },
      { header: "Fits", align: "right" },
      { header: "Max ctx", align: "right" },
      { header: "Decode", align: "right" },
    ],
    rows: options.map((option) => [
      option.quant.label,
      option.quant.bitsPerWeight.toFixed(2),
      formatBytes(option.fit.footprint.weights.totalBytes),
      formatBytes(option.fit.footprint.totalBytes),
      option.fit.fits ? "yes" : "no",
      option.maxContext > 0 ? formatContext(option.maxContext) : "-",
      // A decode figure for a split that needs more host RAM than the tool was
      // told about is not "what you would actually get" -- that configuration
      // does not load at all -- so it is marked rather than printed bare.
      `${formatRate(option.decodeTokensPerSecond)}${isOffloadInfeasible(option) ? " *" : ""}`,
    ]),
  };
}

/** The `vramfit best` table: every quantization, best quality first. */
export function renderBest(
  model: ModelSpec,
  device: DeviceSpec,
  options: readonly QuantOption[],
  ctx: number,
  gpus: number,
  report: ReportOptions = {},
): string[] {
  const heading = `${model.name}  |  ${gpus > 1 ? `${gpus} x ${device.name}` : device.name}`;
  const shape = bestTableShape(options, ctx);

  const starved = options.find(isOffloadInfeasible);
  // A file on disk states its format by being in it, which is a stronger
  // answer than the model's declared `nativeQuant` and the one the table was
  // ranked from.
  const native =
    report.source?.quant ??
    (model.nativeQuant === undefined ? undefined : findQuant(model.nativeQuant));
  const recommended = options.find((option) => option.fit.fits);

  return [
    heading,
    "=".repeat(heading.length),
    "",
    ...sourceLines(report),
    ...renderTable(shape.columns, shape.rows),
    "",
    ...(options.some((option) => !option.fit.fits)
      ? wrap(
          "Rows that do not fit show the decode speed with as many layers as possible offloaded to system RAM, which is what you would actually get.",
          WRAP_WIDTH,
        ).concat("")
      : []),
    ...(native
      ? wrap(
          `${model.name} ${report.source?.quant === undefined ? "ships in" : "is on disk as"} ${native.label}, so that is the best quality there is for it: a wider quantization would be a larger file holding the same ${native.bitsPerWeight.toFixed(2)} bits per weight, and is left out.`,
          WRAP_WIDTH,
        ).concat("")
      : []),
    ...(starved
      ? wrap(
          `* the offloaded remainder needs more system RAM than the ${formatBytes(starved.fit.offload?.systemRamAvailableBytes ?? 0)} assumed here, so that row would not load at all. Say what you have with --ram.`,
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
      // True when nothing has to be offloaded, or when the offloaded
      // remainder fits in the system RAM the caller declared.
      offloadFeasible: !isOffloadInfeasible(option),
      systemRamRequiredBytes: option.fit.offload?.systemRamRequiredBytes ?? 0,
    })),
  };
}

/** Columns and cells for `vramfit devices`. */
export function devicesTableShape(devices: readonly DeviceSpec[]): TableShape {
  return {
    columns: [
      { header: "ID" },
      { header: "Name" },
      { header: "Memory", align: "right" },
      { header: "Usable", align: "right" },
      { header: "Bandwidth", align: "right" },
      { header: "FP16", align: "right" },
      { header: "Family" },
    ],
    rows: devices.map((device) => [
      device.id,
      device.name,
      `${device.vramGiB} GiB`,
      `${(device.vramGiB * device.usableFraction).toFixed(0)} GiB`,
      `${device.bandwidthGBs} GB/s`,
      `${device.fp16Tflops} TF`,
      device.family,
    ]),
  };
}

/** `vramfit devices`. */
export function renderDevices(devices: readonly DeviceSpec[]): string[] {
  const shape = devicesTableShape(devices);
  return renderTable(shape.columns, shape.rows);
}

/** Columns and cells for `vramfit models`. */
export function modelsTableShape(models: readonly ModelSpec[]): TableShape {
  return {
    columns: [
      { header: "ID" },
      { header: "Name" },
      { header: "Params", align: "right" },
      { header: "Active", align: "right" },
      { header: "Layers", align: "right" },
      { header: "KV heads", align: "right" },
      { header: "Max ctx", align: "right" },
      { header: "Attention" },
    ],
    rows: models.map((model) => [
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
          ? `GQA + SWA${model.attentionWindow.fullAttentionEvery === null ? "" : `/${model.attentionWindow.fullAttentionEvery}`}`
          : "GQA",
    ]),
  };
}

/** `vramfit models`. */
export function renderModels(models: readonly ModelSpec[]): string[] {
  const shape = modelsTableShape(models);
  return renderTable(shape.columns, shape.rows);
}

/* -------------------------------------------------------------------------- */
/* compare                                                                     */
/* -------------------------------------------------------------------------- */

/** How a device is named in a table: "RTX 4090" or "2 x RTX 4090". */
function deviceLabel(comparison: DeviceComparison): string {
  return comparison.gpus > 1
    ? `${comparison.gpus} x ${comparison.device.name}`
    : comparison.device.name;
}

/** Columns and cells shared by the text and Markdown renderings of `compare`. */
export function compareTableShape(rows: readonly DeviceComparison[]): TableShape {
  return {
    columns: [
      { header: "" },
      { header: "Device" },
      { header: "Memory", align: "right" },
      { header: "Needed", align: "right" },
      { header: "Free", align: "right" },
      { header: "Fits", align: "right" },
      { header: "Max ctx", align: "right" },
      { header: "Decode", align: "right" },
    ],
    rows: rows.map((row) => [
      row.best ? "->" : "",
      deviceLabel(row),
      formatBytes(row.fit.capacity.totalBytes),
      formatBytes(row.fit.usedBytes),
      formatBytes(row.headroomBytes),
      row.fit.fits ? "yes" : "no",
      row.fit.maxContext > 0 ? formatContext(row.fit.maxContext) : "-",
      `${formatRate(row.decodeTokensPerSecond)}${
        row.fit.offload !== null && !row.fit.offload.feasible ? " *" : ""
      }`,
    ]),
  };
}

/** The `vramfit compare` table: one model, every device, best first. */
export function renderCompare(
  model: ModelSpec,
  quant: QuantSpec,
  rows: readonly DeviceComparison[],
  ctx: number,
  report: ReportOptions = {},
): string[] {
  const heading = `${model.name}  |  ${quant.label}  |  ${formatContext(ctx)} context`;
  const starved = rows.find((row) => row.fit.offload !== null && !row.fit.offload.feasible);

  const shape = compareTableShape(rows);
  const table = renderTable(shape.columns, shape.rows);

  const best = rows.find((row) => row.best);
  // Nothing fits: the ranking put the row that came closest first.
  const closest = rows[0];

  return [
    heading,
    "=".repeat(heading.length),
    "",
    ...sourceLines(report),
    ...table,
    "",
    ...(rows.some((row) => !row.fit.fits)
      ? wrap(
          "Rows that do not fit show the decode speed with as many layers as possible offloaded to system RAM, which is what you would actually get.",
          WRAP_WIDTH,
        ).concat("")
      : []),
    ...(starved
      ? wrap(
          `* the offloaded remainder needs more system RAM than the ${formatBytes(starved.fit.offload?.systemRamAvailableBytes ?? 0)} assumed here, so that row would not load at all. Say what you have with --ram.`,
          WRAP_WIDTH,
        ).concat("")
      : []),
    ...(best
      ? wrap(
          `Best: ${deviceLabel(best)} -- ${formatRate(best.decodeTokensPerSecond)} at ${formatContext(ctx)} with ${formatBytes(best.headroomBytes)} to spare, and room for ${formatContext(best.fit.maxContext)} of context.`,
          WRAP_WIDTH,
        )
      : wrap(
          `Nothing here fits ${model.name} at ${quant.label} and ${formatContext(ctx)} context.${
            closest === undefined
              ? ""
              : ` ${deviceLabel(closest)} came closest, ${formatBytes(-closest.headroomBytes)} short -- a narrower quantization or a shorter context is the cheaper fix than more hardware.`
          }`,
          WRAP_WIDTH,
        )),
  ];
}

/** The `vramfit compare --json` payload. */
export function compareJson(
  model: ModelSpec,
  quant: QuantSpec,
  rows: readonly DeviceComparison[],
  ctx: number,
  version: string,
): unknown {
  return {
    vramfit: version,
    model: { id: model.id, name: model.name, totalParams: model.totalParams },
    config: { quant: quant.id, ctx },
    best: rows.find((row) => row.best)?.device.id ?? null,
    devices: rows.map((row) => ({
      id: row.device.id,
      name: row.device.name,
      count: row.gpus,
      capacityBytes: row.fit.capacity.totalBytes,
      totalBytes: row.fit.usedBytes,
      headroomBytes: row.headroomBytes,
      utilization: row.fit.utilization,
      fits: row.fit.fits,
      maxContext: row.fit.maxContext,
      decodeTokensPerSecond: row.decodeTokensPerSecond,
      offloadFeasible: row.fit.offload === null || row.fit.offload.feasible,
      best: row.best,
    })),
  };
}

/* -------------------------------------------------------------------------- */
/* recommend                                                                   */
/* -------------------------------------------------------------------------- */

/** Columns and cells shared by the text and Markdown renderings of `recommend`. */
export function recommendTableShape(rows: readonly Recommendation[]): TableShape {
  return {
    columns: [
      { header: "#", align: "right" },
      { header: "Model" },
      { header: "Params", align: "right" },
      { header: "Quant" },
      { header: "Total", align: "right" },
      { header: "Max ctx", align: "right" },
      { header: "Decode", align: "right" },
    ],
    rows: rows.map((row, index) => [
      `${index + 1}.`,
      row.model.name,
      formatParams(row.model.totalParams),
      row.quant.label,
      formatBytes(row.fit.footprint.totalBytes),
      formatContext(row.maxContext),
      formatRate(row.decodeTokensPerSecond),
    ]),
  };
}

/**
 * The `vramfit recommend` list.
 *
 * An aligned table would fit more rows on a screen and answer less: the point
 * of the command is the trade-off, not the ranking, so each row carries its
 * own sentence underneath. The header line is still produced by `renderTable`,
 * so the columns line up across rows that have prose between them.
 */
export function renderRecommend(
  device: DeviceSpec,
  profile: UseCaseProfile,
  rows: readonly Recommendation[],
  ctx: number,
  gpus: number,
): string[] {
  const deviceName = gpus > 1 ? `${gpus} x ${device.name}` : device.name;
  const heading = `${deviceName}  |  ${profile.label}  |  ${formatContext(ctx)} context`;
  const lines = [heading, "=".repeat(heading.length), ""];

  if (rows.length === 0) {
    return [
      ...lines,
      ...wrap(
        `Nothing in the bundled database fits ${deviceName} at ${formatContext(ctx)} of context, even at Q2_K. Reduce the context with --ctx, quantize the cache with --kv-quant q8_0, or run "vramfit check <model> -d ${device.id}" to see what a partial offload would cost.`,
        WRAP_WIDTH,
      ),
    ];
  }

  const shape = recommendTableShape(rows);
  const table = renderTable(shape.columns, shape.rows);

  const [header, separator, ...body] = table;
  lines.push(header as string, separator as string);
  body.forEach((line, index) => {
    const row = rows[index];
    lines.push(line, ...(row === undefined ? [] : wrap(row.tradeoff, WRAP_WIDTH, "    ")), "");
  });

  lines.push(
    ...wrap(
      `Ranked by parameter count on a log scale, times a quantization-quality factor, times decode speed against ${profile.comfortableDecode} tok/s -- ${profile.why}. vramfit has no benchmark data and does not rank models by how good they are at anything.`,
      WRAP_WIDTH,
    ),
  );
  return lines;
}

/** The `vramfit recommend --json` payload. */
export function recommendJson(
  device: DeviceSpec,
  profile: UseCaseProfile,
  rows: readonly Recommendation[],
  ctx: number,
  gpus: number,
  version: string,
): unknown {
  return {
    vramfit: version,
    device: { id: device.id, name: device.name, vramGiB: device.vramGiB, count: gpus },
    useCase: {
      id: profile.id,
      comfortableDecodeTokensPerSecond: profile.comfortableDecode,
      ctx,
    },
    models: rows.map((row) => ({
      id: row.model.id,
      name: row.model.name,
      totalParams: row.model.totalParams,
      activeParams: row.model.activeParams,
      quant: row.quant.id,
      totalBytes: row.fit.footprint.totalBytes,
      headroomBytes: row.fit.headroomBytes,
      maxContext: row.maxContext,
      decodeTokensPerSecond: row.decodeTokensPerSecond,
      score: row.score,
      capability: row.capability,
      quality: row.quality,
      speed: row.speed,
      tradeoff: row.tradeoff,
    })),
  };
}

/* -------------------------------------------------------------------------- */
/* fleet                                                                       */
/* -------------------------------------------------------------------------- */

/** How a machine is described in the legend under the fleet table. */
function machineLegend(machine: FleetMachine): string[] {
  const device = machine.gpus > 1 ? `${machine.gpus} x ${machine.device.name}` : machine.device.name;
  const capacity =
    (machine.options.vramGiB ?? machine.device.vramGiB * machine.device.usableFraction) *
    machine.gpus;
  return [
    machine.name,
    device,
    `${capacity.toFixed(capacity < 10 ? 1 : 0)} GiB`,
    machine.options.systemRamGiB === undefined
      ? "-"
      : `${machine.options.systemRamGiB.toFixed(0)} GiB`,
  ];
}

/** Columns and cells shared by the text and Markdown renderings of `fleet`. */
export function fleetTableShape(report: FleetReport): TableShape {
  return {
    columns: [
      { header: "Model" },
      { header: "Quant" },
      { header: "Ctx", align: "right" },
      { header: "Weights+KV", align: "right" },
      ...report.machines.map((machine) => ({ header: machine.name, align: "right" as const })),
      { header: "Served", align: "right" },
    ],
    rows: report.rows.map((row) => [
      row.entry.label,
      row.entry.quant.label,
      formatContext(row.entry.ctx),
      // Weights and cache are the same on every machine; the runtime context
      // and compute buffer are charged per device and so differ between them.
      // Showing one machine's total in a shared column would misreport the rest.
      formatBytes(
        (row.cells[0]?.fit.footprint.weights.totalBytes ?? 0) +
          (row.cells[0]?.fit.footprint.kv.totalBytes ?? 0),
      ),
      ...row.cells.map((cell) =>
        cell.fit.fits ? formatRate(cell.fit.throughput.decode.tokensPerSecond) : "-",
      ),
      `${row.servedBy}/${report.machines.length}`,
    ]),
  };
}

/** The machine legend under a fleet table. */
export function fleetMachineShape(report: FleetReport): TableShape {
  return {
    columns: [
      { header: "Name" },
      { header: "Device" },
      { header: "Usable", align: "right" },
      { header: "System RAM", align: "right" },
    ],
    rows: report.machines.map(machineLegend),
  };
}

/**
 * The `vramfit fleet` matrix: one row per model, one column per machine.
 *
 * The cell is the decode speed rather than a tick, because "yes" and "yes at
 * 3 tok/s" are different answers and the second one is usually a no.
 */
export function renderFleet(report: FleetReport): string[] {
  const shape = fleetTableShape(report);

  const machines = fleetMachineShape(report);
  const heading = `Fleet  |  ${report.machines.length} machines  |  ${report.rows.length} models`;
  const lines = [heading, "=".repeat(heading.length), "", ...renderTable(shape.columns, shape.rows), ""];

  lines.push(
    "Machines",
    ...renderTable(machines.columns, machines.rows).map((line) => `  ${line}`.trimEnd()),
    "",
  );

  if (report.unserved.length > 0) {
    lines.push(
      ...wrap(
        `${report.unserved.length} of ${report.rows.length} models fit nowhere: ${report.unserved.map((row) => row.entry.label).join(", ")}. Run "vramfit check" against the largest machine to see what a partial offload or a narrower quantization would cost.`,
        WRAP_WIDTH,
      ),
      "",
    );
  }
  if (report.idle.length > 0) {
    lines.push(
      ...wrap(
        `${report.idle.map((machine) => machine.name).join(", ")} ${report.idle.length === 1 ? "serves" : "serve"} nothing on this list.`,
        WRAP_WIDTH,
      ),
      "",
    );
  }

  lines.push(
    ...wrap(
      "A dash means the model does not fit in that machine's device memory. Weights+KV is what every machine holds in common; each also pays its own runtime context and compute buffer, once per device. Decode figures are estimates, +/-25%.",
      WRAP_WIDTH,
    ),
  );
  return lines;
}

/** The `vramfit fleet --json` payload. */
export function fleetJson(report: FleetReport, version: string): unknown {
  return {
    vramfit: version,
    machines: report.machines.map((machine) => ({
      name: machine.name,
      device: machine.device.id,
      count: machine.gpus,
      capacityBytes:
        (machine.options.vramGiB ?? machine.device.vramGiB * machine.device.usableFraction) *
        machine.gpus *
        GIB,
    })),
    models: report.rows.map((row) => ({
      label: row.entry.label,
      id: row.entry.model.id,
      quant: row.entry.quant.id,
      ctx: row.entry.ctx,
      servedBy: row.servedBy,
      machines: row.cells.map((cell) => ({
        name: cell.machine.name,
        fits: cell.fit.fits,
        totalBytes: cell.fit.usedBytes,
        headroomBytes: cell.fit.headroomBytes,
        maxContext: cell.fit.maxContext,
        decodeTokensPerSecond: cell.fit.throughput.decode.tokensPerSecond,
      })),
    })),
  };
}

/* -------------------------------------------------------------------------- */
/* launcher flags                                                              */
/* -------------------------------------------------------------------------- */

function launcherBlock(plan: LauncherPlan, runtime: LauncherRuntime): string[] {
  const lines: string[] = [];
  const wants = (id: LauncherRuntime): boolean => runtime === "all" || runtime === id;

  if (wants("llama.cpp")) {
    lines.push("  llama.cpp", `    ${plan.llamaCpp.command}`);
  }
  if (wants("ollama")) {
    if (lines.length > 0) lines.push("");
    lines.push("  Ollama  (Modelfile)", ...plan.ollama.modelfile.map((line) => `    ${line}`));
    // The environment variables are not Modelfile commands: `ollama create`
    // rejects a file that carries them, and this block is meant to be pasted.
    if (plan.ollama.environment.length > 0) {
      lines.push(
        "",
        "  Ollama  (environment, not the Modelfile)",
        ...plan.ollama.environment.map((line) => `    ${line}`),
      );
    }
  }
  if (wants("vllm")) {
    if (lines.length > 0) lines.push("");
    lines.push("  vLLM", `    ${plan.vllm.command}`);
  }
  return lines;
}

/** The `Launch` section of `vramfit check --launcher`. */
export function renderLaunch(plan: LauncherPlan, runtime: LauncherRuntime): string[] {
  const lines = ["Launch", ...launcherBlock(plan, runtime)];
  const notes = notesFor(plan, runtime);
  if (notes.length > 0) {
    lines.push("");
    for (const note of notes) {
      const wrapped = wrap(note, WRAP_WIDTH, "    ");
      lines.push(`  - ${(wrapped[0] ?? "").trimStart()}`, ...wrapped.slice(1));
    }
  }
  return lines;
}

/** The `launcher` block of `vramfit check --json`. */
export function launcherJson(plan: LauncherPlan): unknown {
  return {
    llamaCpp: {
      nGpuLayers: plan.llamaCpp.nGpuLayers,
      ctx: plan.llamaCpp.ctx,
      ubatch: plan.llamaCpp.ubatch,
      parallel: plan.llamaCpp.parallel,
      flashAttention: plan.llamaCpp.flashAttention,
      cacheType: plan.llamaCpp.cacheType,
      args: plan.llamaCpp.args,
      command: plan.llamaCpp.command,
    },
    ollama: {
      numGpu: plan.ollama.numGpu,
      numCtx: plan.ollama.numCtx,
      numBatch: plan.ollama.numBatch,
      modelfile: plan.ollama.modelfile,
      options: plan.ollama.options,
      environment: plan.ollama.environment,
    },
    vllm: {
      gpuMemoryUtilization: plan.vllm.gpuMemoryUtilization,
      maxModelLen: plan.vllm.maxModelLen,
      tensorParallelSize: plan.vllm.tensorParallelSize,
      maxNumSeqs: plan.vllm.maxNumSeqs,
      kvCacheDtype: plan.vllm.kvCacheDtype,
      quantization: plan.vllm.quantization,
      args: plan.vllm.args,
      command: plan.vllm.command,
    },
    notes: plan.notes,
  };
}
