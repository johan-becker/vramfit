import type { DeviceComparison } from "../compare.js";
import type { FitResult, QuantOption } from "../fit.js";
import type { FleetReport } from "../fleet.js";
import { notesFor, type LauncherPlan, type LauncherRuntime } from "../launcher.js";
import type { Recommendation, UseCaseProfile } from "../recommend.js";
import type { DeviceSpec, ModelSpec, QuantSpec } from "../types.js";
import { formatBytes, formatContext, formatParams } from "../units.js";
import { renderMemoryBar } from "./bar.js";
import { PLAIN_PALETTE } from "./color.js";
import { renderExplain } from "./explain.js";
import { formatBandwidth, formatPercent, formatRate, formatSeconds } from "./format.js";
import {
  bestTableShape,
  compareTableShape,
  devicesTableShape,
  fleetMachineShape,
  fleetTableShape,
  modelsTableShape,
  recommendTableShape,
  type ReportOptions,
  type TableShape,
} from "./report.js";

/**
 * `--markdown`: the same reports, for pasting into an issue.
 *
 * The aligned text output is built for a terminal, where every column is
 * padded to its widest cell. Markdown does its own alignment, so pasting the
 * terminal form into an issue produces either a wall of monospace or, worse,
 * a proportional-font mess with the columns gone. This renders the same cells
 * through pipes instead of spaces.
 *
 * The cells themselves are shared with the text renderer rather than rebuilt,
 * so the two cannot drift: a column that gains a figure in one gains it in
 * both. What differs is only the separators, the emphasis, and the couple of
 * places where a fixed-width block -- the memory bar, the launch flags, the
 * arithmetic behind `--explain` -- has to go inside a fenced code block to
 * survive.
 */

/** A pipe inside a cell would end the cell; Markdown escapes it with a slash. */
function escapeCell(value: string): string {
  return value.replaceAll("|", "\\|").replaceAll("\n", " ");
}

/**
 * A GitHub-flavoured Markdown table.
 *
 * The alignment row carries the same left/right choices as the terminal table,
 * so a column of byte counts still reads down its decimal point. An empty
 * header stays empty: Markdown needs the cell, not a name for it.
 */
export function renderMarkdownTable(shape: TableShape): string[] {
  const header = shape.columns.map((column) => escapeCell(column.header));
  const alignment = shape.columns.map((column) => (column.align === "right" ? "---:" : "---"));
  const body = shape.rows.map((row) =>
    shape.columns.map((_, index) => escapeCell(row[index] ?? "")),
  );

  return [
    `| ${header.join(" | ")} |`,
    `| ${alignment.join(" | ")} |`,
    ...body.map((row) => `| ${row.join(" | ")} |`),
  ];
}

/** A fenced block, for output whose alignment is load-bearing. */
function fenced(lines: readonly string[], language = ""): string[] {
  return [`\`\`\`${language}`, ...lines, "```"];
}

function bulletList(items: readonly string[]): string[] {
  return items.map((item) => `- ${item}`);
}

/* -------------------------------------------------------------------------- */
/* check                                                                       */
/* -------------------------------------------------------------------------- */

function memoryShape(fit: FitResult): TableShape {
  const { footprint } = fit;
  return {
    columns: [{ header: "Component" }, { header: "Size", align: "right" }, { header: "Assumption" }],
    rows: [
      [
        "Weights",
        formatBytes(footprint.weights.totalBytes),
        `${formatParams(fit.model.totalParams)} params at ${footprint.weights.effectiveBitsPerWeight.toFixed(2)} effective bits/weight`,
      ],
      [
        "KV cache",
        formatBytes(footprint.kv.totalBytes),
        fit.model.attention === "mla" && fit.model.mla
          ? `${formatContext(fit.ctx)} tokens, ${fit.model.nLayers} layers x ${fit.model.mla.kvLoraRank + fit.model.mla.qkRopeHeadDim}-wide latent (MLA), ${fit.kvQuant.id}`
          : `${formatContext(fit.ctx)} tokens, ${fit.model.nLayers} layers x ${fit.model.nKvHeads} KV heads x ${fit.model.headDim}, ${fit.kvQuant.id}`,
      ],
      [
        "Runtime context",
        formatBytes(footprint.runtimeContextBytes),
        `${fit.device.family} driver and kernels${fit.gpus > 1 ? `, x${fit.gpus} devices` : ""}`,
      ],
      [
        "Compute buffer",
        formatBytes(footprint.activationBytes),
        "activations, logits and graph scratch",
      ],
      ["**Total**", `**${formatBytes(footprint.totalBytes)}**`, `at ${fit.quant.label}`],
      [
        "Available",
        formatBytes(fit.capacity.totalBytes),
        fit.gpus > 1 ? `${fit.gpus} x ${fit.device.name}` : fit.device.name,
      ],
    ],
  };
}

function speedShape(fit: FitResult): TableShape {
  const { throughput } = fit;
  const rows = [
    [
      "Decode",
      formatRate(throughput.decode.tokensPerSecond),
      fit.offload
        ? `blended across ${fit.offload.plan.gpuLayers} device and ${fit.offload.plan.cpuLayers} host layers`
        : `at ${formatContext(fit.ctx)} context, ${fit.batch} sequence${fit.batch > 1 ? "s" : ""}`,
    ],
    [
      "Prefill",
      formatRate(throughput.prefill.tokensPerSecond),
      `${(throughput.prefill.flopsPerToken / 1e9).toFixed(1)} GFLOP per prompt token`,
    ],
    [
      "Time to first token",
      formatSeconds(throughput.timeToFirstTokenSeconds),
      `for a prompt of ${formatContext(throughput.promptTokens)} tokens`,
    ],
  ];
  return {
    columns: [{ header: "Measure" }, { header: "Estimate", align: "right" }, { header: "At" }],
    rows,
  };
}

function offloadShape(fit: FitResult): TableShape | undefined {
  const offload = fit.offload;
  if (!offload) return undefined;
  return {
    columns: [{ header: "Partial offload" }, { header: "Value", align: "right" }, { header: "Note" }],
    rows: [
      [
        "Layers in device memory",
        `${offload.plan.gpuLayers} of ${offload.plan.totalLayers}`,
        offload.plan.vocabOnDevice
          ? "vocabulary tensors included"
          : "vocabulary tensors in system RAM",
      ],
      [
        "Left in system RAM",
        formatBytes(offload.systemRamRequiredBytes),
        `of ${formatBytes(offload.systemRamAvailableBytes)} assumed${offload.feasible ? "" : " -- NOT ENOUGH"}`,
      ],
      [
        "Blended read bandwidth",
        formatBandwidth(offload.bandwidth.peakBytesPerSecond),
        `harmonic mean; the device alone reads at ${formatBandwidth(fit.device.bandwidthGBs * 1e9)}`,
      ],
    ],
  };
}

/** The `vramfit check --markdown` report. */
export function renderCheckMarkdown(
  fit: FitResult,
  recommendation: QuantOption | undefined,
  options: ReportOptions = {},
): string[] {
  const deviceName = fit.gpus > 1 ? `${fit.gpus} x ${fit.device.name}` : fit.device.name;
  const verdict = fit.fits
    ? `**FITS** — ${formatBytes(fit.usedBytes)} of ${formatBytes(fit.capacity.totalBytes)} used, ${formatBytes(fit.headroomBytes)} free (${formatPercent(fit.utilization)} utilised)`
    : `**DOES NOT FIT** — ${formatBytes(fit.usedBytes)} needed, ${formatBytes(fit.capacity.totalBytes)} available, ${formatBytes(-fit.headroomBytes)} short`;

  const lines = [
    `### ${fit.model.name} — ${fit.quant.label} — ${deviceName}`,
    "",
    ...(options.source?.description === undefined ? [] : [`*${options.source.description}*`, ""]),
    verdict,
    "",
    // The bar is fixed-width art: outside a code block a proportional font
    // would leave the segments the wrong lengths, which is the one thing it
    // exists to get right.
    ...fenced(renderMemoryBar(fit, PLAIN_PALETTE).map((line) => line.trimStart())),
    "",
    ...renderMarkdownTable(memoryShape(fit)),
    "",
    "**Capacity**",
    "",
    ...renderMarkdownTable({
      columns: [{ header: "Question" }, { header: "Answer", align: "right" }, { header: "At" }],
      rows: [
        [
          "Largest context that fits",
          fit.maxContext > 0 ? formatContext(fit.maxContext) : "none",
          fit.maxContext >= fit.model.maxCtx
            ? "the architecture's own maximum"
            : `${fit.quant.label}, batch ${fit.batch}, ${fit.kvQuant.id} cache`,
        ],
        [
          `Best quant that fits at ${formatContext(fit.ctx)}`,
          recommendation ? recommendation.quant.label : "none",
          recommendation
            ? `${formatBytes(recommendation.fit.footprint.weights.totalBytes)} of weights, ${formatRate(recommendation.decodeTokensPerSecond)}`
            : "try a smaller model, a longer offload or more memory",
        ],
      ],
    }),
    "",
  ];

  const offload = offloadShape(fit);
  if (offload !== undefined) {
    lines.push(...renderMarkdownTable(offload), "");
  }

  lines.push(
    `**Speed** — estimates: decode ±${Math.round(fit.throughput.decodeErrorBand * 100)}%, prefill ±${Math.round(fit.throughput.prefillErrorBand * 100)}%`,
    "",
    ...renderMarkdownTable(speedShape(fit)),
    "",
  );

  if (options.explain === true) {
    lines.push("<details><summary>The arithmetic</summary>", "", ...fenced(renderExplain(fit)), "", "</details>", "");
  }

  const launcher = options.launcher;
  if (launcher !== undefined) {
    lines.push("**Launch**", "", ...launcherMarkdown(launcher.plan, launcher.runtime));
  }

  const notes = [...(options.source?.notes ?? []), ...fit.warnings];
  if (notes.length > 0) {
    lines.push("**Notes**", "", ...bulletList(notes), "");
  }

  return lines;
}

function launcherMarkdown(plan: LauncherPlan, runtime: LauncherRuntime): string[] {
  const wants = (id: LauncherRuntime): boolean => runtime === "all" || runtime === id;
  const lines: string[] = [];
  // The terminal renderer labels each block; three unlabelled fences leave the
  // reader to infer which runtime is which from the binary names, and that is
  // a content difference rather than the width one this format exists for.
  if (wants("llama.cpp")) {
    lines.push("*llama.cpp*", "", ...fenced([plan.llamaCpp.command], "sh"), "");
  }
  if (wants("ollama")) {
    lines.push("*Ollama — Modelfile*", "", ...fenced(plan.ollama.modelfile, "dockerfile"), "");
    if (plan.ollama.environment.length > 0) {
      lines.push(
        "*Ollama — environment, not the Modelfile*",
        "",
        ...fenced(plan.ollama.environment, "sh"),
        "",
      );
    }
  }
  if (wants("vllm")) {
    lines.push("*vLLM*", "", ...fenced([plan.vllm.command], "sh"), "");
  }
  const notes = notesFor(plan, runtime);
  if (notes.length > 0) lines.push(...bulletList(notes), "");
  return lines;
}

/* -------------------------------------------------------------------------- */
/* the other reports                                                           */
/* -------------------------------------------------------------------------- */

/** The `vramfit best --markdown` table. */
export function renderBestMarkdown(
  model: ModelSpec,
  device: DeviceSpec,
  options: readonly QuantOption[],
  ctx: number,
  gpus: number,
  report: ReportOptions = {},
): string[] {
  const deviceName = gpus > 1 ? `${gpus} x ${device.name}` : device.name;
  const recommended = options.find((option) => option.fit.fits);
  return [
    `### ${model.name} — ${deviceName} — every quantization at ${formatContext(ctx)}`,
    "",
    ...(report.source?.description === undefined ? [] : [`*${report.source.description}*`, ""]),
    ...renderMarkdownTable(bestTableShape(options, ctx)),
    "",
    ...(recommended
      ? [
          `**Recommended: ${recommended.quant.label}** — highest quality that fits at ${formatContext(ctx)}, ${formatRate(recommended.decodeTokensPerSecond)}. ${recommended.quant.notes}`,
        ]
      : [
          `Nothing in this range fits at ${formatContext(ctx)}. Reduce the context, offload layers, or pick a smaller model.`,
        ]),
  ];
}

/** The `vramfit compare --markdown` table. */
export function renderCompareMarkdown(
  model: ModelSpec,
  quant: QuantSpec,
  rows: readonly DeviceComparison[],
  ctx: number,
  report: ReportOptions = {},
): string[] {
  const best = rows.find((row) => row.best);
  return [
    `### ${model.name} — ${quant.label} — ${formatContext(ctx)} context`,
    "",
    ...(report.source?.description === undefined ? [] : [`*${report.source.description}*`, ""]),
    ...renderMarkdownTable(compareTableShape(rows)),
    "",
    ...(best
      ? [
          `**Best: ${best.gpus > 1 ? `${best.gpus} x ` : ""}${best.device.name}** — ${formatRate(best.decodeTokensPerSecond)} at ${formatContext(ctx)} with ${formatBytes(best.headroomBytes)} to spare, and room for ${formatContext(best.fit.maxContext)} of context.`,
        ]
      : [`Nothing here fits ${model.name} at ${quant.label} and ${formatContext(ctx)} context.`]),
  ];
}

/** The `vramfit recommend --markdown` table, with the trade-off as a column. */
export function renderRecommendMarkdown(
  device: DeviceSpec,
  profile: UseCaseProfile,
  rows: readonly Recommendation[],
  ctx: number,
  gpus: number,
  vramGiB?: number,
): string[] {
  const base = gpus > 1 ? `${gpus} x ${device.name}` : device.name;
  // As in the terminal renderer: a --vram override has to appear beside the
  // device, or the nothing-fits line describes a machine nobody asked about.
  const deviceName =
    vramGiB === undefined ? base : `${base} (${vramGiB.toFixed(2)} GiB, from --vram)`;
  const heading = `### ${deviceName} — ${profile.label} — ${formatContext(ctx)} context`;
  if (rows.length === 0) {
    return [
      heading,
      "",
      `Nothing in the bundled database fits ${deviceName} at ${formatContext(ctx)} of context, even at Q2_K.`,
    ];
  }

  // In a terminal the trade-off goes under its row, because a seventh column
  // of prose would wreck the alignment. Markdown wraps cells, so it fits.
  const shape = recommendTableShape(rows);
  for (const [index, row] of shape.rows.entries()) {
    row.push(rows[index]?.tradeoff ?? "");
  }
  const withTradeoff: TableShape = {
    columns: [...shape.columns, { header: "Trade-off" }],
    rows: shape.rows,
  };

  return [
    heading,
    "",
    ...renderMarkdownTable(withTradeoff),
    "",
    `Ranked by parameter count on a log scale, times a quantization-quality factor, times decode speed against ${profile.comfortableDecode} tok/s — ${profile.why}. vramfit has no benchmark data and does not rank models by how good they are at anything.`,
  ];
}

/** The `vramfit fleet --markdown` matrix. */
export function renderFleetMarkdown(report: FleetReport): string[] {
  const lines = [
    `### Fleet — ${report.machines.length} machines, ${report.rows.length} models`,
    "",
    ...renderMarkdownTable(fleetTableShape(report)),
    "",
    "**Machines**",
    "",
    ...renderMarkdownTable(fleetMachineShape(report)),
    "",
  ];

  if (report.unserved.length > 0) {
    lines.push(
      `${report.unserved.length} of ${report.rows.length} models fit nowhere: ${report.unserved.map((row) => row.entry.label).join(", ")}.`,
      "",
    );
  }
  if (report.idle.length > 0) {
    lines.push(
      `${report.idle.map((machine) => machine.name).join(", ")} ${report.idle.length === 1 ? "serves" : "serve"} nothing on this list.`,
      "",
    );
  }
  lines.push(
    "A dash means the model does not fit in that machine's device memory. Weights+KV is what every machine holds in common; each also pays its own runtime context and compute buffer, once per device.",
  );
  return lines;
}

/** `vramfit devices --markdown`. */
export function renderDevicesMarkdown(devices: readonly DeviceSpec[]): string[] {
  return renderMarkdownTable(devicesTableShape(devices));
}

/** `vramfit models --markdown`. */
export function renderModelsMarkdown(models: readonly ModelSpec[]): string[] {
  return renderMarkdownTable(modelsTableShape(models));
}
