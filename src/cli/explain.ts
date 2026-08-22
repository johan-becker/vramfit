import { deriveArchitecture } from "../architecture.js";
import type { FitResult } from "../fit.js";
import { RUNTIME_CONTEXT_BYTES } from "../memory.js";
import { PREFILL_MFU, bandwidthEfficiency } from "../throughput.js";
import { GB_DECIMAL, GIB, bytesToGiB, formatContext } from "../units.js";
import { wrap } from "./format.js";

/**
 * `--explain`: the same numbers, with the arithmetic left in.
 *
 * This is the feature that makes the rest of the tool checkable rather than
 * authoritative. Every figure in the report is a formula from the README
 * applied to this model and this device, and a reader who disagrees with one
 * of the inputs -- the effective bits per weight, the memory-bandwidth
 * efficiency, the 0.7 GiB runtime context -- can see exactly which line it
 * enters on and what it was multiplied by.
 *
 * Nothing here recomputes anything. The values printed are the ones the report
 * printed; the expressions beside them are how those values arose.
 */

const WRAP_WIDTH = 78;

/** Thousands separators without depending on the host's locale data. */
function groupDigits(value: number): string {
  const rounded = Math.round(value);
  return String(rounded).replaceAll(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function gib(bytes: number, decimals = 3): string {
  return `${bytesToGiB(bytes).toFixed(decimals)} GiB`;
}

/** Decimal GB, which is the unit bandwidth and FLOPs are quoted in. */
function gb(bytes: number, decimals = 2): string {
  return `${(bytes / GB_DECIMAL).toFixed(decimals)} GB`;
}

interface Line {
  label: string;
  expression: string;
  value: string;
}

/**
 * Three columns: what it is, how it was computed, what came out. Aligned as a
 * block so the `=` signs line up and the expressions can be read down.
 */
function alignedLines(lines: readonly Line[], indent = "    "): string[] {
  const labelWidth = lines.reduce((widest, line) => Math.max(widest, line.label.length), 0);
  const expressionWidth = lines.reduce(
    (widest, line) => Math.max(widest, line.expression.length),
    0,
  );
  return lines.map((line) =>
    `${indent}${line.label.padEnd(labelWidth)}  ${line.expression.padEnd(expressionWidth)}  = ${line.value}`.trimEnd(),
  );
}

/* -------------------------------------------------------------------------- */
/* Sections                                                                    */
/* -------------------------------------------------------------------------- */

function weightLines(fit: FitResult): string[] {
  const arch = deriveArchitecture(fit.model);
  const { weights } = fit.footprint;
  const vocabParams = arch.inputEmbedParams + arch.outputHeadParams;
  const blockParams = fit.model.totalParams - vocabParams;

  const lines: Line[] = [
    {
      label: "blocks",
      expression: `${groupDigits(blockParams)} x ${fit.quant.bitsPerWeight} / 8`,
      value: gib(weights.blockBytes),
    },
  ];
  if (weights.tokenEmbedBytes > 0) {
    lines.push({
      label: "token_embd",
      expression: `${groupDigits(arch.inputEmbedParams)} x ${fit.quant.tokenEmbedBits} / 8`,
      value: gib(weights.tokenEmbedBytes),
    });
  }
  lines.push({
    label: fit.model.tiedEmbeddings ? "output (tied)" : "output",
    expression: `${groupDigits(arch.logitMatrixParams)} x ${fit.quant.outputHeadBits} / 8`,
    value: gib(weights.logitMatrixBytes),
  });
  lines.push({
    label: "total",
    expression: `${weights.effectiveBitsPerWeight.toFixed(2)} effective bits per weight`,
    value: gib(weights.totalBytes),
  });

  return [
    "  Weights = parameters x bits per weight / 8",
    ...alignedLines(lines),
    ...(fit.model.tiedEmbeddings
      ? wrap(
          "The embedding table is stored once and doubles as the output projection, so it is charged at the promoted rate and not twice.",
          WRAP_WIDTH,
          "    ",
        )
      : []),
  ];
}

function kvLines(fit: FitResult): string[] {
  const { model, footprint } = fit;
  const perElement = footprint.kv.bytesPerElement;
  const windowed = footprint.kv.windowedLayers;
  const fullLayers = model.nLayers - windowed;
  const window = model.attentionWindow;

  if (model.attention === "mla" && model.mla) {
    return [
      "  KV cache = layers x (latent + rope key) x ctx x sequences x bytes/element",
      ...alignedLines([
        {
          label: "latent",
          expression: `${model.nLayers} x (${model.mla.kvLoraRank} + ${model.mla.qkRopeHeadDim}) x ${fit.ctx} x ${fit.batch} x ${perElement}`,
          value: gib(footprint.kv.totalBytes),
        },
      ]),
      ...wrap(
        "Multi-head latent attention caches one compressed vector per token per layer instead of per-head keys and values: no factor of two, and no head count.",
        WRAP_WIDTH,
        "    ",
      ),
    ];
  }

  const lines: Line[] = [];
  if (windowed === 0) {
    lines.push({
      label: "all layers",
      expression: `2 x ${model.nLayers} x ${model.nKvHeads} x ${model.headDim} x ${fit.ctx} x ${fit.batch} x ${perElement}`,
      value: gib(footprint.kv.totalBytes),
    });
  } else {
    const windowCtx = Math.min(fit.ctx, window?.windowSize ?? fit.ctx);
    const perLayer = 2 * model.nKvHeads * model.headDim * perElement * fit.batch;
    // A model that states a window and no period has no full-attention layer
    // at all, and a "0 global = 0.000 GiB" row explains nothing.
    if (fullLayers > 0) {
      lines.push({
        label: `${fullLayers} global`,
        expression: `2 x ${fullLayers} x ${model.nKvHeads} x ${model.headDim} x ${fit.ctx} x ${fit.batch} x ${perElement}`,
        value: gib(perLayer * fit.ctx * fullLayers),
      });
    }
    lines.push({
      label: `${windowed} windowed`,
      expression: `2 x ${windowed} x ${model.nKvHeads} x ${model.headDim} x ${windowCtx} x ${fit.batch} x ${perElement}`,
      value: gib(perLayer * windowCtx * windowed),
    });
    if (fullLayers > 0) {
      lines.push({ label: "total", expression: "", value: gib(footprint.kv.totalBytes) });
    }
  }

  return [
    "  KV cache = 2 x layers x KV heads x head dim x ctx x sequences x bytes/element",
    ...alignedLines(lines),
    ...(model.nKvHeads < model.nHeads
      ? wrap(
          `Grouped-query attention: ${model.nKvHeads} KV heads, not the ${model.nHeads} query heads. Using the query count here is the ${model.nHeads / model.nKvHeads}x mistake this tool exists to avoid.`,
          WRAP_WIDTH,
          "    ",
        )
      : []),
  ];
}

function overheadLines(fit: FitResult): string[] {
  const perDevice = RUNTIME_CONTEXT_BYTES[fit.device.family];
  const ffnWidth = fit.model.moe
    ? fit.model.moe.expertFfnHidden *
      (fit.model.moe.expertsPerToken + fit.model.moe.nSharedExperts)
    : fit.model.ffnHidden;
  const tokensInFlight = Math.min(fit.ctx * fit.batch, fit.physicalBatch);

  return [
    "  Overheads (empirical: +/-0.3 GiB on the first, +/-50% on the second)",
    ...alignedLines([
      {
        label: "runtime context",
        expression: `${(perDevice / GIB).toFixed(2)} GiB x ${fit.gpus} device${fit.gpus > 1 ? "s" : ""}`,
        value: gib(fit.footprint.runtimeContextBytes),
      },
      {
        label: "compute buffer",
        expression: `max(64 MiB, ${tokensInFlight} x (${fit.model.hiddenSize} x 18 + ${ffnWidth} x 6) x 2 + ${fit.batch} x ${fit.model.vocabSize} x 4${fit.flashAttention ? "" : ` + ${tokensInFlight} x ${fit.ctx} x ${fit.model.nHeads} x 4`})`,
        value: gib(fit.footprint.activationBytes),
      },
    ]),
  ];
}

function totalLines(fit: FitResult): string[] {
  const { footprint, capacity } = fit;
  const usable =
    capacity.installedBytes > 0 ? capacity.totalBytes / capacity.installedBytes : 1;

  return [
    "  Total",
    ...alignedLines([
      {
        label: "footprint",
        expression: `${gib(footprint.weights.totalBytes)} + ${gib(footprint.kv.totalBytes)} + ${gib(footprint.runtimeContextBytes)} + ${gib(footprint.activationBytes)}`,
        value: gib(footprint.totalBytes),
      },
      {
        label: "capacity",
        expression: `${(capacity.installedBytes / GIB / capacity.gpus).toFixed(2)} GiB x ${usable.toFixed(2)} usable x ${capacity.gpus} device${capacity.gpus > 1 ? "s" : ""}`,
        value: gib(capacity.totalBytes),
      },
      {
        label: "verdict",
        expression: `${gib(footprint.totalBytes)} ${fit.fits ? "<=" : ">"} ${gib(capacity.totalBytes)}`,
        value: fit.fits ? "FITS" : "DOES NOT FIT",
      },
    ]),
  ];
}

function decodeLines(fit: FitResult): string[] {
  const decode = fit.throughput.decode;
  const peak = decode.efficiency > 0 ? decode.effectiveBandwidthBytesPerSecond / decode.efficiency : 0;
  const derived = bandwidthEfficiency(
    fit.device.family,
    fit.footprint.weights.effectiveBitsPerWeight,
  );

  const lines: Line[] = [];
  if (Math.abs(derived - decode.efficiency) < 1e-9) {
    const narrowness = Math.min(
      Math.max((16 - fit.footprint.weights.effectiveBitsPerWeight) / 12, 0),
      1,
    );
    lines.push({
      label: "efficiency",
      expression: `${bandwidthEfficiency(fit.device.family, 16).toFixed(2)} at 16 bits, less the dequantization penalty at ${narrowness.toFixed(2)} of full`,
      value: decode.efficiency.toFixed(3),
    });
  } else {
    lines.push({
      label: "efficiency",
      expression: fit.offload === null ? "given with --efficiency" : "blended across the offload split",
      value: decode.efficiency.toFixed(3),
    });
  }

  lines.push(
    {
      label: "bandwidth",
      expression: `${gb(peak, 0)}/s x ${decode.efficiency.toFixed(3)}`,
      value: `${gb(decode.effectiveBandwidthBytesPerSecond, 0)}/s`,
    },
    {
      label: "read per token",
      expression: `${gb(decode.weightBytesPerStep)} active weights + ${gb(decode.kvBytesPerStep)} cache`,
      value: gb(decode.bytesPerStep),
    },
    {
      label: "decode",
      expression: `${gb(decode.effectiveBandwidthBytesPerSecond, 0)}/s / ${gb(decode.bytesPerStep)}`,
      value: `${decode.tokensPerSecond.toFixed(decode.tokensPerSecond >= 10 ? 1 : 2)} tok/s`,
    },
  );

  return [
    "  Decode = bandwidth x efficiency / bytes read per token  (estimate, +/-25%)",
    ...alignedLines(lines),
    ...(fit.offload === null
      ? []
      : wrap(
          `The bandwidth is the harmonic mean of ${gb(fit.device.bandwidthGBs * GB_DECIMAL, 0)}/s of device memory and the host's, weighted by the bytes read from each side -- which is why offloading a tenth of a model costs far more than a tenth of the speed.`,
          WRAP_WIDTH,
          "    ",
        )),
  ];
}

function prefillLines(fit: FitResult): string[] {
  const arch = deriveArchitecture(fit.model);
  const prefill = fit.throughput.prefill;
  const peak = prefill.efficiency > 0 ? prefill.effectiveFlopsPerSecond / prefill.efficiency : 0;

  return [
    "  Prefill = device FLOP/s x MFU / FLOPs per prompt token  (estimate, +/-40%)",
    ...alignedLines([
      {
        label: "FLOPs/token",
        expression: `2 x ${groupDigits(arch.activeMatmulParams)} active + ${(prefill.attentionFlopsPerToken / 1e9).toFixed(2)} GFLOP of attention`,
        value: `${(prefill.flopsPerToken / 1e9).toFixed(2)} GFLOP`,
      },
      {
        label: "compute",
        expression: `${(peak / 1e12).toFixed(1)} TFLOP/s x ${prefill.efficiency.toFixed(2)} MFU${prefill.efficiency === PREFILL_MFU[fit.device.family] ? ` (${fit.device.family})` : ""}`,
        value: `${(prefill.effectiveFlopsPerSecond / 1e12).toFixed(1)} TFLOP/s`,
      },
      {
        label: "prefill",
        expression: `${(prefill.effectiveFlopsPerSecond / 1e12).toFixed(1)}e12 / ${(prefill.flopsPerToken / 1e9).toFixed(2)}e9`,
        value: `${Math.round(prefill.tokensPerSecond)} tok/s`,
      },
      {
        label: "first token",
        expression: `${groupDigits(fit.throughput.promptTokens)} prompt tokens / ${Math.round(prefill.tokensPerSecond)} tok/s`,
        value: `${fit.throughput.timeToFirstTokenSeconds.toFixed(1)} s`,
      },
    ]),
  ];
}

/** The `--explain` appendix: every headline number, with its own arithmetic. */
export function renderExplain(fit: FitResult): string[] {
  return [
    `Explain  (${fit.model.name} at ${fit.quant.label}, ${formatContext(fit.ctx)} context, ${fit.kvQuant.id} cache)`,
    ...weightLines(fit),
    "",
    ...kvLines(fit),
    "",
    ...overheadLines(fit),
    "",
    ...totalLines(fit),
    "",
    ...decodeLines(fit),
    "",
    ...prefillLines(fit),
  ];
}
