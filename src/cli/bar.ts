import type { FitResult } from "../fit.js";
import { bytesToGiB, formatBytes } from "../units.js";
import type { Palette, StyleName } from "./color.js";
import { formatPercent } from "./format.js";

/**
 * The memory breakdown as one line.
 *
 * The table above it has the numbers; this has the proportions, which are the
 * part people actually reason with -- "the cache is half of it" is a decision
 * and "4.00 GiB" is a figure you then have to divide.
 *
 * Every segment has its own block character as well as its own colour, so the
 * bar is readable pasted into an issue, piped into a file, or on a terminal
 * with colour turned off. That is the same reason the rest of this package's
 * output is plain: colour here is a second channel, never the only one.
 */

/** Glyph per segment, distinct enough to tell apart without colour. */
const GLYPHS = {
  weights: "█", // full block
  kv: "▓", // dark shade
  overhead: "▒", // medium shade
  free: "░", // light shade
  // Hatched rather than solid: the part that does not fit has to be
  // distinguishable from the weights without relying on colour.
  overflow: "▚",
} as const;

/** Wide enough to resolve a 2% segment, narrow enough for an 80-column terminal. */
export const DEFAULT_BAR_WIDTH = 56;

export interface BarSegment {
  label: string;
  bytes: number;
  style: StyleName;
  glyph: string;
}

/**
 * Split `width` cells between the segments in proportion to their sizes.
 *
 * Largest-remainder apportionment, so the parts add up to exactly `width` --
 * rounding each independently would leave the bar a cell short or a cell long
 * depending on the numbers, and a bar whose length moves is worse than one
 * whose segments are a cell out.
 *
 * A segment with a non-zero size never rounds away to nothing: a 0.15 GiB
 * compute buffer is small, and showing it as absent would be a different
 * claim from showing it as small.
 */
export function apportion(values: readonly number[], width: number): number[] {
  const total = values.reduce((sum, value) => sum + value, 0);
  if (total <= 0 || width <= 0) return values.map(() => 0);

  const exact = values.map((value) => (value / total) * width);
  const floors = exact.map((value) => Math.floor(value));
  // Anything present gets at least one cell, taken from the rounding budget.
  const cells = floors.map((value, index) => (values[index] === 0 ? 0 : Math.max(1, value)));

  let remaining = width - cells.reduce((sum, value) => sum + value, 0);
  const order = exact
    .map((value, index) => ({ index, remainder: value - Math.floor(value) }))
    .toSorted((a, b) => b.remainder - a.remainder);

  // Hand out what is left largest-remainder first; take back from the largest
  // segments when the minimum-one-cell rule has overspent.
  for (let step = 0; remaining > 0 && order.length > 0; step++) {
    const entry = order[step % order.length];
    if (entry === undefined) break;
    if ((values[entry.index] ?? 0) === 0) continue;
    cells[entry.index] = (cells[entry.index] as number) + 1;
    remaining--;
  }
  while (remaining < 0) {
    const largest = cells.reduce(
      (best, value, index) => (value > (cells[best] as number) ? index : best),
      0,
    );
    if ((cells[largest] as number) <= 1) break;
    cells[largest] = (cells[largest] as number) - 1;
    remaining++;
  }
  return cells;
}

/**
 * The parts of the footprint, plus what is left of the card when there is any.
 *
 * The three components always sum to the footprint. Free memory is a fourth
 * segment when the model fits; when it does not there is no fourth segment,
 * because the overflow is not a *part* of the footprint -- it is the tail of
 * it that has nowhere to go, and the bar marks it positionally rather than
 * counting the same bytes twice.
 */
export function memorySegments(fit: FitResult): BarSegment[] {
  const overhead = fit.footprint.runtimeContextBytes + fit.footprint.activationBytes;
  const segments: BarSegment[] = [
    {
      label: "weights",
      bytes: fit.footprint.weights.totalBytes,
      style: "weights",
      glyph: GLYPHS.weights,
    },
    { label: "KV", bytes: fit.footprint.kv.totalBytes, style: "kv", glyph: GLYPHS.kv },
    { label: "overhead", bytes: overhead, style: "overhead", glyph: GLYPHS.overhead },
  ];
  if (fit.headroomBytes > 0) {
    segments.push({ label: "free", bytes: fit.headroomBytes, style: "free", glyph: GLYPHS.free });
  }
  return segments;
}

function legendFor(segments: readonly BarSegment[], palette: Palette): string {
  return segments
    .map(
      (segment) =>
        `${palette.paint(segment.style, segment.glyph)} ${segment.label} ${bytesToGiB(segment.bytes).toFixed(2)}`,
    )
    .join("   ");
}

/** One cell of the bar: what to draw, and in what role. */
interface Cell {
  glyph: string;
  style: StyleName;
}

/** Expand the segments into `width` cells, in order. */
function cellsFor(segments: readonly BarSegment[], width: number): Cell[] {
  const counts = apportion(
    segments.map((segment) => segment.bytes),
    width,
  );
  const cells: Cell[] = [];
  segments.forEach((segment, index) => {
    for (let cell = 0; cell < (counts[index] ?? 0); cell++) {
      cells.push({ glyph: segment.glyph, style: segment.style });
    }
  });
  return cells;
}

/** Group adjacent cells of the same style, so each run is painted once. */
function paintCells(cells: readonly Cell[], palette: Palette): string {
  let out = "";
  let run = "";
  let style: StyleName | undefined;
  for (const cell of cells) {
    if (cell.style !== style) {
      if (style !== undefined) out += palette.paint(style, run);
      style = cell.style;
      run = "";
    }
    run += cell.glyph;
  }
  if (style !== undefined) out += palette.paint(style, run);
  return out;
}

/**
 * Two lines: the bar, and the key to it.
 *
 * When the model does not fit, the scale is the footprint rather than the
 * card, and the part past the card's capacity is drawn in the overflow style
 * -- so the bar shows how far over it went instead of clamping at full and
 * reporting the same picture for 1 GiB short and 40 GiB short.
 */
export function renderMemoryBar(
  fit: FitResult,
  palette: Palette,
  width = DEFAULT_BAR_WIDTH,
): string[] {
  const segments = memorySegments(fit);
  const total = segments.reduce((sum, segment) => sum + segment.bytes, 0);
  const cells = cellsFor(segments, width);

  const legend = [...segments];
  if (!fit.fits && total > 0) {
    // Everything past where the card runs out is redrawn as overflow,
    // whichever component it happened to land in. The cut is a position in
    // the bar, not another segment: those bytes are already counted once.
    const boundary = Math.min(width, Math.max(0, Math.round((fit.capacity.totalBytes / total) * width)));
    for (let index = boundary; index < cells.length; index++) {
      cells[index] = { glyph: GLYPHS.overflow, style: "bad" };
    }
    legend.push({
      label: "over",
      bytes: -fit.headroomBytes,
      style: "bad",
      glyph: GLYPHS.overflow,
    });
  }

  const summary = fit.fits
    ? `${formatPercent(fit.utilization)} of ${formatBytes(fit.capacity.totalBytes)}`
    : `${formatBytes(fit.usedBytes)} needed, ${formatBytes(fit.capacity.totalBytes)} available`;

  return [
    `  ${paintCells(cells, palette)}  ${summary}`.trimEnd(),
    `  ${legendFor(legend, palette)}   (GiB)`.trimEnd(),
  ];
}
