/**
 * Terminal formatting: aligned tables and label/value blocks.
 *
 * No colour and no box drawing. The output of this tool ends up pasted into
 * issues, piped into `grep` and captured in CI logs, all of which are places
 * where ANSI escapes and U+2500 are a liability rather than a flourish.
 */

export type Align = "left" | "right";

export interface Column {
  header: string;
  align?: Align;
}

function pad(value: string, width: number, align: Align): string {
  const padding = " ".repeat(Math.max(0, width - value.length));
  return align === "right" ? padding + value : value + padding;
}

/**
 * Render rows under headers, every column as wide as its widest cell.
 * Trailing whitespace is trimmed so the output diffs cleanly.
 */
export function renderTable(columns: readonly Column[], rows: readonly (readonly string[])[]): string[] {
  const widths = columns.map((column, index) =>
    rows.reduce((widest, row) => Math.max(widest, (row[index] ?? "").length), column.header.length),
  );

  const line = (cells: readonly string[]): string =>
    columns
      .map((column, index) =>
        pad(cells[index] ?? "", widths[index] as number, column.align ?? "left"),
      )
      .join("  ")
      .trimEnd();

  return [
    line(columns.map((column) => column.header)),
    line(widths.map((width) => "-".repeat(width))),
    ...rows.map(line),
  ];
}

export interface Pair {
  label: string;
  value: string;
  /** Parenthetical shown after the value, for the "why" behind a number. */
  note?: string;
}

/** A label/value block with the values aligned into one column. */
export function renderPairs(pairs: readonly Pair[], indent = "  "): string[] {
  const labelWidth = pairs.reduce((widest, pair) => Math.max(widest, pair.label.length), 0);
  const valueWidth = pairs.reduce((widest, pair) => Math.max(widest, pair.value.length), 0);

  return pairs.map((pair) => {
    const head = `${indent}${pad(pair.label, labelWidth, "left")}  ${pad(pair.value, valueWidth, "right")}`;
    return (pair.note === undefined ? head : `${head}  ${pair.note}`).trimEnd();
  });
}

/** Wrap prose to a width, for warnings that would otherwise run off the edge. */
export function wrap(text: string, width: number, indent = ""): string[] {
  const words = text.split(/\s+/).filter((word) => word.length > 0);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current === "" ? word : `${current} ${word}`;
    if (candidate.length + indent.length > width && current !== "") {
      lines.push(indent + current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current !== "") lines.push(indent + current);
  return lines;
}

/** Tokens per second, at a precision that does not imply false accuracy. */
export function formatRate(tokensPerSecond: number): string {
  if (!Number.isFinite(tokensPerSecond)) return "n/a";
  if (tokensPerSecond >= 1000) return `${Math.round(tokensPerSecond)} tok/s`;
  if (tokensPerSecond >= 100) return `${tokensPerSecond.toFixed(0)} tok/s`;
  if (tokensPerSecond >= 10) return `${tokensPerSecond.toFixed(1)} tok/s`;
  return `${tokensPerSecond.toFixed(2)} tok/s`;
}

export function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds)) return "n/a";
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds - minutes * 60)} s`;
}

export function formatPercent(fraction: number, decimals = 0): string {
  if (!Number.isFinite(fraction)) return "n/a";
  return `${(fraction * 100).toFixed(decimals)}%`;
}

/** Decimal GB/s, as vendors quote bandwidth. */
export function formatBandwidth(bytesPerSecond: number): string {
  return `${Math.round(bytesPerSecond / 1e9)} GB/s`;
}
