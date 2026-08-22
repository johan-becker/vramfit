/**
 * Colour, and when not to use it.
 *
 * The rest of this package's output is deliberately plain -- no box drawing,
 * no escapes -- because it ends up in issues, in `grep` and in CI logs. Colour
 * is the one exception worth making, and only for the memory bar, where it
 * carries information rather than decoration. So it is off by default and
 * turned on only when the output is going to a terminal that asked for it.
 *
 * The rules, in order, are the conventions rather than inventions of this
 * package:
 *
 *   1. `--color` / `--no-color` on the command line wins outright.
 *   2. `NO_COLOR` set to anything non-empty disables it (no-color.org).
 *   3. `FORCE_COLOR` set to anything but "0" enables it, which is how CI jobs
 *      that do want colour ask for it.
 *   4. `TERM=dumb` disables it.
 *   5. Otherwise: on when stdout is a terminal, off when it is a pipe.
 *
 * Rule 5 is the one that matters: `vramfit check ... > report.txt` has to
 * produce a file with no escape sequences in it.
 */

/** The roles the report paints, rather than the colours they happen to be. */
export type StyleName =
  | "weights"
  | "kv"
  | "overhead"
  | "free"
  | "good"
  | "bad"
  | "dim"
  | "bold";

/**
 * Basic ANSI SGR codes only. The 8-colour set is the one every terminal that
 * has ever existed agrees on, and nothing here needs more than four hues plus
 * a dim.
 */
const CODES: Readonly<Record<StyleName, string>> = {
  weights: "36", // cyan
  kv: "35", // magenta
  overhead: "33", // yellow
  free: "90", // bright black, i.e. grey
  good: "32", // green
  bad: "31", // red
  dim: "2",
  bold: "1",
};

const RESET = "\u001b[0m";

export interface Palette {
  readonly enabled: boolean;
  /** Wrap `text` in the style, or return it unchanged when colour is off. */
  paint(style: StyleName, text: string): string;
}

export function createPalette(enabled: boolean): Palette {
  return {
    enabled,
    paint(style, text) {
      if (!enabled || text === "") return text;
      return `\u001b[${CODES[style]}m${text}${RESET}`;
    },
  };
}

/** A palette that paints nothing, for tests and for piped output. */
export const PLAIN_PALETTE: Palette = createPalette(false);

export interface ColorEnvironment {
  /** True when stdout is a terminal. */
  isTty?: boolean;
  /** Environment lookup. Absent means an empty environment. */
  env?: (name: string) => string | undefined;
  /** An explicit `--color` / `--no-color`, which overrides everything. */
  requested?: boolean;
}

/** Decide whether to emit escape sequences at all. */
export function shouldUseColor(environment: ColorEnvironment): boolean {
  if (environment.requested !== undefined) return environment.requested;

  const read = environment.env ?? ((): undefined => undefined);
  const noColor = read("NO_COLOR");
  if (noColor !== undefined && noColor !== "") return false;

  const forceColor = read("FORCE_COLOR");
  if (forceColor !== undefined && forceColor !== "" && forceColor !== "0") return true;

  if (read("TERM") === "dumb") return false;
  return environment.isTty === true;
}
