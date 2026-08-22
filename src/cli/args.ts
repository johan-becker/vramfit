/**
 * A hand-rolled argument parser, because a VRAM calculator with a dependency
 * tree would be missing the point.
 *
 * It accepts the forms people actually type: `--device 4090`, `--device=4090`,
 * `-d 4090`, `--json`, `--no-flash-attn`, and `--` to stop flag parsing. It
 * refuses everything else loudly, because a silently-ignored `--ctx 32768` is
 * a wrong answer delivered confidently.
 */

/** A problem with what the user typed, as opposed to a bug in the program. */
export class UsageError extends Error {
  override readonly name = "UsageError";
}

/** Single-letter aliases, expanded to their long form before anything else. */
const SHORT_FLAGS: Readonly<Record<string, string>> = {
  b: "batch",
  c: "ctx",
  d: "device",
  g: "gpus",
  h: "help",
  j: "json",
  q: "quant",
  v: "version",
};

/**
 * Flags that never take a separate value, checked after short flags have been
 * expanded to their long form.
 *
 * Without this list every flag reads the token after it as its value, so
 * `vramfit check --json llama-3.1-8b -d 4090` sets `json` to the model name
 * and then reports that no model was given. `--flag=value` and `--no-flag`
 * still carry a value for all of them.
 */
const VALUELESS_FLAGS: ReadonlySet<string> = new Set([
  "json",
  "help",
  "version",
  "flash-attn",
  "markdown",
  "explain",
  "ngl",
]);

type FlagValue = string | boolean;

function looksLikeValue(token: string | undefined): boolean {
  if (token === undefined) return false;
  if (!token.startsWith("-")) return true;
  // Negative numbers are values, not flags: `--efficiency -1` is a bad value
  // rather than an unknown flag, and should be reported as such.
  return /^-\d/.test(token);
}

export interface ParsedArgv {
  positionals: string[];
  flags: Map<string, FlagValue>;
}

export function parseArgv(argv: readonly string[]): ParsedArgv {
  const positionals: string[] = [];
  const flags = new Map<string, FlagValue>();

  for (let index = 0; index < argv.length; index++) {
    const token = argv[index] as string;

    if (token === "--") {
      positionals.push(...argv.slice(index + 1));
      break;
    }

    let name: string | undefined;
    let inlineValue: string | undefined;

    if (token.startsWith("--")) {
      const body = token.slice(2);
      if (body === "") throw new UsageError('"--" on its own ends flag parsing; "---" is not a flag');
      const equals = body.indexOf("=");
      name = (equals >= 0 ? body.slice(0, equals) : body).toLowerCase();
      if (equals >= 0) inlineValue = body.slice(equals + 1);
    } else if (token.startsWith("-") && token.length > 1) {
      const body = token.slice(1);
      const equals = body.indexOf("=");
      const letters = equals >= 0 ? body.slice(0, equals) : body;
      if (letters.length !== 1) {
        throw new UsageError(`Unknown option "${token}". Short options take one letter at a time.`);
      }
      const expanded = SHORT_FLAGS[letters];
      if (expanded === undefined) throw new UsageError(`Unknown option "${token}"`);
      name = expanded;
      if (equals >= 0) inlineValue = body.slice(equals + 1);
    } else {
      positionals.push(token);
      continue;
    }

    if (inlineValue !== undefined) {
      flags.set(name, inlineValue);
      continue;
    }
    if (name.startsWith("no-") && name.length > 3) {
      flags.set(name.slice(3), false);
      continue;
    }
    if (VALUELESS_FLAGS.has(name)) {
      flags.set(name, true);
      continue;
    }
    const next = argv[index + 1];
    if (looksLikeValue(next)) {
      flags.set(name, next as string);
      index++;
    } else {
      flags.set(name, true);
    }
  }

  return { positionals, flags };
}

export interface NumberRules {
  min?: number;
  max?: number;
  integer?: boolean;
}

/**
 * Context lengths read and type better as `32k` than as `32768`, and in this
 * domain "k" has always meant 1024 -- llama.cpp's `-c 32768` is 32k, not
 * 32000. `m` follows for the few places it makes sense.
 */
export function parseTokenCount(raw: string, flag: string): number {
  const match = /^(\d+(?:\.\d+)?)\s*([km]?)$/i.exec(raw.trim());
  if (!match) {
    throw new UsageError(`--${flag} expects a token count such as 8192 or 32k, got "${raw}"`);
  }
  const magnitude = Number.parseFloat(match[1] as string);
  const suffix = (match[2] ?? "").toLowerCase();
  const scale = suffix === "k" ? 1024 : suffix === "m" ? 1024 * 1024 : 1;
  const value = Math.round(magnitude * scale);
  if (value < 1) throw new UsageError(`--${flag} must be at least 1 token, got "${raw}"`);
  return value;
}

/** Typed, validated access to one parsed command line. */
export class Args {
  readonly positionals: readonly string[];
  private readonly flags: ReadonlyMap<string, FlagValue>;

  constructor(parsed: ParsedArgv) {
    this.positionals = parsed.positionals;
    this.flags = parsed.flags;
  }

  static parse(argv: readonly string[]): Args {
    return new Args(parseArgv(argv));
  }

  has(name: string): boolean {
    return this.flags.has(name);
  }

  /**
   * Reject anything not in the command's own vocabulary. Called by every
   * command, so `vramfit check llama-3.1-8b --ctxx 32k` fails instead of
   * quietly answering the question for the default context.
   */
  assertKnown(known: readonly string[]): void {
    const allowed = new Set(known);
    for (const name of this.flags.keys()) {
      if (!allowed.has(name)) {
        const near = known.filter((candidate) => candidate.startsWith(name.slice(0, 3)));
        const hint = near.length > 0 ? ` Did you mean --${near.join(", --")}?` : "";
        throw new UsageError(`Unknown option "--${name}".${hint}`);
      }
    }
  }

  /**
   * The raw value: a string when one was given, `true` when the flag was
   * present without one. For a flag whose value is optional -- `--launcher`
   * meaning "all three" and `--launcher vllm` meaning one of them -- where
   * `string()` would refuse the bare form as a missing value.
   */
  flag(name: string): string | boolean | undefined {
    return this.flags.get(name);
  }

  string(name: string): string | undefined {
    const value = this.flags.get(name);
    if (value === undefined) return undefined;
    if (typeof value === "boolean") {
      throw new UsageError(`--${name} needs a value`);
    }
    return value;
  }

  requiredString(name: string): string {
    const value = this.string(name);
    if (value === undefined) throw new UsageError(`--${name} is required`);
    return value;
  }

  number(name: string, rules: NumberRules = {}): number | undefined {
    const raw = this.string(name);
    if (raw === undefined) return undefined;
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      throw new UsageError(`--${name} expects a number, got "${raw}"`);
    }
    if (rules.integer === true && !Number.isInteger(value)) {
      throw new UsageError(`--${name} expects a whole number, got "${raw}"`);
    }
    if (rules.min !== undefined && value < rules.min) {
      throw new UsageError(`--${name} must be at least ${rules.min}, got ${value}`);
    }
    if (rules.max !== undefined && value > rules.max) {
      throw new UsageError(`--${name} must be at most ${rules.max}, got ${value}`);
    }
    return value;
  }

  tokens(name: string): number | undefined {
    const raw = this.string(name);
    return raw === undefined ? undefined : parseTokenCount(raw, name);
  }

  boolean(name: string): boolean | undefined {
    const value = this.flags.get(name);
    if (value === undefined) return undefined;
    if (typeof value === "boolean") return value;
    const normalized = value.toLowerCase();
    if (normalized === "true" || normalized === "yes" || normalized === "on") return true;
    if (normalized === "false" || normalized === "no" || normalized === "off") return false;
    throw new UsageError(`--${name} expects true or false, got "${value}"`);
  }
}
