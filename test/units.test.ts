import { describe, expect, it } from "vitest";
import { parseTokenCount } from "../src/cli/args.js";
import { formatBytes, formatContext, formatParams } from "../src/units.js";

/**
 * Formatting is where a correct number becomes a wrong answer. These pin the
 * two properties that matter: memory reads in the binary units the rest of the
 * package computes in, and a context length that is printed can be typed back
 * in without changing what it means.
 */

describe("formatBytes", () => {
  it("reports GiB, and falls back to MiB below a tenth of one", () => {
    expect(formatBytes(4 * 1024 ** 3)).toBe("4.00 GiB");
    expect(formatBytes(1024 ** 3 / 2)).toBe("0.50 GiB");
    expect(formatBytes(64 * 1024 ** 2)).toBe("64.0 MiB");
    expect(formatBytes(0)).toBe("0 MiB");
  });
});

describe("formatParams", () => {
  it("switches unit at a billion and keeps three significant figures", () => {
    expect(formatParams(8_030_000_000)).toBe("8.03B");
    expect(formatParams(116_829_156_672)).toBe("117B");
    expect(formatParams(494_000_000)).toBe("494M");
  });
});

describe("formatContext", () => {
  it('uses one meaning of "K", the same 1024 the parser uses', () => {
    expect(formatContext(131_072)).toBe("128K");
    expect(formatContext(8192)).toBe("8K");
    // 108579 tokens is 106.0 KiB, not the 108.6 a decimal K would print.
    expect(formatContext(108_579)).toBe("106.0K");
    expect(formatContext(5632)).toBe("5.5K");
  });

  it("prints anything under 1K as a plain token count", () => {
    expect(formatContext(1023)).toBe("1023");
    expect(formatContext(128)).toBe("128");
    expect(formatContext(1024)).toBe("1K");
  });

  it("round-trips through the flag parser without ever growing", () => {
    // The headline "largest context that fits" is printed by this function and
    // typed back in as --ctx, so the printed value has to parse to something
    // that still fits: truncated, never rounded up.
    for (const tokens of [512, 1023, 1024, 4096, 5632, 32_768, 108_579, 131_072, 1_000_000]) {
      const printed = formatContext(tokens);
      const reparsed = parseTokenCount(printed.replace(/K$/, "k"), "ctx");
      expect(reparsed, `${tokens} printed as ${printed}`).toBeLessThanOrEqual(tokens);
      expect(tokens - reparsed, `${tokens} printed as ${printed}`).toBeLessThan(1024 / 10);
    }
  });
});
