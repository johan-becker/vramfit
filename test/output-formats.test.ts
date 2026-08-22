import { describe, expect, it } from "vitest";
import { EXIT_OK, run } from "../src/cli/index.js";
import { apportion, memorySegments, renderMemoryBar } from "../src/cli/bar.js";
import { PLAIN_PALETTE, createPalette, shouldUseColor } from "../src/cli/color.js";
import { getDevice } from "../src/db/index.js";
import { checkFit } from "../src/fit.js";
import { getQuant } from "../src/quant.js";
import { FakeIo } from "./fake-io.js";
import { LLAMA_3_1_8B, LLAMA_3_1_70B } from "./fixtures.js";

/**
 * Presentation: the memory bar, and the rules that decide whether anything is
 * coloured at all.
 *
 * The escape sequences matter as much as the glyphs here: `vramfit check >
 * report.txt` has to produce a file with none in it, and a bar pasted into an
 * issue has to still be readable with every colour stripped -- which is why
 * each segment has its own block character as well as its own hue.
 */

const ESC = "\u001b";

function fitOf(model = LLAMA_3_1_8B, deviceId = "4090", options = {}) {
  return checkFit(model, getQuant("q4_k_m"), getDevice(deviceId), { ctx: 8192, ...options });
}

describe("apportion", () => {
  it("splits a width so the parts add up to it exactly", () => {
    for (const width of [10, 24, 56, 80]) {
      const cells = apportion([100, 33, 7, 260], width);
      expect(cells.reduce((sum, value) => sum + value, 0)).toBe(width);
    }
  });

  it("never rounds a present segment away to nothing", () => {
    // A 0.15 GiB compute buffer is small; showing it as absent would be a
    // different claim from showing it as small.
    const cells = apportion([1000, 1, 1], 20);
    expect(cells[1]).toBeGreaterThanOrEqual(1);
    expect(cells[2]).toBeGreaterThanOrEqual(1);
    expect(cells.reduce((sum, value) => sum + value, 0)).toBe(20);
  });

  it("gives an absent segment no cells at all", () => {
    const cells = apportion([10, 0, 10], 10);
    expect(cells[1]).toBe(0);
    expect(cells.reduce((sum, value) => sum + value, 0)).toBe(10);
  });

  it("degrades quietly rather than dividing by zero", () => {
    expect(apportion([], 10)).toEqual([]);
    expect(apportion([0, 0], 10)).toEqual([0, 0]);
    expect(apportion([1, 1], 0)).toEqual([0, 0]);
  });
});

describe("memorySegments", () => {
  it("adds up to the capacity when the model fits", () => {
    const fit = fitOf();
    const segments = memorySegments(fit);
    expect(segments.map((segment) => segment.label)).toEqual([
      "weights",
      "KV",
      "overhead",
      "free",
    ]);
    const total = segments.reduce((sum, segment) => sum + segment.bytes, 0);
    expect(total).toBeCloseTo(fit.capacity.totalBytes, 6);
  });

  it("has no free slice when the model does not fit, and still sums to the footprint", () => {
    const fit = fitOf(LLAMA_3_1_70B, "4090");
    const segments = memorySegments(fit);
    expect(segments.map((segment) => segment.label)).toEqual(["weights", "KV", "overhead"]);
    // The overflow is the tail of the footprint that has nowhere to go, not a
    // fourth part of it: counting it as a segment would count those bytes
    // twice and make the bar 80% longer than the model.
    const total = segments.reduce((sum, segment) => sum + segment.bytes, 0);
    expect(total).toBeCloseTo(fit.usedBytes, 6);
  });
});

describe("renderMemoryBar", () => {
  it("is exactly the width it was asked for, and carries no escapes when plain", () => {
    const lines = renderMemoryBar(fitOf(), PLAIN_PALETTE, 40);
    const bar = (lines[0] ?? "").trim().split("  ")[0] ?? "";
    expect([...bar]).toHaveLength(40);
    expect(lines.join("\n")).not.toContain(ESC);
    for (const line of lines) expect(line).toBe(line.trimEnd());
  });

  it("gives every segment its own glyph, so colour is never the only channel", () => {
    const [bar, legend] = renderMemoryBar(fitOf(), PLAIN_PALETTE, 56);
    expect(bar).toMatch(/█+▓+▒+░+/);
    expect(legend).toMatch(/█ weights \d+\.\d\d {3}▓ KV \d+\.\d\d {3}▒ overhead \d+\.\d\d/);
  });

  it("hatches the part that does not fit", () => {
    const [bar, legend] = renderMemoryBar(fitOf(LLAMA_3_1_70B, "4090"), PLAIN_PALETTE, 56);
    expect(bar).toContain("▚");
    expect(bar).toMatch(/needed, 24\.00 GiB available$/);
    expect(legend).toMatch(/▚ over \d+\.\d\d/);
  });

  it("paints each segment its own colour when colour is on", () => {
    const lines = renderMemoryBar(fitOf(), createPalette(true), 24);
    // cyan weights, magenta cache, yellow overhead, grey free.
    expect(lines[0]).toContain(`${ESC}[36m`);
    expect(lines[0]).toContain(`${ESC}[35m`);
    expect(lines[0]).toContain(`${ESC}[33m`);
    expect(lines[0]).toContain(`${ESC}[90m`);
    expect(lines[0]).toContain(`${ESC}[0m`);
  });
});

/** An environment lookup over a plain object, for the colour rules. */
function env(values: Record<string, string>): (name: string) => string | undefined {
  return (name) => values[name];
}

describe("shouldUseColor", () => {
  it("follows the terminal when nothing says otherwise", () => {
    expect(shouldUseColor({ isTty: true })).toBe(true);
    expect(shouldUseColor({ isTty: false })).toBe(false);
    expect(shouldUseColor({})).toBe(false);
  });

  it("honours NO_COLOR", () => {
    expect(shouldUseColor({ isTty: true, env: env({ NO_COLOR: "1" }) })).toBe(false);
    // An empty value is not a request, per no-color.org.
    expect(shouldUseColor({ isTty: true, env: env({ NO_COLOR: "" }) })).toBe(true);
  });

  it("honours FORCE_COLOR, which is how CI asks for it", () => {
    expect(shouldUseColor({ isTty: false, env: env({ FORCE_COLOR: "1" }) })).toBe(true);
    expect(shouldUseColor({ isTty: false, env: env({ FORCE_COLOR: "0" }) })).toBe(false);
    // NO_COLOR is checked first and wins.
    expect(shouldUseColor({ isTty: true, env: env({ NO_COLOR: "1", FORCE_COLOR: "1" }) })).toBe(
      false,
    );
  });

  it("says no to a dumb terminal", () => {
    expect(shouldUseColor({ isTty: true, env: env({ TERM: "dumb" }) })).toBe(false);
  });

  it("lets an explicit flag override everything", () => {
    expect(shouldUseColor({ isTty: false, env: env({ NO_COLOR: "1" }), requested: true })).toBe(
      true,
    );
    expect(shouldUseColor({ isTty: true, requested: false })).toBe(false);
  });
});

describe("vramfit check, coloured or not", () => {
  it("writes no escape sequences to a pipe", () => {
    const io = new FakeIo();
    expect(run(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k"], io)).toBe(EXIT_OK);
    expect(io.output).not.toContain(ESC);
    expect(io.output).toMatch(/█ weights 4\.62/);
  });

  it("colours the verdict and the bar on a terminal", () => {
    const io = new FakeIo({ isTty: true });
    run(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k"], io);
    expect(io.output).toContain(`${ESC}[32mFITS${ESC}[0m`);
    expect(io.output).toContain(`${ESC}[36m`);
  });

  it("stops at NO_COLOR even on a terminal", () => {
    const io = new FakeIo({ isTty: true, env: { NO_COLOR: "1" } });
    run(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k"], io);
    expect(io.output).not.toContain(ESC);
  });

  it("takes --color and --no-color over both", () => {
    const forced = new FakeIo({ isTty: false, env: { NO_COLOR: "1" } });
    run(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k", "--color"], forced);
    expect(forced.output).toContain(ESC);

    const suppressed = new FakeIo({ isTty: true });
    run(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k", "--no-color"], suppressed);
    expect(suppressed.output).not.toContain(ESC);
  });

  it("keeps JSON free of escapes whatever the terminal says", () => {
    const io = new FakeIo({ isTty: true });
    run(["check", "llama-3.1-8b", "-d", "4090", "--json"], io);
    expect(io.output).not.toContain(ESC);
    expect(() => JSON.parse(io.output)).not.toThrow();
  });
});
