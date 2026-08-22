import { describe, expect, it } from "vitest";
import { EXIT_DOES_NOT_FIT, EXIT_OK, EXIT_USAGE, run } from "../src/cli/index.js";
import { compareDevices, parseDeviceList } from "../src/compare.js";
import { getDevice } from "../src/db/index.js";
import { getQuant } from "../src/quant.js";
import { FakeIo } from "./fake-io.js";
import { LLAMA_3_1_8B, LLAMA_3_1_70B } from "./fixtures.js";

/**
 * `compare`: one model, several devices, one ordering.
 *
 * The arithmetic is `checkFit`, which has its own tests. What is tested here
 * is the part that is new -- the ordering, the tie-breaking and the marking of
 * one row as the answer -- because an unranked list of eight devices is as
 * unhelpful as no list at all.
 */

function candidates(...entries: [string, number][]) {
  return entries.map(([id, gpus]) => ({ device: getDevice(id), gpus }));
}

describe("parseDeviceList", () => {
  it("reads a comma-separated list with optional counts", () => {
    expect(parseDeviceList("4090,3090x2, m4-max ")).toEqual([
      { query: "4090", gpus: 1 },
      { query: "3090", gpus: 2 },
      { query: "m4-max", gpus: 1 },
    ]);
    expect(parseDeviceList("RTX 4090 x 4")).toEqual([{ query: "RTX 4090", gpus: 4 }]);
    expect(parseDeviceList("a100-80*2")).toEqual([{ query: "a100-80", gpus: 2 }]);
  });

  it("refuses an empty list", () => {
    expect(() => parseDeviceList("")).toThrow(/no devices given/);
    expect(() => parseDeviceList(" , , ")).toThrow(/no devices given/);
  });

  it("gives an entry with no count of its own the default", () => {
    // Which is how --gpus reaches a list that does not spell the count out.
    expect(parseDeviceList("4090,3090x4", 2)).toEqual([
      { query: "4090", gpus: 2 },
      { query: "3090", gpus: 4 },
    ]);
  });
});

describe("compareDevices", () => {
  it("puts what fits first, fastest first within that", () => {
    const rows = compareDevices(
      LLAMA_3_1_8B,
      getQuant("q4_k_m"),
      candidates(["rtx-3060-12gb", 1], ["4090", 1], ["h100", 1], ["m1", 1]),
      { ctx: 8192 },
    );

    expect(rows.map((row) => row.device.id)).toEqual([
      // 3.35 TB/s of HBM3 outruns everything here.
      "h100-80gb",
      "rtx-4090",
      "rtx-3060-12gb",
      // 16 GiB of unified memory, 75% wirable: 12 GiB for a 6.5 GiB model
      // plus a 1 GiB cache and the overheads.
      "m1",
    ]);
    expect(rows[0]?.best).toBe(true);
    expect(rows.filter((row) => row.best)).toHaveLength(1);
  });

  it("counts multiple devices as one bigger one, minus the per-device overheads", () => {
    const rows = compareDevices(
      LLAMA_3_1_70B,
      getQuant("q4_k_m"),
      candidates(["3090", 1], ["3090", 2]),
      { ctx: 8192 },
    );
    const pair = rows.find((row) => row.gpus === 2);
    const single = rows.find((row) => row.gpus === 1);
    if (single === undefined || pair === undefined) throw new Error("expected two rows");

    // The pair is the one that fits, so it sorts first.
    expect(rows[0]).toBe(pair);
    expect(pair.fit.fits).toBe(true);
    expect(single.fit.fits).toBe(false);
    // Two cards, but each pays the 0.7 GiB runtime context in full.
    expect(pair.fit.capacity.totalBytes).toBe(2 * single.fit.capacity.totalBytes);
    expect(pair.fit.footprint.runtimeContextBytes).toBe(
      2 * single.fit.footprint.runtimeContextBytes,
    );
  });

  it("orders the rows that do not fit by how close they came", () => {
    const rows = compareDevices(
      LLAMA_3_1_70B,
      getQuant("q8_0"),
      candidates(["rtx-3060-12gb", 1], ["4090", 1], ["a100-40gb", 1]),
      { ctx: 8192 },
    );
    expect(rows.every((row) => !row.fit.fits)).toBe(true);
    expect(rows.map((row) => row.device.id)).toEqual([
      "a100-40gb",
      "rtx-4090",
      "rtx-3060-12gb",
    ]);
    // Nothing fits, so nothing is marked.
    expect(rows.some((row) => row.best)).toBe(false);
  });

  it("orders identical devices the same way every time", () => {
    // Two entries that tie on every numeric column must not shuffle between
    // runs, or the table is not a table.
    const rows = compareDevices(
      LLAMA_3_1_8B,
      getQuant("q4_k_m"),
      candidates(["4090", 1], ["4090", 1]),
      { ctx: 8192 },
    );
    const again = compareDevices(
      LLAMA_3_1_8B,
      getQuant("q4_k_m"),
      candidates(["4090", 1], ["4090", 1]),
      { ctx: 8192 },
    );
    expect(rows.map((row) => row.decodeTokensPerSecond)).toEqual(
      again.map((row) => row.decodeTokensPerSecond),
    );
  });
});

function invoke(argv: string[]) {
  const io = new FakeIo();
  return { code: run(argv, io), io };
}

describe("vramfit compare", () => {
  it("prints one aligned table, best marked", () => {
    const { code, io } = invoke([
      "compare",
      "llama-3.3-70b",
      "--devices",
      "4090,4090x2,a100-80,m3-ultra",
      "--ctx",
      "8k",
    ]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/^Llama 3\.3 70B {2}\| {2}Q4_K_M {2}\| {2}8K context$/m);
    expect(io.output).toMatch(/^->\s+NVIDIA A100 80GB\s+80\.00 GiB/m);
    expect(io.output).toMatch(/^ {4}2 x NVIDIA RTX 4090\s+48\.00 GiB/m);
    expect(io.output).toMatch(/-19\.39 GiB\s+no/);
    expect(io.output).toMatch(/Best: NVIDIA A100 80GB -- 33\.2 tok\/s at 8K/);
    for (const line of io.output.split("\n")) expect(line).toBe(line.trimEnd());
  });

  it("exits 1 when nothing in the list fits, and says what came closest", () => {
    const { code, io } = invoke([
      "compare",
      "qwen3-235b-a22b",
      "--devices",
      "4090,3090",
      "--ctx",
      "8k",
    ]);
    expect(code).toBe(EXIT_DOES_NOT_FIT);
    expect(io.output).toMatch(/Nothing here fits Qwen3 235B-A22B at Q4_K_M and 8K context/);
    expect(io.output).toMatch(/came closest, \d+\.\d\d GiB short/);
  });

  it("emits the same ordering as JSON", () => {
    const { io } = invoke([
      "compare",
      "llama-3.1-8b",
      "--devices",
      "4090,m1,h100",
      "--ctx",
      "8k",
      "--json",
    ]);
    const payload = JSON.parse(io.output) as {
      best: string;
      devices: { id: string; fits: boolean; best: boolean; decodeTokensPerSecond: number }[];
    };
    expect(payload.best).toBe("h100-80gb");
    expect(payload.devices.map((entry) => entry.id)).toEqual(["h100-80gb", "rtx-4090", "m1"]);
    expect(payload.devices[0]?.best).toBe(true);
  });

  it("compares a GGUF file across devices too", () => {
    const io = new FakeIo({ binary: {} });
    expect(run(["compare", "llama-3.1-8b", "--devices", "4090"], io)).toBe(EXIT_OK);
    expect(io.output).toMatch(/NVIDIA RTX 4090/);
  });

  it("asks for the list rather than guessing one", () => {
    const { code, io } = invoke(["compare", "llama-3.1-8b"]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/--devices is required/);
  });

  it("names an unknown device in the list", () => {
    const { code, io } = invoke(["compare", "llama-3.1-8b", "--devices", "4090,rtx-4091"]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/Unknown device "rtx-4091"/);
  });

  it("refuses a device count that is not one", () => {
    const { code, io } = invoke(["compare", "llama-3.1-8b", "--devices", "4090x0"]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/asks for 0 devices/);
  });

  it("accepts a name that itself ends in x and a number", () => {
    // rtx4090 and 4090x2 are the same shape. Splitting the count off first
    // amputated the alias and then reported an unknown device "rt" -- a name
    // the user never typed, for a spelling "check -d rtx4090" accepts.
    for (const spelling of ["rtx4090", "rtx3090", "NVIDIA RTX 4090"]) {
      const { code, io } = invoke(["compare", "llama-3.1-8b", "--devices", spelling, "--ctx", "8k"]);
      expect(code, spelling).toBe(EXIT_OK);
      expect(io.errors, spelling).toBe("");
    }
    // The count still parses when the whole token is not a device.
    const pair = invoke(["compare", "llama-3.1-8b", "--devices", "rtx4090x2", "--ctx", "8k"]);
    expect(pair.code).toBe(EXIT_OK);
    expect(pair.io.output).toMatch(/2 x NVIDIA RTX 4090/);
  });

  it("applies --gpus to every entry that does not carry its own count", () => {
    // compare accepts --gpus, so it has to mean the same thing here as it
    // does in check: the two commands cannot answer the same question with
    // opposite exit codes.
    const compared = invoke([
      "compare",
      "llama-3.3-70b",
      "--devices",
      "rtx-4090",
      "--gpus",
      "2",
      "--ctx",
      "8k",
    ]);
    expect(compared.code).toBe(EXIT_OK);
    expect(compared.io.output).toMatch(/2 x NVIDIA RTX 4090\s+48\.00 GiB\s+44\.39 GiB/);

    const checked = invoke(["check", "llama-3.3-70b", "-d", "rtx-4090", "--gpus", "2", "--ctx", "8k"]);
    expect(checked.code).toBe(EXIT_OK);
    expect(checked.io.output).toMatch(/FITS {2}- {2}44\.39 GiB of 48\.00 GiB used/);

    // An entry with its own count keeps it.
    const explicit = invoke([
      "compare",
      "llama-3.3-70b",
      "--devices",
      "rtx-4090x1",
      "--gpus",
      "2",
      "--ctx",
      "8k",
    ]);
    expect(explicit.code).toBe(EXIT_DOES_NOT_FIT);
  });
});
