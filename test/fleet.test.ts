import { describe, expect, it } from "vitest";
import { EXIT_DOES_NOT_FIT, EXIT_OK, EXIT_USAGE, run } from "../src/cli/index.js";
import { SpecValidationError } from "../src/db/validate.js";
import { getDevice } from "../src/db/index.js";
import { parseFleetConfig, planFleet, type FleetMachine } from "../src/fleet.js";
import { getQuant } from "../src/quant.js";
import { FakeIo } from "./fake-io.js";
import { llamaGgufBytes } from "./gguf-fixtures.js";
import { LLAMA_3_1_8B, LLAMA_3_1_70B } from "./fixtures.js";

/**
 * `fleet`: which of several machines can serve which of several models.
 *
 * Every cell is a `checkFit`, which is tested elsewhere. What is tested here
 * is the file format, the placement summary and the exit code -- a fleet
 * report is the kind of thing a CI job runs, so "one of these models fits
 * nowhere" has to be visible to a shell.
 */

const FLEET = JSON.stringify({
  ctx: 8192,
  quant: "q4_k_m",
  machines: [
    { name: "workstation", device: "4090", gpus: 2, ram: 128 },
    { name: "laptop", device: "m3-max", ram: 128 },
    { name: "old-box", device: "rtx-3060-12gb", ram: 32 },
  ],
  models: ["llama-3.1-8b", "llama-3.3-70b", { model: "gemma-3-27b", ctx: 4096 }],
});

function machine(name: string, deviceId: string, gpus = 1): FleetMachine {
  return { name, device: getDevice(deviceId), gpus, options: { gpus } };
}

function invoke(argv: string[], files: Record<string, string> = {}) {
  const io = new FakeIo({ text: files });
  return { code: run(argv, io), io };
}

describe("parseFleetConfig", () => {
  it("reads machines, models and the defaults they share", () => {
    const config = parseFleetConfig(JSON.parse(FLEET));
    expect(config.ctx).toBe(8192);
    expect(config.quant).toBe("q4_k_m");
    expect(config.machines[0]).toEqual({
      name: "workstation",
      device: "4090",
      gpus: 2,
      vramGiB: undefined,
      systemRamGiB: 128,
    });
    expect(config.models[0]).toEqual({ model: "llama-3.1-8b", quant: undefined, ctx: undefined });
    expect(config.models[2]).toEqual({ model: "gemma-3-27b", quant: undefined, ctx: 4096 });
  });

  it("names a machine after its device when it does not name itself", () => {
    const config = parseFleetConfig({ machines: [{ device: "4090" }], models: ["llama-3.1-8b"] });
    expect(config.machines[0]?.name).toBe("4090");
    expect(config.machines[0]?.gpus).toBe(1);
  });

  it("refuses two machines with the same name", () => {
    // The table has one column per machine; two columns with one heading is a
    // report nobody can act on.
    expect(() =>
      parseFleetConfig({
        machines: [
          { name: "box", device: "4090" },
          { name: "box", device: "3090" },
        ],
        models: ["llama-3.1-8b"],
      }),
    ).toThrow(/fleet\.machines\[1\]\.name is "box", which is already taken/);
  });

  it("names the field and the index when the file is wrong", () => {
    expect(() => parseFleetConfig({ models: ["llama-3.1-8b"] })).toThrow(
      /fleet\.machines must be an array, got nothing/,
    );
    expect(() => parseFleetConfig({ machines: [], models: ["x"] })).toThrow(
      /fleet\.machines must not be empty/,
    );
    expect(() =>
      parseFleetConfig({ machines: [{ device: "4090" }], models: [{ quant: "q4_k_m" }] }),
    ).toThrow(/fleet\.models\[0\]\.model must be a non-empty string/);
    expect(() =>
      parseFleetConfig({ machines: [{ device: "4090", gpus: 1.5 }], models: ["x"] }),
    ).toThrow(/fleet\.machines\[0\]\.gpus must be a whole number/);
    expect(() => parseFleetConfig({ machines: [{ device: "4090" }], models: ["x"] })).not.toThrow();
    expect(() => parseFleetConfig("nope")).toThrow(SpecValidationError);
  });

  it("refuses a machine name that could forge a line of the report", () => {
    // An ANSI erase-display sequence in a name would let a fleet file clear
    // the screen and print a line of its own above the real table.
    expect(() =>
      parseFleetConfig({
        machines: [{ name: "box\u001b[2J", device: "4090" }],
        models: ["llama-3.1-8b"],
      }),
    ).toThrow(/must not contain control characters/);
  });
});

describe("planFleet", () => {
  it("evaluates every model on every machine", () => {
    const machines = [machine("small", "rtx-3060-12gb"), machine("big", "a100-80gb")];
    const report = planFleet(machines, [
      { label: "8B", model: LLAMA_3_1_8B, quant: getQuant("q4_k_m"), ctx: 8192 },
      { label: "70B", model: LLAMA_3_1_70B, quant: getQuant("q4_k_m"), ctx: 8192 },
    ]);

    expect(report.rows).toHaveLength(2);
    expect(report.rows[0]?.servedBy).toBe(2);
    expect(report.rows[1]?.servedBy).toBe(1);
    expect(report.rows[1]?.cells[0]?.fit.fits).toBe(false);
    expect(report.unserved).toHaveLength(0);
    expect(report.idle).toHaveLength(0);
  });

  it("reports models nowhere can hold and machines nothing needs", () => {
    const machines = [machine("tiny", "rtx-3060-12gb"), machine("also-tiny", "m1")];
    const report = planFleet(machines, [
      { label: "70B", model: LLAMA_3_1_70B, quant: getQuant("f16"), ctx: 8192 },
    ]);
    expect(report.unserved.map((row) => row.entry.label)).toEqual(["70B"]);
    expect(report.idle.map((entry) => entry.name)).toEqual(["tiny", "also-tiny"]);
  });

  it("honours each row's own context", () => {
    const machines = [machine("box", "4090")];
    const report = planFleet(machines, [
      { label: "short", model: LLAMA_3_1_8B, quant: getQuant("q4_k_m"), ctx: 1024 },
      { label: "long", model: LLAMA_3_1_8B, quant: getQuant("q4_k_m"), ctx: 65_536 },
    ]);
    const short = report.rows[0]?.cells[0]?.fit;
    const long = report.rows[1]?.cells[0]?.fit;
    expect(long?.footprint.kv.totalBytes).toBe((short?.footprint.kv.totalBytes ?? 0) * 64);
  });

  it("charges each device of a multi-GPU machine its own overhead", () => {
    const report = planFleet(
      [machine("one", "3090"), machine("two", "3090", 2)],
      [{ label: "70B", model: LLAMA_3_1_70B, quant: getQuant("q4_k_m"), ctx: 8192 }],
    );
    const [single, pair] = report.rows[0]?.cells ?? [];
    expect(pair?.fit.footprint.runtimeContextBytes).toBe(
      2 * (single?.fit.footprint.runtimeContextBytes ?? 0),
    );
    expect(pair?.fit.fits).toBe(true);
    expect(single?.fit.fits).toBe(false);
  });
});

describe("vramfit fleet", () => {
  it("prints the matrix, the machine legend and the placement summary", () => {
    const { code, io } = invoke(["fleet", "--config", "fleet.json"], { "fleet.json": FLEET });
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/^Fleet {2}\| {2}3 machines {2}\| {2}3 models$/m);
    expect(io.output).toMatch(
      /^Model\s+Quant\s+Ctx\s+Weights\+KV\s+workstation\s+laptop\s+old-box\s+Served$/m,
    );
    expect(io.output).toMatch(/^Llama 3\.1 8B\s+Q4_K_M\s+8K\s+5\.62 GiB.*3\/3$/m);
    expect(io.output).toMatch(/^Llama 3\.3 70B\s+Q4_K_M\s+8K.*-\s+2\/3$/m);
    expect(io.output).toMatch(/^ {2}old-box\s+NVIDIA RTX 3060 12GB\s+12 GiB\s+32 GiB$/m);
    for (const line of io.output.split("\n")) expect(line).toBe(line.trimEnd());
  });

  it("exits 1 when a model fits nowhere, and names it", () => {
    const config = JSON.stringify({
      ctx: 8192,
      machines: [{ name: "old-box", device: "rtx-3060-12gb" }],
      models: ["llama-3.1-8b", "qwen3-235b-a22b"],
    });
    const { code, io } = invoke(["fleet", "--config", "fleet.json"], { "fleet.json": config });
    expect(code).toBe(EXIT_DOES_NOT_FIT);
    expect(io.output).toMatch(/1 of 2 models fit nowhere: Qwen3 235B-A22B/);
  });

  it("takes a model from a path as readily as from the database", () => {
    const io = new FakeIo({
      text: {
        "fleet.json": JSON.stringify({
          ctx: 8192,
          machines: [{ name: "box", device: "4090" }],
          models: ["./local.gguf"],
        }),
      },
      binary: { "./local.gguf": llamaGgufBytes() },
    });
    expect(run(["fleet", "--config", "fleet.json"], io)).toBe(EXIT_OK);
    // Labelled by the path it came from, and quantized as the file itself is.
    expect(io.output).toMatch(/^\.\/local\.gguf\s+Q4_K_M\s+8K/m);
  });

  it("emits the same placement as JSON", () => {
    const { io } = invoke(["fleet", "--config", "fleet.json", "--json"], { "fleet.json": FLEET });
    const payload = JSON.parse(io.output) as {
      machines: { name: string; count: number; capacityBytes: number }[];
      models: { label: string; servedBy: number; machines: { name: string; fits: boolean }[] }[];
    };
    expect(payload.machines.map((entry) => entry.name)).toEqual([
      "workstation",
      "laptop",
      "old-box",
    ]);
    expect(payload.machines[0]?.capacityBytes).toBe(48 * 1024 ** 3);
    expect(payload.models[1]?.label).toBe("Llama 3.3 70B");
    expect(payload.models[1]?.servedBy).toBe(2);
    expect(payload.models[1]?.machines[2]).toEqual(
      expect.objectContaining({ name: "old-box", fits: false }),
    );
  });

  it("asks for the file rather than looking for one", () => {
    expect(invoke(["fleet"]).io.errors).toMatch(/--config is required/);
    expect(invoke(["fleet", "--config", "missing.json"]).io.errors).toMatch(
      /Cannot read missing\.json/,
    );
  });

  it("reports a bad fleet file with the path and the field", () => {
    const { code, io } = invoke(["fleet", "--config", "fleet.json"], {
      "fleet.json": JSON.stringify({
        machines: [{ device: "rtx-9090" }],
        models: ["llama-3.1-8b"],
      }),
    });
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/Unknown device "rtx-9090"/);
  });
});
