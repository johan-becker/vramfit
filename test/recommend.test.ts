import { describe, expect, it } from "vitest";
import { EXIT_DOES_NOT_FIT, EXIT_OK, EXIT_USAGE, run } from "../src/cli/index.js";
import { getDevice } from "../src/db/index.js";
import {
  USE_CASES,
  USE_CASE_IDS,
  effectiveCapabilityParams,
  findUseCase,
  recommendModels,
} from "../src/recommend.js";
import { FakeIo } from "./fake-io.js";
import { LLAMA_3_1_8B, LLAMA_3_1_70B, LLAMA_3_2_1B, MIXTRAL_8X7B } from "./fixtures.js";

/**
 * `recommend`: what to run on the box you have.
 *
 * The ranking is a judgement, so what is tested is that the judgement is the
 * one documented -- capability on a log scale, a quantization-quality factor,
 * and speed measured against a bar that the use case sets -- and that each
 * factor can actually change the answer.
 */

function invoke(argv: string[]) {
  const io = new FakeIo();
  return { code: run(argv, io), io };
}

describe("use case profiles", () => {
  it("sets a context and a speed bar for each, and nothing else", () => {
    expect(USE_CASE_IDS).toEqual(["chat", "code", "long-context"]);
    expect(USE_CASES.chat.comfortableDecode).toBe(15);
    expect(USE_CASES.code.comfortableDecode).toBe(30);
    expect(USE_CASES["long-context"].defaultContext).toBe(32_768);
    expect(findUseCase("LONG-CONTEXT")).toBe(USE_CASES["long-context"]);
    expect(findUseCase("vibes")).toBeUndefined();
  });
});

describe("effectiveCapabilityParams", () => {
  it("is the parameter count for a dense model", () => {
    expect(effectiveCapabilityParams(LLAMA_3_1_8B)).toBe(8_030_000_000);
  });

  it("puts a mixture of experts between what it stores and what it computes", () => {
    // sqrt(46.703e9 x 12.879e9) = 24.5e9: Mixtral behaves like a mid-20s
    // dense model, not like a 47B and not like a 13B.
    expect(effectiveCapabilityParams(MIXTRAL_8X7B) / 1e9).toBeCloseTo(24.5, 1);
  });
});

describe("recommendModels", () => {
  const candidates = [LLAMA_3_2_1B, LLAMA_3_1_8B, LLAMA_3_1_70B, MIXTRAL_8X7B];

  it("leaves out what does not fit at all", () => {
    const rows = recommendModels(getDevice("rtx-3060-12gb"), {
      models: candidates,
      ctx: 8192,
    });
    expect(rows.map((row) => row.model.id)).toEqual([
      "fixture-llama-3.1-8b",
      "fixture-llama-3.2-1b",
    ]);
  });

  it("prefers the bigger model when both clear the speed bar", () => {
    const rows = recommendModels(getDevice("h100"), { models: candidates, ctx: 8192 });
    expect(rows[0]?.model.id).toBe("fixture-llama-3.1-70b");
    // Capability is logarithmic: a 70B is about one unit above a 1B, not 70.
    const seventy = rows.find((row) => row.model.id === "fixture-llama-3.1-70b");
    const one = rows.find((row) => row.model.id === "fixture-llama-3.2-1b");
    // log10(70.554) - log10(1.236) = 1.757, not 57.
    expect((seventy?.capability ?? 0) - (one?.capability ?? 0)).toBeCloseTo(1.757, 3);
  });

  it("lets the speed bar demote a model that fits but crawls", () => {
    // On a 3090 the 70B only fits by dropping to a narrow quantization and
    // decodes in single digits; the 8B is the better answer for chat.
    const chat = recommendModels(getDevice("3090"), { models: candidates, useCase: "chat" });
    const seventy = chat.find((row) => row.model.id === "fixture-llama-3.1-70b");
    if (seventy !== undefined) {
      expect(seventy.speed).toBeLessThan(1);
      expect(chat[0]?.model.id).not.toBe("fixture-llama-3.1-70b");
    }
    expect(chat[0]?.speed).toBe(1);
  });

  it("charges a narrow quantization a quality factor", () => {
    const rows = recommendModels(getDevice("3090"), { models: candidates, ctx: 8192 });
    for (const row of rows) {
      // Q4_K_M and above are treated as lossless; below that the factor bites.
      if (row.quant.qualityRank >= 60) expect(row.quality).toBe(1);
      else expect(row.quality).toBeLessThan(1);
      expect(row.quality).toBeGreaterThanOrEqual(0.6);
      expect(row.quality).toBeLessThanOrEqual(1);
      expect(row.score).toBeCloseTo(row.capability * row.quality * row.speed, 10);
    }
  });

  it("refuses to recommend a model past its own trained context", () => {
    // The fixture 8B is a 128K model; the 1B fixture below is capped at 8K.
    const short = { ...LLAMA_3_2_1B, maxCtx: 8192, defaultCtx: 8192 };
    const rows = recommendModels(getDevice("4090"), {
      models: [short, LLAMA_3_1_8B],
      ctx: 32_768,
    });
    expect(rows.map((row) => row.model.id)).toEqual(["fixture-llama-3.1-8b"]);
  });

  it("explains each row in one sentence", () => {
    const rows = recommendModels(getDevice("4090"), { models: candidates, ctx: 8192 });
    const eight = rows.find((row) => row.model.id === "fixture-llama-3.1-8b");
    expect(eight?.tradeoff).toMatch(/^8\.03B parameters at /);
    expect(eight?.tradeoff).toMatch(/tok\/s/);

    const mixtral = rows.find((row) => row.model.id === "fixture-mixtral-8x7b");
    if (mixtral !== undefined) {
      expect(mixtral.tradeoff).toMatch(/of which 12\.88B run per token/);
    }
  });

  it("honours a limit", () => {
    const rows = recommendModels(getDevice("h100"), { models: candidates, limit: 2 });
    expect(rows).toHaveLength(2);
  });
});

describe("vramfit recommend", () => {
  it("ranks the bundled models and explains each", () => {
    const { code, io } = invoke(["recommend", "-d", "4090", "--use-case", "code", "--limit", "3"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/^NVIDIA RTX 4090 {2}\| {2}code {2}\| {2}16K context$/m);
    expect(io.output).toMatch(/^1\.\s+Qwen2\.5 14B\s+14\.77B\s+Q8_0/m);
    expect(io.output).toMatch(/comfortably above the 30 tok\/s code wants/);
    expect(io.output).toMatch(/does not rank models by how good they are at\s+anything/);
    for (const line of io.output.split("\n")) expect(line).toBe(line.trimEnd());
  });

  it("starts the trade-off under the Model column however wide the index is", () => {
    // The unlimited form is the default, and from the tenth row the index
    // column is three wide: a hard-coded four-space indent left every
    // trade-off one column short of the row it belongs to.
    for (const argv of [["recommend", "-d", "4090"], ["recommend", "-d", "4090", "--limit", "3"]]) {
      const { io } = invoke(argv);
      const lines = io.output.split("\n");
      const header = lines.find((line) => line.includes("#  Model"));
      if (header === undefined) throw new Error("expected a table header");
      const modelColumn = header.indexOf("Model");

      const rowPattern = /^\s*\d+\.\s{2}\S/;
      const body = lines.slice(lines.indexOf(header) + 2);
      const tradeoffs = body.filter(
        (line) => line.startsWith(" ") && line.trim() !== "" && !rowPattern.test(line),
      );
      expect(tradeoffs.length).toBeGreaterThan(0);
      for (const line of tradeoffs) {
        expect(line.search(/\S/), line).toBe(modelColumn);
      }
    }
  });

  it("says so, usefully, when nothing fits", () => {
    // A gigabyte of usable memory is gone before a weight is placed.
    const { code, io } = invoke(["recommend", "-d", "4090", "--vram", "1"]);
    expect(code).toBe(EXIT_DOES_NOT_FIT);
    expect(io.output).toMatch(/Nothing in the bundled database fits/);
    expect(io.output).toMatch(/--kv-quant q8_0/);
  });

  it("says the capacity is an override rather than blaming the card", () => {
    // "Nothing fits an RTX 4090 at 8K" is false -- plenty does. What did not
    // fit is the 1 GiB the run was capped at, so both the header and the
    // message have to say so, and the follow-up command has to carry the
    // override rather than silently dropping the user back to 24 GiB.
    for (const format of [[], ["--markdown"]]) {
      const { io } = invoke(["recommend", "-d", "4090", "--vram", "1", ...format]);
      expect(io.output).toMatch(/NVIDIA RTX 4090 \(1\.00 GiB, from --vram\)/);
    }

    const { io } = invoke(["recommend", "-d", "4090", "--vram", "1"]);
    expect(io.output).toMatch(/vramfit check <model> -d rtx-4090 --vram\s+1"/);

    // Without the override the device is named plainly, as before.
    const plain = invoke(["recommend", "-d", "4090", "--limit", "1"]);
    expect(plain.io.output).toMatch(/^NVIDIA RTX 4090 {2}\| {2}chat/m);
    expect(plain.io.output).not.toMatch(/--vram/);
  });

  it("names a native quantization as native rather than as a compromise", () => {
    const { io } = invoke(["recommend", "-d", "m3-ultra", "--use-case", "long-context"]);
    expect(io.output).toMatch(/in its native MXFP4/);
  });

  it("carries the ranking's own numbers into JSON", () => {
    const { io } = invoke(["recommend", "-d", "4090", "--limit", "2", "--json"]);
    const payload = JSON.parse(io.output) as {
      useCase: { id: string; ctx: number; comfortableDecodeTokensPerSecond: number };
      models: { id: string; score: number; capability: number; quality: number; speed: number }[];
    };
    expect(payload.useCase).toEqual({
      id: "chat",
      ctx: 8192,
      comfortableDecodeTokensPerSecond: 15,
    });
    expect(payload.models).toHaveLength(2);
    const first = payload.models[0];
    if (first === undefined) throw new Error("expected a row");
    expect(first.score).toBeCloseTo(first.capability * first.quality * first.speed, 10);
    expect((payload.models[1]?.score ?? 0) <= first.score).toBe(true);
  });

  it("refuses a use case it does not have a profile for", () => {
    const { code, io } = invoke(["recommend", "-d", "4090", "--use-case", "vibes"]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/Unknown use case "vibes"\. Expected one of chat, code, long-context/);
  });

  it("still needs a device", () => {
    expect(invoke(["recommend"]).io.errors).toMatch(/--device is required/);
  });
});
