import { describe, expect, it } from "vitest";
import { Args, UsageError, parseArgv, parseTokenCount } from "../src/cli/args.js";

describe("parseArgv", () => {
  it("reads long flags with a separate value", () => {
    const parsed = parseArgv(["check", "llama-3.1-8b", "--device", "4090"]);
    expect(parsed.positionals).toEqual(["check", "llama-3.1-8b"]);
    expect(parsed.flags.get("device")).toBe("4090");
  });

  it("reads long flags with an inline value", () => {
    expect(parseArgv(["--device=4090"]).flags.get("device")).toBe("4090");
    expect(parseArgv(["--kv-quant=q8_0"]).flags.get("kv-quant")).toBe("q8_0");
    // An empty inline value is still a value, not a missing one.
    expect(parseArgv(["--device="]).flags.get("device")).toBe("");
  });

  it("treats a trailing long flag as a boolean", () => {
    const parsed = parseArgv(["check", "--json"]);
    expect(parsed.flags.get("json")).toBe(true);
  });

  it("reads --no-x as false", () => {
    expect(parseArgv(["--no-flash-attn"]).flags.get("flash-attn")).toBe(false);
    // "--no" and "--not" are flags in their own right, not negations.
    expect(parseArgv(["--no"]).flags.get("no")).toBe(true);
  });

  it("expands short flags, alone and with values", () => {
    const parsed = parseArgv(["check", "-d", "4090", "-q=q8_0", "-j"]);
    expect(parsed.flags.get("device")).toBe("4090");
    expect(parsed.flags.get("quant")).toBe("q8_0");
    expect(parsed.flags.get("json")).toBe(true);
  });

  it("stops flag parsing at --", () => {
    const parsed = parseArgv(["check", "--", "--device", "4090"]);
    expect(parsed.positionals).toEqual(["check", "--device", "4090"]);
    expect(parsed.flags.size).toBe(0);
  });

  it("treats a negative number as a value, so it can be rejected as one", () => {
    expect(parseArgv(["--efficiency", "-0.5"]).flags.get("efficiency")).toBe("-0.5");
  });

  it("rejects clustered and unknown short flags", () => {
    expect(() => parseArgv(["-abc"])).toThrow(/Short options take one letter at a time/);
    expect(() => parseArgv(["-z"])).toThrow(/Unknown option "-z"/);
  });

  it("does not let a value-less flag swallow the next argument", () => {
    // Flag-before-positional is normal GNU-style ordering, and --json takes no
    // value: reading one left `vramfit check --json llama-3.1-8b` with no
    // model and a json flag set to "llama-3.1-8b".
    const parsed = parseArgv(["check", "--json", "llama-3.1-8b", "-d", "4090"]);
    expect(parsed.flags.get("json")).toBe(true);
    expect(parsed.positionals).toEqual(["check", "llama-3.1-8b"]);

    expect(parseArgv(["--help", "check"]).positionals).toEqual(["check"]);
    expect(parseArgv(["-v", "check"]).flags.get("version")).toBe(true);
    expect(parseArgv(["-v", "check"]).positionals).toEqual(["check"]);
    expect(parseArgv(["--flash-attn", "llama-3.1-8b"]).positionals).toEqual(["llama-3.1-8b"]);
  });

  it("still reads a value-less flag's value in the inline and no- forms", () => {
    expect(parseArgv(["--json=false"]).flags.get("json")).toBe("false");
    expect(parseArgv(["--no-json"]).flags.get("json")).toBe(false);
    expect(parseArgv(["--flash-attn=off"]).flags.get("flash-attn")).toBe("off");
  });

  it("lower-cases flag names so --CTX works", () => {
    expect(parseArgv(["--CTX", "8192"]).flags.get("ctx")).toBe("8192");
  });
});

describe("parseTokenCount", () => {
  it("accepts a plain count", () => {
    expect(parseTokenCount("8192", "ctx")).toBe(8192);
  });

  it("treats k as 1024, which is what it means in this domain", () => {
    expect(parseTokenCount("32k", "ctx")).toBe(32_768);
    expect(parseTokenCount("128K", "ctx")).toBe(131_072);
    expect(parseTokenCount("1m", "ctx")).toBe(1_048_576);
    expect(parseTokenCount("1.5k", "ctx")).toBe(1536);
  });

  it("rejects nonsense with the flag name in the message", () => {
    expect(() => parseTokenCount("lots", "ctx")).toThrow(
      /--ctx expects a token count such as 8192 or 32k, got "lots"/,
    );
    expect(() => parseTokenCount("0", "ctx")).toThrow(/at least 1 token/);
    expect(() => parseTokenCount("-5", "ctx")).toThrow(/--ctx expects a token count/);
  });
});

describe("Args", () => {
  it("validates numbers against their bounds", () => {
    const args = Args.parse(["--gpus", "3", "--efficiency", "1.5", "--batch", "2.5"]);
    expect(args.number("gpus", { integer: true, min: 1 })).toBe(3);
    expect(() => args.number("efficiency", { max: 1 })).toThrow(/--efficiency must be at most 1/);
    expect(() => args.number("batch", { integer: true })).toThrow(/whole number/);
    expect(args.number("missing")).toBeUndefined();
  });

  it("refuses a flag that was given without its value", () => {
    const args = Args.parse(["check", "--device"]);
    expect(() => args.string("device")).toThrow(/--device needs a value/);
  });

  it("reports a missing required flag", () => {
    expect(() => Args.parse([]).requiredString("device")).toThrow(/--device is required/);
  });

  it("reads booleans in the spellings people use", () => {
    expect(Args.parse(["--json"]).boolean("json")).toBe(true);
    expect(Args.parse(["--no-json"]).boolean("json")).toBe(false);
    expect(Args.parse(["--json=false"]).boolean("json")).toBe(false);
    expect(Args.parse(["--json=yes"]).boolean("json")).toBe(true);
    expect(Args.parse([]).boolean("json")).toBeUndefined();
    expect(() => Args.parse(["--json=maybe"]).boolean("json")).toThrow(/expects true or false/);
  });

  it("rejects a flag the command does not accept, and suggests the near one", () => {
    const args = Args.parse(["check", "--ctxx", "8k"]);
    expect(() => args.assertKnown(["ctx", "device"])).toThrow(UsageError);
    expect(() => args.assertKnown(["ctx", "device"])).toThrow(
      /Unknown option "--ctxx"\. Did you mean --ctx\?/,
    );
  });

  it("names the option the way it was typed", () => {
    // `--no-color` reaches the flag map as `color`, and reporting the
    // normalised name tells the user off for a flag they never wrote.
    const negated = Args.parse(["devices", "--no-colour"]);
    expect(() => negated.assertKnown(["json"])).toThrow(/Unknown option "--no-colour"/);

    const short = Args.parse(["check", "-d", "4090"]);
    expect(() => short.assertKnown(["json"])).toThrow(/Unknown option "-d"/);

    const shouted = Args.parse(["check", "--CTXX", "8k"]);
    expect(() => shouted.assertKnown(["ctx"])).toThrow(
      /Unknown option "--CTXX"\. Did you mean --ctx\?/,
    );
  });

  it("accepts every flag the command declares", () => {
    const args = Args.parse(["check", "m", "--ctx", "8k", "--json"]);
    expect(() => args.assertKnown(["ctx", "json"])).not.toThrow();
  });
});
