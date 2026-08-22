import { describe, expect, it } from "vitest";
import {
  EXIT_DOES_NOT_FIT,
  EXIT_OK,
  EXIT_USAGE,
  run,
  type Io,
} from "../src/cli/index.js";
import { renderTable, wrap } from "../src/cli/format.js";

/**
 * The CLI is a pure function of argv and an IO object, so the whole surface --
 * exit codes, report text, JSON payloads, error messages -- is covered here
 * without spawning a process or touching the filesystem.
 */
class FakeIo implements Io {
  readonly stdout: string[] = [];
  readonly stderr: string[] = [];
  private readonly files: Map<string, string>;

  constructor(files: Record<string, string> = {}) {
    this.files = new Map(Object.entries(files));
  }

  out(text: string): void {
    this.stdout.push(text);
  }

  err(text: string): void {
    this.stderr.push(text);
  }

  readFile(path: string): string {
    const content = this.files.get(path);
    if (content === undefined) throw new Error(`ENOENT: ${path}`);
    return content;
  }

  get output(): string {
    return this.stdout.join("\n");
  }

  get errors(): string {
    return this.stderr.join("\n");
  }
}

function invoke(argv: string[], files?: Record<string, string>) {
  const io = new FakeIo(files);
  const code = run(argv, io);
  return { code, io };
}

const CUSTOM_MODEL = JSON.stringify({
  id: "my-finetune-13b",
  name: "My Finetune 13B",
  aliases: [],
  totalParams: 13_000_000_000,
  activeParams: 13_000_000_000,
  nLayers: 40,
  hiddenSize: 5120,
  nHeads: 40,
  nKvHeads: 40,
  headDim: 128,
  ffnHidden: 13_824,
  vocabSize: 32_000,
  tiedEmbeddings: false,
  attention: "gqa",
  mla: null,
  moe: null,
  attentionWindow: null,
  maxCtx: 4096,
  defaultCtx: 4096,
  source: "my own config.json",
});

const CUSTOM_DEVICE = JSON.stringify({
  id: "prototype-gpu",
  name: "Prototype GPU",
  aliases: [],
  family: "cuda-consumer",
  vramGiB: 48,
  bandwidthGBs: 1500,
  fp16Tflops: 200,
  unifiedMemory: false,
  usableFraction: 1,
  source: "made up for a test",
});

describe("vramfit check", () => {
  it("reports a fit and exits 0", () => {
    const { code, io } = invoke(["check", "llama-3.1-8b", "--device", "4090", "--ctx", "8k"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/^Llama 3\.1 8B {2}\| {2}Q4_K_M {2}\| {2}NVIDIA RTX 4090$/m);
    expect(io.output).toMatch(/FITS {2}- {2}6\.\d\d GiB of 24\.00 GiB used/);
    expect(io.errors).toBe("");
  });

  it("exits 1 when it does not fit, so it can gate a deploy script", () => {
    const { code, io } = invoke(["check", "llama-3.3-70b", "-d", "4090"]);
    expect(code).toBe(EXIT_DOES_NOT_FIT);
    expect(io.output).toMatch(/DOES NOT FIT/);
    expect(io.output).toMatch(/Partial offload/);
    expect(io.output).toMatch(/Layers in device memory\s+\d+ of 80/);
  });

  it("shows every term of the memory breakdown with its assumption", () => {
    const { io } = invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k"]);
    expect(io.output).toMatch(/Weights\s+4\.62 GiB\s+8\.03B params at 4\.94 effective bits\/weight/);
    expect(io.output).toMatch(/KV cache\s+1\.00 GiB\s+8K tokens, 32 layers x 8 KV heads x 128, f16/);
    expect(io.output).toMatch(/Runtime context\s+0\.70 GiB\s+cuda-consumer driver and kernels/);
    expect(io.output).toMatch(/Compute buffer/);
    expect(io.output).toMatch(/Largest context that fits\s+128K/);
  });

  it("annotates time to first token with the prompt it was computed from", () => {
    // The figure comes from --prompt when given, so the note beside it has to
    // say so: labelling a 128-token prefill "for a prompt of 32K tokens" is
    // wrong by a factor of 256 against the number it explains.
    const long = invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "32k"]).io.output;
    expect(long).toMatch(/Time to first token\s+13\.\d s\s+for a prompt of 32K tokens/);

    const short = invoke([
      "check",
      "llama-3.1-8b",
      "-d",
      "4090",
      "--ctx",
      "32k",
      "--prompt",
      "128",
    ]).io.output;
    expect(short).toMatch(/Time to first token\s+\d+ ms\s+for a prompt of 128 tokens/);

    const payload = JSON.parse(
      invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "32k", "--prompt", "128", "--json"])
        .io.output,
    ) as { throughput: { promptTokens: number; timeToFirstTokenSeconds: number } };
    expect(payload.throughput.promptTokens).toBe(128);
    expect(payload.throughput.timeToFirstTokenSeconds).toBeLessThan(1);
  });

  it("turns a malformed --version or --help into exit 2, never a crash", () => {
    // Both were read outside the try/catch, so `vramfit -v check` printed a
    // raw V8 stack trace and exited 1 -- the code a deploy gate reads as
    // "does not fit". A usage mistake has to stay a usage mistake.
    for (const argv of [
      ["-v", "check"],
      ["--version=abc"],
      ["check", "llama-3.1-8b", "-d", "4090", "--help=x"],
    ]) {
      const { code, io } = invoke(argv);
      expect(code, argv.join(" ")).toBe(EXIT_USAGE);
      expect(io.errors, argv.join(" ")).toMatch(/^vramfit: /);
      expect(io.output, argv.join(" ")).toBe("");
    }
  });

  it("prints help for --help wherever it appears, and exits 0", () => {
    const early = invoke(["--help", "check"]);
    expect(early.code).toBe(EXIT_OK);
    expect(early.io.output).toMatch(/USAGE/);
    expect(early.io.errors).toBe("");

    const late = invoke(["check", "llama-3.1-8b", "--help"]);
    expect(late.code).toBe(EXIT_OK);
    expect(late.io.output).toMatch(/USAGE/);
  });

  it("accepts a value-less flag before the model, as every other CLI does", () => {
    const { code, io } = invoke(["check", "--json", "llama-3.1-8b", "-d", "4090", "--ctx", "8k"]);
    expect(code).toBe(EXIT_OK);
    const payload = JSON.parse(io.output) as { model: { id: string } };
    expect(payload.model.id).toBe("llama-3.1-8b");
  });

  it("refuses an argument the command has no meaning for", () => {
    // --flash-attn takes no separate value, so "false" here is a stray word.
    // Silently ignoring it would answer a question the user did not ask.
    const { code, io } = invoke([
      "check",
      "llama-3.1-8b",
      "-d",
      "4090",
      "--flash-attn",
      "false",
    ]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/Unexpected argument "false"/);
    expect(io.errors).toMatch(/--flag=value/);

    expect(invoke(["models", "extra"]).code).toBe(EXIT_USAGE);
  });

  it("reports a --vram override as the whole usable budget", () => {
    const capped = invoke(["check", "llama-3.1-8b", "-d", "m4-max", "--ctx", "8k"]).io.output;
    expect(capped).toMatch(/Available\s+96\.00 GiB\s+Apple M4 Max, 75% of 128 GiB wirable/);
    expect(capped).toMatch(/iogpu\.wired_limit_mb/);

    const raised = invoke([
      "check",
      "llama-3.1-8b",
      "-d",
      "m4-max",
      "--vram",
      "120",
      "--ctx",
      "8k",
    ]).io.output;
    expect(raised).toMatch(/Available\s+120\.00 GiB\s+Apple M4 Max\s*$/m);
    expect(raised).not.toMatch(/wirable/);
    // The user has already told us their budget; repeating the how-to-raise-it
    // advice would be telling them to do what they have just done.
    expect(raised).not.toMatch(/iogpu\.wired_limit_mb/);
  });

  it("describes latent attention and sliding windows in the cache line", () => {
    expect(invoke(["check", "deepseek-v2-lite", "-d", "4090"]).io.output).toMatch(
      /27 layers x 576-wide latent \(MLA\)/,
    );
    expect(invoke(["check", "gemma-3-27b", "-d", "h100", "--ctx", "32k"]).io.output).toMatch(
      /\d+ windowed at 1024/,
    );
  });

  it("names both parameter counts for a mixture of experts", () => {
    const { io } = invoke(["check", "mixtral-8x7b", "-d", "h100", "--ctx", "8k"]);
    expect(io.output).toMatch(/46\.70B total \/ 12\.88B active/);
  });

  it("honours quant, context, batch, cache type and device count", () => {
    const { code, io } = invoke([
      "check",
      "llama-3.3-70b",
      "-d",
      "3090",
      "-g",
      "2",
      "-q",
      "q4_k_m",
      "--ctx",
      "8k",
      "--batch",
      "2",
      "--kv-quant",
      "q8_0",
    ]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/2 x NVIDIA RTX 3090/);
    expect(io.output).toMatch(/2 sequences, q8_0/);
    expect(io.output).toMatch(/Decode, aggregate/);
    expect(io.output).toMatch(/capacity, not tokens per second/);
  });

  it("emits a stable JSON document under --json", () => {
    const { code, io } = invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "8k", "--json"]);
    expect(code).toBe(EXIT_OK);

    const payload = JSON.parse(io.output) as Record<string, Record<string, unknown>>;
    expect(payload["fits"]).toBe(true);
    expect(payload["model"]?.["id"]).toBe("llama-3.1-8b");
    expect(payload["device"]?.["id"]).toBe("rtx-4090");
    expect(payload["config"]?.["ctx"]).toBe(8192);
    expect(payload["config"]?.["kvQuant"]).toBe("f16");
    expect(payload["memory"]?.["kvCacheBytes"]).toBe(1024 * 1024 * 1024);
    expect(payload["capacity"]?.["maxContext"]).toBe(131_072);
    expect(payload["offload"]).toBeNull();
    expect(payload["warnings"]).toEqual([]);

    // The four terms must still add up to the headline number.
    const memory = payload["memory"] as unknown as {
      weightsBytes: number;
      kvCacheBytes: number;
      runtimeContextBytes: number;
      activationBytes: number;
      totalBytes: number;
      capacityBytes: number;
      headroomBytes: number;
    };
    expect(
      memory.weightsBytes +
        memory.kvCacheBytes +
        memory.runtimeContextBytes +
        memory.activationBytes,
    ).toBeCloseTo(memory.totalBytes, 6);
    expect(memory.capacityBytes - memory.totalBytes).toBeCloseTo(memory.headroomBytes, 6);
  });

  it("reports the offload split in JSON too", () => {
    const { code, io } = invoke([
      "check",
      "llama-3.3-70b",
      "-d",
      "4090",
      "--ctx",
      "8k",
      "--ram",
      "64",
      "--json",
    ]);
    expect(code).toBe(EXIT_DOES_NOT_FIT);
    const payload = JSON.parse(io.output) as Record<string, Record<string, unknown>>;
    expect(payload["fits"]).toBe(false);
    expect(payload["offload"]?.["feasible"]).toBe(true);
    expect(payload["offload"]?.["gpuLayers"]).toBeGreaterThan(0);
    expect(payload["offload"]?.["systemRamAvailableBytes"]).toBe(64 * 1024 ** 3);
  });

  it("accepts a model and a device from JSON files", () => {
    const { code, io } = invoke(
      ["check", "--model-json", "model.json", "--device-json", "device.json", "--ctx", "4k"],
      { "model.json": CUSTOM_MODEL, "device.json": CUSTOM_DEVICE },
    );
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/My Finetune 13B {2}\| {2}Q4_K_M {2}\| {2}Prototype GPU/);
    expect(io.output).toMatch(/40 layers x 40 KV heads x 128/);
  });

  it("validates a user-supplied model exactly like a bundled one", () => {
    const broken = JSON.stringify({ ...JSON.parse(CUSTOM_MODEL), nKvHeads: 7 });
    const { code, io } = invoke(["check", "--model-json", "model.json", "-d", "4090"], {
      "model.json": broken,
    });
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/model\.json\.nKvHeads must divide nHeads \(40\) evenly/);
  });

  it("refuses a spec whose parameter count contradicts its architecture", () => {
    // The 1000x slip that used to produce "FITS - 2.55 GiB of 24.00 GiB used"
    // and exit 0 for a model that needs 4.62 GiB of weights alone.
    const typo = JSON.stringify({
      ...(JSON.parse(CUSTOM_MODEL) as Record<string, unknown>),
      totalParams: 13_000_000,
      activeParams: 13_000_000,
    });
    const { code, io } = invoke(["check", "--model-json", "model.json", "-d", "4090"], {
      "model.json": typo,
    });
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/model\.json\.totalParams is 13000000/);
    expect(io.output).toBe("");
  });

  it("reports an unreadable or malformed file clearly", () => {
    expect(invoke(["check", "--model-json", "missing.json", "-d", "4090"]).io.errors).toMatch(
      /Cannot read missing\.json/,
    );
    expect(
      invoke(["check", "--model-json", "bad.json", "-d", "4090"], { "bad.json": "{oops" }).io
        .errors,
    ).toMatch(/bad\.json is not valid JSON/);
  });

  it("names the file that would not parse without quoting what is in it", () => {
    // A mistyped path in a CI step must not echo the head of whatever file was
    // named into the build log; V8's own JSON.parse message quotes the first
    // ten bytes of the input.
    const { code, io } = invoke(["check", "--model-json", "secrets.env", "-d", "4090"], {
      "secrets.env": "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG\n",
    });
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/secrets\.env is not valid JSON/);
    expect(io.errors).not.toMatch(/AWS_SECRET/);
    expect(io.errors).not.toMatch(/wJalrXUtnFEMI/);
  });
});

describe("vramfit best", () => {
  it("ranks every quantization and recommends the best that fits", () => {
    const { code, io } = invoke(["best", "llama-3.1-8b", "-d", "3060", "--ctx", "16k"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/Quant\s+bpw\s+Weights\s+Total at 16K\s+Fits\s+Max ctx\s+Decode/);
    expect(io.output).toMatch(/^F16\s+16\.00\s+14\.96 GiB\s+\S+ GiB\s+no/m);
    expect(io.output).toMatch(/^Q8_0\s+8\.50\s+7\.95 GiB\s+\S+ GiB\s+yes/m);
    expect(io.output).toMatch(/Recommended: Q8_0 -- highest quality that fits at 16K/);
  });

  it("lists candidates best quality first", () => {
    const { io } = invoke(["best", "mistral-7b", "-d", "4090"]);
    const quantColumn = io.output
      .split("\n")
      .filter((line) => /^(F16|BF16|Q\d)/.test(line))
      .map((line) => line.split(/\s+/)[0]);
    expect(quantColumn.slice(0, 4)).toEqual(["F16", "BF16", "Q8_0", "Q6_K"]);
    expect(quantColumn.at(-1)).toBe("Q2_K");
  });

  it("exits 1 and says what to do when nothing fits", () => {
    const { code, io } = invoke(["best", "qwen3-235b-a22b", "-d", "3060"]);
    expect(code).toBe(EXIT_DOES_NOT_FIT);
    expect(io.output).toMatch(/Nothing in this range fits/);
    expect(io.output).toMatch(/Reduce the context, offload layers, or pick a smaller model/);
  });

  it("emits the same ranking as JSON", () => {
    const { io } = invoke(["best", "llama-3.1-8b", "-d", "3060", "--ctx", "16k", "--json"]);
    const payload = JSON.parse(io.output) as {
      recommended: string;
      quants: { id: string; fits: boolean; maxContext: number }[];
    };
    expect(payload.recommended).toBe("q8_0");
    expect(payload.quants[0]?.id).toBe("f16");
    expect(payload.quants.find((q) => q.id === "q4_k_m")?.fits).toBe(true);
  });
});

describe("vramfit devices and models", () => {
  it("lists devices with their memory, bandwidth and family", () => {
    const { code, io } = invoke(["devices"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/^rtx-4090\s+NVIDIA RTX 4090\s+24 GiB\s+24 GiB\s+1008 GB\/s/m);
    // Apple silicon shows the wired-memory limit as usable memory.
    expect(io.output).toMatch(/^m3-max\s+Apple M3 Max\s+128 GiB\s+96 GiB/m);
  });

  it("lists models with the KV head count that actually matters", () => {
    const { io } = invoke(["models"]);
    expect(io.output).toMatch(/^llama-3\.3-70b\s+Llama 3\.3 70B\s+70\.55B\s+dense\s+80\s+8\s+128K\s+GQA$/m);
    expect(io.output).toMatch(/^deepseek-v2-lite\s+.*\s+MLA$/m);
    expect(io.output).toMatch(/^gemma-3-27b\s+.*GQA \+ SWA\/6$/m);
    expect(io.output).toMatch(/^mixtral-8x7b\s+Mixtral 8x7B\s+46\.70B\s+12\.88B/m);
  });

  it("emits the raw database under --json", () => {
    const devices = JSON.parse(invoke(["devices", "--json"]).io.output) as { id: string }[];
    const models = JSON.parse(invoke(["models", "--json"]).io.output) as { id: string }[];
    expect(devices.some((device) => device.id === "rtx-4090")).toBe(true);
    expect(models.some((model) => model.id === "gpt-oss-120b")).toBe(true);
  });
});

describe("help, version and errors", () => {
  it("prints help on --help and exits 0", () => {
    const { code, io } = invoke(["--help"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/USAGE/);
    expect(io.output).toMatch(/vramfit check <model> --device <device>/);
    expect(io.output).toMatch(/0 {2}fits {4}1 {2}does not fit {4}2 {2}bad usage/);
  });

  it("prints help and exits 2 when called with nothing", () => {
    const { code, io } = invoke([]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.output).toMatch(/USAGE/);
  });

  it("prints the package version", () => {
    const { code, io } = invoke(["--version"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/^\d+\.\d+\.\d+/);
    expect(invoke(["version"]).io.output).toBe(io.output);
  });

  it("suggests near matches for an unknown model or device", () => {
    expect(invoke(["check", "llama-3.1-9b", "-d", "4090"]).io.errors).toMatch(
      /Unknown model "llama-3\.1-9b"\. Did you mean: llama-3\.1-8b/,
    );
    expect(invoke(["check", "llama-3.1-8b", "-d", "rtx-4091"]).io.errors).toMatch(
      /Unknown device "rtx-4091"\. Did you mean: rtx-4090/,
    );
  });

  it("refuses an unknown command, option or flag value", () => {
    expect(invoke(["frobnicate"]).code).toBe(EXIT_USAGE);
    expect(invoke(["frobnicate"]).io.errors).toMatch(/Unknown command "frobnicate"/);
    expect(invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctxx", "8k"]).io.errors).toMatch(
      /Unknown option "--ctxx"/,
    );
    expect(invoke(["check", "llama-3.1-8b", "-d", "4090", "--ctx", "lots"]).io.errors).toMatch(
      /--ctx expects a token count/,
    );
    expect(invoke(["check", "llama-3.1-8b", "-d", "4090", "-q", "q9_k"]).io.errors).toMatch(
      /Unknown quantization "q9_k"/,
    );
  });

  it("asks for the missing piece rather than guessing", () => {
    expect(invoke(["check", "llama-3.1-8b"]).io.errors).toMatch(/--device is required/);
    expect(invoke(["check", "-d", "4090"]).io.errors).toMatch(/A model is required/);
  });

  it("rejects --json on a command that does not take other flags", () => {
    expect(invoke(["devices", "--ctx", "8k"]).io.errors).toMatch(/Unknown option "--ctx"/);
  });

  it("writes errors to stderr and nothing to stdout", () => {
    const { io } = invoke(["check", "nope-9000", "-d", "4090"]);
    expect(io.stdout).toEqual([]);
    expect(io.stderr.length).toBeGreaterThan(0);
  });
});

describe("formatting primitives", () => {
  it("aligns a table to its widest cell and trims trailing space", () => {
    const lines = renderTable(
      [{ header: "Name" }, { header: "Size", align: "right" }],
      [
        ["a", "1"],
        ["longer", "1000"],
      ],
    );
    expect(lines).toEqual(["Name    Size", "------  ----", "a          1", "longer  1000"]);
    for (const line of lines) expect(line).toBe(line.trimEnd());
  });

  it("wraps prose without breaking words", () => {
    const lines = wrap("one two three four five six", 12);
    expect(lines).toEqual(["one two", "three four", "five six"]);
    for (const line of lines) expect(line.length).toBeLessThanOrEqual(12);
  });

  it("keeps an indent inside the wrap width", () => {
    const lines = wrap("alpha beta gamma delta", 16, "    ");
    for (const line of lines) {
      expect(line.startsWith("    ")).toBe(true);
      expect(line.length).toBeLessThanOrEqual(16);
    }
  });
});
