import { describe, expect, it } from "vitest";
import { listDevices, listModels } from "../src/db/index.js";
import { EXIT_DOES_NOT_FIT, EXIT_OK, EXIT_USAGE, run, type Io } from "../src/cli/index.js";
import { looksLikePath } from "../src/cli/source.js";
import type { ClosableByteSource, PathKind } from "../src/cli/source.js";
import { GgufBuilder, llamaGgufBytes, moeGgufBuilder, str } from "./gguf-fixtures.js";

/**
 * `vramfit check ./model.gguf`, end to end, against headers built in memory.
 *
 * The IO object is the seam: the fake below serves bytes out of a Map, so the
 * command runs exactly as it would against a 5 GB file on disk without one
 * existing. `closed` records that the file handle was released, including on
 * the failure paths.
 */
class FakeIo implements Io {
  readonly stdout: string[] = [];
  readonly stderr: string[] = [];
  readonly closed: string[] = [];
  private readonly text: Map<string, string>;
  private readonly binary: Map<string, Uint8Array>;
  private readonly directories: Set<string>;

  constructor(
    files: { text?: Record<string, string>; binary?: Record<string, Uint8Array>; directories?: string[] } = {},
  ) {
    this.text = new Map(Object.entries(files.text ?? {}));
    this.binary = new Map(Object.entries(files.binary ?? {}));
    this.directories = new Set(files.directories ?? []);
  }

  out(text: string): void {
    this.stdout.push(text);
  }

  err(text: string): void {
    this.stderr.push(text);
  }

  readFile(path: string): string {
    const content = this.text.get(path);
    if (content === undefined) throw new Error(`ENOENT: ${path}`);
    return content;
  }

  openBytes(path: string): ClosableByteSource {
    const bytes = this.binary.get(path);
    if (bytes === undefined) throw new Error(`ENOENT: ${path}`);
    const closed = this.closed;
    return {
      size: bytes.length,
      read(offset: number, length: number): Uint8Array {
        return bytes.subarray(offset, offset + Math.max(0, length));
      },
      close(): void {
        closed.push(path);
      },
    };
  }

  pathKind(path: string): PathKind {
    if (this.directories.has(path)) return "directory";
    if (this.binary.has(path) || this.text.has(path)) return "file";
    return "missing";
  }

  get output(): string {
    return this.stdout.join("\n");
  }

  get errors(): string {
    return this.stderr.join("\n");
  }
}

const LLAMA_PATH = "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

function invoke(argv: string[], binary: Record<string, Uint8Array> = {}) {
  const io = new FakeIo({ binary });
  const code = run(argv, io);
  return { code, io };
}

function withLlama(argv: string[]) {
  return invoke(argv, { [LLAMA_PATH]: llamaGgufBytes() });
}

describe("looksLikePath", () => {
  it("recognises the forms people type", () => {
    expect(looksLikePath("./model.gguf")).toBe(true);
    expect(looksLikePath("../a/b.gguf")).toBe(true);
    expect(looksLikePath("/srv/models/x.gguf")).toBe(true);
    expect(looksLikePath("C:\\models\\x.gguf")).toBe(true);
    expect(looksLikePath("models/x.gguf")).toBe(true);
    expect(looksLikePath("x.GGUF")).toBe(true);
    expect(looksLikePath("./checkpoint/")).toBe(true);
    expect(looksLikePath("config.json")).toBe(true);
  });

  it("leaves every bundled name alone", () => {
    // The two namespaces have to stay disjoint, or a model would shadow a
    // path or the other way round.
    for (const model of listModels()) {
      for (const key of [model.id, model.name, ...model.aliases]) {
        expect(looksLikePath(key)).toBe(false);
      }
    }
    for (const device of listDevices()) {
      for (const key of [device.id, device.name, ...device.aliases]) {
        expect(looksLikePath(key)).toBe(false);
      }
    }
  });
});

describe("vramfit check <path>.gguf", () => {
  it("checks a file the same way it checks a bundled model", () => {
    const { code, io } = withLlama(["check", LLAMA_PATH, "-d", "4090", "--ctx", "8k"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/Meta Llama 3\.1 8B Instruct {2}\| {2}Q4_K_M {2}\| {2}NVIDIA RTX 4090/);
    expect(io.output).toMatch(
      /Read from \.\/models\/Meta-Llama-3\.1-8B-Instruct-Q4_K_M\.gguf {2}\(GGUF v3, 292 tensors, llama architecture, Q4_K_M\)/,
    );
    expect(io.output).toMatch(/FITS/);
    // 8.03B parameters counted from the shape table, not from a name.
    expect(io.output).toMatch(/8\.03B params at 5\.15 effective bits\/weight/);
    expect(io.output).toMatch(/32 layers x 8 KV heads x 128, f16/);
    expect(io.errors).toBe("");
  });

  it("closes the file, on the way out and on the way through an error", () => {
    const good = new FakeIo({ binary: { [LLAMA_PATH]: llamaGgufBytes() } });
    run(["check", LLAMA_PATH, "-d", "4090"], good);
    expect(good.closed).toEqual([LLAMA_PATH]);

    const truncated = new FakeIo({ binary: { "./cut.gguf": llamaGgufBytes().slice(0, 900) } });
    run(["check", "./cut.gguf", "-d", "4090"], truncated);
    expect(truncated.closed).toEqual(["./cut.gguf"]);
  });

  it("uses the file's own quantization, and lets -q override it", () => {
    const measured = withLlama(["check", LLAMA_PATH, "-d", "4090", "--json"]);
    const payload = JSON.parse(measured.io.output) as {
      model: Record<string, unknown>;
      config: Record<string, unknown>;
      memory: Record<string, number>;
    };
    expect(payload.model["origin"]).toBe("gguf");
    expect(payload.model["from"]).toBe(LLAMA_PATH);
    expect(payload.config["quant"]).toBe("file");
    // The weights the file itself occupies, to the byte.
    expect(payload.memory["weightsBytes"]).toBeCloseTo(5_172_420_864, 0);

    const overridden = withLlama(["check", LLAMA_PATH, "-d", "4090", "-q", "q8_0", "--json"]);
    const other = JSON.parse(overridden.io.output) as { config: Record<string, unknown> };
    expect(other.config["quant"]).toBe("q8_0");
  });

  it("ranks requantizations of a file with best", () => {
    const { code, io } = withLlama(["best", LLAMA_PATH, "-d", "4090", "--ctx", "8k"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/Meta Llama 3\.1 8B Instruct {2}\| {2}NVIDIA RTX 4090/);
    expect(io.output).toMatch(/Read from \.\/models\//);
    expect(io.output).toMatch(/^Q4_K_M\s+4\.83/m);
  });

  it("reads a mixture of experts out of a file", () => {
    const { code, io } = invoke(["check", "./moe.gguf", "-d", "4090", "--ctx", "4k"], {
      "./moe.gguf": moeGgufBuilder().build(),
    });
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/55M total \/ 16M active/);
  });

  it("takes the path from --gguf when the name says nothing", () => {
    const { code, io } = invoke(["check", "--gguf", "checkpoint.bin", "-d", "4090"], {
      "checkpoint.bin": llamaGgufBytes(),
    });
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/Read from checkpoint\.bin/);
  });

  it("still reports a model that does not fit, and exits 1", () => {
    const { code, io } = withLlama(["check", LLAMA_PATH, "-d", "rtx-3060-12gb", "--ctx", "128k"]);
    expect(code).toBe(EXIT_DOES_NOT_FIT);
    expect(io.output).toMatch(/DOES NOT FIT/);
    expect(io.output).toMatch(/Partial offload/);
  });
});

describe("vramfit check <path> diagnostics", () => {
  it("names a file it cannot open", () => {
    const { code, io } = invoke(["check", "./missing.gguf", "-d", "4090"]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/Cannot read \.\/missing\.gguf/);
    expect(io.stdout).toEqual([]);
  });

  it("passes the reader's diagnostic through with the path in front of it", () => {
    const { code, io } = invoke(["check", "./notes.gguf", "-d", "4090"], {
      "./notes.gguf": new TextEncoder().encode("just some notes about models"),
    });
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/\.\/notes\.gguf: not a GGUF file: it starts with "just"/);
  });

  it("refuses a path whose format it does not read", () => {
    const { code, io } = invoke(["check", "./model.safetensors", "-d", "4090"]);
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/is not a GGUF file\. Pass a \.gguf path/);
  });

  it("refuses a GGUF whose metadata is missing the shape it needs", () => {
    const bytes = new GgufBuilder()
      .kv("general.architecture", str("mystery"))
      .tensor("token_embd.weight", [64, 128], 1)
      .build();
    const { code, io } = invoke(["check", "./mystery.gguf", "-d", "4090"], {
      "./mystery.gguf": bytes,
    });
    expect(code).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/no numeric "mystery\.block_count" in its metadata/);
  });
});
