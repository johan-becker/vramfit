import { describe, expect, it } from "vitest";
import { EXIT_OK, EXIT_USAGE, run } from "../src/cli/index.js";
import { directoryOf, joinPath } from "../src/cli/source.js";
import { FakeIo } from "./fake-io.js";

/**
 * `vramfit check ./checkpoint/`, end to end.
 *
 * The directory below is the file listing of a real sharded Llama 3.1 8B
 * repository, minus the weights: a config.json, a safetensors index and
 * nothing else the command is allowed to need.
 */

const DIRECTORY = "./Meta-Llama-3.1-8B-Instruct";

const CONFIG = JSON.stringify({
  _name_or_path: "meta-llama/Meta-Llama-3.1-8B-Instruct",
  architectures: ["LlamaForCausalLM"],
  model_type: "llama",
  hidden_size: 4096,
  intermediate_size: 14_336,
  max_position_embeddings: 131_072,
  num_attention_heads: 32,
  num_hidden_layers: 32,
  num_key_value_heads: 8,
  tie_word_embeddings: false,
  torch_dtype: "bfloat16",
  vocab_size: 128_256,
});

const INDEX = JSON.stringify({
  metadata: { total_size: 16_060_530_688 },
  weight_map: {
    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    "model.layers.20.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    "lm_head.weight": "model-00004-of-00004.safetensors",
  },
});

function safetensorsBytes(header: Record<string, unknown>): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  return out;
}

function shardedRepo(): FakeIo {
  return new FakeIo({
    text: {
      [joinPath(DIRECTORY, "config.json")]: CONFIG,
      [joinPath(DIRECTORY, "model.safetensors.index.json")]: INDEX,
    },
    directories: [DIRECTORY],
  });
}

function invoke(io: FakeIo, argv: string[]) {
  return { code: run(argv, io), io };
}

describe("path helpers", () => {
  it("joins and splits with forward slashes on every platform", () => {
    expect(joinPath("./models/llama", "config.json")).toBe("./models/llama/config.json");
    expect(joinPath("./models/llama/", "config.json")).toBe("./models/llama/config.json");
    expect(joinPath("", "config.json")).toBe("config.json");
    expect(directoryOf("./models/llama/config.json")).toBe("./models/llama");
    expect(directoryOf("config.json")).toBe(".");
  });
});

describe("vramfit check <checkpoint directory>", () => {
  it("reads config.json out of a directory", () => {
    const { code, io } = invoke(shardedRepo(), ["check", DIRECTORY, "-d", "4090", "--ctx", "8k"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(
      /Read from \.\/Meta-Llama-3\.1-8B-Instruct\/config\.json {2}\(32 layers, 8 KV heads, parameters from model\.safetensors\.index\.json \(3 shards\)\)/,
    );
    expect(io.output).toMatch(/Meta-Llama-3\.1-8B-Instruct {2}\| {2}Q4_K_M {2}\| {2}NVIDIA RTX 4090/);
    expect(io.output).toMatch(/32 layers x 8 KV heads x 128, f16/);
    expect(io.errors).toBe("");
  });

  it("accepts a trailing separator without stat-ing anything", () => {
    const io = new FakeIo({
      text: { [joinPath(DIRECTORY, "config.json")]: CONFIG },
      // No directory entry: the trailing slash is enough to say what it is.
    });
    expect(run(["check", `${DIRECTORY}/`, "-d", "4090", "--ctx", "8k"], io)).toBe(EXIT_OK);
    expect(io.output).toMatch(/parameters derived from the architecture/);
  });

  it("takes the parameter count from the safetensors index", () => {
    const { io } = invoke(shardedRepo(), ["check", DIRECTORY, "-d", "4090", "--json"]);
    const payload = JSON.parse(io.output) as {
      model: Record<string, unknown>;
    };
    expect(payload.model["origin"]).toBe("huggingface");
    expect(payload.model["from"]).toBe("./Meta-Llama-3.1-8B-Instruct/config.json");
    // 16,060,530,688 bytes of bfloat16.
    expect(payload.model["totalParams"]).toBe(8_030_265_344);
  });

  it("takes it exactly from a single-file checkpoint's header", () => {
    const io = new FakeIo({
      text: { [joinPath(DIRECTORY, "config.json")]: CONFIG },
      binary: {
        [joinPath(DIRECTORY, "model.safetensors")]: safetensorsBytes({
          __metadata__: { format: "pt" },
          "model.embed_tokens.weight": { dtype: "BF16", shape: [128_256, 4096] },
          "lm_head.weight": { dtype: "BF16", shape: [128_256, 4096] },
          "model.norm.weight": { dtype: "BF16", shape: [4096] },
        }),
      },
      directories: [DIRECTORY],
    });
    // The header describes only the vocabulary tensors, which cannot be
    // reconciled with a 32-layer decoder: the architecture's count wins and
    // the difference is reported rather than swallowed.
    expect(run(["check", DIRECTORY, "-d", "4090", "--ctx", "8k"], io)).toBe(EXIT_OK);
    expect(io.output).toMatch(/parameters derived from the architecture/);
    expect(io.output).toMatch(/Notes/);
    expect(io.output).toMatch(/weight files hold 1050677248 parameters/);
    expect(io.closed).toEqual([joinPath(DIRECTORY, "model.safetensors")]);
  });

  it("assumes 16-bit weights when the config declares no dtype, and says so", () => {
    const io = new FakeIo({
      text: {
        [joinPath(DIRECTORY, "config.json")]: JSON.stringify({
          ...(JSON.parse(CONFIG) as Record<string, unknown>),
          torch_dtype: undefined,
        }),
        [joinPath(DIRECTORY, "model.safetensors.index.json")]: INDEX,
      },
      directories: [DIRECTORY],
    });
    expect(run(["check", DIRECTORY, "-d", "4090", "--ctx", "8k"], io)).toBe(EXIT_OK);
    expect(io.output).toMatch(/gives bytes, not/);
    expect(io.output).toMatch(/16-bit weights are/);
  });

  it("reads a config.json named directly, and via --hf-config", () => {
    const positional = shardedRepo();
    expect(
      run(["check", joinPath(DIRECTORY, "config.json"), "-d", "4090", "--ctx", "8k"], positional),
    ).toBe(EXIT_OK);
    expect(positional.output).toMatch(/parameters from model\.safetensors\.index\.json/);

    const flagged = shardedRepo();
    expect(
      run(["check", "--hf-config", joinPath(DIRECTORY, "config.json"), "-d", "4090"], flagged),
    ).toBe(EXIT_OK);
    expect(flagged.output).toMatch(/Read from \.\/Meta-Llama-3\.1-8B-Instruct\/config\.json/);
  });

  it("still reads a vramfit spec that happens to be a .json path", () => {
    // The two are told apart by looking: nLayers is vramfit's spelling,
    // num_hidden_layers is HuggingFace's.
    const spec = JSON.stringify({
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
    const io = new FakeIo({ text: { "./my-finetune.json": spec } });
    expect(run(["check", "./my-finetune.json", "-d", "4090", "--ctx", "4k"], io)).toBe(EXIT_OK);
    expect(io.output).toMatch(/My Finetune 13B {2}\| {2}Q4_K_M/);
  });

  it("ranks quantizations of a checkpoint with best", () => {
    const { code, io } = invoke(shardedRepo(), ["best", DIRECTORY, "-d", "3090", "--ctx", "8k"]);
    expect(code).toBe(EXIT_OK);
    expect(io.output).toMatch(/Read from \.\/Meta-Llama-3\.1-8B-Instruct\/config\.json/);
    expect(io.output).toMatch(/^Q4_K_M\s+4\.83/m);
  });
});

describe("vramfit check <checkpoint> diagnostics", () => {
  it("names a directory with no config.json in it", () => {
    const io = new FakeIo({ directories: ["./empty"] });
    expect(run(["check", "./empty", "-d", "4090"], io)).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/Cannot read \.\/empty\/config\.json/);
  });

  it("reports invalid JSON without echoing the file", () => {
    const io = new FakeIo({
      text: { [joinPath(DIRECTORY, "config.json")]: "AWS_SECRET=hunter2\n{" },
      directories: [DIRECTORY],
    });
    expect(run(["check", DIRECTORY, "-d", "4090"], io)).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/config\.json is not valid JSON/);
    expect(io.errors).not.toMatch(/hunter2/);
  });

  it("names the field a config is missing", () => {
    const io = new FakeIo({
      text: { [joinPath(DIRECTORY, "config.json")]: JSON.stringify({ hidden_size: 4096 }) },
      directories: [DIRECTORY],
    });
    expect(run(["check", DIRECTORY, "-d", "4090"], io)).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/config\.json: the config has no layer count/);
  });

  it("refuses a broken safetensors index rather than guessing a count", () => {
    const io = new FakeIo({
      text: {
        [joinPath(DIRECTORY, "config.json")]: CONFIG,
        [joinPath(DIRECTORY, "model.safetensors.index.json")]: JSON.stringify({ metadata: {} }),
      },
      directories: [DIRECTORY],
    });
    expect(run(["check", DIRECTORY, "-d", "4090"], io)).toBe(EXIT_USAGE);
    expect(io.errors).toMatch(/no positive metadata\.total_size/);
  });
});
