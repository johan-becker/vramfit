#!/usr/bin/env node
// End-to-end check of the built artifact.
//
// The unit tests call `run(argv, io)` in-process, which is the right place to
// test behaviour but cannot catch the things that only go wrong once the code
// is compiled and executed as a program: a missing dist/data copy, a broken
// relative import, an ESM/CJS mismatch, a shebang that never made it through
// tsc, a package.json that cannot be found from dist/cli. So this spawns the
// real binary and checks both what it prints and what it exits with.
//
// No network, no fixtures, no temporary files: it runs the same commands the
// README documents and asserts the documented exit codes.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const bin = join(root, "dist", "bin", "vramfit.js");

if (!existsSync(bin)) {
  console.error(`smoke: ${bin} does not exist. Run "npm run build" first.`);
  process.exit(1);
}

const cases = [
  {
    name: "check, fits",
    argv: ["check", "llama-3.1-8b", "--device", "4090", "--ctx", "8k"],
    exit: 0,
    expect: [/FITS/, /Weights\s+4\.62 GiB/, /KV cache\s+1\.00 GiB/, /Decode/],
  },
  {
    name: "check, does not fit, offloads",
    argv: ["check", "llama-3.3-70b", "-d", "4090", "--ctx", "8k", "--ram", "64"],
    exit: 1,
    expect: [/DOES NOT FIT/, /Partial offload/, /Layers in device memory\s+\d+ of 80/],
  },
  {
    name: "check, multi-GPU",
    argv: ["check", "llama-3.3-70b", "-d", "3090", "-g", "2", "--ctx", "8k"],
    exit: 0,
    expect: [/2 x NVIDIA RTX 3090/, /FITS/],
  },
  {
    name: "check, json",
    argv: ["check", "mixtral-8x7b", "-d", "h100", "--ctx", "8k", "--json"],
    exit: 0,
    json: (payload) => {
      if (payload.fits !== true) throw new Error("expected mixtral to fit an H100");
      if (payload.model.moe !== true) throw new Error("expected moe: true");
      const memory = payload.memory;
      const sum =
        memory.weightsBytes +
        memory.kvCacheBytes +
        memory.runtimeContextBytes +
        memory.activationBytes;
      if (Math.abs(sum - memory.totalBytes) > 1) {
        throw new Error(`memory terms sum to ${sum}, not ${memory.totalBytes}`);
      }
    },
  },
  {
    name: "best",
    argv: ["best", "qwen2.5-32b", "-d", "4090", "--ctx", "8k"],
    exit: 0,
    expect: [/Recommended: Q4_K_M/, /^Q4_K_M\s+4\.83/m],
  },
  { name: "devices", argv: ["devices"], exit: 0, expect: [/^rtx-4090\s+NVIDIA RTX 4090/m] },
  { name: "models", argv: ["models"], exit: 0, expect: [/^llama-3\.1-8b/m] },
  { name: "help", argv: ["--help"], exit: 0, expect: [/USAGE/] },
  { name: "version", argv: ["--version"], exit: 0, expect: [/^\d+\.\d+\.\d+/] },
  {
    name: "unknown model",
    argv: ["check", "llama-3.1-9b", "-d", "4090"],
    exit: 2,
    expectStderr: [/Did you mean: llama-3\.1-8b/],
  },
  {
    name: "unknown option",
    argv: ["check", "llama-3.1-8b", "-d", "4090", "--ctxx", "8k"],
    exit: 2,
    expectStderr: [/Unknown option "--ctxx"/],
  },
  {
    // A bad --version used to escape the CLI's own error handling and be
    // reported by Node as an uncaught exception with exit 1 -- the code that
    // means "does not fit" to a deploy gate.
    name: "misused --version is a usage error, not a crash",
    argv: ["-v", "check"],
    exit: 2,
    expectStderr: [/^vramfit: /m],
    rejectStderr: [/^\s+at /m, /UsageError:/],
  },
  {
    name: "flag before the model name",
    argv: ["check", "--json", "llama-3.1-8b", "-d", "4090", "--ctx", "8k"],
    exit: 0,
    json: (payload) => {
      if (payload.model.id !== "llama-3.1-8b") {
        throw new Error(`--json before the model lost it: ${JSON.stringify(payload.model)}`);
      }
    },
  },
];

let failures = 0;

for (const testCase of cases) {
  const result = spawnSync(process.execPath, [bin, ...testCase.argv], { encoding: "utf8" });
  const problems = [];

  if (result.error) problems.push(`failed to spawn: ${result.error.message}`);
  if (result.status !== testCase.exit) {
    problems.push(`exit ${result.status}, expected ${testCase.exit}`);
  }
  for (const pattern of testCase.expect ?? []) {
    if (!pattern.test(result.stdout)) problems.push(`stdout did not match ${pattern}`);
  }
  for (const pattern of testCase.expectStderr ?? []) {
    if (!pattern.test(result.stderr)) problems.push(`stderr did not match ${pattern}`);
  }
  for (const pattern of testCase.rejectStderr ?? []) {
    if (pattern.test(result.stderr)) problems.push(`stderr matched ${pattern}, which it must not`);
  }
  if (testCase.json) {
    try {
      testCase.json(JSON.parse(result.stdout));
    } catch (cause) {
      problems.push(`json: ${cause.message}`);
    }
  }

  if (problems.length > 0) {
    failures++;
    console.error(`FAIL  ${testCase.name}  (vramfit ${testCase.argv.join(" ")})`);
    for (const problem of problems) console.error(`        ${problem}`);
    if (result.stderr) console.error(`        stderr: ${result.stderr.trim()}`);
  } else {
    console.log(`ok    ${testCase.name}`);
  }
}

// The library entry point has to load from the build too, and it is the one
// that reads the bundled JSON relative to its own compiled location.
const library = await import(new URL("../dist/index.js", import.meta.url));
if (library.listModels().length < 20 || library.listDevices().length < 20) {
  console.error("FAIL  library entry point: bundled database did not load from dist");
  failures++;
} else {
  console.log("ok    library entry point");
}

if (failures > 0) {
  console.error(`\nsmoke: ${failures} of ${cases.length + 1} checks failed`);
  process.exit(1);
}
console.log(`\nsmoke: ${cases.length + 1} checks passed`);
