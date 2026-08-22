# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Compute buffer no longer scales with `--batch`.** `n_ubatch` counts the
  tokens in one graph pass across every sequence, not per sequence, so
  `check llama-3.1-8b -d 4090 --ctx 4k -b 32` reported a 4.89 GiB compute
  buffer and refused a configuration that needs ~21.5 GiB on a 24 GiB card.
- **`--efficiency` and `--prefill-efficiency` apply to the device only.** They
  were also raising the system-RAM side of a partial offload, inflating a
  70B's blended decode estimate by 81% when a GPU measurement was passed in.
- **Time to first token is labelled with the prompt it was computed from**,
  not with the context length, and `promptTokens` is exposed in JSON.
- **Context lengths print in binary K**, matching the parser, so the largest
  context that fits can be typed back in; the fractional form truncates
  rather than rounding up.
- **`--vram` is the usable budget**, not another figure to scale by the
  device's usable fraction -- which is what the Apple entries advertise it as.
- **Value-less flags no longer swallow the next argument**, so
  `check --json <model>` and `--json models` work; a positional the command
  has no use for is now refused rather than ignored.
- **A malformed `--version` or `--help` exits 2 with a message** instead of
  escaping as an uncaught exception with a stack trace and exit 1, the code
  that means "does not fit".
- **A model spec's parameter count is reconciled with its architecture.** A
  1000x typo in `totalParams` was schema-valid and produced a plausible-looking
  weight figure and a green verdict; `computeWeightBytes` no longer clamps the
  degenerate case into silence either.
- **User specs are refused when they carry control characters** -- an ANSI
  sequence in a device name could forge a verdict line in the report -- or
  shape fields orders of magnitude beyond anything published, which used to
  cost seconds of CPU and gigabytes of RSS.
- **The invalid-JSON message no longer quotes the file's contents**, which
  echoed the head of whatever file was named into the log.
- **`best` marks rows whose offloaded remainder needs more system RAM than
  assumed**, instead of printing a decode speed for a configuration that
  cannot load; `offloadFeasible` and `systemRamRequiredBytes` are in the JSON.
- **`npm run typecheck` really does cover `scripts/`**: the files were listed
  in `tsconfig.json` but silently discarded without `allowJs`.
- **README** documents `best`'s exit 1, the `--json` payload, and the
  behaviour of `--vram`, `--efficiency` and `-q`.

### Added

- **`vramfit check ./model.gguf`** — read a real checkpoint instead of the
  bundled database. The streaming header reader pulls one window at a time and
  skips what it does not need, so a 40 GB file costs a few hundred kilobytes
  of reads; the model's shape, attention flavour (GQA, MLA, sliding window)
  and expert geometry come from the metadata, the parameter count is summed
  from the shape table, and the bits per weight are measured from the ggml
  type of every tensor — separately for the blocks, the embedding table and
  the output head. Little-endian GGUF v2 and v3, every metadata value type
  including nested arrays. `--gguf <path>` for a file that is not named
  `.gguf`; `model.origin` and `model.from` in the JSON payload.
- `nativeQuant` on `ModelSpec`: a model released in a quantization of its own
  is checked in that format by default and ranked by it in `best`, and wider
  requantizations of it are left out. Both gpt-oss entries declare MXFP4, which
  the recommender previously ignored in favour of a Q8_0 twice its size.

## [0.1.0] - 2026-08-22

First release. Library and CLI, TypeScript/ESM, Node 20+, zero runtime
dependencies.

### Added

**Memory model**

- Weight sizing from a quantization table of *effective* bits per weight, not
  nominal names: `Q4_K_M` is 4.83 bpw, `Q6_K` is 6.56, `Q8_0` is 8.5. Fourteen
  entries across the GGUF k-quant, MXFP4, AWQ and GPTQ ecosystems.
- Embedding and output-head promotion (k-quant mixes store the logit matrix at
  `Q6_K`) and correct handling of tied embedding tables, which together bring
  predicted file sizes within 2% of the published GGUF downloads for every
  dense model in the test table.
- KV cache sized from `n_kv_heads`, with distinct formulas for DeepSeek-style
  multi-head latent attention and for the interleaved sliding-window attention
  used by Gemma 2/3 and gpt-oss.
- KV quantization at the real block layouts: `f16` and `bf16` 2 bytes/element,
  `q8_0` 1.0625, `q5_1` 0.75, `q5_0` 0.6875, `q4_1` 0.625, `q4_0` 0.5625.
- MoE models tracked as total parameters (memory) and active parameters per
  token (speed), derived from the architecture rather than a headline figure.
- Empirical per-family runtime-context overhead and a compute buffer that
  scales with the physical batch, with the flash-attention-off score matrix
  modelled separately.

**Throughput**

- Bandwidth-bound decode: `bandwidth x efficiency / (active weight bytes + KV
  bytes)`, with efficiency modelled as a per-family 16-bit MBU table plus a
  per-family dequantization penalty, interpolated linearly in bits. The pair
  reproduces published llama.cpp `tg128` figures for M2 Ultra F16/Q4_0 and RTX
  4090 Q4_0 within 1–5%.
- Compute-bound prefill from `2 x active_params` plus the attention FLOPs
  against device FP16 TFLOPs at a documented MFU, and time to first token.
- Both estimates carry a declared error band (±25% decode, ±40% prefill) that
  the report prints alongside the numbers.

**Partial offload and multi-GPU**

- `planOffload` reproduces llama.cpp's `--n-gpu-layers` placement: tail-first
  layer selection, per-layer KV slices, vocabulary tensors placed last.
- Blended read bandwidth as a byte-weighted harmonic mean of device and system
  RAM, with prefill blended by layer share.
- Homogeneous multi-GPU capacity, charging the runtime context and compute
  buffer once per device and refusing to pretend a layer split raises decode
  speed.

**Fit**

- `checkFit` composes the above into a verdict, headroom, utilisation, the
  exact largest context that fits (found by bisection), the highest-quality
  quantization that fits within one ecosystem, and contextual warnings.

**Data**

- Bundled database of 29 devices (NVIDIA RTX 3060 12GB–5090, A100 40/80GB,
  H100 80GB SXM5, L40S, AMD RX 7900 XTX, Apple M1–M4 Pro/Max/Ultra with the
  macOS wired-memory limit, generic CPU memory tiers) and 24 models (Llama
  3.1/3.2/3.3, Qwen 2.5, Qwen 3 including two MoEs, Mistral, Mixtral, Gemma
  2/3, Phi-4, DeepSeek-V2-Lite, gpt-oss 20B/120B), every entry carrying its
  source.
- A validator that enforces cross-field invariants at load — GQA head
  divisibility, MLA geometry, expert counts — and is reused verbatim for
  user-supplied `--model-json` / `--device-json`.

**CLI**

- `check`, `best`, `devices` and `models`, hand-rolled argument parsing with no
  dependency, aligned terminal report and `--json`.
- Exit codes 0 fits / 1 does not fit / 2 bad usage, so `check` can gate a
  deploy script.
- `run(argv, io)` is a pure function, which is how the CLI is tested without
  spawning a process.

**Project**

- CI on Node 20, 22 and 24 on Linux plus Node 22 on macOS and Windows, running
  lint, build, test and a smoke test of the built binary; a second job packs
  the tarball, installs it into a clean project and runs the installed binary
  and library from there.
- 242 tests across 12 files. No test touches the network.

[Unreleased]: https://github.com/johan-becker/vramfit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/johan-becker/vramfit/releases/tag/v0.1.0
