# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

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
