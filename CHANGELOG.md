# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Licensing is now documented in full.** vramfit is distributed under the
  [Apache License, Version 2.0](LICENSE) — OSI-approved open source. Use,
  modification and redistribution are free for any purpose and at any
  organisation size, including commercially and inside proprietary products.
  The conditions are the usual Apache ones: keep the copyright and licence
  notices, state significant changes, and ship the `NOTICE` file with any
  redistribution. The licence also carries an express patent grant from every
  contributor, with the standard termination on patent litigation. Licensing
  contact: jo_becker@mailbox.org.

### Added

- **`NOTICE`** — the attribution notice required by section 4(d) of the Apache
  License, carrying the copyright line and the trade mark reservation. It must
  be included with any redistribution of vramfit.
- **`TRADEMARKS.md`** — "vramfit" and its logo are unregistered trade marks of
  Johan Becker and are not licensed by the code licence. Nominative use is
  fine; shipping a fork, distribution, hosted service or product under the
  name, or claiming an official namespace, is not.
- **`DCO`** — the Developer Certificate of Origin 1.1, which contributions are
  now accepted under. `CONTRIBUTING.md` gains a contributor-terms section:
  `Signed-off-by` is required on every commit, and contributions are taken
  under the same Apache License, Version 2.0 the project ships under, with
  contributors keeping their own copyright.

### Fixed

- **`compare` and device names that end in a digit.** The `xN` count suffix was
  split off before the name was looked up, so `--devices rtx4090` became an
  unknown device `"rt"` -- a spelling `check -d` accepts, and six bundled
  aliases and four names have that shape. The whole token is now resolved as a
  name first.
- **`compare --gpus`.** The flag is in `compare`'s own vocabulary and was then
  overwritten by the per-entry default of 1, so `compare llama-3.3-70b
  --devices rtx-4090 --gpus 2` exited 1 where `check` with the same arguments
  exited 0. An entry with no count of its own now takes it.
- **`tie_word_embeddings` absent means tied**, matching `PretrainedConfig`'s
  default. Reading it as untied charged a second `vocab x hidden` matrix that
  is not on disk: +22.6% of the parameters of a gemma-2-2b, and a safetensors
  cross-check that then failed and blamed a vision tower for it.
- **A bare `sliding_window` windows every layer**, which is what the field
  means in transformers. Modelling it as Gemma 2's alternation put Mistral 7B's
  32K cache at 2.25 GiB against llama.cpp's 4.00 GiB -- the dangerous
  direction. `AttentionWindowSpec.fullAttentionEvery` is nullable for it, both
  readers say so in the notes, and a `layer_types` list that is not one
  repeating pattern is sized without a window instead of assuming a period.
- **`--explain`'s compute buffer** printed an expression with no per-device
  factor beside a value that has one: on a two-card fit the arithmetic came to
  half the number next to it.
- **The weight-count note has a direction.** Weight files smaller than the
  decoder derivation cannot be an extra head, and are no longer explained as
  one.
- **Nested GGUF arrays are bounded** (`maxArrayDepth`). A 1.2 MB header nested
  100,000 deep threw a `RangeError` past every handler, with no file name and
  no help line.
- **The Ollama block holds Modelfile commands only.** `OLLAMA_FLASH_ATTENTION`,
  `OLLAMA_KV_CACHE_TYPE` and the parallelism (now `OLLAMA_NUM_PARALLEL`, which
  is not a `PARAMETER`) are printed under their own heading, so the block can
  be pasted into a Modelfile as invited.
- **`OLLAMA_KV_CACHE_TYPE` is only printed for the types Ollama takes** -- f16,
  q8_0, q4_0. `--kv-quant q5_1`, `q5_0` and `q4_1` produced a value its daemon
  rejects; they now print none and say why.
- **`--quantization gguf` only when vLLM is serving the GGUF.** A safetensors
  directory or a repository id got a flag contradicting the note printed under
  it.
- **`--launcher <runtime>` filters the notes too**, so naming one runtime no
  longer leads with three notes about another and a flag name absent from the
  output.
- **`--markdown` keeps the runtime labels** above each launch fence.
- **`best` and `check` respect the format a file is already in.** A Q4_K_M GGUF
  was offered F16, BF16, Q8_0, Q6_K and both Q5_Ks and recommended F16; `best`
  also sized its row from the table's nominal 4.83 bits per weight while
  `check` sized the same file from its own 5.10, so the two commands reported
  totals 4.3% apart for one checkpoint.
- **`--color` / `--no-color` work on `fleet`, `devices` and `models`**, which
  the README's options table never scoped away from them, and a rejected flag
  is now quoted the way it was typed rather than after normalisation.
- **The `recommend` trade-off lines up with its column** at ten rows or more,
  which is the unlimited form the quickstart uses.
- **A directory pointed at the GGUF reader is named**, instead of a bare
  `EISDIR` with no path and no help line.
- **Two model sources are refused**, the way `--json` with `--markdown` is: a
  positional model with `--gguf` silently reported on the file and never
  mentioned dropping the other.

### Changed

- The GGUF transcripts in the README name a local artefact and say that they
  come from the synthetic header the test fixtures build, with the figure a
  real conversion of the same model produces beside them.

## [0.2.0] - 2026-08-22

The question was "will this model fit on this device". This release answers it
for the checkpoint you already have -- a GGUF file or a HuggingFace directory,
read from its own header -- and answers the three questions that follow it:
which of my machines should run this, what should I run on the machine I have,
and what exactly do I type to start it.

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
- **`vramfit check ./checkpoint/`** — read a HuggingFace `config.json` from a
  local path, with `--hf-config <path>` for one named directly. Maps every
  spelling the ecosystem uses for the same field (`num_local_experts` /
  `num_experts` / `n_routed_experts`, a shared-expert count or a combined
  width, `sliding_window` with and without `use_sliding_window`), descends
  into `text_config` for multimodal releases, and takes the true parameter
  count from a safetensors index or a single-file safetensors header beside
  it. A count that cannot be reconciled with the decoder the config describes
  is reported under *Notes* rather than charged to it.
- **`vramfit compare <model> --devices 4090,3090x2,m4-max`** — one model
  across several devices in one ranked table: fits, headroom, largest context
  and decode speed, what fits first and fastest first within that, with the
  rows that do not fit ordered by how close they came.
- **`vramfit recommend --device <d> [--use-case chat|code|long-context]`** —
  every bundled model that fits, ranked by `capability x quality x speed` with
  all three factors exposed in the JSON, and each row explained in a sentence.
  The use case sets the context to check at and the decode bar to clear, and
  nothing else: vramfit has no benchmark data and does not rank models by how
  good they are at anything.
- **`vramfit fleet --config fleet.json`** — heterogeneous machines against a
  list of models, one row per model and one column per machine, exiting 1 when
  any model fits nowhere. Model entries may be bundled names or paths.
- **`--launcher` and `--ngl` on `check`** — the exact flags a fit implies, for
  llama.cpp (`-ngl`, `-c`, `-fa`, `--cache-type-k/-v`, `--parallel`), Ollama
  (`num_gpu`, `num_ctx`, and the environment variables it keeps them in) and
  vLLM (`--gpu-memory-utilization` computed from the footprint over installed
  memory, `--max-model-len`, `--tensor-parallel-size`, `--kv-cache-dtype`).
  `--ngl` prints the layer count alone, for a shell substitution.
- **A memory-breakdown bar** under `check`'s memory table — weights, cache,
  overhead and free, each with its own block character as well as its own
  colour, so it reads with colour stripped. ANSI is emitted only for a
  terminal that wants it: `--color` / `--no-color`, then `NO_COLOR`, then
  `FORCE_COLOR`, then `TERM`, then whether stdout is a TTY.
- **`--explain`** — every headline figure with the formula and the numbers
  that produced it, switching KV formula with the model's attention flavour.
- **`--markdown`** on every command, for pasting into an issue. The cells are
  shared with the terminal renderer so the two cannot drift; the bar, the
  launch flags and the `--explain` block go inside fences, the last folded
  into a `<details>`. Refused together with `--json`.
- `nativeQuant` on `ModelSpec`: a model released in a quantization of its own
  is checked in that format by default and ranked by it in `best`, and wider
  requantizations of it are left out. Both gpt-oss entries declare MXFP4, which
  the recommender previously ignored in favour of a Q8_0 twice its size.
- The smoke test covers the new commands and formats against the built
  binary, including that a pipe receives no escape sequences and that the
  memory bar's block characters survive on every supported platform.
- 464 tests across 23 files, up from 242 across 12. The GGUF and safetensors
  fixtures are built byte by byte in the test suite, so the readers are tested
  against real container layouts without a download: still no test touches the
  network.

### Changed

- **The README presents the expansion as first-class** rather than as
  subsections of the reference: section 6 reads a checkpoint, 7 is `compare`,
  `recommend` and `fleet`, 8 is the launch flags, 9 is the three output
  formats, and the reference moves down to 10. Every console block in it is a
  pasted run of the built binary.
- **`--help` says which flags belong to which command** -- `--explain`,
  `--launcher` and `--ngl` to `check` alone, `--json` and `--markdown` to all
  seven -- and what "does not fit" means for each of them.

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

[Unreleased]: https://github.com/johan-becker/vramfit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/johan-becker/vramfit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/johan-becker/vramfit/releases/tag/v0.1.0
