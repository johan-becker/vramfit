![vramfit banner](docs/assets/banner.svg)

[![CI](https://github.com/johan-becker/vramfit/actions/workflows/ci.yml/badge.svg)](https://github.com/johan-becker/vramfit/actions/workflows/ci.yml)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Node](https://img.shields.io/badge/node-%E2%89%A5%2020-339933?style=flat&logo=node.js&logoColor=white)](https://nodejs.org/)
[![runtime dependencies](https://img.shields.io/badge/runtime%20dependencies-0-1f6feb?style=flat)](package.json)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat)](LICENSE)

`vramfit` answers the most-asked question in local LLM self-hosting — *will
this model run on my machine, and how fast?* — with arithmetic instead of
forum guesswork.

I built it because every VRAM calculator I tried was wrong in a way that cost
gigabytes: it treated `Q4_K_M` as four bits per weight, or sized the KV cache
from the query-head count instead of the KV-head count, or forgot that the
CUDA runtime takes 0.7 GiB before a single tensor is allocated. Every formula
here is written down with its assumption, unit-tested against a hand-computed
value, and — where external ground truth exists — calibrated against published
GGUF file sizes and llama.cpp benchmark tables.

It answers the question about the checkpoint you already have, not only about
the models it happens to know: point `check` at a GGUF file or a HuggingFace
directory and the parameter count and the bits per weight are measured from
that file's own tables. And it answers the three questions that follow the
first one — which of my machines should run this (`compare`), what should I
run on the machine I have (`recommend`), and what exactly do I type to start
it (`--launcher`). TypeScript, ESM, Node 20+, library and CLI, zero runtime
dependencies.

## 1. The problem

You have 24 GB of VRAM and you want to run a 70B at 32K context. Somebody on a
forum says it fits. Somebody else says you need two cards. The model card
quotes a parameter count, the GGUF repo quotes a file size, and neither tells
you what the KV cache costs at your context length or what happens to your
tokens per second when three layers spill into system RAM.

Every one of these is a real mistake in a widely used calculator, and each is
worth gigabytes:

| Mistake | What it costs |
| --- | --- |
| Treating `Q4_K_M` as 4 bits per weight | 18% low on Llama 3.1 8B — the mix is 4.83 bpw |
| Sizing the KV cache from `n_heads` instead of `n_kv_heads` | Up to **8x** too large on any modern GQA model |
| Ignoring that k-quants promote the output head to Q6_K | 0.11 GB on an 8B — the head alone is 0.43 GB of a 4.96 GB file |
| Charging a tied embedding table twice | 12% of a Llama 3.2 3B file |
| Ignoring interleaved sliding-window attention | 5x too large on Gemma 3 at 32K |
| Ignoring MLA on DeepSeek | 7x too large |
| Forgetting the ~0.7 GiB CUDA runtime context | The difference between "fits" and OOM at 95% utilisation |
| Sizing MoE speed by total parameters | Mixtral decodes like a 13B, not a 47B |

`vramfit` gets all eight right, and says out loud which of its numbers are
arithmetic and which are estimates with an error band.

## 2. Quickstart

```sh
npx vramfit check llama-3.1-8b --device 4090 --ctx 32k   # no install
npx vramfit check ./model.gguf --device 4090 --ctx 32k   # or read the file
npm install -g vramfit                                    # CLI
npm install vramfit                                       # library
```

`check` exits **0** when the configuration fits, **1** when it does not and
**2** on bad usage, so it can gate a deploy script:

```sh
vramfit check "$MODEL" --device 4090 --ctx 32k --json > plan.json \
  || { echo "won't fit, refusing to deploy"; exit 1; }
```

The other three questions, in the order people ask them:

```sh
vramfit compare llama-3.3-70b --devices 4090,4090x2,a100-80  # which machine
vramfit recommend --device 4090 --use-case code              # which model
vramfit check llama-3.3-70b -d 4090 --ctx 8k --launcher      # what to type
```

## 3. A worked example

Everything below is real output from the build in this repository, pasted
unedited.

```console
$ vramfit check llama-3.1-8b --device 4090 --ctx 32k
Llama 3.1 8B  |  Q4_K_M  |  NVIDIA RTX 4090
===========================================

FITS  -  9.47 GiB of 24.00 GiB used, 14.53 GiB free (39% utilised)

Memory
  Weights           4.62 GiB  8.03B params at 4.94 effective bits/weight
  KV cache          4.00 GiB  32K tokens, 32 layers x 8 KV heads x 128, f16
  Runtime context   0.70 GiB  cuda-consumer driver and kernels
  Compute buffer    0.15 GiB  activations, logits and graph scratch
  Total             9.47 GiB  at Q4_K_M
  Available        24.00 GiB  NVIDIA RTX 4090

  ███████████▓▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  39% of 24.00 GiB
  █ weights 4.62   ▓ KV 4.00   ▒ overhead 0.85   ░ free 14.53   (GiB)

Capacity
  Largest context that fits    128K  the architecture's own maximum
  Best quant that fits at 32K   F16  14.96 GiB of weights, 39.2 tok/s

Speed (estimates: decode +/-25%, prefill +/-40%)
  Decode               67.4 tok/s  at 32K context, 1 sequence
  Prefill              2450 tok/s  23.6 GFLOP per prompt token
  Time to first token      13.4 s  for a prompt of 32K tokens
```

The same question for every quantization at once, with the largest context and
the decode speed each one leaves room for:

```console
$ vramfit best qwen2.5-32b -d 4090 --ctx 8k
Qwen2.5 32B  |  NVIDIA RTX 4090
===============================

Quant     bpw    Weights  Total at 8K  Fits  Max ctx        Decode
------  -----  ---------  -----------  ----  -------  ------------
F16     16.00  61.03 GiB    63.97 GiB    no        -  1.43 tok/s *
BF16    16.00  61.03 GiB    63.97 GiB    no        -  1.43 tok/s *
Q8_0     8.50  32.42 GiB    35.37 GiB    no        -    3.75 tok/s
Q6_K     6.56  25.02 GiB    27.97 GiB    no        -    8.73 tok/s
Q5_K_M   5.67  21.71 GiB    24.65 GiB    no     5.3K    18.9 tok/s
Q5_K_S   5.52  21.15 GiB    24.10 GiB    no     7.6K    19.2 tok/s
Q4_K_M   4.83  18.58 GiB    21.53 GiB   yes    17.8K    27.8 tok/s
Q4_K_S   4.57  17.61 GiB    20.56 GiB   yes    21.7K    29.0 tok/s
Q4_0     4.55  17.54 GiB    20.48 GiB   yes    22.0K    29.1 tok/s
Q3_K_M   3.91  15.07 GiB    18.02 GiB   yes    31.9K    32.8 tok/s
Q2_K     3.35  12.91 GiB    15.86 GiB   yes    40.5K    37.6 tok/s

Rows that do not fit show the decode speed with as many layers as possible
offloaded to system RAM, which is what you would actually get.

* the offloaded remainder needs more system RAM than the 32.00 GiB assumed
here, so that row would not load at all. Say what you have with --ram.

Recommended: Q4_K_M -- highest quality that fits at 8K, 27.8 tok/s.
  The default recommendation: best quality-per-byte in the GGUF lineup for
  most models.
```

## 4. How it works

### 4.1 Weights

```
bytes = params x bits_per_weight / 8
```

with three corrections that decide whether the answer is useful. GGUF k-quants
are **not** their nominal width — a `Q4_K_M` stores super-blocks of 256 weights
with two levels of scales and promotes selected tensor types, landing at 4.83
bits per weight. Every k-quant mix promotes the logit matrix to `Q6_K`. And a
tied embedding table is stored once, not twice.

Predicted file size against the GGUF you would actually download, next to the
naive `params x nominal_bits / 8`:

| Model | Quant | Published | `vramfit` | Naive |
| --- | --- | ---: | ---: | ---: |
| Llama 3.2 3B | Q4_K_M | 2.02 GB | 2.02 GB (+0.2%) | 1.61 GB (−20.5%) |
| Llama 3.1 8B | Q4_K_M | 4.92 GB | 4.96 GB (+0.9%) | 4.02 GB (−18.4%) |
| Llama 3.1 8B | Q8_0 | 8.54 GB | 8.53 GB (−0.1%) | 8.03 GB (−6.0%) |
| Llama 3.3 70B | Q4_K_M | 42.52 GB | 42.82 GB (+0.7%) | 35.28 GB (−17.0%) |
| Mistral 7B v0.3 | Q4_K_M | 4.37 GB | 4.41 GB (+0.8%) | 3.62 GB (−17.1%) |
| Qwen2.5 32B | Q4_K_M | 19.85 GB | 19.95 GB (+0.5%) | 16.38 GB (−17.5%) |
| Gemma 2 9B | Q4_K_M | 5.76 GB | 5.78 GB (+0.3%) | 4.62 GB (−19.8%) |
| Phi-4 14B | Q4_K_M | 9.05 GB | 8.96 GB (−1.0%) | 7.33 GB (−19.0%) |

Published sizes are the file listings of the `bartowski` conversions; the full
table lives in `test/gguf-sizes.test.ts`, which asserts 2% on every dense row.

### 4.2 KV cache

```
bytes = 2 x n_layers x n_kv_heads x head_dim x ctx x batch x bytes_per_element
        ^-- K and V
```

`n_kv_heads`, never `n_heads`. Llama 3.3 70B has 64 query heads and 8 KV heads;
using 64 makes the cache come out exactly eight times too large, which turns
"a 70B at 32K fits in 48 GB" into "you need an H100". Two architectures need a
different formula entirely:

- **Multi-head Latent Attention** (DeepSeek V2/V3) caches one compressed latent
  plus a decoupled RoPE key, shared across every head:
  `n_layers x (kv_lora_rank + qk_rope_head_dim) x ctx x batch x bpe`. No factor
  of two and no head count — 576 elements per token per layer instead of 4096.
- **Sliding-window attention** caps a local layer at `window_size` tokens.
  Gemma 2/3 and gpt-oss *interleave*, keeping every Nth layer global; a
  checkpoint that states a window and no period at all (Mistral, Phi-3) windows
  every layer, and is modelled that way rather than assumed to alternate.

At 32K tokens, batch 1, `f16` — measured against what the naive `n_heads`
formula would have charged:

| Model | Attention | `vramfit` | Naive `n_heads` |
| --- | --- | ---: | ---: |
| Llama 3.3 70B | GQA 64→8 | 10.00 GiB | 80.00 GiB (8.0x) |
| Qwen2.5 7B | GQA 28→4 | 1.75 GiB | 12.25 GiB (7.0x) |
| Gemma 3 27B | GQA + SWA, 52 of 62 layers windowed at 1024 | 2.91 GiB | 31.00 GiB (10.7x) |
| DeepSeek-V2-Lite | MLA | 0.95 GiB | 6.75 GiB (7.1x) |

KV quantization uses the exact block layouts rather than the nominal width:
`f16` = 2 bytes/element, `q8_0` = 1.0625, `q4_0` = 0.5625.

### 4.3 Mixture of experts

Total parameters drive **memory**; active parameters per token drive **speed**.
Mixtral 8x7B occupies 46.70B parameters of memory and multiplies 12.88B of them
per token, so it needs the VRAM of a 47B and decodes at roughly the speed of a
13B. Both figures are derived from the architecture fields, not copied from a
headline.

### 4.4 Overheads

Two empirical terms, labelled as such in the report:

- **Runtime context** — the CUDA primary context, kernel image and cuBLAS
  workspaces: ~0.7 GiB consumer, ~1.0 GiB datacenter, ~0.5 GiB Metal. Charged
  once *per device*, which is why the second GPU in a pair gives you less than
  its sticker capacity.
- **Compute buffer** — activations, FP32 logits and graph scratch, scaling with
  the physical batch (`--ubatch`) rather than the context, plus the
  `ubatch x ctx x n_heads` score matrix when flash attention is off.

Assume ±0.3 GiB on the first and ±50% on the second.

### 4.5 Throughput

Decode is memory-bandwidth bound: every active weight is read once and
multiplied against a single vector, roughly one FLOP per byte.

```
tokens/second = bandwidth x efficiency / (active_weight_bytes + kv_bytes)
```

The KV term is why long conversations get slower — attention re-reads the whole
cache for every token it emits.

Efficiency is two tables rather than one constant, because one constant cannot
fit the data. At 16 bits decode is purely bandwidth-bound and reaches 0.70–0.85
of peak, depending on the backend. At 4 bits it does not, because unpacking the
weights is real work and how much it costs depends on the backend's kernels;
the penalty is interpolated linearly in bits between those two anchors.
Calibrated against published llama.cpp `tg128` results for the original
LLaMA 7B, the model those tables are quoted for:

| Device | Quant | Measured | `vramfit` |
| --- | --- | ---: | ---: |
| M2 Ultra | F16 | 42.75 tok/s | 42.2 (−1.4%) |
| M2 Ultra | Q4_0 | 88.64 tok/s | 92.7 (+4.6%) |
| RTX 4090 | Q4_0 | 152.2 tok/s | 155.0 (+1.8%) |

The interesting row is the middle one: Metal only gets 2.07x from F16 → Q4_0
where the byte counts imply 3.5x, and reproducing that gap is exactly what the
per-family dequantization table is for.

Prefill is compute bound — `2 x active_params` FLOPs per token for the matmuls
plus the attention matmuls, against the device's FP16 TFLOPs at a documented
model-FLOPs-utilisation figure.

**These are estimates**, and the report says so on the same line: ±25% on
decode, ±40% on prefill. They answer "which order of magnitude, and would a
different quant help", not "what will my benchmark print".

## 5. When it does not fit

"It does not fit" is almost never the end of the story, so `vramfit` models
llama.cpp's `--n-gpu-layers` placement: as many transformer blocks as fit go
into device memory, each taking its own slice of the KV cache, filled from the
tail, with the vocabulary tensors placed only once every layer is already
resident.

```console
$ vramfit check llama-3.3-70b -d 3090 --ctx 8k --ram 64
Llama 3.3 70B  |  Q4_K_M  |  NVIDIA RTX 3090
============================================

DOES NOT FIT  -  43.39 GiB needed, 24.00 GiB available, 19.39 GiB short

Memory
  Weights          39.88 GiB  70.55B params at 4.86 effective bits/weight
  KV cache          2.50 GiB  8K tokens, 80 layers x 8 KV heads x 128, f16
  Runtime context   0.70 GiB  cuda-consumer driver and kernels
  Compute buffer    0.31 GiB  activations, logits and graph scratch
  Total            43.39 GiB  at Q4_K_M
  Available        24.00 GiB  NVIDIA RTX 3090

  ███████████████████████████████▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚  43.39 GiB needed, 24.00 GiB available
  █ weights 39.88   ▓ KV 2.50   ▒ overhead 1.01   ▚ over 19.39   (GiB)

Capacity
  Largest context that fits   none  Q4_K_M, batch 1, f16 cache
  Best quant that fits at 8K  none  try a smaller model, a longer offload or more memory

Partial offload
  Layers in device memory   44 of 80  vocabulary tensors in system RAM
  Left in system RAM       19.84 GiB  of 64.00 GiB assumed available
  Blended read bandwidth    175 GB/s  harmonic mean; the device alone reads at 936 GB/s

Speed (estimates: decode +/-25%, prefill +/-40%)
  Decode                2.05 tok/s  blended across 44 device and 36 host layers
  Prefill               8.65 tok/s  149.7 GFLOP per prompt token
  Time to first token  15 min 47 s  for a prompt of 8K tokens
```

The blended bandwidth is a harmonic mean weighted by the bytes read from each
side — harmonic, because the slow side dominates the time. Moving 10% of a
model off a 1008 GB/s RTX 4090 into 89.6 GB/s DDR5 does not cost 10% of the
speed: the blended figure falls to 498 GB/s, less than half.

`--gpus N` splits across an identical set of devices instead. Weights and cache
are shared, but each device pays the runtime context and compute buffer in
full, and a layer split does not raise decode speed — so the report says that
rather than quietly multiplying:

```console
$ vramfit check llama-3.3-70b -d 3090 -g 2 --ctx 8k
Llama 3.3 70B  |  Q4_K_M  |  2 x NVIDIA RTX 3090
================================================

FITS  -  44.39 GiB of 48.00 GiB used, 3.61 GiB free (92% utilised)
...
  Runtime context   1.40 GiB  cuda-consumer driver and kernels, x2 devices
...
Notes
  - Each of the 2 devices pays the runtime context and compute buffer in full,
    and a layer split does not raise decode speed -- extra devices buy
    capacity, not tokens per second.
```

## 6. Point it at the checkpoint

The bundled database is a convenience for the models everyone runs. The file
already on your disk is the one you actually have a question about, so `check`
takes a path wherever it takes a name — a GGUF file, or a HuggingFace
checkpoint directory — and reads the model out of that file's own tables
instead of looking it up. Nothing is downloaded, no `transformers` import
happens, and neither reader touches the network.

### 6.1 A GGUF file

```console
$ vramfit check ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -d 4090 --ctx 32k
Meta Llama 3.1 8B Instruct  |  Q4_K_M  |  NVIDIA RTX 4090
=========================================================

Read from ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  (GGUF v3, 292 tensors, llama architecture, Q4_K_M)

FITS  -  9.67 GiB of 24.00 GiB used, 14.33 GiB free (40% utilised)

Memory
  Weights           4.82 GiB  8.03B params at 5.15 effective bits/weight
  KV cache          4.00 GiB  32K tokens, 32 layers x 8 KV heads x 128, f16
  Runtime context   0.70 GiB  cuda-consumer driver and kernels
  Compute buffer    0.15 GiB  activations, logits and graph scratch
  Total             9.67 GiB  at Q4_K_M
  Available        24.00 GiB  NVIDIA RTX 4090

  ███████████▓▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  40% of 24.00 GiB
  █ weights 4.82   ▓ KV 4.00   ▒ overhead 0.85   ░ free 14.33   (GiB)

Capacity
  Largest context that fits    128K  the architecture's own maximum
  Best quant that fits at 32K   F16  14.96 GiB of weights, 39.2 tok/s

Speed (estimates: decode +/-25%, prefill +/-40%)
  Decode               66.0 tok/s  at 32K context, 1 sequence
  Prefill              2450 tok/s  23.6 GFLOP per prompt token
  Time to first token      13.4 s  for a prompt of 32K tokens
```

Every number in that report is **measured, not looked up**. The parameter
count is summed from the file's own shape table — all 292 tensors, including
the norms a published "8B" rounds away — and the bits per weight are computed
from the ggml type of each tensor, separately for the transformer blocks
(`Q4_K` at 144 bytes per 256 weights), the token embedding table and the
output head (`Q6_K` at 210). Feed those back into `computeWeightBytes` and the
result comes back to the byte, because the prediction *is* the file.

The reader is streaming: it pulls a window at a time and skips everything it
does not need — a tokenizer's 128k-entry token list and all of the tensor
data — so checking a 40 GB checkpoint reads a few hundred kilobytes of it.
Little-endian GGUF v2 and v3, every metadata value type including nested
arrays, GQA/MLA/sliding-window attention and MoE expert geometry. A
big-endian file, a v1 file, a truncated one or an unknown ggml type each fail
by name rather than producing a plausible wrong answer.

A checkpoint that is not named `.gguf` — an Ollama blob, a file a downloader
renamed — is read with `--gguf <path>`, which skips the extension check and
changes nothing else:

```console
$ vramfit check --gguf ./blobs/sha256-6a0746a1ec1aa3d1d1a7b1c7b1ee1eae -d 4090 --ctx 8k
Meta Llama 3.1 8B Instruct  |  Q4_K_M  |  NVIDIA RTX 4090
=========================================================

Read from ./blobs/sha256-6a0746a1ec1aa3d1d1a7b1c7b1ee1eae  (GGUF v3, 292 tensors, llama architecture, Q4_K_M)

FITS  -  6.67 GiB of 24.00 GiB used, 17.33 GiB free (28% utilised)
...
```

### 6.2 A HuggingFace checkpoint

A HuggingFace checkpoint works the same way — point at the directory, or at
its `config.json` with `--hf-config`:

```console
$ vramfit check ./Qwen3-30B-A3B/ -d m4-max --ctx 32k
Qwen3-30B-A3B  |  Q4_K_M  |  Apple M4 Max
=========================================

Read from ./Qwen3-30B-A3B/config.json  (48 layers, 4 KV heads, parameters from model.safetensors.index.json (2 shards))

FITS  -  20.80 GiB of 96.00 GiB used, 75.20 GiB free (22% utilised)

Memory
  Weights          17.23 GiB  30.53B total / 3.04B active, 4.85 effective bits/weight
  KV cache          3.00 GiB  32K tokens, 48 layers x 4 KV heads x 128, f16
  Runtime context   0.50 GiB  metal driver and kernels
  Compute buffer    72.6 MiB  activations, logits and graph scratch
  Total            20.80 GiB  at Q4_K_M
  Available        96.00 GiB  Apple M4 Max, 75% of 128 GiB wirable

  ██████████▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  22% of 96.00 GiB
  █ weights 17.23   ▓ KV 3.00   ▒ overhead 0.57   ░ free 75.20   (GiB)

Capacity
  Largest context that fits    40K  the architecture's own maximum
  Best quant that fits at 32K  F16  56.87 GiB of weights, 41.1 tok/s

Speed (estimates: decode +/-25%, prefill +/-40%)
  Decode               48.2 tok/s  at 32K context, 1 sequence
  Prefill               582 tok/s  19.0 GFLOP per prompt token
  Time to first token      56.3 s  for a prompt of 32K tokens

Notes
  - Apple M4 Max shares memory with the OS: only 75% is wirable for the GPU by
    default. Raise it with "sudo sysctl iogpu.wired_limit_mb=N" and pass the
    result as --vram.
```

The config is read from the path you give and nothing else is opened. The
parameter count comes from the safetensors index beside it
(`metadata.total_size` divided by the checkpoint's dtype) or, for a
single-file repository, exactly from its header's tensor shapes.

What makes this more than a field rename is that the fields that matter most
are the ones the ecosystem is least consistent about, and each way of getting
one wrong is worth gigabytes:

| Field | The trap |
| --- | --- |
| `num_key_value_heads` | Absent means *equal to the query heads*. Reading it as 1 undersizes a pre-GQA cache by the head count |
| `head_dim` | Absent means `hidden_size / num_attention_heads` — except on the models that state it and disagree |
| expert count | `num_local_experts` on Mixtral, `num_experts` on Qwen3-MoE, `n_routed_experts` on DeepSeek |
| shared experts | Qwen publishes a combined *width*, DeepSeek publishes a *count* |
| `sliding_window` | Set but inert on Qwen2 unless `use_sliding_window` is true — honouring it undersizes the cache |
| `sliding_window_pattern` / `layer_types` | Only these state the interleaving. A bare `sliding_window` windows *every* layer; assuming Gemma 2's alternation instead undersizes Mistral 7B's 32K cache by 44% |
| `tie_word_embeddings` | Absent means **true**: `PretrainedConfig` defaults it that way and `save_pretrained` writes only what differs, so the checkpoints that tie are the ones that never mention it |
| `text_config` | Multimodal releases nest the decoder inside it, with the vision tower beside it |

The last one has a consequence worth stating: when the weight files hold more
parameters than the decoder in the config accounts for — a vision tower, an
extra head — the decoder's own count is used and the difference is printed
under *Notes*, rather than charged to a model that does not exist.

## 7. Beyond one model on one device

Three commands for the questions that need more than one `check`.

### 7.1 compare — one model, several devices

One model, every device you might use, ranked. What fits comes before what
does not, then fastest first, because once a configuration fits, decode speed
is what you actually feel. `4090x2` means two of them.

```console
$ vramfit compare llama-3.3-70b --devices 4090,4090x2,a100-80,m3-ultra,3090x2 --ctx 8k
Llama 3.3 70B  |  Q4_K_M  |  8K context
=======================================

    Device                   Memory     Needed        Free  Fits  Max ctx      Decode
--  -------------------  ----------  ---------  ----------  ----  -------  ----------
->  NVIDIA A100 80GB      80.00 GiB  43.69 GiB   36.31 GiB   yes   124.1K  33.2 tok/s
    2 x NVIDIA RTX 4090   48.00 GiB  44.39 GiB    3.61 GiB   yes    19.5K  13.4 tok/s
    2 x NVIDIA RTX 3090   48.00 GiB  44.39 GiB    3.61 GiB   yes    19.5K  12.4 tok/s
    Apple M3 Ultra       384.00 GiB  43.19 GiB  340.81 GiB   yes     128K  8.27 tok/s
    NVIDIA RTX 4090       24.00 GiB  43.39 GiB  -19.39 GiB    no        -  2.06 tok/s

Rows that do not fit show the decode speed with as many layers as possible
offloaded to system RAM, which is what you would actually get.

Best: NVIDIA A100 80GB -- 33.2 tok/s at 8K with 36.31 GiB to spare, and room
for 124.1K of context.
```

Two 4090s hold the model and decode at 13.4 tok/s; one A100 holds it and
decodes 2.5x faster, because a layer split does not raise decode speed. The
M3 Ultra has 340 GiB spare and is the slowest thing in the table that fits.
Neither of those is obvious from a spec sheet.

A path works here too, and is read once for the whole table:

```sh
vramfit compare ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --devices 4090,3060 --ctx 8k
```

### 7.2 recommend — one device, every model

The inverse question, and the one people actually arrive with. Ranked by

```
score = capability x quality x speed
```

where *capability* is `log10` of the parameter count (scaling laws are
logarithmic — a 70B is one unit above a 7B, not ten times it), with a mixture
of experts entering as the geometric mean of total and active parameters;
*quality* is 1.0 for `Q4_K_M` and above and falls off below it; and *speed* is
decode against a bar the use case sets, capped at 1 — which is what stops the
ranking from simply naming the largest model that technically fits.

```console
$ vramfit recommend -d 4090 --use-case chat --limit 3
NVIDIA RTX 4090  |  chat  |  8K context
=======================================

 #  Model        Params  Quant       Total  Max ctx      Decode
--  -----------  ------  ------  ---------  -------  ----------
1.  Qwen2.5 32B  32.76B  Q4_K_M  21.53 GiB    17.8K  27.8 tok/s
    32.76B parameters at Q4_K_M, the best quality-per-byte in the GGUF lineup,
    and 27.8 tok/s, comfortably above the 15 tok/s chat wants.

2.  Qwen3 32B    32.76B  Q4_K_M  21.51 GiB    17.9K  27.8 tok/s
    32.76B parameters at Q4_K_M, the best quality-per-byte in the GGUF lineup,
    and 27.8 tok/s, comfortably above the 15 tok/s chat wants.

3.  Gemma 2 27B  27.23B  Q6_K    23.94 GiB       8K  25.4 tok/s
    27.23B parameters at Q6_K, which is effectively lossless, and 25.4 tok/s,
    comfortably above the 15 tok/s chat wants.

Ranked by parameter count on a log scale, times a quantization-quality factor,
times decode speed against 15 tok/s -- reading speed is 10-15 tokens a second,
so anything above that is buffered by your eyes rather than enjoyed. vramfit
has no benchmark data and does not rank models by how good they are at
anything.
```

`--use-case` sets two numbers and nothing else: the context to check at and
the decode speed to clear. `chat` is 8K and 15 tok/s, because reading speed is
10–15 tokens a second; `code` is 16K and 30 tok/s, because a completion is
hundreds of tokens you wait for in full; `long-context` is 32K and 8 tok/s,
because summarising is dominated by prefill. **vramfit has no benchmark data
and does not rank models by how good they are at anything** — the report says
so, on the page, every time.

### 7.3 fleet — several machines, several models

Heterogeneous machines against a list of models, from a file:

```json
{
  "ctx": 8192,
  "quant": "q4_k_m",
  "machines": [
    { "name": "workstation", "device": "4090", "gpus": 2, "ram": 128 },
    { "name": "laptop", "device": "m3-max", "ram": 128 },
    { "name": "rack", "device": "a100-80", "ram": 512 },
    { "name": "old-box", "device": "rtx-3060-12gb", "ram": 32 }
  ],
  "models": [
    "llama-3.1-8b",
    "qwen2.5-32b",
    "llama-3.3-70b",
    { "model": "gemma-3-27b", "ctx": 4096 },
    "qwen3-235b-a22b"
  ]
}
```

```console
$ vramfit fleet --config ./fleet.json
Fleet  |  4 machines  |  5 models
=================================

Model            Quant   Ctx  Weights+KV  workstation      laptop        rack     old-box  Served
---------------  ------  ---  ----------  -----------  ----------  ----------  ----------  ------
Llama 3.1 8B     Q4_K_M   8K    5.62 GiB    105 tok/s  31.8 tok/s   261 tok/s  37.6 tok/s     4/4
Qwen2.5 32B      Q4_K_M   8K   20.58 GiB   27.8 tok/s  8.38 tok/s  69.0 tok/s           -     3/4
Llama 3.3 70B    Q4_K_M   8K   42.38 GiB   13.4 tok/s  4.04 tok/s  33.2 tok/s           -     3/4
Gemma 3 27B      Q4_K_M   4K   16.19 GiB   34.7 tok/s  10.5 tok/s  85.9 tok/s           -     3/4
Qwen3 235B-A22B  Q4_K_M   8K  133.78 GiB            -           -           -           -     0/4

Machines
  Name         Device                Usable  System RAM
  -----------  --------------------  ------  ----------
  workstation  2 x NVIDIA RTX 4090   48 GiB     128 GiB
  laptop       Apple M3 Max          96 GiB     128 GiB
  rack         NVIDIA A100 80GB      80 GiB     512 GiB
  old-box      NVIDIA RTX 3060 12GB  12 GiB      32 GiB

1 of 5 models fit nowhere: Qwen3 235B-A22B. Run "vramfit check" against the
largest machine to see what a partial offload or a narrower quantization would
cost.

A dash means the model does not fit in that machine's device memory.
Weights+KV is what every machine holds in common; each also pays its own
runtime context and compute buffer, once per device. Decode figures are
estimates, +/-25%.
```

The cell is a decode figure rather than a tick, because "yes" and "yes at
4 tok/s" are different answers and the second one is usually a no. A model
entry is a name or a path, so a fleet file can mix the bundled database with
the checkpoints actually sitting on those machines. `fleet` exits **1** when
any model fits nowhere, which is what makes it useful in CI.

## 8. The flags to actually type

Everything else here answers *will it fit*. `--launcher` answers *then what do
I run*, which is the step where the arithmetic usually gets thrown away and
replaced by `-ngl 99` and a shrug. Nothing new is computed: the layer count,
the context and the memory fraction are already decided by the time the
verdict is printed.

```console
$ vramfit check llama-3.3-70b -d 4090 --ctx 8k --ram 64 --launcher
...
Launch
  llama.cpp
    llama-server -m <model.gguf> -ngl 44 -c 8192 -fa on

  Ollama  (Modelfile)
    FROM llama-3.3-70b
    PARAMETER num_gpu 44
    PARAMETER num_ctx 8192
    OLLAMA_FLASH_ATTENTION=1

  vLLM
    vllm serve <org/model> --max-model-len 8192 --gpu-memory-utilization 0.95 --quantization gguf

  - -ngl 44 leaves 36 of 80 layers on the CPU. That is the most that fits; a
    higher number will load and then run out of memory.
  - vLLM does not offload to system RAM: these flags describe the memory
    budget, not a configuration it can serve.
  - --gpu-memory-utilization is capped at 0.95: this deployment wants 181% of
    the card, and above 95% there is no room left for CUDA graph capture and
    allocator fragmentation.
  - vLLM's GGUF loader is experimental and single-file only; the usual path is
    to serve the safetensors checkpoint and let --quantization pick the
    kernel.
```

### 8.1 The three runtimes

The three of them spell the same three decisions differently, and each
translation has a trap in it:

- **`-ngl`** is `n_layer + 1` when the model is resident — llama.cpp counts one
  extra offloadable layer for the output tensors, which is what `-ngl 99`
  means spelled exactly — and the number of layers the offload plan actually
  placed when it is not. That number is a *ceiling*: a higher one loads and
  then runs out of memory partway through.
- **Ollama** takes the same two numbers as `num_gpu` and `num_ctx`, but keeps
  flash attention and the cache type in environment variables rather than in
  the Modelfile, so both are printed.
- **vLLM** does not take a layer count at all. It takes the fraction of each
  card it may occupy and fills whatever is left after the weights with KV
  blocks, so the figure is this deployment's footprint over the card's
  *installed* memory — rounded **up**, because rounding down asks for less
  than the plan needs, and capped at 0.95, because above that there is no room
  for CUDA graph capture and the server fails at startup.

The paths are kept apart on purpose: `-m` opens a GGUF and nothing else, vLLM
wants a repository or a directory of safetensors, and Ollama's `FROM` takes
either. Where there is nothing of the right kind to give — a model looked up
in the bundled database, as above — a placeholder is printed rather than a
path that would look right and fail. Check the file instead and the flags come
out ready to run:

```console
$ vramfit check ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -d 4090 --ctx 8k --launcher
...
Launch
  llama.cpp
    llama-server -m ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 33 -c 8192 -fa on

  Ollama  (Modelfile)
    FROM ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    PARAMETER num_gpu 33
    PARAMETER num_ctx 8192
    OLLAMA_FLASH_ATTENTION=1

  vLLM
    vllm serve ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --max-model-len 8192 --gpu-memory-utilization 0.28 --quantization gguf

  - vLLM's GGUF loader is experimental and single-file only; the usual path is
    to serve the safetensors checkpoint and let --quantization pick the
    kernel.
```

That `-ngl 33` is the model's 32 blocks plus the one llama.cpp counts for the
output tensors, and the vLLM fraction is this deployment's 6.67 GiB over the
card's installed 24 GiB.

### 8.2 The layer count on its own

`--ngl` prints the number and nothing else, which is the form a script wants:

```sh
llama-server -m ./model.gguf -c 8192 \
  -ngl "$(vramfit check ./model.gguf -d 4090 --ctx 8k --ngl)"
```

The exit code still reports the fit — **0** resident, **1** offloaded — so
under `set -e` a model that no longer fits stops the script instead of quietly
starting a server that decodes at 2 tok/s.

## 9. Presentation

Three things that exist so the output can be read, argued with and passed on.

### 9.1 The memory bar

The table has the numbers; the bar has the proportions, which is the part
people actually reason with — "the cache is half of it" is a decision, and
"4.00 GiB" is a figure you then have to divide. Every segment
carries its own block character as well as its own colour, so it survives
being pasted into an issue, piped into a file, or read with colour off:

```
  ███████████▓▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  39% of 24.00 GiB
  █ weights 4.62   ▓ KV 4.00   ▒ overhead 0.85   ░ free 14.53   (GiB)
```

When the model does not fit, the part past the card's capacity is redrawn as
overflow **in place** rather than appended, because those bytes are already
counted once — so the bar shows how far over it went instead of clamping at
full and looking the same at 1 GiB short and 40 GiB short.

Colour is off unless the output is going to a terminal that wants it:
`--color` / `--no-color` win outright, then `NO_COLOR`, then `FORCE_COLOR`,
then `TERM=dumb`, then whether stdout is a TTY. `vramfit check > report.txt`
produces a file with no escape sequences in it.

### 9.2 The formulas, with the numbers in them

`--explain` prints the formulas from section 4 with this model's own numbers
substituted, so the maths can be checked rather than taken:

```console
$ vramfit check llama-3.1-8b -d 4090 --ctx 32k --explain
...
Explain  (Llama 3.1 8B at Q4_K_M, 32K context, f16 cache)
  Weights = parameters x bits per weight / 8
    blocks      6,979,588,096 x 4.83 / 8        = 3.925 GiB
    token_embd  525,336,576 x 4.83 / 8          = 0.295 GiB
    output      525,336,576 x 6.56 / 8          = 0.401 GiB
    total       4.94 effective bits per weight  = 4.621 GiB

  KV cache = 2 x layers x KV heads x head dim x ctx x sequences x bytes/element
    all layers  2 x 32 x 8 x 128 x 32768 x 1 x 2  = 4.000 GiB
    Grouped-query attention: 8 KV heads, not the 32 query heads. Using the
    query count here is the 4x mistake this tool exists to avoid.

  Overheads (empirical: +/-0.3 GiB on the first, +/-50% on the second)
    runtime context  0.70 GiB x 1 device                                              = 0.700 GiB
    compute buffer   max(64 MiB, 512 x (4096 x 18 + 14336 x 6) x 2 + 1 x 128256 x 4)  = 0.153 GiB

  Total
    footprint  4.621 GiB + 4.000 GiB + 0.700 GiB + 0.153 GiB  = 9.474 GiB
    capacity   24.00 GiB x 1.00 usable x 1 device             = 24.000 GiB
    verdict    9.474 GiB <= 24.000 GiB                        = FITS

  Decode = bandwidth x efficiency / bytes read per token  (estimate, +/-25%)
    efficiency      0.75 at 16 bits, less the dequantization penalty at 0.92 of full  = 0.598
    bandwidth       1008 GB/s x 0.598                                                 = 603 GB/s
    read per token  4.64 GB active weights + 4.29 GB cache                            = 8.94 GB
    decode          603 GB/s / 8.94 GB                                                = 67.4 tok/s

  Prefill = device FLOP/s x MFU / FLOPs per prompt token  (estimate, +/-40%)
    FLOPs/token  2 x 7,504,658,432 active + 8.59 GFLOP of attention  = 23.60 GFLOP
    compute      165.2 TFLOP/s x 0.35 MFU (cuda-consumer)            = 57.8 TFLOP/s
    prefill      57.8e12 / 23.60e9                                   = 2450 tok/s
    first token  32,768 prompt tokens / 2450 tok/s                   = 13.4 s
```

Nothing is recomputed for this: the values printed are the ones the report
printed, and the expressions beside them are how those values arose. The KV
section switches formula with the architecture, so a DeepSeek model shows
`27 x (512 + 64) x ctx` and a Gemma 3 shows its global and windowed layers
apart.

### 9.3 Markdown

`--markdown` renders the same reports through pipes instead of spaces, for
pasting into an issue or a README. The head of
`check llama-3.1-8b -d 4090 --ctx 8k --markdown`, as it arrives on stdout:

````markdown
### Llama 3.1 8B — Q4_K_M — NVIDIA RTX 4090

**FITS** — 6.47 GiB of 24.00 GiB used, 17.53 GiB free (27% utilised)

```
███████████▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  27% of 24.00 GiB
█ weights 4.62   ▓ KV 1.00   ▒ overhead 0.85   ░ free 17.53   (GiB)
```

| Component | Size | Assumption |
| --- | ---: | --- |
| Weights | 4.62 GiB | 8.03B params at 4.94 effective bits/weight |
| KV cache | 1.00 GiB | 8K tokens, 32 layers x 8 KV heads x 128, f16 |
| Runtime context | 0.70 GiB | cuda-consumer driver and kernels |
| Compute buffer | 0.15 GiB | activations, logits and graph scratch |
| **Total** | **6.47 GiB** | at Q4_K_M |
| Available | 24.00 GiB | NVIDIA RTX 4090 |
````

The cells come from the same place the terminal table's do, so the two cannot
drift apart. Only three things change shape: the memory bar, the launch flags
and the `--explain` block are fixed-width, so they go inside fences — and the
last of those folds into a `<details>` block so the report above it stays
readable. `recommend --markdown` gains a trade-off column, which a terminal
has no room for. `--json` and `--markdown` together are refused rather than
one quietly winning.

## 10. Reference

The whole surface in one place. Sections 6 to 9 are where each part of it is
worked through.

### 10.1 Commands

| Command | Purpose | Exit |
| --- | --- | --- |
| `vramfit check <model> --device <d>` | Full report for one model, quant, context and device | 0 fits / 1 does not / 2 usage |
| `vramfit best <model> --device <d>` | Every quantization ranked by quality, with max context and decode speed | 0 something fits / 1 nothing in the range fits at this context / 2 usage |
| `vramfit compare <model> --devices <list>` | One model on several devices, best first | 0 something fits / 1 nothing does / 2 usage |
| `vramfit recommend --device <d>` | Every bundled model that fits, ranked, each with its trade-off | 0 something fits / 1 nothing does / 2 usage |
| `vramfit fleet --config <file>` | Which of several machines can serve which of several models | 0 every model placed / 1 one fits nowhere / 2 usage |
| `vramfit devices` | List the 29 bundled devices | 0 / 2 |
| `vramfit models` | List the 24 bundled models | 0 / 2 |

### 10.2 Options

`<model>` is a bundled id, name or alias — or a path: a GGUF file, or a
HuggingFace checkpoint directory. Both are read from their own headers.

| Option | Meaning |
| --- | --- |
| `-d, --device <id>` | Bundled device id, name or alias (`4090`, `"RTX 4090"`, `m3-max`) |
| `--gguf <path>` | Read the model from a GGUF file whatever it is named; only the header is read |
| `--hf-config <path>` | Read the model from a HuggingFace checkpoint directory or its `config.json`; a safetensors index beside it gives the true parameter count |
| `--devices <list>` | `compare` only: comma-separated devices, each with an optional count — `4090,3090x2,m4-max` |
| `--use-case <id>` | `recommend` only: `chat`, `code` or `long-context`. Sets the context to check at and the decode speed to clear |
| `--limit <n>` | `recommend` only: show the top n |
| `--config <path>` | `fleet` only: the JSON description of the machines and the models |
| `--launcher [runtime]` | `check` only: print the exact flags this fit implies -- `llama.cpp`, `ollama`, `vllm`, or all three |
| `--ngl` | `check` only: print the llama.cpp `-ngl` value and nothing else, for a shell substitution |
| `--explain` | `check` only: show every headline number with the arithmetic that produced it |
| `--markdown` | Markdown tables on any command, for pasting into an issue. Mutually exclusive with `--json` |
| `--color` / `--no-color` | Force ANSI colour on or off; the default follows the terminal, and `NO_COLOR`, `FORCE_COLOR` and `TERM=dumb` are honoured |
| `-q, --quant <id>` | Weight quantization; defaults to the format the model ships in (MXFP4 for gpt-oss), else `q4_k_m` |
| `-c, --ctx <n>` | Context length; `32768` or `32k`. Defaults to the model's own default |
| `-b, --batch <n>` | Concurrent sequences, default 1 |
| `-g, --gpus <n>` | Identical devices sharing the model, default 1 |
| `--kv-quant <id>` | `f16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0` |
| `--vram <GiB>` | Usable memory per device, used as given -- not scaled again by the device's usable fraction |
| `--ubatch <n>` | Physical batch (llama.cpp `--ubatch-size`), default 512 |
| `--no-flash-attn` | Model the compute buffer without flash attention |
| `--prompt <n>` | Prompt length for time-to-first-token |
| `--ram <GiB>` | System RAM available for offloaded layers |
| `--ram-bandwidth <GB/s>` | System RAM bandwidth, default 89.6 (DDR5-5600) |
| `--cpu-tflops <n>` | CPU dense FP16 throughput, for offloaded prefill |
| `--efficiency <0-1>` | Override the device's memory-bandwidth efficiency; offloaded layers keep their derived figure |
| `--prefill-efficiency <0-1>` | Override the device's prefill MFU, on the same terms |
| `--model-json <path>` | Use a model spec from a file instead of the database |
| `--device-json <path>` | Use a device spec from a file instead of the database |
| `--json` | Machine-readable output |
| `-h, --help` / `-v, --version` | Usage text / version |

### 10.3 Quantizations

Effective bits per weight, before the embedding and output-head promotions:

| Quant | bpw | Quant | bpw | Quant | bpw |
| --- | ---: | --- | ---: | --- | ---: |
| `f16` / `bf16` | 16.00 | `q4_k_m` | 4.83 | `mxfp4` | 4.25 |
| `q8_0` | 8.50 | `q4_k_s` | 4.57 | `awq-4bit` | 4.25 |
| `q6_k` | 6.56 | `q4_0` | 4.55 | `gptq-4bit` | 4.25 |
| `q5_k_m` | 5.67 | `q3_k_m` | 3.91 | | |
| `q5_k_s` | 5.52 | `q2_k` | 3.35 | | |

### 10.4 JSON output

`--json` writes one object to stdout and nothing else, so it pipes straight
into `jq`. Every computed quantity is in base units: bytes for memory, decimal
bytes per second for bandwidth, tokens per second for rates, seconds for
durations, tokens for context and prompt lengths — GiB appear only in the human
report. The `device` block is the stored spec, where the field name carries the
unit (`vramGiB`, `bandwidthGBs`). Keys are added over time, but the ones below
keep their name and meaning.

`vramfit check --json`:

| Key | Contents |
| --- | --- |
| `vramfit` | Version that produced the payload |
| `fits` | The verdict, matching the exit code |
| `model` | `id`, `name`, `origin` (`database`, `gguf`, `huggingface` or `json`), `from` (the path or id it was read from), `totalParams`, `activeParams`, `nLayers`, `nKvHeads`, `headDim`, `attention`, `moe` |
| `device` | `id`, `name`, `family`, `vramGiB`, `bandwidthGBs`, `usableFraction`, `count` |
| `config` | `quant`, `bitsPerWeight`, `effectiveBitsPerWeight`, `ctx`, `batch`, `kvQuant` |
| `memory` | `weightsBytes`, `kvCacheBytes`, `runtimeContextBytes`, `activationBytes` — which sum to `totalBytes` — plus `capacityBytes`, `headroomBytes`, `utilization` |
| `capacity` | `maxContext`, `recommendedQuant` (id or `null`) |
| `throughput` | `decodeTokensPerSecond`, `aggregateDecodeTokensPerSecond`, `prefillTokensPerSecond`, `promptTokens`, `timeToFirstTokenSeconds`, `decodeErrorBand`, `prefillErrorBand` |
| `offload` | `null` when the model is fully resident, otherwise `gpuLayers`, `cpuLayers`, `vocabOnDevice`, `systemRamRequiredBytes`, `systemRamAvailableBytes`, `feasible`, `blendedBandwidthBytesPerSecond` |
| `warnings` | The strings the report prints under *Notes* |

`--launcher` adds a `launcher` key to `check --json` — `llamaCpp`, `ollama`
and `vllm`, each with its own fields plus an `args` array and a `command`
string, and the `notes` the report prints beside them. The key is absent
without the flag.

`vramfit compare --json` returns `vramfit`, `model`, `config`, `best` (a
device id or `null`) and `devices[]` in the table's own order, each with
`capacityBytes`, `totalBytes`, `headroomBytes`, `utilization`, `fits`,
`maxContext`, `decodeTokensPerSecond`, `offloadFeasible` and `best`.
`recommend --json` returns `device`, `useCase` (`id`, `ctx`,
`comfortableDecodeTokensPerSecond`) and `models[]` with the three ranking
factors — `capability`, `quality`, `speed` — beside the `score` they multiply
to, so the ranking can be recomputed or argued with. `fleet --json` returns
`machines[]` and `models[]`, each model carrying `servedBy` and one entry per
machine.

`vramfit best --json` returns `vramfit`, `model`, `device`, `ctx`,
`recommended` (a quant id or `null`) and `quants[]`, one entry per candidate
with `id`, `label`, `bitsPerWeight`, `qualityRank`, `weightsBytes`,
`totalBytes`, `fits`, `maxContext`, `decodeTokensPerSecond`, `offloadFeasible`
and `systemRamRequiredBytes`. `devices --json` and `models --json` print the
bundled `DeviceSpec[]` and `ModelSpec[]` as they are.

```sh
vramfit check llama-3.1-8b -d 4090 --ctx 32k --json \
  | jq '.memory.totalBytes / 1073741824, .throughput.decodeTokensPerSecond'
```

### 10.5 Library

```ts
import { checkFit, getDevice, getModel, getQuant, recommendQuant } from "vramfit";

const fit = checkFit(getModel("llama-3.1-8b"), getQuant("q4_k_m"), getDevice("4090"), {
  ctx: 32_768,
});

fit.fits;                               // true
fit.headroomBytes;                      // 15597252060.16
fit.utilization;                        // 0.3947469606002172
fit.maxContext;                         // 131072
fit.footprint.kv.totalBytes;            // 4294967296
fit.throughput.decode.tokensPerSecond;  // 67.4244625650359
fit.offload;                            // null -- it is fully resident
fit.warnings;                           // []

recommendQuant(getModel("qwen2.5-32b"), getDevice("4090"), { ctx: 8192 })?.quant.label;
// "Q4_K_M"
```

Every layer is exported and usable on its own — `computeWeightBytes`,
`computeKvCacheBytes`, `computeActivationBytes`, `computeFootprint`,
`estimateThroughput`, `planOffload`, `blendBandwidth`, `maxContextFor`,
`evaluateQuants`, `deriveArchitecture` — along with the quantization tables,
the device and model registries, and the validators that parse user-supplied
specs. The arithmetic is pure: plain data in, plain data out, no global state.
The only I/O in the package is a lazy, memoised read of the bundled JSON, and
it only happens if you ask for a bundled model or device by name.

### 10.6 Bundled data, and bringing your own

`vramfit devices` lists 29 devices: NVIDIA RTX 3060 12GB through 5090, A100
40/80GB, H100 80GB SXM5, L40S, AMD RX 7900 XTX, Apple M1–M4 in Pro/Max/Ultra
variants with their unified-memory bandwidth and the macOS wired-memory limit,
and generic CPU tiers from dual-channel DDR4-3200 to 8-channel DDR5-4800.

`vramfit models` lists 24 models: Llama 3.1/3.2/3.3 (1B–70B), Qwen 2.5 (7B–72B),
Qwen 3 including the 30B-A3B and 235B-A22B MoEs, Mistral 7B v0.3, Mixtral
8x7B, Gemma 2 (9B/27B), Gemma 3 (4B/12B/27B), Phi-4, DeepSeek-V2-Lite (MLA)
and gpt-oss 20B/120B.

Every device figure comes from a vendor datasheet and every model figure from
that model's published `config.json`; both carry their source string in the
JSON. The test suite re-derives each model's parameter count from its own
architecture fields and requires the result within 0.5% of the published count
— the worst of the 24 is Gemma 3 12B at 0.040%.

The database is a convenience, not a limit. A checkpoint on disk is read
directly (section 6), and anything matching `ModelSpec` / `DeviceSpec` works,
from a file or from code:

```sh
vramfit check --model-json ./my-finetune.json --device-json ./my-gpu.json --ctx 4k
```

User-supplied specs go through the same validator as the bundled data, which
enforces the invariants that keep the arithmetic honest — the declared
parameter count has to match the count the shape fields imply, query heads must
divide evenly into KV head groups, an MLA model must carry MLA geometry, a
router cannot pick more experts than exist — and names the exact field path
when they do not.

A model released in a quantization of its own says so with `nativeQuant`, as
the two gpt-oss entries do. That format is then the default for `check` and the
top of the `best` table, and wider quantizations of it are left out: a Q8_0 of
an MXFP4 checkpoint is twice the bytes for weights that were never wider than
4.25 bits.

## 11. Accuracy and limitations

- Memory figures are **arithmetic**, and only as good as their inputs. Weight
  sizes land within 2% of published GGUF files for every dense model in the
  test table.
- **MoE k-quant mixes** quantize expert tensors more narrowly than the dense
  `_M` recipe, so Mixtral 8x7B Q4_K_M is predicted about 7% high. High is the
  safe direction for a "will it fit" tool, and the gap is pinned by a test
  rather than left to be discovered.
- Throughput is an **estimate** with a stated band (±25% decode, ±40%
  prefill). It is most optimistic for a small model on a datacenter GPU at
  batch 1, where decode is latency-bound rather than bandwidth-bound.
  `--efficiency` lets you calibrate against your own measurement.
- Multi-GPU assumes a **homogeneous layer split**. Tensor-parallel runtimes
  scale decode throughput; this does not model that, and it does not model
  heterogeneous cards.
- The runtime context and compute buffer are **empirical**, not derived. They
  are backend- and version-dependent; treat them as ±0.3 GiB and ±50%.
- There is no model download and no device probing. `vramfit` never touches
  the network, at runtime or in its tests: a GGUF header or a `config.json` is
  read from a path you already have, and the test suite builds its GGUF and
  safetensors fixtures byte by byte.
- A `config.json` describes an **architecture**, not a checkpoint. Its
  parameter count leaves out biases, layer norms and rotary tables — about
  0.003% on Llama 3.1 8B — so the weight files are preferred when they are
  there to read.

## 12. Development

```sh
npm install
npm run lint       # oxlint, warnings are errors
npm run typecheck  # tsc --noEmit over src, test, scripts and the vitest config
npm run build      # tsc + copy the bundled JSON into dist/
npm test           # vitest -- 464 tests across 23 files
npm run smoke      # spawn the built binary and assert its output and exit codes
```

Tests never touch the network: everything is arithmetic over fixtures and
bundled JSON, so the suite passes offline. The fixtures in `test/fixtures.ts`
are deliberately independent of `src/data/models.json`, because a test that
reads the same file the code reads can only catch a bad lookup, never a bad
formula.

CI runs the whole chain on Node 20, 22 and 24 on Linux plus Node 22 on macOS
and Windows, then packs the tarball, installs it into a clean project, and runs
the installed binary and library from there — the only way to catch a `files`
list that forgot `dist/data`, a `bin` path that does not resolve, or an
`exports` map a consumer cannot import.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the commit convention and what a
pull request is expected to carry, [SECURITY.md](SECURITY.md) for private
vulnerability reports, and [CHANGELOG.md](CHANGELOG.md) for the release record.
Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## 13. License

vramfit is **open source**, licensed under the
[Apache License, Version 2.0](LICENSE).

**What you may do:** use, modify and redistribute vramfit for any purpose,
including commercially and inside proprietary products, at any organisation
size — there is no separate licence to buy and nobody to ask.

**What the licence asks in return:** keep the copyright and licence notices,
state any significant changes you made, and include the [NOTICE](NOTICE) file
with any redistribution. The full conditions are in section 4 of the
[LICENSE](LICENSE).

**Patents are covered.** Apache-2.0 grants an express, royalty-free patent
licence from every contributor, and that grant terminates for anyone who
brings a patent claim alleging that the project infringes their patents.

Contributions are taken under the Developer Certificate of Origin and are
licensed under the same Apache License, Version 2.0 — see
[CONTRIBUTING.md](CONTRIBUTING.md).

Copyright © 2026 Johan Becker.
