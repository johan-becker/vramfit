# vramfit

**Will this model run on my machine, and how fast?**

The most-asked question in local LLM self-hosting, answered with arithmetic
instead of forum guesswork. Give vramfit a model, a quantization, a context
length and a device, and it tells you whether it fits, how much headroom is
left, the largest context that *would* fit, the best quantization that fits,
and an estimated tokens per second — and when it does not fit, how many layers
you can offload and what that costs you.

Zero runtime dependencies. TypeScript, ESM, Node 20+. Library and CLI.

```console
$ npx vramfit check llama-3.1-8b --device 4090 --ctx 32k
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

Capacity
  Largest context that fits    128K  the architecture's own maximum
  Best quant that fits at 32K   F16  14.96 GiB of weights, 39.2 tok/s

Speed (estimates: decode +/-25%, prefill +/-40%)
  Decode               67.4 tok/s  at 32K context, 1 sequence
  Prefill              2450 tok/s  23.6 GFLOP per prompt token
  Time to first token      13.4 s  for a prompt of 32K tokens
```

## Install

```sh
npm install -g vramfit    # CLI
npm install vramfit       # library
npx vramfit check ...     # or don't install it at all
```

## Why this exists

Almost every VRAM calculator gets at least one of these wrong, and each one is
worth gigabytes:

| Mistake | Cost |
| --- | --- |
| Treating `Q4_K_M` as 4 bits per weight | 18% low on a Llama 3.1 8B — it is 4.83 bpw |
| Sizing the KV cache with `n_heads` instead of `n_kv_heads` | Up to **8x** too large on any modern GQA model |
| Ignoring that the output head is promoted to Q6_K | 0.4 GB on an 8B, 4 GB on a 262k-vocab Gemma 3 |
| Charging a tied embedding table twice | 12% of a Llama 3.2 3B file |
| Ignoring sliding-window attention | 5x too large on Gemma 3 at 32K |
| Ignoring MLA on DeepSeek | 7x too large |
| Forgetting the ~0.7 GB CUDA runtime context | The difference between "fits" and OOM at 95% utilisation |
| Sizing MoE speed by total parameters | Mixtral decodes like a 13B, not a 47B |

vramfit gets all of them right, and every formula is documented with its
assumption in the source and unit-tested against hand-computed values.

## The math

### 1. Weights

```
bytes = params x bits_per_weight / 8
```

with three refinements that matter. GGUF k-quants are **not** their nominal
width — a Q4_K_M stores super-blocks of 256 weights with two levels of scales
and promotes some tensor types, landing at 4.83 bits per weight. The logit
matrix is promoted to Q6_K by every llama.cpp k-quant mix. And a tied embedding
table is stored once, not twice.

Predicted file sizes land within 2% of the published GGUF downloads for every
dense model in `test/gguf-sizes.test.ts`:

| Model | Quant | Predicted | Published |
| --- | --- | --- | --- |
| Llama 3.1 8B | Q4_K_M | 4.96 GB | 4.92 GB |
| Llama 3.1 8B | Q8_0 | 8.53 GB | 8.54 GB |
| Llama 3.3 70B | Q4_K_M | 42.82 GB | 42.52 GB |
| Mistral 7B v0.3 | Q4_K_M | 4.41 GB | 4.37 GB |
| Gemma 2 9B | Q4_K_M | 5.78 GB | 5.76 GB |

### 2. KV cache

```
bytes = 2 x n_layers x n_kv_heads x head_dim x ctx x batch x bytes_per_element
        ^-- K and V
```

`n_kv_heads`, never `n_heads`. Llama 3.1 70B has 64 query heads and 8 KV heads:
using 64 makes the cache come out exactly 8x too large, which turns "70B at 32K
fits in 48 GB" into "you need an H100".

Two architectures need their own formula:

- **Multi-head Latent Attention** (DeepSeek V2/V3) caches one compressed latent
  plus a decoupled RoPE key, shared by every head:
  `n_layers x (kv_lora_rank + qk_rope_head_dim) x ctx x batch x bpe`. No factor
  of two, no head count — 576 elements per token per layer instead of 4096.
- **Interleaved sliding-window attention** (Gemma 2/3, gpt-oss) caps most
  layers at `window_size` tokens and keeps only every Nth layer global.

KV quantization uses the exact block layouts: f16 = 2 bytes/element, q8_0 =
1.0625, q4_0 = 0.5625.

### 3. Mixture of experts

Total parameters drive **memory**; active parameters per token drive **speed**.
Mixtral 8x7B occupies 46.7B parameters of memory and multiplies 12.9B of them
per token, so it needs the VRAM of a 47B and decodes at roughly the speed of a
13B. Both numbers are derived from the architecture, not from a headline figure.

### 4. Overheads

Two empirical terms, and they are labelled as such:

- **Runtime context** — the CUDA primary context, kernel image and cuBLAS
  workspaces (~0.7 GiB consumer, ~1.0 GiB datacenter, ~0.5 GiB Metal), charged
  once *per device*.
- **Compute buffer** — activations, FP32 logits and graph scratch, scaling with
  the physical batch (`--ubatch`) rather than the context, plus the
  `ubatch x ctx x n_heads` score matrix when flash attention is off.

Assume ±0.3 GiB on the first and ±50% on the second.

### 5. Throughput

Decode is memory-bandwidth bound — every weight is read and multiplied against
a single vector, about one FLOP per byte:

```
tokens/second = bandwidth x efficiency / (active_weight_bytes + kv_bytes)
```

The KV term is why long conversations slow down: attention re-reads the whole
cache for every token it emits.

Efficiency is two tables rather than one constant, because one constant cannot
fit the data. At 16 bits decode is purely bandwidth-bound and reaches 0.70–0.85
of peak. At 4 bits it does not, because unpacking the weights is real work, and
how much it costs depends on the backend's kernels. Calibrated against
published llama.cpp `tg128` results for LLaMA 7B:

| Device | Quant | Measured | vramfit |
| --- | --- | --- | --- |
| M2 Ultra | F16 | 42.75 tok/s | 42.2 (−1%) |
| M2 Ultra | Q4_0 | 88.64 tok/s | 92.7 (+5%) |
| RTX 4090 | Q4_0 | 152.2 tok/s | 155.0 (+2%) |

Prefill is compute bound: `2 x active_params` FLOPs per token for the matmuls,
plus the attention matmuls, against the device's FP16 TFLOPs at a documented
model-FLOPs-utilisation figure.

**These are estimates**, and the report says so: ±25% on decode, ±40% on
prefill. They are for "which order of magnitude, and will changing the quant
help", not for benchmarking.

### 6. Partial offload

"It does not fit" is nearly never the end of the story. vramfit models
llama.cpp's `--n-gpu-layers`: as many blocks as fit go in device memory, each
taking its own slice of the KV cache, filled from the tail, with the vocabulary
tensors placed only once every layer is already there.

The resulting speed is a harmonic mean of the two bandwidths weighted by the
bytes read from each — harmonic, because the slow side dominates. Moving 10% of
a model off a 1008 GB/s RTX 4090 into 89.6 GB/s DDR5 does not cost 10% of the
speed; the blended bandwidth falls to 498 GB/s, less than half.

```console
$ vramfit check llama-3.3-70b -d 3090 --ctx 8k --ram 64
...
DOES NOT FIT  -  43.39 GiB needed, 24.00 GiB available, 19.39 GiB short

Partial offload
  Layers in device memory   44 of 80  vocabulary tensors in system RAM
  Left in system RAM       19.84 GiB  of 64.00 GiB assumed available
  Blended read bandwidth    175 GB/s  harmonic mean; the device alone reads at 936 GB/s

Speed (estimates: decode +/-25%, prefill +/-40%)
  Decode                2.05 tok/s  blended across 44 device and 36 host layers
```

### 7. Multi-GPU

`--gpus N` splits an identical set of devices. Weights and cache are shared
between them, but each device pays the runtime context and compute buffer in
full, so a pair of 3090s holds a 70B Q4_K_M where one does not — with the
second card's overheads eating a gigabyte of the 24 it added. A layer split
does not raise decode speed; extra devices buy capacity, not tokens per second,
and vramfit says so rather than quietly multiplying.

## CLI

```
vramfit check <model> --device <device> [options]
vramfit best  <model> --device <device> [options]
vramfit devices [--json]
vramfit models  [--json]
```

`check` exits **0** when the configuration fits, **1** when it does not and
**2** for bad usage, so it can gate a deploy script:

```sh
vramfit check "$MODEL" --device 4090 --ctx 32k --json > plan.json \
  || { echo "won't fit, refusing to deploy"; exit 1; }
```

### Options

| Option | Meaning |
| --- | --- |
| `-d, --device <id>` | Bundled device id, name or alias (`4090`, `"RTX 4090"`, `m3-max`) |
| `-q, --quant <id>` | Weight quantization, default `q4_k_m` |
| `-c, --ctx <n>` | Context length; `32768` or `32k` |
| `-b, --batch <n>` | Concurrent sequences |
| `-g, --gpus <n>` | Identical devices sharing the model |
| `--kv-quant <id>` | `f16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0` |
| `--vram <GiB>` | Override the device's memory, per device |
| `--ram <GiB>` | System RAM available for offloaded layers |
| `--ram-bandwidth <GB/s>` | System RAM bandwidth, default 89.6 (DDR5-5600) |
| `--cpu-tflops <n>` | CPU FP16 throughput, for offloaded prefill |
| `--ubatch <n>` | Physical batch (llama.cpp `--ubatch-size`), default 512 |
| `--no-flash-attn` | Model the compute buffer without flash attention |
| `--prompt <n>` | Prompt length for time-to-first-token |
| `--model-json <path>` | Use a model spec from a file instead of the database |
| `--device-json <path>` | Use a device spec from a file instead of the database |
| `--efficiency <0-1>` | Override the memory-bandwidth efficiency |
| `--prefill-efficiency <0-1>` | Override the prefill MFU |
| `--json` | Machine-readable output |

### `vramfit best`

```console
$ vramfit best qwen2.5-32b -d 4090 --ctx 8k
Qwen2.5 32B  |  NVIDIA RTX 4090
===============================

Quant     bpw    Weights  Total at 8K  Fits  Max ctx      Decode
------  -----  ---------  -----------  ----  -------  ----------
F16     16.00  61.03 GiB    63.97 GiB    no        -  1.43 tok/s
BF16    16.00  61.03 GiB    63.97 GiB    no        -  1.43 tok/s
Q8_0     8.50  32.42 GiB    35.37 GiB    no        -  3.75 tok/s
Q6_K     6.56  25.02 GiB    27.97 GiB    no        -  8.73 tok/s
Q5_K_M   5.67  21.71 GiB    24.65 GiB    no     5.5K  18.9 tok/s
Q5_K_S   5.52  21.15 GiB    24.10 GiB    no     7.8K  19.2 tok/s
Q4_K_M   4.83  18.58 GiB    21.53 GiB   yes    18.3K  27.8 tok/s
Q4_K_S   4.57  17.61 GiB    20.56 GiB   yes    22.3K  29.0 tok/s
Q4_0     4.55  17.54 GiB    20.48 GiB   yes    22.6K  29.1 tok/s
Q3_K_M   3.91  15.07 GiB    18.02 GiB   yes    32.7K  32.8 tok/s
Q2_K     3.35  12.91 GiB    15.86 GiB   yes    41.5K  37.6 tok/s

Rows that do not fit show the decode speed with as many layers as possible
offloaded to system RAM, which is what you would actually get.

Recommended: Q4_K_M -- highest quality that fits at 8K, 27.8 tok/s.
  The default recommendation: best quality-per-byte in the GGUF lineup for
  most models.
```

## Library

```ts
import { checkFit, getDevice, getModel, getQuant, recommendQuant } from "vramfit";

const fit = checkFit(getModel("llama-3.1-8b"), getQuant("q4_k_m"), getDevice("4090"), {
  ctx: 32_768,
});

fit.fits;                                  // true
fit.headroomBytes;                         // 15_597_252_060
fit.maxContext;                            // 131072
fit.footprint.kv.totalBytes;               // 4 GiB
fit.throughput.decode.tokensPerSecond;     // 67.4
fit.offload;                               // null -- it is fully resident

recommendQuant(getModel("qwen2.5-32b"), getDevice("4090"), { ctx: 8192 })?.quant.label;
// "Q4_K_M"
```

Every layer of the model is exported and usable on its own —
`computeWeightBytes`, `computeKvCacheBytes`, `computeActivationBytes`,
`computeFootprint`, `estimateThroughput`, `planOffload`, `blendBandwidth`,
`maxContextFor`, `evaluateQuants` — along with the quantization tables, the
device and model registries, and the validators that parse user-supplied specs.

### Bring your own model or device

The bundled database is a convenience, not a limit. Anything matching
`ModelSpec` / `DeviceSpec` works, from a file or from code:

```sh
vramfit check --model-json ./my-finetune.json --device-json ./my-gpu.json --ctx 4k
```

Both go through the same validator as the bundled data, which enforces the
invariants that keep the arithmetic honest — query heads must divide evenly
into KV head groups, an MLA model must carry MLA geometry, a router cannot pick
more experts than exist — and reports the exact field path when they do not.

## Bundled data

**Devices** (`vramfit devices`): NVIDIA RTX 3060 12GB, 3090, 4070 Ti SUPER,
4080, 4090, 5090; A100 40/80GB, H100 80GB, L40S; AMD RX 7900 XTX; Apple M1–M4
in Pro/Max/Ultra variants with correct unified-memory bandwidth and the macOS
wired-memory limit; generic CPU memory tiers from dual-channel DDR4 to
8-channel DDR5.

**Models** (`vramfit models`): Llama 3.1/3.2/3.3 (1B–70B), Qwen 2.5 (7B–72B),
Qwen 3 including 30B-A3B and 235B-A22B MoEs, Mistral 7B, Mixtral 8x7B, Gemma 2
(9B/27B), Gemma 3 (4B/12B/27B), Phi-4, DeepSeek-V2-Lite (MLA) and gpt-oss
20B/120B.

Every device figure comes from a vendor datasheet and every model figure from
that model's published `config.json`; both are stored with their source string.
The test suite re-derives each model's parameter count from its own
architecture fields and requires the result within 0.5% of the published count
— all 24 land inside 0.04%.

## Accuracy and known limitations

- Memory figures are arithmetic and are as good as the inputs. Weight sizes are
  within 2% of published GGUF files for dense models.
- **MoE k-quant mixes** quantize expert tensors more narrowly than the dense
  `_M` recipe, so Mixtral 8x7B Q4_K_M is predicted ~7% high. High is the safe
  direction for a "will it fit" tool, and the gap is pinned by a test.
- Throughput is an **estimate** with a stated band (±25% decode, ±40% prefill).
  It is most optimistic for a small model on a datacenter GPU at batch 1, where
  decode is latency-bound rather than bandwidth-bound. `--efficiency` lets you
  calibrate against your own measurement.
- Multi-GPU assumes a homogeneous layer split. Tensor-parallel runtimes scale
  decode; this does not model that.
- The runtime context and compute buffer are empirical, not derived.

## Development

```sh
npm install
npm run lint      # oxlint
npm run build     # tsc
npm test          # vitest
```

Tests never touch the network. Everything is arithmetic over fixtures and
bundled JSON, so the suite passes offline.

## License

MIT © 2026 Johan Becker — see [LICENSE](LICENSE).
