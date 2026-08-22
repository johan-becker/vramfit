/**
 * Core data shapes. These are the contract between the bundled JSON database,
 * the arithmetic in `memory.ts` / `throughput.ts` / `offload.ts`, and the CLI.
 */

/** Attention flavours that change how the KV cache is sized. */
export type AttentionKind =
  /** Multi-head, grouped-query and multi-query attention. All three are the
   *  same formula; MHA is nKvHeads === nHeads and MQA is nKvHeads === 1. */
  | "gqa"
  /** DeepSeek-style Multi-head Latent Attention: a single compressed latent
   *  vector per token per layer instead of per-head K and V. */
  | "mla";

/** Mixture-of-experts geometry. Absent (null) for dense models. */
export interface MoeSpec {
  /** Routed experts per MoE layer. */
  nExperts: number;
  /** Routed experts actually evaluated per token (top-k). */
  expertsPerToken: number;
  /** Intermediate width of one routed expert's FFN. */
  expertFfnHidden: number;
  /** Always-on shared experts (DeepSeek, Qwen3-MoE variants). 0 if none. */
  nSharedExperts: number;
  /** Leading layers that use a plain dense FFN instead of an MoE block. */
  denseLayers: number;
  /** Intermediate width of those leading dense layers. */
  denseFfnHidden: number | null;
}

/** Multi-head Latent Attention geometry (DeepSeek V2/V3 family). */
export interface MlaSpec {
  /** Width of the compressed KV latent that is actually cached. */
  kvLoraRank: number;
  /** Width of the decoupled RoPE key, also cached. */
  qkRopeHeadDim: number;
  /** Non-positional part of the query/key head, reconstructed, not cached. */
  qkNopeHeadDim: number;
  /** Value head width, reconstructed from the latent, not cached. */
  vHeadDim: number;
  /** Query down-projection rank, or null when queries are not compressed. */
  qLoraRank: number | null;
}

/**
 * Interleaved sliding-window attention. Gemma 2/3 and gpt-oss alternate cheap
 * local-attention layers with a smaller number of full-attention layers, and
 * the local layers only ever cache `windowSize` tokens. Ignoring this
 * overestimates Gemma 3 27B's cache at 128K by roughly 4x.
 */
export interface AttentionWindowSpec {
  /** Tokens retained by a local-attention layer. */
  windowSize: number;
  /** One layer in every N is full attention; the rest are windowed. */
  fullAttentionEvery: number;
}

export interface ModelSpec {
  id: string;
  name: string;
  aliases: string[];
  /** Every parameter in the checkpoint. Drives memory. */
  totalParams: number;
  /** Parameters touched per decoded token. Equals totalParams for dense
   *  models; drives speed for MoE models. */
  activeParams: number;
  nLayers: number;
  hiddenSize: number;
  /** Query heads. NOT the number used for the KV cache. */
  nHeads: number;
  /** Key/value heads. This is the one the KV cache formula needs. */
  nKvHeads: number;
  headDim: number;
  /** Dense FFN intermediate width. For MoE models this is the width used by
   *  any dense layers; per-expert width lives in `moe.expertFfnHidden`. */
  ffnHidden: number;
  vocabSize: number;
  /** True when the output head reuses the input embedding matrix. */
  tiedEmbeddings: boolean;
  attention: AttentionKind;
  mla: MlaSpec | null;
  moe: MoeSpec | null;
  attentionWindow: AttentionWindowSpec | null;
  /** Largest context the architecture supports, per its published config. */
  maxCtx: number;
  /** A sensible default to check, usually what the vendor ships enabled. */
  defaultCtx: number;
  /** Where the numbers came from, so they can be audited. */
  source: string;
  notes?: string;
}

export type DeviceFamily =
  | "cuda-consumer"
  | "cuda-datacenter"
  | "rocm"
  | "metal"
  | "cpu";

export interface DeviceSpec {
  id: string;
  name: string;
  aliases: string[];
  family: DeviceFamily;
  /** Physical memory in GiB (2^30 bytes) -- see units.ts on why not "GB". */
  vramGiB: number;
  /** Peak theoretical memory bandwidth, decimal GB/s as vendors quote it. */
  bandwidthGBs: number;
  /** Peak dense FP16/BF16 tensor throughput, TFLOP/s, without sparsity. */
  fp16Tflops: number;
  /** True for Apple silicon and other shared CPU/GPU memory systems. */
  unifiedMemory: boolean;
  /**
   * Fraction of `vramGiB` an inference process can actually allocate.
   * 1.0 for a headless discrete GPU; ~0.75 on macOS, where Metal's
   * recommendedMaxWorkingSetSize caps GPU-wired memory well below installed
   * RAM unless `iogpu.wired_limit_mb` is raised by hand.
   */
  usableFraction: number;
  source: string;
  notes?: string;
}

/** Where a quantization scheme comes from; used to keep recommendations
 *  inside one ecosystem (you cannot load an AWQ file in llama.cpp). */
export type QuantFamily = "float" | "gguf-k" | "gguf-legacy" | "mxfp4" | "awq" | "gptq";

export interface QuantSpec {
  id: string;
  label: string;
  family: QuantFamily;
  /**
   * Effective bits per weight for the bulk (repeating transformer block)
   * tensors, including the block-wise scales and mins the format stores.
   */
  bitsPerWeight: number;
  /**
   * Effective bits per weight for the token embedding table (`token_embd`).
   * Equal to `bitsPerWeight` for GGUF k-quants -- the embedding table itself
   * is not promoted -- but FP16 for GPTQ/AWQ/MXFP4, which quantize only the
   * linear layers.
   */
  tokenEmbedBits: number;
  /**
   * Effective bits per weight for the matrix that produces logits
   * (`output.weight`, or `token_embd` again when embeddings are tied).
   * llama.cpp's k-quant mixes promote this to Q6_K because quantizing the
   * vocabulary projection costs far more quality than quantizing a block.
   */
  outputHeadBits: number;
  /**
   * Quality ordering, higher is better. Derived from the perplexity tables
   * llama.cpp publishes for its quantization mixes, not from bits alone --
   * a k-quant beats a legacy quant of the same width.
   */
  qualityRank: number;
  notes: string;
}

/** KV cache element type. */
export interface KvQuantSpec {
  id: string;
  bytesPerElement: number;
  notes: string;
}
