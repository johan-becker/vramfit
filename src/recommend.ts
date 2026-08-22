import { listModels } from "./db/index.js";
import { recommendQuant, type FitResult, type QuantSearchOptions } from "./fit.js";
import { findQuant } from "./quant.js";
import type { DeviceSpec, ModelSpec, QuantSpec } from "./types.js";
import { formatParams } from "./units.js";

/**
 * "What should I run on this box?"
 *
 * The inverse of `check`, and the question people actually arrive with. Every
 * bundled model is evaluated at the best quantization that fits on the given
 * device, and the ones that fit are ranked.
 *
 * The ranking is a judgement, so it is written down rather than buried:
 *
 *     score = capability x quality x speed
 *
 * with each factor derived from something checkable.
 *
 *   capability  log10 of the parameter count in billions. Model quality tracks
 *               parameters logarithmically -- that is what every scaling-law
 *               curve since Kaplan and Chinchilla says -- so a 70B is not ten
 *               times a 7B, it is about one unit better on this scale. For a
 *               mixture of experts the count used is the geometric mean of
 *               total and active parameters, which is the usual rough stand-in
 *               for the dense model an MoE behaves like: Mixtral 8x7B lands
 *               near 24B rather than at either 47B or 13B.
 *
 *   quality     1.0 for Q4_K_M and anything above it, falling off below. The
 *               README's own position, and llama.cpp's perplexity tables: the
 *               step from Q6_K to Q4_K_M costs very little and the step from
 *               Q4_K_M to Q3_K_M costs a lot.
 *
 *   speed       decode tokens per second against what the use case needs,
 *               capped at 1. Above the bar, more speed does not make the model
 *               better; below it, it is proportionally worse. This is the
 *               factor that stops the ranking from simply recommending the
 *               largest model that technically fits.
 *
 * What this deliberately does NOT do is rank models by how good they are at
 * anything. vramfit has no benchmark data and inventing a coding score would
 * be exactly the kind of confident guess the rest of the package exists to
 * replace. `--use-case` changes the context to check at and the speed bar to
 * clear, and says so.
 */

export type UseCase = "chat" | "code" | "long-context";

export interface UseCaseProfile {
  id: UseCase;
  label: string;
  /** Decode speed below which the experience breaks down, tokens/second. */
  comfortableDecode: number;
  /** Context to evaluate at, when the caller does not name one. */
  defaultContext: number;
  /** Why those two numbers, in the report's own words. */
  why: string;
}

/**
 * The three profiles.
 *
 * Chat is bounded by reading speed: people read 5-8 words a second, which is
 * 10-15 tokens, and anything faster than that is being buffered by your eyes
 * rather than enjoyed. Code generation is not read as it arrives -- you wait
 * for the whole completion or diff, several hundred tokens of it -- so the bar
 * is higher and the context has to hold a couple of files. Long-context work
 * is summarising and retrieval over a document you have already got: the
 * prompt dominates the wall clock and the decode bar is correspondingly low.
 */
export const USE_CASES: Readonly<Record<UseCase, UseCaseProfile>> = {
  chat: {
    id: "chat",
    label: "chat",
    comfortableDecode: 15,
    defaultContext: 8192,
    why: "reading speed is 10-15 tokens a second, so anything above that is buffered by your eyes rather than enjoyed",
  },
  code: {
    id: "code",
    label: "code",
    comfortableDecode: 30,
    defaultContext: 16_384,
    why: "a completion or a diff is hundreds of tokens you wait for in full, and the context has to hold a couple of files",
  },
  "long-context": {
    id: "long-context",
    label: "long context",
    comfortableDecode: 8,
    defaultContext: 32_768,
    why: "summarising a document you already have is dominated by prefill, and the cache is what decides whether it fits at all",
  },
};

export const USE_CASE_IDS: readonly UseCase[] = ["chat", "code", "long-context"];

export function findUseCase(id: string): UseCaseProfile | undefined {
  return USE_CASES[id.trim().toLowerCase() as UseCase];
}

/** Quality rank at and above which vramfit treats a quantization as lossless. */
const LOSSLESS_QUALITY_RANK = 60;
/** Floor of the quality factor, at the narrowest quantization in the table. */
const NARROWEST_QUALITY_FACTOR = 0.6;

/**
 * The dense model a mixture of experts behaves like, for ranking purposes.
 * The geometric mean of what it stores and what it computes: Mixtral 8x7B
 * comes out near 24B, which is roughly where it sits against dense models.
 */
export function effectiveCapabilityParams(model: ModelSpec): number {
  if (model.moe === null) return model.totalParams;
  return Math.sqrt(model.totalParams * model.activeParams);
}

function capabilityOf(model: ModelSpec): number {
  const billions = effectiveCapabilityParams(model) / 1e9;
  // A 1B model scores 1, an 8B 1.9, a 70B 2.85. Floored so that a very small
  // model still has a positive score rather than a negative one.
  return Math.max(0.1, Math.log10(billions) + 1);
}

/** True when this is the format the checkpoint is published in. */
function isNativeQuant(model: ModelSpec, quant: QuantSpec): boolean {
  return model.nativeQuant !== undefined && findQuant(model.nativeQuant)?.id === quant.id;
}

function qualityOf(model: ModelSpec, quant: QuantSpec): number {
  // A model released in a quantization is not being compromised by running in
  // it: MXFP4 is the only format gpt-oss exists in, so charging it the
  // quality penalty of a 4-bit requantization would rank it below models that
  // really have been narrowed.
  if (isNativeQuant(model, quant)) return 1;
  if (quant.qualityRank >= LOSSLESS_QUALITY_RANK) return 1;
  return (
    NARROWEST_QUALITY_FACTOR +
    (1 - NARROWEST_QUALITY_FACTOR) * (quant.qualityRank / LOSSLESS_QUALITY_RANK)
  );
}

export interface Recommendation {
  model: ModelSpec;
  /** The highest-quality quantization that fits at the requested context. */
  quant: QuantSpec;
  fit: FitResult;
  maxContext: number;
  decodeTokensPerSecond: number;
  /** capability x quality x speed. Comparable only within one run. */
  score: number;
  capability: number;
  quality: number;
  speed: number;
  /** One sentence saying what this row costs you, in plain terms. */
  tradeoff: string;
}

export interface RecommendOptions extends QuantSearchOptions {
  /** Defaults to chat. */
  useCase?: UseCase;
  /** Candidates. Defaults to the bundled database. */
  models?: readonly ModelSpec[];
  /** How many rows to return. All of them by default. */
  limit?: number;
}

function quantClause(model: ModelSpec, quant: QuantSpec): string {
  if (isNativeQuant(model, quant)) {
    return `in its native ${quant.label}, the only format it is published in`;
  }
  if (quant.qualityRank >= 80) return `at ${quant.label}, which is effectively lossless`;
  if (quant.qualityRank >= LOSSLESS_QUALITY_RANK) {
    return `at ${quant.label}, the best quality-per-byte in the GGUF lineup`;
  }
  if (quant.qualityRank >= 45) return `at ${quant.label}, a measurable step down from Q4_K_M`;
  return `at ${quant.label}, narrow enough that the quality drop is easy to see`;
}

function speedClause(decode: number, profile: UseCaseProfile): string {
  const rate = decode >= 10 ? decode.toFixed(1) : decode.toFixed(2);
  const bar = `the ${profile.comfortableDecode} tok/s ${profile.label} wants`;
  if (decode >= profile.comfortableDecode * 2) return `and ${rate} tok/s, well clear of ${bar}`;
  if (decode >= profile.comfortableDecode) return `and ${rate} tok/s, comfortably above ${bar}`;
  return `but only ${rate} tok/s, under ${bar}`;
}

function tradeoffFor(
  model: ModelSpec,
  quant: QuantSpec,
  decode: number,
  profile: UseCaseProfile,
): string {
  const size =
    model.moe === null
      ? `${formatParams(model.totalParams)} parameters`
      : `${formatParams(model.totalParams)} parameters of which ${formatParams(model.activeParams)} run per token`;
  return `${size} ${quantClause(model, quant)}, ${speedClause(decode, profile)}.`;
}

/**
 * Rank what fits on this device, best first.
 *
 * A model appears only if some quantization in the GGUF range fits it at the
 * requested context with room for the cache and the overheads. Everything that
 * does not fit is left out rather than listed with a cross: the question is
 * what to run, and `check` already answers what it would take to run something
 * that does not.
 */
export function recommendModels(
  device: DeviceSpec,
  options: RecommendOptions = {},
): Recommendation[] {
  const profile = USE_CASES[options.useCase ?? "chat"];
  const ctx = options.ctx ?? profile.defaultContext;

  const ranked = (options.models ?? listModels())
    .map((model): Recommendation | undefined => {
      // A model whose architecture stops short of the requested context is
      // not a candidate, however well it would fit in memory: running Gemma 2
      // at 16K is running it past its trained maximum, and recommending that
      // would be recommending degraded output.
      if (model.maxCtx < ctx) return undefined;
      const option = recommendQuant(model, device, { ...options, ctx });
      if (option === undefined) return undefined;

      const capability = capabilityOf(model);
      const quality = qualityOf(model, option.quant);
      const speed = Math.min(1, option.decodeTokensPerSecond / profile.comfortableDecode);

      return {
        model,
        quant: option.quant,
        fit: option.fit,
        maxContext: option.maxContext,
        decodeTokensPerSecond: option.decodeTokensPerSecond,
        score: capability * quality * speed,
        capability,
        quality,
        speed,
        tradeoff: tradeoffFor(model, option.quant, option.decodeTokensPerSecond, profile),
      };
    })
    .filter((entry): entry is Recommendation => entry !== undefined)
    .toSorted(
      (a, b) =>
        b.score - a.score ||
        b.model.totalParams - a.model.totalParams ||
        a.model.id.localeCompare(b.model.id),
    );

  const limit = options.limit;
  return limit === undefined ? ranked : ranked.slice(0, Math.max(0, Math.floor(limit)));
}
