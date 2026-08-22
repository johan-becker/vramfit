# Contributing to vramfit

Thanks for looking. This is a small, opinionated package: an arithmetic core
with a CLI on top and **zero runtime dependencies**. Most of what follows
exists to keep those two properties true.

## Ground rules

1. **No runtime dependencies.** `dependencies` in `package.json` stays empty. A
   change that needs a library at runtime needs a very good argument first —
   open an issue before writing it. `devDependencies` are fine.
2. **No network.** Not at runtime, not in the test suite, not in CI beyond the
   package registry. The whole suite must pass on a laptop in aeroplane mode.
3. **Every formula carries its assumption.** If you add or change arithmetic,
   the comment above it says where the number comes from and what it assumes.
   "Empirical" is an acceptable answer; leaving it unsaid is not.
4. **Every formula carries a test.** New arithmetic needs a unit test with a
   hand-computed expected value, not a snapshot of what the code currently
   returns.

## Getting set up

Node 20 or newer, then:

```sh
git clone https://github.com/johan-becker/vramfit.git
cd vramfit
npm install
```

The verification chain, which is exactly what CI runs:

```sh
npm run lint       # oxlint, warnings are errors
npm run typecheck  # tsc --noEmit, covers test/ and scripts/ as well as src/
npm run build      # tsc + copy src/data into dist/data
npm test           # vitest
npm run smoke      # spawn the built binary, assert output and exit codes
```

`npm run build` compiles `src/` only, so `npm run typecheck` is the pass that
keeps the test files and build scripts honest.

## Where things live

| Path | What it holds |
| --- | --- |
| `src/quant.ts` | Quantization tables: effective bits per weight, KV block layouts, quality ranking |
| `src/architecture.ts` | Derives layer geometry, active parameters and MoE structure from a `ModelSpec` |
| `src/memory.ts` | Weight bytes, KV cache bytes, runtime context and compute buffer |
| `src/throughput.ts` | Bandwidth-bound decode, compute-bound prefill, efficiency tables |
| `src/offload.ts` | llama.cpp-style layer placement and blended bandwidth |
| `src/fit.ts` | Composes the above into a verdict, max context and quant recommendation |
| `src/compare.ts` | One model across several devices, and the ordering that makes it a table |
| `src/recommend.ts` | The capability x quality x speed ranking, and the use-case profiles |
| `src/fleet.ts` | The fleet file format and the machine-by-model placement matrix |
| `src/launcher.ts` | The llama.cpp, Ollama and vLLM flags a fit implies |
| `src/db/` | The bundled registry and the validator that every spec goes through |
| `src/gguf/` | Streaming GGUF header reader, and the mapping onto a `ModelSpec` and a `QuantSpec` |
| `src/hf/` | HuggingFace `config.json` mapping, and the safetensors headers that give the parameter count |
| `src/cli/` | Hand-rolled argument parsing, formatting and the pure `run(argv, io)` |
| `src/cli/bar.ts`, `color.ts` | The memory bar, and the rules that decide whether anything is coloured |
| `src/cli/explain.ts` | `--explain`: every headline number with its own arithmetic |
| `src/cli/markdown.ts` | `--markdown`: the same cells, through pipes instead of spaces |
| `src/data/*.json` | The bundled devices and models, each with a `source` string |

## Adding a device or a model

Both files are validated at load, so a bad entry fails the test suite rather
than producing a plausible wrong answer.

- Take every figure from a primary source: a vendor datasheet for a device, the
  model's published `config.json` for a model. Put that source in the `source`
  field. "A forum post said" is not a source.
- A model's parameter count is re-derived from its own architecture fields and
  must land within 0.5% of the count you wrote down (`test/db.test.ts`). If it
  does not, one of the two numbers is wrong — find out which before adjusting
  the tolerance.
- Give the entry the aliases people actually type (`4090`, `llama3.1:8b`).

## Changing the arithmetic

Calibration figures are ground truth and are not to be nudged to make a change
pass:

- `test/gguf-sizes.test.ts` holds published GGUF file sizes; predictions must
  stay within 2% on every dense row.
- `test/throughput.test.ts` holds published llama.cpp `tg128` results;
  estimates must stay inside the declared error band.

If your change moves those, the change is wrong or the error band is wrong —
say which in the pull request, and argue for it.

## Commits

Conventional commit subjects, imperative mood, lower case, no trailing period:

```
feat: estimate prefill speed and time to first token
fix: keep the CLI test suite type-clean under noUncheckedIndexedAccess
test: calibrate throughput against published benchmarks
docs: document the CI matrix and the smoke check
ci: run the verification chain on every supported platform
refactor: ...
chore: ...
```

Write a body whenever the *why* is not obvious from the subject — which, for
anything touching a formula, is always. Reference the source you used.

## Pull requests

Branch off `main`, one topic per branch. A pull request is expected to carry:

- the four-command verification chain passing locally,
- tests for anything new or changed, with hand-computed expectations,
- a `CHANGELOG.md` entry under `## [Unreleased]` for anything user-visible,
- a README update if you changed CLI behaviour, output, or a documented figure.
  Any example output printed in the README must be pasted from a real run.

CI runs the same chain on Node 20, 22 and 24 on Linux plus Node 22 on macOS and
Windows, and additionally packs the tarball and installs it into a clean
project. Both jobs must be green.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Contributor terms

vramfit is **open source** under the
[Apache License, Version 2.0](LICENSE). There is no CLA, no copyright
assignment and no separate commercial licence: inbound contributions are taken
under exactly the same terms the project ships under. Two things follow from
that, and both are stated here rather than buried in a form.

### Sign off every commit

Contributions are accepted under the
[Developer Certificate of Origin 1.1](DCO) — the full text is in the `DCO` file
at the root of this repository. There is no CLA to sign and no form to fill in;
you sign off in the commit itself:

```sh
git commit -s -m "fix: keep the KV block layout table in sync with quant.ts"
```

`-s` appends a line to the commit message:

```
Signed-off-by: Jane Doe <jane@example.com>
```

Use your real name and an address where you can be reached. By adding that
line you certify the DCO: in short, that you wrote the contribution or
otherwise have the right to submit it, and that you understand it will be
kept and redistributed as part of a public record. Commits without a
`Signed-off-by` line cannot be merged — if you forgot, `git commit --amend -s`
fixes the last one and `git rebase --signoff main` fixes a branch.

### What you grant, and what you keep

Section 5 of the [Apache License 2.0](LICENSE) already settles this: unless you
state otherwise in writing, anything you deliberately submit for inclusion in
vramfit is contributed under the terms of that licence. There is no extra grant
to sign, because none is needed.

You **keep the copyright in your contribution**. This is a licence, not an
assignment: you are not signing your work over, and you remain free to use your
own contribution however you like, elsewhere, for any purpose. The patent
licence in section 3 travels with the code, so everyone downstream receives the
same patent grant from you as from every other contributor.

If you are contributing as part of your job, make sure your employer is fine
with it before you open the pull request — that is exactly what clause (a) of
the DCO is asking you to confirm.

Bug reports, reproductions, benchmark data, device and model figures with
sources, and documentation corrections are all very welcome, and none of them
are blocked on any of the above.
