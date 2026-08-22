## What this changes

<!-- One or two sentences. If it changes a number anyone could rely on, say
     which number and by how much. -->

## Why

<!-- For anything touching a formula: where the number comes from and what it
     assumes. Link the datasheet, config.json, or benchmark table. -->

## Verification

<!-- Paste the tail of each command, not just a claim that it passed. -->

- [ ] `npm run lint`
- [ ] `npm run build`
- [ ] `npm test`
- [ ] `npm run smoke`
- [ ] `npm run typecheck`

```console
$ npm test
...
```

## Checklist

- [ ] No new runtime dependency (`dependencies` is still empty).
- [ ] No network access at runtime or in tests.
- [ ] New or changed arithmetic has a unit test with a hand-computed expectation.
- [ ] Any formula I added or changed carries its assumption in a comment.
- [ ] The published-GGUF and llama.cpp calibration tests still pass unmodified,
      or I explain below why the ground truth or the error band had to move.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]` for anything
      user-visible.
- [ ] README updated if CLI behaviour, output, or a documented figure changed —
      and every example printed there is pasted from a real run.

## Notes for the reviewer

<!-- Anything you are unsure about, or a decision you would like challenged. -->
