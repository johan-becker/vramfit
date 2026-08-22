# Security Policy

## Supported versions

`vramfit` is developed on `main` and released from tags. Security fixes land on
the latest minor release; older minors are not backported.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| < 0.1 | No |

## Reporting a vulnerability

Report privately to **jo_becker@mailbox.org**. Do not open a public issue for a
vulnerability.

Please include:

- what the issue is and which version or commit you saw it on,
- the smallest input that reproduces it (a CLI invocation, a `--model-json` /
  `--device-json` file, or a few lines of library code),
- what you believe an attacker gains.

You can expect an acknowledgement within 72 hours and an assessment within
seven days. If the report is accepted, I will agree a disclosure date with you
and credit you in the release notes unless you ask me not to.

## Threat model

`vramfit` is an offline arithmetic tool. It has **zero runtime dependencies**,
opens **no network connections** at runtime or during its test suite, spawns no
subprocesses, and writes no files. Its only I/O is reading JSON: the bundled
`dist/data/*.json` database, and any file you point `--model-json` or
`--device-json` at.

That makes the interesting surface small and specific:

- **Untrusted spec files.** `--model-json` and `--device-json` parse arbitrary
  JSON. Every field goes through the same validator as the bundled database,
  which rejects wrong types, non-finite numbers, negative sizes and violated
  cross-field invariants before any arithmetic runs. A parse or validation
  failure exits 2 with a field path. A crafted spec that causes a hang, an
  unhandled exception, a prototype-pollution effect, or a stack overflow is a
  vulnerability — please report it.
- **Library consumers.** `parseModelSpec` and `parseDeviceSpec` are exported
  precisely so that an application can validate untrusted input with them. A
  value that gets past them and then breaks the arithmetic is a vulnerability.

Out of scope: the accuracy of an estimate, a bundled device or model figure
being out of date, and anything requiring you to already be able to run code as
the same user. Those are ordinary issues — open one.
