#!/usr/bin/env node
import { EXIT_USAGE, run } from "../cli/index.js";

// The CLI proper is a pure function of argv; this file exists only to connect
// it to the process, so that everything worth testing is testable without
// spawning one.
//
// `run` turns every failure it knows about into an exit code, and this guard
// is here so that one it does not know about still cannot exit 1: that code
// means "does not fit", and a crash reported as a verdict would silently fail
// a deploy gate written against it.
try {
  process.exitCode = run(process.argv.slice(2));
} catch (error) {
  process.stderr.write(`vramfit: ${(error as Error).message}\n`);
  process.exitCode = EXIT_USAGE;
}
