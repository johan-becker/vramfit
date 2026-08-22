#!/usr/bin/env node
import { run } from "../cli/index.js";

// The CLI proper is a pure function of argv; this file exists only to connect
// it to the process, so that everything worth testing is testable without
// spawning one.
process.exitCode = run(process.argv.slice(2));
