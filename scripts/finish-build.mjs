#!/usr/bin/env node
// Post-`tsc` build step.
//
// 1. tsc does not copy non-TS assets, and the bundled device/model database is
//    plain JSON loaded at runtime relative to `import.meta.url`. Copying
//    src/data -> dist/data keeps that relative path identical in both the
//    source tree (vitest) and the published tree (node dist/...).
// 2. npm sets the executable bit on `bin` entries at install time, but not for
//    a local `npm run build`, so we set it here to keep ./dist/bin/vramfit.js
//    runnable straight out of the build.
import { chmodSync, cpSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

const dataSrc = join(root, "src", "data");
const dataDest = join(root, "dist", "data");
cpSync(dataSrc, dataDest, { recursive: true });

const bin = join(root, "dist", "bin", "vramfit.js");
if (existsSync(bin)) {
  chmodSync(bin, 0o755);
}

console.log(`finish-build: copied ${dataSrc} -> ${dataDest}`);
