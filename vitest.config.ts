import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
    // The whole point of this package is arithmetic; nothing here talks to the
    // network, and nothing here is allowed to. Tests must pass offline.
    globals: false,
  },
});
