import { cloudflareTest } from "@cloudflare/vitest-plugin";
import { defineConfig } from "vitest/config";
export default defineConfig({
  plugins: [
    cloudflareTest({
      main: "./test/entry.ts",
      miniflare: {
        compatibilityDate: "2026-09-20",
        compatibilityFlags: ["nodejs_compat"],
        bindings: {
          ENABLED: "true",
          INSTALLATION_ID: "123",
          ISSUE_LIMIT_MICROS: "20000000",
          MONTH_LIMIT_MICROS: "200000000",
          GITHUB_WEBHOOK_SECRET: "test-secret",
        },
        durableObjects: {
          LEDGER: { className: "IssueLedger", useSQLite: true },
          PUBLICATIONS: { className: "PublicationJournal", useSQLite: true },
        },
      },
    }),
  ],
  test: { include: ["test/**/*.test.ts"], setupFiles: ["./test/setup.ts"] },
});
