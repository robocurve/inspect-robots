import { readFileSync } from 'node:fs';
import { cloudflareTest } from '@cloudflare/vitest-plugin';
import { defineConfig } from 'vitest/config';
export default defineConfig({
  plugins: [
    { name: 'review-policy-text', enforce: 'pre', load(id) { if (id.endsWith('.md')) return `export default ${JSON.stringify(readFileSync(id, 'utf8'))}`; } },
    cloudflareTest({ main: './test/entry.ts', miniflare: {
      compatibilityDate: '2026-09-18', compatibilityFlags: ['nodejs_compat'],
      bindings: { REVIEW_PR_LIMITS_JSON: JSON.stringify({ '456': 25_000_000 }), REVIEW_HEAD_LIMITS_JSON: JSON.stringify({ '456-696fbaa9a00d7c345a81dd179fa10934f51ade89': 25_000_000 }) },
      durableObjects: { LEDGER: { className: 'ReviewLedger', useSQLite: true } },
    } }),
  ],
  test: { include: ['test/**/*.test.ts'], setupFiles: ['./test/setup.ts'] },
});
