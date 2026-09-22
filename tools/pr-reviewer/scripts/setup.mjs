// Local operator utility. Credentials go only to the configured provider via stdin/HTTPS.
import { readFileSync, writeFileSync, mkdirSync, existsSync, chmodSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { createPrivateKey, randomBytes } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { SignJWT } from 'jose';

const directory = join(homedir(), '.config/robocurve-pr-reviewer');
const privateKeyPath = process.env.REVIEWER_PRIVATE_KEY_PATH ?? join(homedir(), 'Downloads/robocurve-pr-reviewer.2026-09-20.private-key.pem');
const secretPath = join(directory, 'github-webhook-secret');
const mode = process.argv[2];
try {
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  if (!existsSync(secretPath)) writeFileSync(secretPath, randomBytes(32).toString('hex'), { mode: 0o600, flag: 'wx' });
  chmodSync(secretPath, 0o600);
  const key = createPrivateKey(readFileSync(privateKeyPath));
  const secret = readFileSync(secretPath, 'utf8').trim();
  if (mode === 'secrets') {
    const apiKey = readFileSync(join(directory, 'openai-api-key'), 'utf8').trim();
    if (!apiKey.startsWith('sk-') || /\s/.test(apiKey)) throw new Error('invalid_api_key_file');
    for (const [config, name, value] of [
      ['publisher.wrangler.jsonc', 'GITHUB_PRIVATE_KEY', key.export({ type: 'pkcs8', format: 'pem' })],
      ['wrangler.jsonc', 'GITHUB_WEBHOOK_SECRET', secret],
      ['wrangler.jsonc', 'OPENAI_API_KEY', apiKey],
    ]) {
      const result = spawnSync(process.execPath, ['node_modules/wrangler/bin/wrangler.js', 'secret', 'put', name, '--config', config], { input: value, encoding: 'utf8' });
      if (result.status !== 0) throw new Error(`secret_upload_failed_${name}`);
      console.log(`Uploaded ${name} to ${config}.`);
    }
  } else if (mode === 'webhook') {
    const url = new URL(process.argv[3]);
    if (url.protocol !== 'https:' || !url.hostname.endsWith('.workers.dev') || url.pathname !== '/webhook') throw new Error('unexpected_webhook_url');
    const now = Math.floor(Date.now() / 1000);
    const jwt = await new SignJWT({}).setProtectedHeader({ alg: 'RS256' }).setIssuer('5012304').setIssuedAt(now - 60).setExpirationTime(now + 300).sign(key);
    const response = await fetch('https://api.github.com/app/hook/config', { method: 'PATCH', headers: { Authorization: `Bearer ${jwt}`, Accept: 'application/vnd.github+json', 'Content-Type': 'application/json', 'X-GitHub-Api-Version': '2022-11-28' }, body: JSON.stringify({ url: url.href, content_type: 'json', insecure_ssl: '0', secret }), signal: AbortSignal.timeout(30_000) });
    if (!response.ok) throw new Error(`webhook_config_http_${response.status}`);
    console.log(`Configured signed GitHub deliveries to ${url.href}.`);
  } else throw new Error('usage_setup_secrets_or_webhook_url');
} catch (error) {
  // Do not emit provider responses or exception objects that could contain credentials.
  console.error(error instanceof Error && /^[a-zA-Z0-9_]+$/.test(error.message) ? error.message : 'setup_failed');
  process.exitCode = 1;
}
