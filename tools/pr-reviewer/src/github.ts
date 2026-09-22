import { importPKCS8, SignJWT } from 'jose';
import { boundedText, REPO } from './common';

export function allowedRead(path: string): boolean {
  if (/^\/compare\/[a-f0-9]{40}\.\.\.[a-f0-9]{40}\?per_page=1$/.test(path)) return true;
  if (path.includes('..') || path.includes('\\') || /[\r\n#]/.test(path)) return false;
  const url = new URL(`https://api.github.com/repos/${REPO}${path}`);
  return url.origin === 'https://api.github.com' && url.pathname.startsWith(`/repos/${REPO}/`) &&
    /^\/(pulls(?:\/\d+(?:\/(?:files|commits))?)?|issues\/\d+(?:\/comments)?|contents\/[^?]+|compare\/[a-f0-9]{40}\.\.\.[a-f0-9]{40}|git\/trees\/[a-f0-9]{40}|commits\/[a-f0-9]{40}\/(?:check-runs|status)|actions\/runs(?:\/\d+)?)($|\?)/.test(path);
}

export async function github(token: string, path: string, method = 'GET', body?: unknown): Promise<any> {
  const request = new Request(`https://api.github.com${path}`, {
    method, redirect: 'manual', signal: AbortSignal.timeout(30000),
    headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'inspect-robots-reviewer', 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const response = await fetch(request);
  if (!response.ok) throw new Error(`github_http_${response.status}`);
  return JSON.parse(await boundedText(response));
}

export async function installationToken(env: PublisherEnv, write: boolean): Promise<string> {
  let phase = 'import_key';
  try {
    const key = await importPKCS8(env.GITHUB_PRIVATE_KEY, 'RS256');
    phase = 'sign_jwt';
    const now = Math.floor(Date.now() / 1000);
    const jwt = await new SignJWT({}).setProtectedHeader({ alg: 'RS256' }).setIssuer(env.GITHUB_APP_ID).setIssuedAt(now - 60).setExpirationTime(now + 300).sign(key);
    phase = 'request_installation_token';
    const result = await github(jwt, `/app/installations/${env.GITHUB_INSTALLATION_ID}/access_tokens`, 'POST', {
      repositories: ['inspect-robots'],
      permissions: { contents: 'read', issues: 'read', actions: 'read', pull_requests: write ? 'write' : 'read', checks: write ? 'write' : 'read' },
    });
    return result.token;
  } catch (error) {
    console.error(JSON.stringify({ event: 'github_auth_failed', phase, name: error instanceof Error ? error.name : 'unknown' }));
    throw error;
  }
}

export async function ciGreen(read: (path: string) => Promise<any>, sha: string): Promise<boolean> {
  // Accept only the aggregate check produced by GitHub Actions, not an arbitrary status.
  const checks = await read(`/commits/${sha}/check-runs?filter=latest&per_page=100`);
  const candidates = checks.check_runs?.filter((c: any) => c.name === 'ci-ok' && c.app?.slug === 'github-actions' && c.head_sha === sha) ?? [];
  return candidates.length > 0 && candidates.every((c: any) => c.status === 'completed' && c.conclusion === 'success');
}
