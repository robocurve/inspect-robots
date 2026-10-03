import { importPKCS8, SignJWT } from "jose";
import { boundedText, REPO } from "./contracts";

export interface AppConfig {
  GITHUB_PRIVATE_KEY: string;
  GITHUB_APP_ID: string;
  GITHUB_INSTALLATION_ID: string;
}

export async function github(
  token: string,
  path: string,
  method = "GET",
  body?: unknown,
): Promise<any> {
  if (!path.startsWith("/") || path.includes("://") || /[\r\n]/.test(path))
    throw new Error("invalid_github_path");
  const response = await fetch(
    new Request(`https://api.github.com${path}`, {
      method,
      redirect: "manual",
      signal: AbortSignal.timeout(30000),
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "robocurve-issue-bot",
        "Content-Type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
  if (!response.ok) throw new Error(`github_http_${response.status}`);
  const text = await boundedText(response, 4_000_000);
  return text ? JSON.parse(text) : null;
}

export async function installationToken(
  env: AppConfig,
  write = false,
): Promise<string> {
  const key = await importPKCS8(env.GITHUB_PRIVATE_KEY, "RS256");
  const now = Math.floor(Date.now() / 1000);
  const jwt = await new SignJWT({})
    .setProtectedHeader({ alg: "RS256" })
    .setIssuer(env.GITHUB_APP_ID)
    .setIssuedAt(now - 60)
    .setExpirationTime(now + 300)
    .sign(key);
  const result = await github(
    jwt,
    `/app/installations/${env.GITHUB_INSTALLATION_ID}/access_tokens`,
    "POST",
    {
      repositories: ["inspect-robots"],
      permissions: {
        contents: write ? "write" : "read",
        issues: write ? "write" : "read",
        pull_requests: write ? "write" : "read",
        checks: "read",
      },
    },
  );
  return result.token;
}

export function allowedRead(path: string): boolean {
  return /^\/(issues\/[1-9][0-9]*(?:\/comments\?per_page=100(?:&page=[1-9][0-9]*)?)?|commits\/main|pulls\?state=open&per_page=100(?:&page=[1-9][0-9]*)?)$/.test(
    path,
  );
}

export const repoPath = (path: string) => `/repos/${REPO}${path}`;
