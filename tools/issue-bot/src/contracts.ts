import { z } from "zod";

export const REPO = "robocurve/inspect-robots";
export const MAINTAINER_ID = 42904912;
export const MODEL = "gpt-6-astra";
export function linksIssue(body: string, number: number): boolean {
  return [
    ...body.matchAll(
      /\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+(?:https:\/\/github\.com\/robocurve\/inspect-robots\/issues\/|#)(\d+)\b/gi,
    ),
  ].some((match) => Number(match[1]) === number);
}
export const SHA = /^[a-f0-9]{40}$/;
export const HASH = /^[a-f0-9]{64}$/;
export const StageKind = z.enum([
  "triage",
  "plan",
  "plan_review",
  "implement",
  "code_review",
]);
export type StageKind = z.infer<typeof StageKind>;
export const FileChange = z
  .object({
    path: z.string().min(1).max(300),
    content: z.string().max(200_000).nullable(),
    mode: z.enum(["100644", "100755"]),
  })
  .strict();
export type FileChange = z.infer<typeof FileChange>;
export const StageResult = z
  .object({
    status: z.enum([
      "CONFIRMED",
      "NEEDS_INFO",
      "NOT_REPRODUCED",
      "DUPLICATE",
      "FIX_PROPOSED",
      "REQUIRE_REVIEWER",
      "PLAN",
      "APPROVE",
      "REQUEST_CHANGES",
      "IMPLEMENTED",
    ]),
    serious: z.boolean(),
    summary: z.string().min(1).max(3000),
    evidence: z.array(z.string().max(3000)).max(20),
    plan: z.string().max(16000),
    findings: z.array(z.string().max(3000)).max(20),
    checks: z.array(z.string().max(1000)).max(30),
    limitations: z.array(z.string().max(2000)).max(20),
  })
  .strict();
export type StageResult = z.infer<typeof StageResult>;
export const ExecutionRecord = z
  .object({
    command: z.string().max(12000),
    exitCode: z.number().int().nullable(),
  })
  .strict();
export const StageOutput = z
  .object({
    exitCode: z.number().int(),
    failure: z.string().max(100).nullable(),
    result: StageResult.nullable(),
    files: z.array(FileChange).max(80),
    executions: z.array(ExecutionRecord).max(150),
  })
  .strict();
export type StageOutput = z.infer<typeof StageOutput>;
export interface IssueSnapshot {
  number: number;
  title: string;
  body: string;
  author: string;
  authorId: number;
  state: string;
  revision: string;
  base: string;
}
export interface StageRequest {
  id: string;
  jobId: string;
  kind: StageKind;
  base: string;
  issue: IssueSnapshot;
  context: string;
  plan: string;
  feedback: string;
  files: FileChange[];
  inputDigest: string;
  token: string;
  checkpointToken: string;
  sandbox: string;
}
export interface Publication {
  jobId: string;
  issue: IssueSnapshot;
  status: string;
  summary: string;
  details: string[];
  costMicros: number;
  pr?: number;
}
export interface FixPublication {
  jobId: string;
  issue: IssueSnapshot;
  files: FileChange[];
  artifactDigest: string;
  approvedDigest: string;
  plan: string;
  summary: string;
  checks: string[];
}
export interface PublishedFix {
  number: number;
  nodeId: string;
  head: string;
  url: string;
}

export async function digest(value: string): Promise<string> {
  return Array.from(
    new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
    ),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
}
export async function artifactDigest(files: FileChange[]): Promise<string> {
  return digest(
    JSON.stringify([...files].sort((a, b) => a.path.localeCompare(b.path))),
  );
}
export async function semanticRevision(issue: {
  title: string;
  body?: string | null;
  state: string;
}): Promise<string> {
  return digest(JSON.stringify([issue.title, issue.body ?? "", issue.state]));
}
export async function boundedText(
  input: Request | Response,
  max = 2_000_000,
): Promise<string> {
  if (!input.body) return "";
  const reader = input.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    size += part.value.length;
    if (size > max) {
      await reader.cancel();
      throw new Error("body_too_large");
    }
    chunks.push(part.value);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return new TextDecoder().decode(bytes);
}

export function validateFiles(input: unknown): FileChange[] {
  const files = z.array(FileChange).max(80).parse(input);
  const seen = new Set<string>();
  let size = 0;
  for (const file of files) {
    if (
      !/^(src|tests|plugins|docs|examples|plans)\/[A-Za-z0-9_./-]+$/.test(
        file.path,
      ) ||
      file.path
        .split("/")
        .some((p) => !p || p === "." || p === ".." || p.startsWith(".")) ||
      /(^|\/)(AGENTS\.md|CLAUDE\.md|pyproject\.toml|uv\.lock|package(?:-lock)?\.json|.*\.(?:pem|key|env))$/i.test(
        file.path,
      ) ||
      seen.has(file.path) ||
      file.content?.includes("\u0000")
    )
      throw new Error("unsafe_artifact");
    seen.add(file.path);
    size += new TextEncoder().encode(file.content ?? "").length;
  }
  if (size > 1_000_000) throw new Error("artifact_too_large");
  return files;
}
