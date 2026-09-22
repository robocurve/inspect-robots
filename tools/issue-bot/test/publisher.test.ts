import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  artifactDigest,
  semanticRevision,
  validateFiles,
  type FixPublication,
  type Publication,
} from "../src/contracts";
import { PublicationJournal, renderNotice } from "../src/publisher";
import { allowedRead, github } from "../src/github";

vi.mock("../src/github", async (original) => ({
  ...(await original<typeof import("../src/github")>()),
  github: vi.fn(),
  installationToken: vi.fn(async () => "test-token"),
}));
const base = "a".repeat(40),
  head = "b".repeat(40),
  jobId = "c".repeat(48);
let input: FixPublication;
let comments: any[];
let pr: any;
let branch: any;
let ci: string;
let failComment: boolean;
let calls: string[];
let issueChanged: boolean;
let competitor: boolean;
function journal(): PublicationJournal {
  const values = new Map<string, unknown>();
  return Object.assign(Object.create(PublicationJournal.prototype), {
    chain: Promise.resolve(),
    env: { GITHUB_APP_ID: "123", GITHUB_APP_SLUG: "robocurve-issue-bot" },
    ctx: {
      storage: {
        get: async (k: string) => values.get(k),
        put: async (k: string, v: unknown) => {
          values.set(k, v);
        },
      },
    },
  });
}
beforeEach(async () => {
  comments = [];
  pr = null;
  branch = null;
  ci = "pending";
  failComment = false;
  calls = [];
  issueChanged = false;
  competitor = false;
  const issue = {
    number: 401,
    title: "Logs lost",
    body: "Failure loses logs",
    author: "jeqcho",
    authorId: 42904912,
    state: "open",
    revision: "",
    base,
  };
  issue.revision = await semanticRevision(issue);
  const files = [
    {
      path: "src/inspect_robots/fix.py",
      content: "fixed\n",
      mode: "100644" as const,
    },
  ];
  input = {
    jobId,
    issue,
    files,
    artifactDigest: await artifactDigest(files),
    approvedDigest: await artifactDigest(files),
    plan: "Preserve logs and add regression.",
    summary: "Preserve failure logs.",
    checks: ["pytest tests/test_fix.py: exit 0"],
  };
  vi.mocked(github).mockImplementation(
    async (_token, path, method = "GET", body?: any) => {
      calls.push(`${method} ${path}`);
      if (path.endsWith("/issues/401"))
        return {
          ...issue,
          ...(issueChanged ? { body: "Changed report" } : {}),
          user: { login: "jeqcho", type: "User" },
        };
      if (path.endsWith("/commits/main"))
        return { sha: base, commit: { tree: { sha: "d".repeat(40) } } };
      if (path.includes("/comments?")) return comments;
      if (path.endsWith("/issues/401/comments")) {
        const comment = {
          id: 1,
          body: body.body,
          performed_via_github_app: { id: 123 },
        };
        comments.push(comment);
        if (failComment) {
          failComment = false;
          throw new Error("github_http_502");
        }
        return comment;
      }
      if (path.endsWith("/git/trees")) return { sha: "e".repeat(40) };
      if (path.endsWith("/git/commits")) return { sha: head };
      if (path.includes("/git/ref/heads/")) {
        if (!branch) throw new Error("github_http_404");
        return branch;
      }
      if (path.endsWith("/git/refs")) {
        branch = { object: { sha: body.sha } };
        return branch;
      }
      if (path.includes("/pulls?state=open"))
        return [
          ...(pr ? [pr] : []),
          ...(competitor
            ? [
                {
                  number: 404,
                  head: { ref: "contributor-fix" },
                  body: "Fixes #401",
                },
              ]
            : []),
        ];
      if (path.includes("/pulls?")) return pr ? [pr] : [];
      if (path.endsWith("/pulls") && method === "POST") {
        pr = {
          number: 999,
          node_id: "node999",
          html_url: "https://github.com/robocurve/inspect-robots/pull/999",
          draft: true,
          state: "open",
          body: body.body,
          head: {
            sha: head,
            ref: `issue-bot/401-${jobId.slice(0, 12)}`,
            repo: { full_name: "robocurve/inspect-robots" },
          },
          base: {
            ref: "main",
            repo: { full_name: "robocurve/inspect-robots" },
          },
          user: { type: "Bot", login: "robocurve-issue-bot[bot]" },
        };
        return pr;
      }
      if (path.endsWith("/pulls/999")) return pr;
      if (path.includes("/check-runs?"))
        return {
          check_runs: [
            {
              name: "ci-ok",
              head_sha: head,
              app: { slug: "github-actions" },
              status: ci === "pending" ? "in_progress" : "completed",
              conclusion: ci,
            },
          ],
        };
      if (path === "/graphql") {
        pr.draft = false;
        return {
          data: {
            markPullRequestReadyForReview: { pullRequest: { isDraft: false } },
          },
        };
      }
      throw new Error(`unexpected mock path ${path}`);
    },
  );
});
describe("trusted issue publication", () => {
  it("routes information requests only to a trusted human author and escapes injected mentions", () => {
    const notice: Publication = {
      jobId,
      issue: input.issue,
      status: "NEEDS_INFO",
      summary: "@attacker <script>x</script>",
      details: ["@another"],
      costMicros: 10,
    };
    const text = renderNotice(notice, { login: "reporter", type: "User" });
    expect(text).toContain("\n\n@reporter\n\n**Your action:** Provide");
    expect(text).not.toContain("@attacker");
    expect(text).not.toContain("<script>");
    expect(
      renderNotice(notice, { login: "thing[bot]", type: "Bot" }),
    ).toContain("@jeqcho");
    expect(renderNotice(notice, { login: "ghost", type: "User" })).toContain(
      "@jeqcho",
    );
    expect(
      renderNotice(
        { ...notice, status: "CONFIRMED" },
        { login: "reporter", type: "User" },
      ),
    ).toContain("@jeqcho");
  });
  it("delivers stale stop notices but retires obsolete assessments after an issue edit", async () => {
    const j = journal();
    issueChanged = true;
    const notice: Publication = {
      jobId,
      issue: input.issue,
      status: "CONFIRMED",
      summary: "Old assessment",
      details: [],
      costMicros: 0,
    };
    expect(await j.publish(notice)).toBe(true);
    expect(comments).toHaveLength(0);
    expect(await j.publish({ ...notice, status: "REQUIRE_REVIEWER" })).toBe(
      true,
    );
    expect(comments).toHaveLength(1);
    expect(comments[0].body).toContain("issue changed");
    await expect(j.createFix(input)).rejects.toThrow("issue_changed");
  });
  it("caps escaped prose within GitHub limits even with adversarial expansion", () => {
    const notice: Publication = {
      jobId,
      issue: input.issue,
      status: "NEEDS_INFO",
      summary: "@".repeat(3000),
      details: ["&".repeat(60000), "@".repeat(40000)],
      costMicros: 0,
    };
    const body = renderNotice(notice, { login: "reporter", type: "User" });
    expect(body.length).toBeLessThan(60000);
    expect(body).toContain("Truncated for GitHub");
  });
  it("reconciles a comment saved by GitHub before its acknowledgement was lost", async () => {
    const j = journal();
    failComment = true;
    const notice: Publication = {
      jobId,
      issue: input.issue,
      status: "REQUIRE_REVIEWER",
      summary: "Validation incomplete",
      details: ["Missing dependencies"],
      costMicros: 0,
    };
    await expect(j.publish(notice)).rejects.toThrow("github_http_502");
    await expect(j.publish(notice)).resolves.toBe(true);
    expect(comments).toHaveLength(1);
  });
  it("creates one draft PR on replay, then waits for exact-head CI before ready", async () => {
    const j = journal();
    const first = await j.createFix(input);
    const second = await j.createFix(input);
    expect(second).toEqual(first);
    expect(
      calls.filter((c) => c === "POST /repos/robocurve/inspect-robots/pulls"),
    ).toHaveLength(1);
    expect(pr.draft).toBe(true);
    expect(await j.ready(input, first)).toBe(false);
    ci = "success";
    expect(await j.ready(input, first)).toBe(true);
    expect(pr.draft).toBe(false);
    expect(await j.ready(input, first)).toBe(true);
    expect(calls.filter((c) => c === "POST /graphql")).toHaveLength(1);
  });
  it("refuses an unapproved artifact before making any GitHub write", async () => {
    await expect(
      journal().createFix({ ...input, approvedDigest: "f".repeat(64) }),
    ).rejects.toThrow("unapproved_artifact");
    expect(calls.some((c) => c.startsWith("POST"))).toBe(false);
  });
  it("never readies a changed head or a failed CI run", async () => {
    const j = journal();
    const fix = await j.createFix(input);
    ci = "failure";
    await expect(j.ready(input, fix)).rejects.toThrow("ci_failed");
    ci = "success";
    pr.head.sha = "0".repeat(40);
    await expect(j.ready(input, fix)).rejects.toThrow("pull_request_changed");
    expect(calls).not.toContain("POST /graphql");
  });
  it("does not take over a preexisting unrelated branch", async () => {
    branch = { object: { sha: "f".repeat(40) } };
    await expect(journal().createFix(input)).rejects.toThrow("branch_changed");
    expect(calls.some((c) => c.startsWith("PATCH"))).toBe(false);
  });
  it("stops when a competing fix appears after the original triage", async () => {
    competitor = true;
    await expect(journal().createFix(input)).rejects.toThrow("competing_fix");
    expect(calls.some((c) => c.startsWith("POST"))).toBe(false);
  });
  it("rejects automation, credentials, traversal and duplicate paths", () => {
    for (const path of [
      ".github/workflows/ci.yml",
      "tools/issue-bot/src/worker.ts",
      "src/../secret",
      "src/.env",
      "src/AGENTS.md",
      "plugins/x/pyproject.toml",
    ])
      expect(() =>
        validateFiles([{ path, content: "x", mode: "100644" }]),
      ).toThrow();
    expect(() => validateFiles([input.files[0], input.files[0]])).toThrow();
    expect(allowedRead("/issues/401/comments?per_page=100")).toBe(true);
    expect(allowedRead("/pulls/455/merge")).toBe(false);
  });
});
