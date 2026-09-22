import { DurableObject, WorkerEntrypoint } from "cloudflare:workers";
import {
  artifactDigest,
  linksIssue,
  HASH,
  REPO,
  semanticRevision,
  SHA,
  validateFiles,
  type FixPublication,
  type Publication,
  type PublishedFix,
} from "./contracts";
import { allowedRead, github, installationToken, repoPath } from "./github";

const STATUSES = new Set([
  "CONFIRMED",
  "FIXING",
  "NEEDS_INFO",
  "NOT_REPRODUCED",
  "DUPLICATE",
  "FIX_PROPOSED",
  "REQUIRE_REVIEWER",
  "PLAN_APPROVED",
  "PR_READY",
]);
export function safeText(value: string, limit = 30000): string {
  const escaped = value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/@/g, "&#64;")
    .replace(/[\\`*_\[\]~]/g, "\\$&");
  return escaped.length <= limit
    ? escaped
    : escaped.slice(0, limit - 64) +
        "\n[Truncated for GitHub; full evidence is retained in the workflow.]";
}
export function renderNotice(
  input: Publication,
  author: { login: string; type: string },
): string {
  if (
    !STATUSES.has(input.status) ||
    !Number.isSafeInteger(input.costMicros) ||
    input.costMicros < 0
  )
    throw new Error("invalid_publication");
  const target =
    input.status === "NEEDS_INFO" &&
    author.type === "User" &&
    author.login !== "ghost" &&
    /^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,38})$/.test(author.login)
      ? author.login
      : "jeqcho";
  const action =
    input.status === "NEEDS_INFO"
      ? "Provide the missing information below."
      : input.status === "PR_READY"
        ? `Review PR #${input.pr}. The independent PR reviewer will assess the ready revision.`
        : input.status === "FIXING"
          ? "No action needed yet. The confirmed bug is entering the plan and independent review workflow."
          : input.status === "DUPLICATE"
            ? "Check the referenced original issue and decide whether to close this report as a duplicate."
            : input.status === "FIX_PROPOSED"
              ? "Review the existing fix PR and verify that it resolves this issue. No competing fix was started."
              : input.status === "REQUIRE_REVIEWER"
                ? "Resolve the specific blocker below before this workflow continues."
                : input.status === "NOT_REPRODUCED"
                  ? "Review the checks and limitations below and decide what further investigation is needed."
                  : "Review the assessment below.";
  return `**STATUS:** ${input.status}\n\n**Issue summary:** ${safeText(input.summary, 4000)}\n\n@${target}\n\n**Your action:** ${action}\n\n<details>\n<summary>Evidence and checks</summary>\n\n${safeText(input.details.join("\n\n"), 45000)}\n\nAutomated Astra/Codex assessment of base \`${input.issue.base}\`.\n</details>\n\nRecorded issue workflow spending: $${(input.costMicros / 1_000_000).toFixed(3)}. No merge or issue closure was performed.`;
}

function validateIdentity(input: {
  jobId: string;
  issue: { number: number; revision: string; base: string };
}): void {
  if (
    !/^[a-f0-9]{32,64}$/.test(input.jobId) ||
    !Number.isSafeInteger(input.issue.number) ||
    input.issue.number < 1 ||
    !HASH.test(input.issue.revision) ||
    !SHA.test(input.issue.base)
  )
    throw new Error("invalid_publication");
}

/** One serialized publication journal per issue job; remote writes reconcile after a crash. */
export class PublicationJournal extends DurableObject<PublisherEnv> {
  private chain: Promise<unknown> = Promise.resolve();
  private serialize<T>(fn: () => Promise<T>): Promise<T> {
    const task = this.chain.then(fn);
    this.chain = task.catch(() => undefined);
    return task;
  }
  private async readIssue(token: string, input: Publication | FixPublication) {
    const issue = await github(
      token,
      repoPath(`/issues/${input.issue.number}`),
    );
    if (
      issue.pull_request ||
      (await semanticRevision(issue)) !== input.issue.revision ||
      issue.state !== "open"
    )
      throw new Error("issue_changed");
    return issue;
  }
  async publish(input: Publication): Promise<boolean> {
    return this.serialize(async () => {
      validateIdentity(input);
      const token = await installationToken(this.env, true);
      const issue = await github(
        token,
        repoPath(`/issues/${input.issue.number}`),
      );
      if (issue.pull_request) throw new Error("invalid_issue");
      // Superseded assessments are retired rather than retried forever. A trusted
      // stop notice must still explain why an edited issue's workflow stopped.
      if (issue.state !== "open") return true;
      const stale = (await semanticRevision(issue)) !== input.issue.revision;
      if (stale && input.status !== "REQUIRE_REVIEWER") return true;
      if (stale)
        input = {
          ...input,
          details: [
            "The issue changed after this workflow began. This stop notice concerns the earlier saved revision; no fix was published for the changed report.",
            ...input.details,
          ],
        };
      const marker = `<!-- robocurve-issue-bot:${input.jobId}:${input.status} -->`;
      const body = `${renderNotice(input, issue.user)}\n\n${marker}`;
      await this.ctx.storage.put(`intent-${input.status}`, body);
      let previous: any;
      for (let page = 1; page <= 10; page++) {
        const comments = await github(
          token,
          repoPath(
            `/issues/${input.issue.number}/comments?per_page=100&page=${page}`,
          ),
        );
        previous = comments.find(
          (c: any) =>
            String(c.performed_via_github_app?.id) === this.env.GITHUB_APP_ID &&
            c.body?.includes(marker),
        );
        if (previous || comments.length < 100) break;
        if (page === 10) throw new Error("comment_pagination_limit");
      }
      if (previous?.body !== body) {
        const latest = await github(
          token,
          repoPath(`/issues/${input.issue.number}`),
        );
        if (latest.state !== "open") return true;
        if (
          input.status !== "REQUIRE_REVIEWER" &&
          (await semanticRevision(latest)) !== input.issue.revision
        )
          return true;
        previous = await github(
          token,
          repoPath(
            previous
              ? `/issues/comments/${previous.id}`
              : `/issues/${input.issue.number}/comments`,
          ),
          previous ? "PATCH" : "POST",
          { body },
        );
      }
      await this.ctx.storage.put(`delivered-${input.status}`, previous.id);
      return true;
    });
  }
  private async verifyFix(token: string, input: FixPublication) {
    validateIdentity(input);
    const files = validateFiles(input.files);
    if (
      !files.length ||
      (await artifactDigest(files)) !== input.artifactDigest ||
      input.approvedDigest !== input.artifactDigest ||
      !input.plan.trim() ||
      !input.checks.length
    )
      throw new Error("unapproved_artifact");
    await this.readIssue(token, input);
    const main = await github(token, repoPath("/commits/main"));
    if (main.sha !== input.issue.base) throw new Error("base_changed");
    return main;
  }
  async createFix(input: FixPublication): Promise<PublishedFix> {
    return this.serialize(async () => {
      const token = await installationToken(this.env, true);
      const main = await this.verifyFix(token, input);
      const branch = `issue-bot/${input.issue.number}-${input.jobId.slice(0, 12)}`;
      await this.verifyNoCompetingFix(token, input.issue.number, branch);
      const marker = `<!-- robocurve-issue-fix:${input.jobId}:${input.artifactDigest} -->`;
      let date = await this.ctx.storage.get<string>("commit-date");
      if (!date) {
        date = new Date().toISOString();
        await this.ctx.storage.put("commit-date", date);
      }
      const tree = await github(token, repoPath("/git/trees"), "POST", {
        base_tree: main.commit.tree.sha,
        tree: input.files.map((f) => ({
          path: f.path,
          mode: f.mode,
          type: "blob",
          ...(f.content === null ? { sha: null } : { content: f.content }),
        })),
      });
      const identity = {
        name: "robocurve-issue-bot",
        email: "robocurve-issue-bot@users.noreply.github.com",
        date,
      };
      const commit = await github(token, repoPath("/git/commits"), "POST", {
        message: `Fix issue #${input.issue.number}\n\n${marker}`,
        tree: tree.sha,
        parents: [input.issue.base],
        author: identity,
        committer: identity,
      });
      let ref;
      try {
        ref = await github(token, repoPath(`/git/ref/heads/${branch}`));
      } catch (error) {
        if (!(error instanceof Error) || error.message !== "github_http_404")
          throw error;
      }
      if (ref && ref.object.sha !== commit.sha)
        throw new Error("branch_changed");
      if (!ref)
        await github(token, repoPath("/git/refs"), "POST", {
          ref: `refs/heads/${branch}`,
          sha: commit.sha,
        });
      const prs = await github(
        token,
        repoPath(
          `/pulls?state=all&head=robocurve:${branch}&base=main&per_page=100`,
        ),
      );
      if (prs.length > 1) throw new Error("ambiguous_pull_request");
      let pr = prs[0];
      if (!pr) {
        const body = `${safeText(input.summary, 4000)}\n\nFixes #${input.issue.number}.\n\nAutomated fix by robocurve-issue-bot. @jeqcho owns the merge decision.\n\n<details>\n<summary>Reviewed plan and validation</summary>\n\n${safeText(input.plan, 20000)}\n\n${safeText(input.checks.join("\n\n"), 24000)}\n\nApproved artifact: \`${input.artifactDigest}\`\n</details>\n\n${marker}`;
        pr = await github(token, repoPath("/pulls"), "POST", {
          title: `Fix #${input.issue.number}: ${input.issue.title.slice(0, 180)}`,
          body,
          head: branch,
          base: "main",
          draft: true,
        });
      }
      this.verifyPr(pr, commit.sha, marker);
      const result = {
        number: pr.number,
        nodeId: pr.node_id,
        head: commit.sha,
        url: pr.html_url,
      };
      await this.ctx.storage.put("fix", result);
      return result;
    });
  }
  private verifyPr(pr: any, head: string, marker: string): void {
    if (
      pr.state !== "open" ||
      pr.head?.sha !== head ||
      pr.head?.repo?.full_name !== REPO ||
      pr.base?.ref !== "main" ||
      pr.base?.repo?.full_name !== REPO ||
      pr.user?.type !== "Bot" ||
      pr.user?.login !== `${this.env.GITHUB_APP_SLUG}[bot]` ||
      !pr.body?.includes(marker)
    )
      throw new Error("pull_request_changed");
  }
  private async verifyNoCompetingFix(
    token: string,
    issue: number,
    branch: string,
  ): Promise<void> {
    for (let page = 1; page <= 5; page++) {
      const pulls = await github(
        token,
        repoPath(`/pulls?state=open&per_page=100&page=${page}`),
      );
      if (
        pulls.some(
          (p: any) =>
            p.head?.ref !== branch && linksIssue(String(p.body ?? ""), issue),
        )
      )
        throw new Error("competing_fix");
      if (pulls.length < 100) return;
    }
    throw new Error("duplicate_search_incomplete");
  }
  async ready(
    input: FixPublication,
    published: PublishedFix,
  ): Promise<boolean> {
    return this.serialize(async () => {
      const saved = await this.ctx.storage.get<PublishedFix>("fix");
      if (!saved || JSON.stringify(saved) !== JSON.stringify(published))
        throw new Error("unrecognized_pull_request");
      const token = await installationToken(this.env, true);
      await this.verifyFix(token, input);
      await this.verifyNoCompetingFix(
        token,
        input.issue.number,
        `issue-bot/${input.issue.number}-${input.jobId.slice(0, 12)}`,
      );
      const marker = `<!-- robocurve-issue-fix:${input.jobId}:${input.artifactDigest} -->`;
      const pr = await github(token, repoPath(`/pulls/${published.number}`));
      this.verifyPr(pr, published.head, marker);
      const checks = await github(
        token,
        repoPath(
          `/commits/${published.head}/check-runs?filter=latest&per_page=100`,
        ),
      );
      const ci =
        checks.check_runs?.filter(
          (c: any) =>
            c.name === "ci-ok" &&
            c.app?.slug === "github-actions" &&
            c.head_sha === published.head,
        ) ?? [];
      if (!ci.length || ci.some((c: any) => c.status !== "completed"))
        return false;
      if (ci.some((c: any) => c.conclusion !== "success"))
        throw new Error("ci_failed");
      if (pr.draft) {
        const data = await github(token, "/graphql", "POST", {
          query:
            "mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id}){pullRequest{isDraft}}}",
          variables: { id: published.nodeId },
        });
        if (
          data.errors?.length ||
          data.data?.markPullRequestReadyForReview?.pullRequest?.isDraft !==
            false
        )
          throw new Error("ready_failed");
      }
      await this.ctx.storage.put("ready", published.head);
      return true;
    });
  }
}

export class IssuePublisher extends WorkerEntrypoint<PublisherEnv> {
  async read(path: string): Promise<string> {
    if (!allowedRead(path)) throw new Error("read_not_allowed");
    return JSON.stringify(
      await github(await installationToken(this.env), repoPath(path)),
    );
  }
  async publish(input: Publication) {
    validateIdentity(input);
    return this.env.PUBLICATIONS.getByName(input.jobId).publish(input);
  }
  async createFix(input: FixPublication) {
    validateIdentity(input);
    return this.env.PUBLICATIONS.getByName(input.jobId).createFix(input);
  }
  async ready(input: FixPublication, published: PublishedFix) {
    validateIdentity(input);
    return this.env.PUBLICATIONS.getByName(input.jobId).ready(input, published);
  }
}
export default {
  fetch() {
    return new Response("Not found", { status: 404 });
  },
};
