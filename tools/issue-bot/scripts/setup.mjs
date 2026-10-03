// Register a separate GitHub App. Secrets never enter logs, browser HTML, or argv.
import { createServer } from "node:http";
import { randomBytes, createPrivateKey, sign } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { parse as parseJsonc } from "jsonc-parser";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const secretDir = join(root, ".secrets");
const credentialsPath = join(secretDir, "github-app.json");
const mode = process.argv[2];
const endpoint = "https://robocurve-issue-bot.jay-7f4.workers.dev/webhook";
const headers = {
  Accept: "application/vnd.github+json",
  "X-GitHub-Api-Version": "2022-11-28",
  "Content-Type": "application/json",
  "User-Agent": "robocurve-issue-bot-setup",
};
const html = (text) =>
  text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
function save(value) {
  mkdirSync(secretDir, { recursive: true, mode: 0o700 });
  writeFileSync(credentialsPath, JSON.stringify(value), { mode: 0o600 });
}
function jwt(app) {
  const now = Math.floor(Date.now() / 1000);
  const payload = [
    { alg: "RS256", typ: "JWT" },
    { iat: now - 60, exp: now + 300, iss: String(app.id) },
  ]
    .map((x) => Buffer.from(JSON.stringify(x)).toString("base64url"))
    .join(".");
  return `${payload}.${sign("RSA-SHA256", Buffer.from(payload), createPrivateKey(app.pem)).toString("base64url")}`;
}
async function api(path, token, method = "GET", body) {
  // Assign only the pathname: a callback-derived value cannot change the origin.
  const url = new URL("https://api.github.com");
  url.pathname = path;
  const response = await fetch(url, {
    method,
    headers: {
      ...headers,
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    redirect: "error",
    signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) throw new Error(`github_http_${response.status}`);
  return response.json();
}
async function installation(app) {
  const installations = await api("/app/installations", jwt(app));
  const found = installations.filter((i) => i.account?.login === "robocurve");
  if (found.length !== 1) throw new Error("install_on_robocurve_first");
  const item = found[0];
  if (
    item.permissions.contents !== "write" ||
    item.permissions.issues !== "write" ||
    item.permissions.pull_requests !== "write" ||
    item.permissions.checks !== "read" ||
    Object.keys(item.permissions).some(
      (key) =>
        !["contents", "issues", "pull_requests", "checks", "metadata"].includes(
          key,
        ),
    ) ||
    item.repository_selection !== "selected"
  )
    throw new Error("incorrect_app_permissions");
  const auth = await api(
    `/app/installations/${item.id}/access_tokens`,
    jwt(app),
    "POST",
    {
      permissions: { metadata: "read" },
    },
  );
  const repos = await api("/installation/repositories", auth.token);
  if (
    repos.total_count !== 1 ||
    repos.repositories.length !== 1 ||
    repos.repositories[0].full_name !== "robocurve/inspect-robots"
  )
    throw new Error("inspect_robots_not_selected");
  app.installation_id = item.id;
  save(app);
  console.log(
    `Installed ${app.slug}: App ${app.id}, installation ${item.id}. Credentials saved privately.`,
  );
  return app;
}
function configure(app) {
  for (const filename of ["wrangler.jsonc", "publisher.wrangler.jsonc"]) {
    const path = join(root, filename);
    const errors = [];
    const config = parseJsonc(readFileSync(path, "utf8"), errors, {
      allowTrailingComma: true,
    });
    if (errors.length) throw new Error("invalid_worker_config");
    if (filename === "wrangler.jsonc")
      config.vars.INSTALLATION_ID = String(app.installation_id);
    else
      Object.assign(config.vars, {
        GITHUB_APP_ID: String(app.id),
        GITHUB_INSTALLATION_ID: String(app.installation_id),
        GITHUB_APP_SLUG: app.slug,
      });
    writeFileSync(path, JSON.stringify(config, null, 2) + "\n");
  }
}
async function register() {
  if (existsSync(credentialsPath))
    throw new Error("app_already_registered_use_install");
  const state = randomBytes(32).toString("hex");
  const local = "http://127.0.0.1:8766";
  const manifest = {
    name: "robocurve-issue-bot",
    url: "https://github.com/robocurve/inspect-robots",
    description:
      "Astra/Codex issue triage and independently reviewed bug-fix PRs. Never merges or closes issues.",
    public: false,
    hook_attributes: { url: endpoint, active: true },
    redirect_url: `${local}/callback`,
    setup_url: `${local}/installed`,
    default_permissions: {
      contents: "write",
      issues: "write",
      pull_requests: "write",
      checks: "read",
    },
    default_events: ["issues", "issue_comment"],
  };
  const server = createServer(async (req, res) => {
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Referrer-Policy", "no-referrer");
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    if (req.headers.host !== "127.0.0.1:8766" || req.method !== "GET") {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }
    const url = new URL(req.url, local);
    try {
      if (url.pathname === "/" && url.searchParams.get("state") === state) {
        res.end(
          `<!doctype html><title>Register robocurve-issue-bot</title><h1>Register robocurve-issue-bot</h1><p>This separate App writes issue comments and fix branches/PRs, and reads CI checks. Install it only on robocurve/inspect-robots. It does not change the PR reviewer App.</p><form method="post" action="https://github.com/organizations/robocurve/settings/apps/new?state=${state}"><input type="hidden" name="manifest" value="${html(JSON.stringify(manifest))}"><button>Create GitHub App</button></form>`,
        );
      } else if (
        url.pathname === "/callback" &&
        url.searchParams.get("state") === state &&
        /^[a-zA-Z0-9_-]{10,200}$/.test(url.searchParams.get("code") ?? "")
      ) {
        if (existsSync(credentialsPath))
          throw new Error("app_already_registered");
        const raw = await api(
          `/app-manifests/${url.searchParams.get("code")}/conversions`,
          null,
          "POST",
        );
        if (
          !raw.pem ||
          !raw.webhook_secret ||
          !raw.id ||
          !/^[a-z0-9-]+$/.test(raw.slug)
        )
          throw new Error("invalid_manifest_response");
        save({
          id: raw.id,
          slug: raw.slug,
          pem: raw.pem,
          webhook_secret: raw.webhook_secret,
        });
        const installUrl = `https://github.com/apps/${raw.slug}/installations/new`;
        console.log(
          `Registered ${raw.slug}, App ${raw.id}. Install only on inspect-robots: ${installUrl}`,
        );
        res.end(
          `<h1>App registered</h1><p>Private credentials were saved locally.</p><a href="${installUrl}">Install on robocurve, selecting only inspect-robots</a>`,
        );
      } else if (url.pathname === "/installed" && existsSync(credentialsPath)) {
        const app = await installation(
          JSON.parse(readFileSync(credentialsPath, "utf8")),
        );
        configure(app);
        res.end(
          "<h1>Installation verified</h1><p>Setup is complete. Codex can now deploy and test the bot.</p>",
        );
        server.close();
      } else {
        res.writeHead(404);
        res.end("Not found");
      }
    } catch (error) {
      const code =
        error instanceof Error && /^[a-z0-9_]+$/.test(error.message)
          ? error.message
          : "setup_failed";
      console.error(code);
      res.writeHead(500);
      res.end(html(code));
    }
  });
  server.listen(8766, "127.0.0.1", () =>
    console.log(`Registration link: ${local}/?state=${state}`),
  );
  server.on("error", () => {
    console.error("registration_server_failed");
    process.exitCode = 1;
  });
  setTimeout(() => {
    server.close();
    console.log("Registration server expired. Rerun setup if needed.");
  }, 55 * 60_000).unref();
}
try {
  if (mode === "register") await register();
  else if (mode === "install")
    configure(
      await installation(JSON.parse(readFileSync(credentialsPath, "utf8"))),
    );
  else if (mode === "secrets") {
    const app = await installation(
      JSON.parse(readFileSync(credentialsPath, "utf8")),
    );
    configure(app);
    const apiKey = readFileSync(
      process.env.ISSUE_OPENAI_KEY_FILE ??
        join(secretDir, "openai-api-key"),
      "utf8",
    ).trim();
    if (!apiKey.startsWith("sk-") || /\s/.test(apiKey))
      throw new Error("invalid_api_key");
    for (const [config, name, value] of [
      [
        "publisher.wrangler.jsonc",
        "GITHUB_PRIVATE_KEY",
        createPrivateKey(app.pem).export({ type: "pkcs8", format: "pem" }),
      ],
      ["wrangler.jsonc", "GITHUB_WEBHOOK_SECRET", app.webhook_secret],
      ["wrangler.jsonc", "OPENAI_API_KEY", apiKey],
    ]) {
      const result = spawnSync(
        process.execPath,
        [
          join(root, "node_modules/wrangler/bin/wrangler.js"),
          "secret",
          "put",
          name,
          "--config",
          config,
        ],
        { cwd: root, input: value, encoding: "utf8" },
      );
      if (result.status !== 0)
        throw new Error(`secret_upload_failed_${name.toLowerCase()}`);
      console.log(`Uploaded ${name} to ${config}.`);
    }
  } else throw new Error("usage_register_install_or_secrets");
} catch (error) {
  console.error(
    error instanceof Error && /^[a-z0-9_]+$/.test(error.message)
      ? error.message
      : "setup_failed",
  );
  process.exitCode = 1;
}
