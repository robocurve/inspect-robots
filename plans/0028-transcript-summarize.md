# 0028 — `summarize`: distill an eval log into a learnings file

Issue: #195. Status: revised after one critique round (transcript sourcing
corrected to the actual log schema; urllib coverage strategy and request
shape pinned).

## Problem

The CLI can pretty-print a log (`inspect` at `cli.py:1150`) and render it as
HTML (`view` at `cli.py:1220`), but both are deterministic formatters. When an
agent run fails, understanding *why* means a human reading the policy
transcript JSONL under `<log_dir>/transcripts/<run_id>/` line by line. Nothing
turns a run into the artifact we actually want afterwards: a compact,
natural-language account of what the agent tried, where it went wrong, and
what a next attempt should do differently.

This is the producer half of a retry-with-learning loop. The consumer half
(plan 0029, issue #196) injects a markdown "learnings file" into the agent's
system prompt on a later run. The two halves are coupled only by a contract:

- **Format:** plain markdown, treated as opaque by the consumer.
- **Default location:** `<log_dir>/learnings/<log_stem>.md`, where
  `<log_stem>` is the log's filename without `.json` (e.g.
  `logs/adhoc_084f91f0.json` → `logs/learnings/adhoc_084f91f0.md`).
- The consumer always takes an explicit path; nothing breaks if the producer
  was never run or the file was hand-written.

## Design

New subcommand:

```
inspect-robots summarize LOG [--model MODEL] [--base-url URL]
                             [--api-key-env VAR] [-o FILE]
```

Two modes, decided by `--model`:

1. **Digest mode (default, no `--model`):** deterministic markdown digest
   built from the log alone — no network, no key, works offline. Useful on
   its own and doubles as the grounding context for mode 2.
2. **LLM mode (`--model` given):** the digest plus the recorded transcripts
   are sent to an OpenAI-compatible `/chat/completions` endpoint; the model's
   markdown reply is the output.

Output goes to the contract path by default, `-o FILE` overrides, `-o -`
writes to stdout. In file mode the command prints the written path on
success; in stdout mode it prints only the document (so piping stays clean),
matching `_cmd_view`'s split (`cli.py:1256-1259`).

### Why the OpenAI-compatible chat wire, and why stdlib

Core is deliberately numpy + stdlib only (`pyproject.toml`: "Core depends
ONLY on numpy + the standard library"). The agent plugin's httpx wire clients
cannot be imported from core without inverting the dependency. A single
OpenAI-compatible chat POST is ~30 lines of `urllib.request` and covers
Anthropic (`https://api.anthropic.com/v1/`, the OpenAI-compat endpoint),
OpenRouter, OpenAI, and local servers — one wire, every provider we use.

Defaults: `--base-url https://api.anthropic.com/v1`,
`--api-key-env ANTHROPIC_API_KEY`, both overridable — the same pairing the
agent plugin already ships against this endpoint (`_llm.py:37,184`). Request
shape is pinned to what that proven client does: URL is
`base_url.rstrip("/") + "/chat/completions"`, auth header is
`Authorization: Bearer` (the OpenAI-compat wire — not `x-api-key`, which is
the native-Messages wire). Unlike the agent's short tool-call turns, a
summary is a long document, so the request sends an explicit generous
`max_tokens` (8192) rather than trusting the compat layer's default output
cap — a silently truncated learnings file is worse than an error.

A missing key with `--model` set raises `errors.ConfigError` with a `fix:`
line (house convention, #168; same shape as `_llm.py:123-128`), caught in
`_cmd_summarize` the way `_cmd_run` catches it — never a traceback.

### New module: `src/inspect_robots/_summarize.py`

- `load_transcripts(log: EvalLog, log_path: Path) -> list[TrialTranscript]` —
  one entry per **trial**, keyed `(scene_id, epoch)`. The unit here is the
  trial, not the scene: `SceneResult` stores everything per-trial in tuples
  parallel to `epochs` (`log.py:76-90`), and a `--epochs 3` run has three
  conversations per scene. Two sources, in order:
  1. **Inline first:** `scene.policy_transcripts[i]` — `rollout()` captures
     the conversation for any policy with a `transcript()` hook
     (`rollout.py:106-130`) and `eval()` persists it in the log itself
     (`eval.py:423`). This is what `inspect --transcript` and the HTML viewer
     already read, so all three surfaces share one source of truth.
  2. **Sidecar fallback:** when the inline entry is missing or was dropped
     for size (`_TRANSCRIPT_BYTE_LIMIT`, 2 MiB, `rollout.py:47` — the entry
     becomes `{"transcript_dropped": True, ...}`), follow
     `scene.trial_metadata[i]["transcript"]` (the JSONL path the agent
     plugin's `on_trial_end` records, relative to the log's directory;
     written *before* `trial_metadata` is captured, `eval.py:407-421`, so
     the pointer is always in the saved log when the sidecar exists).
  Trials with neither are represented with an empty transcript, noted in the
  digest, never an error: logs from non-LLM policies have no transcripts and
  digest mode must still work on them.
- `build_digest(log: EvalLog, transcripts) -> str` — deterministic markdown:
  run header (task, policy, embodiment, status, and model when
  `policy_config` carries one — only the LLM policies do), then a per-trial
  line each with outcome, steps, termination reason, operator
  judgement/notes when present, and error text; then per-trial transcript
  stats (message count, tool-call count, last assistant note). Reuses the
  outcome vocabulary of `_outcome_line` (`cli.py:619`) so the surfaces never
  disagree.
- `build_messages(digest: str, transcripts) -> list[dict]` — chat messages
  for LLM mode. System prompt fixes the output structure:
  `## What happened`, `## Failure modes`, `## Lessons for next attempt` —
  the last section written as direct imperatives to a future agent attempting
  the same task. Transcripts are appended under a per-trial char budget,
  keeping the *tail* of each trial when truncating (failures concentrate at
  the end). `transcript()` is already image-free, so no payload concerns.
- `chat_completion(base_url, api_key, model, messages, *, http_post=None)
  -> str` — POST via `http_post`, an injectable
  `Callable[[url, headers, body_bytes], tuple[int, bytes]]` defaulting to a
  named module-level `_urllib_post` built on `urllib.request`. The split is
  the coverage strategy for the `fail_under = 100` gate: business-logic
  tests inject fakes, and `_urllib_post` itself gets direct tests with
  `urllib.request.urlopen` monkeypatched (success, HTTPError with body,
  URLError) — every line executes, no network. Non-2xx or malformed replies
  raise `ConfigError` with a response-body excerpt in the guided message.
- `summarize(log_path, *, model, base_url, api_key_env, http_post=None)
  -> str` — orchestrates the above, returns the markdown.

### CLI wiring

`build_parser()` gains `p_summarize` next to `inspect`/`view` (same "reads a
saved log" family). `_cmd_summarize` mirrors `_cmd_view`'s shape: resolve
output path, call `summarize()`, write atomically (temp created *in the
destination directory* + `os.replace`, same pattern as `json_log.py`, so the
replace stays same-filesystem), print the path. `learnings/` is created on
demand. `_cmd_view`'s output guards carry over verbatim: `-o` pointing at a
directory is a guided error (`cli.py:1236-1237`), and `-o` pointing at the
input log itself is refused (`cli.py:1238-1240` — the one data-loss path).
`ConfigError` from `summarize()` is caught and rendered as a guided message
the way `_cmd_run` does.

## Files

- `src/inspect_robots/_summarize.py` — new (all logic above).
- `src/inspect_robots/cli.py` — parser entry + `_cmd_summarize` + dispatch.
- `tests/test_summarize.py` — new.
- `src/inspect_robots/CLAUDE.md` — module table gains `_summarize.py`.
- `docs/` — CLI reference regenerated if applicable; README section under the
  existing CLI table.

## Testing

- Digest mode: golden-style assertions on a synthetic `EvalLog` fixture with
  two scenes and `--epochs 2` shape (per-trial tuples populated): one trial
  with an inline transcript, one with `transcript_dropped` + a JSONL sidecar
  (exercises the fallback), one with neither.
- LLM mode: fake `http_post` returning a canned completion; assert request
  shape (url join, `Authorization: Bearer` header, model and `max_tokens`
  fields) and that the reply lands in the output file verbatim.
- `_urllib_post` directly, with `urlopen` monkeypatched: 2xx, `HTTPError`
  carrying a body, `URLError`.
- Error paths: missing key env, non-2xx reply, malformed JSON reply, log with
  zero samples, `-o` at a directory, `-o` at the input log, `-o -`.
- CLI: `main(["summarize", ...])` end-to-end against a tmp log dir.

## Out of scope

- Summarizing *sets* of logs in one call (natural follow-up once single-log
  output proves useful).
- Automating the summarize→inject loop (`retry_attempts`, issue #196's
  follow-up).
