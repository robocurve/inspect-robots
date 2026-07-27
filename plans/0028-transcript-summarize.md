# 0028 — `summarize`: distill an eval log into a learnings file

Issue: #195. Status: draft.

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
writes to stdout. The command prints the written path on success.

### Why the OpenAI-compatible chat wire, and why stdlib

Core is deliberately numpy + stdlib only (`pyproject.toml`: "Core depends
ONLY on numpy + the standard library"). The agent plugin's httpx wire clients
cannot be imported from core without inverting the dependency. A single
OpenAI-compatible chat POST is ~30 lines of `urllib.request` and covers
Anthropic (`https://api.anthropic.com/v1/`, the OpenAI-compat endpoint),
OpenRouter, OpenAI, and local servers — one wire, every provider we use.

Defaults: `--base-url https://api.anthropic.com/v1`,
`--api-key-env ANTHROPIC_API_KEY`, both overridable. A missing key with
`--model` set is a `ConfigError`-style guided message (house convention,
#168), not a traceback.

### New module: `src/inspect_robots/_summarize.py`

- `load_transcripts(log: EvalLog, log_path: Path) -> list[TrialTranscript]` —
  follows each sample's `metadata["transcript"]` pointer (written by the agent
  plugin's `on_trial_end`, relative to the log's directory). Missing or
  unparseable sidecars are skipped with a note in the digest, not an error:
  logs from non-LLM policies have no transcripts and digest mode must still
  work on them.
- `build_digest(log: EvalLog, transcripts) -> str` — deterministic markdown:
  run header (task, policy, embodiment, model from `policy_config`, status),
  a per-scene line each with outcome, steps, termination reason, operator
  judgement/notes when present, and error text; then per-trial transcript
  stats (message count, tool-call count, last assistant note). Reuses the
  outcome vocabulary of `_outcome_line` (`cli.py:619`) so the two surfaces
  never disagree.
- `build_messages(digest: str, transcripts) -> list[dict]` — chat messages
  for LLM mode. System prompt fixes the output structure:
  `## What happened`, `## Failure modes`, `## Lessons for next attempt` —
  the last section written as direct imperatives to a future agent attempting
  the same task. Transcripts are appended under a per-trial char budget,
  keeping the *tail* of each trial when truncating (failures concentrate at
  the end). `transcript()` is already image-free, so no payload concerns.
- `chat_completion(base_url, api_key, model, messages, *, http_post=None)
  -> str` — stdlib `urllib.request` POST. `http_post` is an injectable
  `Callable[[url, headers, body_bytes], tuple[int, bytes]]` so tests never
  touch the network and the coverage gate (`fail_under = 100`) is satisfiable.
  Non-2xx or malformed replies raise with the response body excerpt in the
  guided message.
- `summarize(log_path, *, model, base_url, api_key_env, http_post=None)
  -> str` — orchestrates the above, returns the markdown.

### CLI wiring

`build_parser()` gains `p_summarize` next to `inspect`/`view` (same "reads a
saved log" family). `_cmd_summarize` mirrors `_cmd_view`'s shape: resolve
output path, call `summarize()`, write atomically (temp + `os.replace`, same
pattern as `json_log.py`), print the path. `learnings/` is created on demand.

## Files

- `src/inspect_robots/_summarize.py` — new (all logic above).
- `src/inspect_robots/cli.py` — parser entry + `_cmd_summarize` + dispatch.
- `tests/test_summarize.py` — new.
- `docs/` — CLI reference regenerated if applicable; README section under the
  existing CLI table.

## Testing

- Digest mode: golden-style assertions on a synthetic `EvalLog` fixture with
  two scenes (one success, one step-limit failure), one with a transcript
  sidecar and one without.
- LLM mode: fake `http_post` returning a canned completion; assert request
  shape (url join, auth header, model field) and that the reply lands in the
  output file verbatim.
- Error paths: missing key env, non-2xx reply, malformed JSON reply, log with
  zero samples, `-o -`.
- CLI: `main(["summarize", ...])` end-to-end against a tmp log dir.

## Out of scope

- Summarizing *sets* of logs in one call (natural follow-up once single-log
  output proves useful).
- Automating the summarize→inject loop (`retry_attempts`, issue #196's
  follow-up).
