# 0077: Actuate demo launcher script and Kimi K3 test-taker

Demo-branch only (`demo/actuate-conference`, PR #397; never merges to main).
All paths relative to `examples/actuate/`.

## Goal

Two additions to the conference demo:

1. A single launcher script that starts the whole demo (display server plus
   the eval loop) instead of the README's two hand-typed tmux commands.
2. Kimi K3 joins the roster as a **test-taker only**: it can be drawn to
   drive the robot and it appears on the leaderboard, but it is never drawn
   as task maker or grader.

## Facts (verified 2026-08-19)

- Kimi K3 released 2026-07-16 (Moonshot AI). OpenAI-compatible chat
  completions; `reasoning_effort` accepts `low`/`high`/`max` (default
  `max`), so the demo's shared `effort=high` needs no special-casing.
  The model always reasons and returns `reasoning_content` alongside
  `content`; the agent plugin's chat wire ignores unknown response fields,
  so no plugin change is needed.
- Served as `kimi-k3` at `https://api.moonshot.ai/v1` (needs a Moonshot
  account) or as `moonshotai/kimi-k3` at `https://openrouter.ai/api/v1`.
  This rig already has `OPENROUTER_API_KEY` in `~/robocurve/test-dir/.env`,
  so the roster entry uses OpenRouter; the demo's own `.env` (symlink to the
  checkout root's `.env`) currently lacks that key and must gain it before
  K3 can score (operator copies the existing key; never committed).
- The demo currently draws all three roles from one `ROSTER` via
  `random.choice` in `run.py`; `serve.py` builds the leaderboard from
  `ROSTER` keys; `monitor.html` hardcodes a JS `order` array and `accents`
  map. All three surfaces need the new entry; only `run.py` needs role
  restriction.

## Changes

### `_roster.py`: role eligibility plus the K3 entry

- New module constant `ALL_ROLES = ("tasker", "test_taker", "grader")`.
- Each roster entry may carry `"roles": tuple[str, ...]`; absent means all
  three (existing entries stay untouched, restating the invariant: GPT,
  Opus, and Gemini remain eligible for every role and their entries do not
  change).
- New entry, after the existing three:

```python
"Kimi K3": {
    # test-taker only: drives the robot but never authors tasks or grades.
    "roles": ("test_taker",),
    # OpenRouter because the rig already holds OPENROUTER_API_KEY; for
    # Moonshot direct use model=kimi-k3, base_url=https://api.moonshot.ai/v1,
    # api_key_env=MOONSHOT_API_KEY.
    "model": "moonshotai/kimi-k3",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "policy": {
        "model": "moonshotai/kimi-k3",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
},
```

  (`model`/`base_url`/`api_key_env` at the top level stay present even
  though K3 is never a tasker/grader: `_role_args` never reads them for K3,
  but keeping the shape uniform means a booth edit that widens `roles`
  cannot half-work. The `policy` dict speaks the default chat wire; no
  `wire` key.)
- `ACCENTS` gains `"Kimi K3": "#9067e8"` (violet; distinct from the
  existing green/orange/blue on the dark background).

### `run.py`: per-role draws

Replace the single `roster = list(ROSTER.items())` with three eligibility
lists built once in `main()`:

```python
def _eligible(role: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (name, model)
        for name, model in ROSTER.items()
        if role in model.get("roles", ALL_ROLES)
    ]
```

and draw `tasker` from `_eligible("tasker")`, `test_taker` from
`_eligible("test_taker")`, `grader` from `_eligible("grader")`
(`random.choice` per role, unchanged semantics otherwise — the same model
may still hold two or three roles when eligible). Startup sanity: if any
role's list is empty, `SystemExit` with a fix-the-roster message before the
loop starts. The import at the top of run.py extends to
`from _roster import ALL_ROLES, EFFORT, ROSTER`.

### `monitor.html`: new model on the display

- `order` array gains `"Kimi K3"` (leaderboard row order; append last).
- JS `accents` map gains `"Kimi K3": "#9067e8"` matching `_roster.py`.

### New file: `start.sh` (the launcher)

Bash, executable, `#!/usr/bin/env bash`, `set -euo pipefail`. Mirrors the
README's recipe as one command. tmux specifics that are load-bearing, not
implementer judgment:

- **Exact-match session targets everywhere.** `-t actuate` prefix-matches
  `actuate-serve`, so a bare target makes the loop-session guard succeed
  against the serve session and the launcher can never start the loop. Every
  `has-session`/`attach` target uses the `=` prefix: `-t =actuate-serve`,
  `-t =actuate`.
- **If-guard every `has-session`** (`if tmux has-session -t =... 2>/dev/null`):
  under `set -e` a bare call kills the script, and on a fresh boot with no
  tmux server it also prints noise to stderr.
- **Absolute interpreter path in the inner commands.** A pre-existing tmux
  server hands new sessions the *server's* environment, not the launching
  shell's, so bare `python` can resolve to system python even though the
  step-2 guard passed. Capture `PYTHON="$(command -v python)"` up front and
  build both inner commands with it, `%q`-quoted.
- **Quoting:** build each inner command string with `printf '%q '` over the
  argv (interpreter, script path, `--`, passthrough args). Never interpolate
  `$*` raw. (`%q` emits bash-flavored quoting; tmux runs commands via its
  default-shell, bash on this rig — fine.)

Steps:

1. Resolve the repo root from the script's own path; `cd` there (the CLI
   auto-loads `.env` from the working directory, and `run.py` resolves its
   own siblings; repo root matches where the README puts `.env`).
2. Guard: `command -v inspect-robots` must succeed, else exit 1 with
   "activate the rig venv first (source .../bin/activate)". Warn (do not
   exit) if `.env` is missing at the repo root.
3. If a passthrough arg list is given and its first element is `--`, strip
   it (the old README recipe taught `run.py -- --config ...`; muscle-memory
   `./start.sh -- --config ...` must not forward a literal `--` into
   `inspect-robots run`).
4. Serve session: if `=actuate-serve` absent, start it detached with the
   `%q`-built `"$PYTHON" examples/actuate/serve.py` command; then sleep 1
   and re-check `has-session -t =actuate-serve` — if it died (port already
   bound, wrong python), exit 1 telling the operator to check
   `tmux new -s actuate-serve` manually / port 8377. If the session already
   existed, print that it is being reused (it may have been started from
   another checkout; the operator can `tmux kill-session -t =actuate-serve`
   to reset).
5. Print the display URL (`http://localhost:8377/`; the launcher does not
   forward serve.py args — booth simplicity, README notes `--port` requires
   the manual recipe).
6. If `=actuate` exists, exit with "demo loop already running; tmux attach
   -t =actuate" rather than stacking a second loop — two concurrent loops
   would fight over the rig.
7. Start the eval loop **attached** (the operator needs the terminal: Esc
   ends an episode early): `tmux new -s actuate <inner>` where `<inner>` is
   the `%q`-built `"$PYTHON" examples/actuate/run.py -- <args...>` (omit
   the `--` when no args).

No log-file redirection: the loop's console is the operator surface at the
booth, and `state/results.jsonl` plus the eval logs already persist
outcomes.

### `README.md`

- "Running it" section: replace the two tmux lines with
  `./examples/actuate/start.sh` (README keeps the manual recipe as the
  "what the script does" explanation; arguments after the script name reach
  `inspect-robots run` verbatim, same as before).
- Booth checklist, model IDs bullet: add Kimi K3, test-taker only, served
  via OpenRouter, `OPENROUTER_API_KEY` must be present in the demo `.env`
  (copy the existing key from `~/robocurve/test-dir/.env`), not yet
  validated live: force one eval with it as test-taker during setup.

## Invariants retained

- Existing three roster entries byte-identical; every existing role
  pairing still possible. The thermal gate (plan 0076) is untouched: the
  gate still runs at the top of each iteration before the role draw, and
  the per-role draws replace only the `random.choice(roster)` calls.
- `serve.py` needs no change: the leaderboard iterates `ROSTER` keys, so
  K3 appears automatically with n=0 until it scores.
- The launcher never runs git, never edits `.env`, and is idempotent for
  the serve session.

## Out of scope

- Wiring `MOONSHOT_API_KEY` / Moonshot-direct (documented in the roster
  comment as the booth-editable alternative).
- Any plugin change for `reasoning_content` (chat wire ignores it).

## Validation

- `ruff check examples/actuate/` and `ruff format --check` pass;
  `bash -n examples/actuate/start.sh` and a `shellcheck` pass if available.
- Draw sanity offline: import run.py with a stubbed roster and confirm
  10k draws never produce K3 as tasker or grader but do produce it as
  test-taker; confirm empty-role SystemExit fires on a roster with no
  eligible grader.
- Launcher on this machine (no rig): guard message without the venv;
  with the venv, serve session starts, URL prints, and a second invocation
  refuses to stack the `actuate` session.
- On the rig (operator): add `OPENROUTER_API_KEY` to the demo `.env`, force
  one K3 test-taker eval, confirm it scores and the leaderboard row fills.
