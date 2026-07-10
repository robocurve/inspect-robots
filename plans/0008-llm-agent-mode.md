# 0008 — Agent mode: LLMs drive robots through tool calls

## 1. Goal

Let a frontier LLM (Claude, GPT, anything behind an OpenAI-compatible API)
control a robot interactively — outside the eval loop — by calling tools
against any registered `Embodiment`. Pilot target: the bimanual YAM arms.

```bash
pip install inspect-robots inspect-robots-yam inspect-robots-agent
inspect-robots config set embodiment yam        # once, per machine
export ANTHROPIC_API_KEY=sk-ant-...

inspect-robots agent "place the fork on the plate" --model anthropic/claude-fable-5
```

Same command with `--embodiment mock/cubepick` runs hardware-free, which is
also how the whole feature is tested in CI.

This is **not** a `Policy`. The eval loop's policy contract (observation →
`ActionChunk` at control rate) is wrong for an LLM that thinks for seconds per
decision. Agent mode is a separate vertical: an agentic tool-calling loop where
each tool call executes a smooth, approver-checked motion primitive spanning
many `embodiment.step()` calls. An LLM-backed `Policy` for scored evals can come
later and is out of scope here.

## 2. Placement: what goes where

Follows the family convention (hardware adapters in satellite repos, policy-side
adapters as in-repo plugins; `plugins/inspect-robots-xpolicylab` is the
precedent — plan 0007).

| Piece | Where | Why |
|---|---|---|
| Safety approvers (max-delta limiter, chaining) | core `approver.py` | NumPy-only; protects every action source; the `Approver` seam already exists and its docstring reserves exactly this hardening |
| `dim_labels` on `ActionSemantics` | core `spaces.py` | NumPy-only; lets the agent (and future logging/viz) name action dimensions |
| Plugin CLI subcommands + `config set/show` | core `cli.py` / `_defaults.py` | tiny, dependency-free, benefits every plugin |
| Agent loop, LLM client, tool surface, `agent` subcommand | new `plugins/inspect-robots-agent` | needs an HTTP client dep, so it cannot live in the NumPy-only core |
| YAM pilot | `inspect-robots-yam` repo (docs/quickstart) | its bimanual `Embodiment` (two CAN channels, `can0`/`can1` defaults, flat 14-D action space) already exists; the pilot is documentation, not code |

## 3. Core changes

### 3a. Safety approvers (`approver.py`)

- **`DeltaLimitApprover(max_delta, reference)`** — the "no wild swings" gate.
  Clamps each action dimension to at most `max_delta` away from a reference
  (last approved action, falling back to the current proprioceptive state on
  the first action). Per-dimension `max_delta` array or scalar. NaN → `SafetyAbort`
  (same contract as `ClampApprover`). A clamped action is flagged via
  `action.meta["delta_clamped"]` so the transcript records an approval event.
- **`ChainApprover(approvers)`** — runs approvers in sequence (bounds clamp,
  then delta limit). Trivial, but gives "guardrails" one name.

Both are generic: they harden scored evals with VLA policies exactly as much as
agent mode. Registered defaults change nothing for existing users — `eval()`
keeps `AutoApprover` unless configured otherwise; **agent mode** is where the
chain is on by default (§4d).

### 3b. `ActionSemantics.dim_labels` (`spaces.py`)

Optional `dim_labels: tuple[str, ...] | None = None`; when present its length
must equal the owning action `Box.dim` (validated in `Box.__post_init__`, which
is where the semantics/shape pairing is visible). YAM will set
`("left_j0", …, "left_j5", "left_gripper", "right_j0", …, "right_gripper")` in
its repo. Embodiments without labels still work — the agent falls back to
index-keyed targets (`"3": 0.4`).

### 3c. CLI plugin subcommands + `config` (`cli.py`, `_defaults.py`)

- New entry-point group **`inspect_robots.cli`**: each entry point resolves to a
  `register(subparsers) -> None` callable. Core loads the group after building
  its own subparsers; a plugin that fails to import is skipped with a warning
  (same tolerance the component registry uses).
- New **`config`** subcommand in core: `config set KEY VALUE` writes
  `[defaults]` keys (`policy`, `embodiment`, `sim_embodiment`, `model`) into
  `~/.config/inspect-robots/config.ini` via `configparser` (atomic
  write-temp-then-rename, preserving unknown sections); `config show` prints the
  resolved defaults with their sources. `_defaults.py` gains `model` /
  `INSPECT_ROBOTS_MODEL` alongside the existing keys so `--model` participates
  in the same flag > env > config chain.

The unconfigured experience stays the existing guided `SystemExit` ("registered
embodiments: …; fix: pass --embodiment, set $…, or run `inspect-robots config
set embodiment NAME`" — the fix line gains the `config set` spelling). No
shipped hardware default, deliberately: core cannot name an optional package,
and a tool that moves physical robots must not pick its target implicitly
(same trust stance as plan 0005's rejection of project-local config).

## 4. The plugin: `inspect-robots-agent`

```
plugins/inspect-robots-agent/
  pyproject.toml            # deps: inspect-robots, httpx; static version 0.1.0
  src/inspect_robots_agent/
    __init__.py
    _llm.py                 # OpenAI-compatible chat client (httpx), provider resolution
    _tools.py               # tool schemas + dispatch over an Embodiment
    _motion.py              # joint-space interpolation primitive
    _loop.py                # the agentic loop (LLM ↔ tools ↔ embodiment)
    _cli.py                 # registers the `agent` subcommand (entry point)
  tests/                    # mock transport + mock/cubepick; no network, no hardware
```

### 4a. LLM client (`_llm.py`)

Speaks the OpenAI chat-completions wire format (tools + tool_choice), which
covers OpenRouter, OpenAI, local vLLM/Ollama, and Anthropic's OpenAI-compat
endpoint. No provider SDKs — one `httpx` client, same "speak the protocol,
don't import the package" doctrine as plan 0007.

Model strings are OpenRouter-style `provider/model`. Key/base-url resolution,
first match wins:

1. `--base-url` flag (+ `--api-key-env NAME`, default `OPENROUTER_API_KEY`) — any compatible endpoint
2. `anthropic/*` model + `ANTHROPIC_API_KEY` → Anthropic compat endpoint
3. `openai/*` model + `OPENAI_API_KEY` → OpenAI
4. `OPENROUTER_API_KEY` → OpenRouter (any model string)

No key that matches → guided `SystemExit` naming the env vars, mirroring the
embodiment error. This is the whole "works out of the box with an OpenRouter /
Claude / OpenAI key" requirement.

### 4b. Tool surface (`_tools.py`)

Generated from the embodiment's spaces — the plugin never knows what a "YAM"
or an "arm" is:

- `get_observation()` → proprioceptive state (labeled when `dim_labels` exist)
  and camera frames, sent to the LLM as image content blocks.
- `move_joints(targets: dict[str, float], duration_s: float)` — **named partial
  targets**: unnamed joints hold position. `{"right_j2": 0.4, "right_gripper": 1.0}`
  moves one arm of a bimanual robot; naming joints from both arms coordinates
  them in one call. Falls back to index keys without labels. Unknown label,
  out-of-bounds `duration_s` → structured tool error the LLM can react to,
  never an exception.
- `done(summary)` / `give_up(reason)` — terminate the loop.

The tool JSON schema embeds the action space's bounds, labels, and semantics
(control mode, gripper kind), so the system prompt stays generic.

### 4c. Motion primitive (`_motion.py`)

One tool call → one smooth trajectory: linear interpolation in joint space from
current position to target over `duration_s` at the embodiment's control rate,
each interpolated step passing through the approver chain, then
`embodiment.step()`. Blocks until the trajectory finishes; returns the
achieved state. This layer is what bridges "LLM thinks in seconds" to "robot
steps at 10 Hz" — and interpolation itself is a guardrail (no teleports even if
the target is far).

### 4d. Guardrails-by-default (`_loop.py`, `_cli.py`)

The loop wires `ChainApprover(ClampApprover(action_space), DeltaLimitApprover(...))`
between the primitive and the embodiment. **Always on.** `--disable-guardrails`
is the explicit opt-out flag; it prints a prominent warning to stderr. There is
no code path from the LLM to `embodiment.step()` that bypasses the approver —
the safety boundary is below the model, so no prompt injection or model error
can emit an unclamped action. Tunables: `--max-joint-delta` (rad/step, sane
default), `--max-steps`, `--max-llm-calls`.

Loop shape: system prompt (embodiment description, tool docs, safety notes) +
user goal → LLM → tool calls executed sequentially → results (text + images)
appended → repeat until `done`/`give_up`/budget. Every LLM call, tool call, and
approval event is echoed to the console and written to a JSONL transcript under
`logs/` (reusing core transcript event types where they fit).

### 4e. CLI (`_cli.py`)

```
inspect-robots agent "GOAL" [--model P/M] [--embodiment NAME] [-E k=v]
    [--base-url URL] [--api-key-env NAME] [--max-joint-delta R]
    [--max-steps N] [--max-llm-calls N] [--disable-guardrails] [--log-dir D]
```

`--model` and `--embodiment` resolve through the §3c defaults chain; both
unconfigured cases exit with the guided message. Registered via the
`inspect_robots.cli` entry point, so the command appears whenever the plugin is
installed.

## 5. Testing (no network, no hardware)

- Core: approvers and `dim_labels` are pure NumPy — straight into the 100%
  gate. `config set/show` tested against a tmp `XDG_CONFIG_HOME`; the
  entry-point hook tested with a synthetic entry point (same technique as
  existing registry tests).
- Plugin: `httpx.MockTransport` scripts LLM conversations (tool calls as
  canned JSON) against `mock/cubepick`; asserts end-to-end that goals run,
  guardrails clamp a scripted wild swing, `--disable-guardrails` doesn't, NaN
  aborts, budgets terminate, and each provider resolution rule picks the right
  base URL/key. Plugin keeps its own coverage scope outside the core gate,
  like the other plugins.

## 6. CI, workspace, release

Same integration as plan 0007 §8: uv workspace member; a `test-agent-plugin` CI
job added to `ci-ok.needs`; core-only-import job already proves core never
imports it; `publish-inspect-robots-agent` job in `release.yml` with
`skip-existing`; PyPI trusted-publisher environment for the new package.

## 7. Risks & mitigations

- **LLMs are poor low-level controllers.** Expected; the pilot's bar is
  "coherent, safe, visibly reasoned motion", not VLA-grade manipulation.
  Named partial targets + images per step give the model its best shot;
  budgets bound the failure mode.
- **Latency between decisions** — the arm idles while the LLM thinks. Fine for
  a pilot; hold-position is the safe idle.
- **Prompt injection / model misbehavior** — cannot bypass the approver chain
  (enforced below the model); worst case is in-bounds, rate-limited motion.
  `--max-steps`/`--max-llm-calls` bound runaway loops; Ctrl-C must always
  stop cleanly (`finally: embodiment.close()`).
- **Anthropic compat-endpoint drift** — the client is plain OpenAI wire
  format; if the compat endpoint diverges, OpenRouter remains the universal
  path and a native adapter can be added behind the same interface later.

## 8. Execution steps (each a commit / PR-sized slice)

1. Core: `DeltaLimitApprover` + `ChainApprover` (+ tests, API snapshot).
2. Core: `ActionSemantics.dim_labels` (+ validation, tests, snapshot).
3. Core: `config set/show` subcommand; `model` default key; guided-error text
   gains the `config set` fix line.
4. Core: `inspect_robots.cli` entry-point group.
5. Plugin scaffold: pyproject, workspace membership, CI job, release job.
6. Plugin: `_llm.py` client + provider resolution (mock-transport tests).
7. Plugin: `_tools.py` + `_motion.py` over mock/cubepick.
8. Plugin: `_loop.py` + `_cli.py`; end-to-end scripted-conversation tests.
9. Docs: core README section; yam repo quickstart PR (separate repo).
