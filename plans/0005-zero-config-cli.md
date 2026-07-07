# 0005 — Zero-config CLI: `inspect-robots "place the spoon on the plate"`

## 1. Goal & motivation

Today the only way to run an eval from the CLI is the fully explicit form:

```bash
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick
```

The target UX is the `ollama run`-style one-liner for robotics:

```bash
inspect-robots "place the spoon on the plate"
```

which runs a **single ad-hoc scene** built from that language instruction using a
**default policy and embodiment** the user configured once (e.g. a MolmoAct2-YAM
checkpoint on a bimanual YAM rig, installed as plugins). The value is demo/
smoke-test velocity: "point the framework at my robot and give it a command"
without writing a Task or memorizing registry names.

Non-goals (YAGNI):

- No new benchmark semantics — an ad-hoc run is a normal `Task` with one `Scene`.
- No plugin auto-selection magic ("if exactly one policy is installed, use it").
  Defaults are explicit, user-set configuration.
- No natural-language → Target/scorer synthesis. The instruction is passed to the
  policy verbatim; success judgement comes from the operator (or an explicitly
  chosen scorer).
- No changes to plugin packaging or entry points.

## 2. What exists today (grounding)

- `cli.py`: argparse subcommands `list` / `run` / `inspect`; `run` requires
  `--task/--policy/--embodiment` (registry names) and passes `-T/-P/-E k=v`
  constructor args (`_parse_kvs`/`_parse_value`).
- `Scene` already carries `instruction: str` (`scene.py`). `Task` is
  `name + scenes + scorer + max_steps [+ epochs/control_hz/metadata]` (`task.py`).
- Registry (`registry.py`) resolves by `(kind, name)` from builtins + entry
  points. No notion of "default" components.
- `TrialRecord.operator_judgement` exists (R6) and `operator_scorer` **reads** it,
  but nothing in core ever *sets* it. Scoring happens inline in `eval.py`'s run
  loop immediately after each trial (~L303–308); **errored trials are recorded
  but never scored** (~L297–301, a CLAUDE.md invariant); the `LogSink.on_trial_end`
  hook fires *after* scoring (~L309). `transcript.operator_event(t, verdict)`
  exists, unused.
- The persisted `EvalLog` (`log.py`) contains only `EvalSpec` (names/config/seed),
  `EvalResults`, `EvalStats`, and per-scene `SceneResult(scene_id, status,
  reduced, epochs, error)`. **Neither the scene instruction nor
  `operator_judgement` is persisted today**; `JsonLogSink.on_trial_end` is a
  no-op. `EvalLog.from_dict` constructs dataclasses with `**kwargs`, so adding
  defaulted fields keeps the "newer reads older" guarantee at `SCHEMA_VERSION=1`.
- Binding constraints: plan 0001 §9 — especially **R6** (operator judgement is
  captured once during rollout as a recorded event; scorers only read it; v0
  keeps CI/unattended runs non-blocking) and **R10** (YAGNI reservations).
- Core is NumPy-only; py3.10–3.13 claimed (so **no `tomllib`**, which is 3.11+).
- Gates: ruff, mypy `--strict` (src+tests), pytest at **100 % coverage**.

## 3. Design

### 3.1 CLI surface

```bash
# zero-config form (positional instruction; sugar for `run --instruction`)
inspect-robots "place the spoon on the plate"

# explicit form, mixable with any run flag
inspect-robots run --instruction "place the spoon on the plate" \
    --policy molmoact2-yam --embodiment yam-bimanual --max-steps 400
```

- **Bare-instruction sugar** lives in `main()`: let `tok = argv[0].strip()`;
  if `argv` is non-empty, `tok` is not a known subcommand (`list`/`run`/
  `inspect`), does not start with `-`, **and contains interior whitespace**,
  rewrite to `["run", "--instruction", argv[0], *argv[1:]]`. The whitespace
  requirement is a safety gate: a mistyped subcommand (`inspect-robots
  isnpect`, `runs`) is a single token and must *not* silently start a robot
  rollout — it falls through to argparse's normal "invalid choice" error.
  Stripping before both checks means a whitespace-padded subcommand
  (`inspect-robots " list "`, e.g. from a shell variable with a stray space)
  is still treated as the subcommand, not as an instruction. Single-word
  instructions use the explicit `run --instruction` form (documented). `build_parser()` stays
  untouched; `--help`/`--version` behave as before.
- **`run` changes** (all backward compatible):
  - `--task` is no longer `required=True`. Exactly one of `--task` /
    `--instruction` must be given (checked in `_cmd_run`, clear `SystemExit`
    message otherwise — argparse mutually-exclusive groups can't express
    "required XOR" across an option that used to be required, so validate
    manually).
  - `--policy` / `--embodiment` are no longer `required=True`; when omitted they
    resolve through the defaults chain (§3.2). If the chain produces nothing →
    `SystemExit` with a message that lists registered policies/embodiments and
    shows both remedies (flag and config file).
  - New flags `--max-steps` and `--scorer` (registry name), **`default=None`
    sentinels** so "was it passed?" is detectable: with `--instruction` the
    effective value is `flag | config | fallback` (fallbacks: 300 / `operator`).
    Flag/argument validity is symmetric and explicit — with `--task`, passing
    `--max-steps` or `--scorer` is a `SystemExit` error (a registered Task owns
    its horizon and scorers); with `--instruction`, passing `-T k=v` is a
    `SystemExit` error (the ad-hoc Task is constructed directly — there is no
    task factory to receive `-T`). Silently ignoring flags would be a lie.
    `-T` with `--task` keeps working exactly as today (backward compatible).
  - New flag `--no-prompt`: disables the operator prompt (§3.4).
- The run header prints what was resolved and from where **before** the eval
  starts (i.e. before the embodiment resets), so defaults are never silent:
  `policy: molmoact2-yam (from $XDG_CONFIG_HOME/inspect-robots/config.ini)`.

### 3.2 Defaults resolution (new `src/inspect_robots/_defaults.py`)

Precedence, first hit wins, per component kind:

1. explicit CLI flag (`--policy` / `--embodiment`)
2. environment: `INSPECT_ROBOTS_POLICY`, `INSPECT_ROBOTS_EMBODIMENT`
3. user config file `<config-home>/inspect-robots/config.ini`, where
   `<config-home>` = `env["XDG_CONFIG_HOME"]` if set, else `env["HOME"]/.config`
   if `HOME` is set, else no config file. Derived **only from the injected env
   mapping** — never `Path.home()` — so the fallback branch is testable without
   touching the real home directory (and behaves predictably on Windows CI).
4. nothing → caller raises the guidance error (§3.1)

There is **no project-local config file**. A `./inspect-robots.ini` outranking
user config would mean `cd untrusted-checkout && inspect-robots "..."` runs
repo-chosen policy/embodiment/constructor args on the user's hardware — a
`.envrc`-class footgun unacceptable for a tool that moves physical robots.
(Revisit only with an explicit trust/allowlist mechanism.)

Config file format is **INI via stdlib `configparser`** (works on py3.10; TOML
would need `tomllib`≥3.11 or a new dep — rejected to keep the core NumPy-only).
The parser is constructed with `inline_comment_prefixes=(";", "#")` so the
documented examples with trailing comments parse as expected.

```ini
[defaults]
policy = molmoact2-yam
embodiment = yam-bimanual
scorer = operator      ; optional, ad-hoc runs only
max_steps = 300        ; optional, ad-hoc runs only

[policy.args]          ; default -P k=v pairs, same value parsing as the CLI
checkpoint = ~/ckpts/molmoact2-yam.pt

[embodiment.args]      ; default -E k=v pairs
cameras = wrist,front
```

- Values go through the existing `_parse_value` scalar parsing; additionally,
  string values in `[policy.args]`/`[embodiment.args]` that start with `~` get
  `os.path.expanduser` applied (checkpoint paths are the flagship use case and a
  literal `~/...` string would fail silently deep inside a plugin).
- Explicit `-P/-E k=v` flags **override** same-named config args (dict merge,
  CLI wins); other config args still apply.
- `_defaults.py` exposes one function:
  `load_defaults(env: Mapping[str, str]) -> Defaults` (a frozen dataclass:
  per-kind name + human-readable source description + args dicts). "Never a
  traceback" covers value validation too, not just parse errors: malformed INI,
  a non-integer or `< 1` `[defaults] max_steps`, and any other type-invalid
  `[defaults]` value → `SystemExit` naming the file, key, and problem. Unknown
  sections/keys are ignored (forward compatibility). Registry-name resolution
  failures (unknown policy/embodiment/scorer, whether from flags, env, or
  config) are caught in `_cmd_run` and re-raised as `SystemExit` listing the
  available names — no raw `KeyError` traceback from the zero-config path.
- **Not public API**: module is underscore-private, nothing added to
  `inspect_robots.__all__` (no API-snapshot change — the snapshot is name-based).

### 3.3 Ad-hoc task synthesis (in `cli.py`)

```python
Task(
    name="adhoc",
    scenes=[Scene(id="scene-0", instruction=<text>)],
    scorer=<--scorer | config | "operator">,
    max_steps=<--max-steps | config | 300>,
    metadata={"instruction": <text>, "adhoc": True},
)
```

`--epochs`, `--seed`, `--log-dir`, `--fail-on-error`, `--store-frames` all keep
working — the ad-hoc task flows through the same `_cmd_run` path and produces a
normal immutable `EvalLog` (log file name derives from the task name, `adhoc_*`).
An ad-hoc `Scene` has `target=None`/`setup=None`, so it passes R7 realizability
checks on any embodiment.

### 3.4 Operator verdict capture

Default scorer `operator` is useless unless something records the judgement. Per
R6 the verdict must be captured before scoring, recorded on the trial, and merely
read by the scorer. Today there is no seam. Add one:

- `eval(..., before_scoring: Callable[[TrialRecord, Scene], None] | None = None)`
  — keyword-only, default `None` → zero behavior change for every existing
  caller. Named `before_scoring` (not `on_trial_end`) because a `LogSink` hook of
  that name already exists and fires on the *other* side of scoring. **Firing
  rule: invoked exactly once per trial that will be scored** — i.e. only for
  trials with `record.status == "success"`. Errored trials (continuing
  `PolicyError`, halting `EmbodimentFault`/`SafetyAbort`) are recorded but never
  scored (existing invariant), so prompting a human for a verdict no scorer can
  read would be dead data and a blocked terminal on a crashed trial. An exception
  raised by the hook is not swallowed (caller-owned code; wrapping it would hide
  bugs). `eval_set` forwards it.
- The CLI passes a prompt hook **only on the ad-hoc (`--instruction`) path**,
  and only when the resolved scorers include the `operator` scorer, stdin is a
  TTY, and `--no-prompt` was not given. Registered `--task` runs are never
  prompted — R6's binding "non-blocking, unattended-safe" property for benchmark
  runs is untouched (a 50-scene × 5-epoch task must not produce 250 modal
  prompts because its scorer list mentions `operator`). `--no-prompt` recovers
  unattended behavior for ad-hoc runs left in a tmux TTY.
- **Prompt contract**: `did the robot succeed? [y/n/partial/skip]` via
  `input()`; answers are stripped/lower-cased; re-prompt on anything not in
  `{y, yes, n, no, partial, skip}` (a typo like `yse` must not be recorded as a
  failing verdict). `EOFError` counts as `skip`. `skip` → `operator_judgement`
  stays `None` (scores as "no operator judgement recorded") and **no** operator
  event is appended (`operator_event` requires a `str` verdict). Any other
  answer is recorded verbatim (normalized) via `record.operator_judgement` plus
  `transcript.operator_event(t=len(record.steps), verdict=...)` appended to
  `record.events`. The prompt text notes that `partial` counts as failure in the
  binary success metric (existing `_OPERATOR_SUCCESS` semantics; richer partial
  credit is R10-reserved).

### 3.5 Log persistence (small additive schema change)

The persisted log today records neither what the robot was asked nor what the
operator answered — which would make the ad-hoc log unreproducible and the
operator flow untestable end-to-end. Two **additive, defaulted** fields on
`SceneResult` (schema stays `SCHEMA_VERSION=1`; `from_dict` fills missing keys
with defaults, so newer-reads-older still holds; the golden read-back test in
`tests/test_eval_log.py` and the strict-JSON round-trips in
`tests/test_strict_json.py` are updated):

- `instruction: str | None = None` — copied from the `Scene` by `eval()` for
  every task (useful beyond ad-hoc: a benchmark log becomes self-describing).
- `operator_judgements: list[str | None] = field(default_factory=list)` —
  **always strictly parallel to `epochs`** in newly written logs: exactly one
  entry per recorded trial, `None` for a trial that was errored/unscored *or*
  scored-but-unjudged (skip / no prompt), the verdict string otherwise. (Note
  `epochs` itself gets `{}` for errored trials — eval.py appends per *recorded*
  trial.) The `[]` default exists only so old logs without the field read back;
  a mixed errored/judged scene is pinned by a test.

Defaults *provenance* (flag vs env vs config) is printed in the run header but
deliberately **not** persisted: the log already records the resolved
policy/embodiment names and configs in `EvalSpec`, which is what reproduction
needs; where the name came from is session UX.

### 3.6 Alternatives considered

- **Plugin-declared defaults** (entry-point group `inspect_robots.defaults`):
  rejected — two installed plugins fight over the default, and "install changes
  behavior" is spooky action.
- **Auto-pick the only installed non-mock policy/embodiment**: rejected — adding
  a second plugin silently changes what `inspect-robots "..."` does.
- **TOML config**: rejected for py3.10 support without new deps (revisit when
  the floor moves to 3.11).
- **Default scorer `success_at_end`**: rejected as the ad-hoc default — real
  robots have no success oracle for an arbitrary instruction, so it would report
  a confident-looking `0.0` forever. `operator` is honest in both modes.
- **Project-local config**: rejected (trust footgun, §3.2).

## 4. File-level changes

```
inspect-robots/
├── plans/0005-zero-config-cli.md            (this doc)
├── src/inspect_robots/
│   ├── _defaults.py                         (new: Defaults dataclass + load_defaults)
│   ├── cli.py                               (argv sugar; run: instruction path,
│   │                                         defaults chain, operator prompt hook)
│   ├── eval.py                              (eval/eval_set: before_scoring param;
│   │                                         populate SceneResult.instruction /
│   │                                         operator_judgements)
│   ├── log.py                               (SceneResult: two additive fields)
│   └── CLAUDE.md                            (module map rows for _defaults/cli)
├── tests/
│   ├── test_defaults.py                     (new: precedence, INI parsing, errors)
│   ├── test_registry_cli.py                 (extend: sugar, ad-hoc run e2e on mock,
│   │                                         operator prompt, flag validation)
│   ├── test_eval_orchestration.py           (extend: before_scoring hook)
│   ├── test_eval_log.py                     (extend: golden read-back with and
│   │                                         without the new SceneResult fields)
│   └── test_strict_json.py                  (extend: round-trip new fields)
├── docs/guide/cli.md                        (zero-config section + config file)
├── README.md                                (one-liner in the CLI example block)
└── CLAUDE.md                                (mention zero-config CLI where apt)
```

## 5. Testing (TDD; 100 % coverage; no vacuous tests)

Every test asserts observable behavior (exit codes, stdout, written logs,
recorded judgements) — not merely "function returns without raising".

- **argv sugar**: `main(["place the spoon"])` runs the ad-hoc path (assert via
  the written `adhoc_*` log); `main(["isnpect"])` (single token, no whitespace)
  → argparse invalid-choice error, **no** eval runs; `main([])` prints help,
  exit 0; `main(["list"])` still lists; `--version` unaffected.
- **defaults precedence** (`test_defaults.py`, all through the injected `env`
  mapping — no real `$HOME`/`$XDG_CONFIG_HOME` reads): flag > env > user INI >
  error; `XDG_CONFIG_HOME` set / unset-with-`HOME` / neither (no config read);
  args merge (CLI `-P` overrides same key from `[policy.args]`, non-colliding
  config keys survive); `~` expansion on arg values; inline comments parse;
  malformed INI → `SystemExit` naming the file; missing file → empty defaults;
  value parsing (bool/int/none/str) matches `-P` parsing.
- **run validation**: `--task` + `--instruction` together → error; neither →
  error; `--max-steps` / `--scorer` / `-T` with `--task`+ad-hoc mismatches →
  error (each direction); omitted policy/embodiment with no defaults → error
  message lists registered names; sentinel logic: config `max_steps` wins when
  the flag is omitted, flag wins when passed (including `--max-steps 300`
  passed explicitly with a different config value — distinguishable only via
  the `None` sentinel).
- **ad-hoc e2e on the mock** (no hardware): env or flags select
  `scripted`/`cubepick`; `inspect-robots "reach the cube" --scorer
  success_at_end` runs, writes an `adhoc_*` log, exit code reflects status;
  the written log's `samples[0].instruction` equals the CLI text;
  `--max-steps`/`--epochs` honored (assert in the written log); resolved-
  defaults header printed before the run with the correct source label.
- **operator flow** (ad-hoc path): TTY + `operator` scorer → prompt called
  (monkeypatched `input` and `isatty`), judgement lands in the written log's
  `operator_judgements` and the `operator` scorer scores it; transcript
  `operator` event appended with `t == len(record.steps)`; re-prompt on invalid
  input (`yse` → re-ask, then `y` recorded); `skip` and `EOFError` → judgement
  `None`, no event; non-TTY or `--no-prompt` → no prompt, score `False` with
  the existing explanation; `--task` path never prompts even on a TTY with
  `operator` scorer.
- **`before_scoring` hook** (eval-level, independent of CLI): called exactly
  once per **scored** trial with the record and scene, before scorers run
  (observable: hook sets `operator_judgement`, operator scorer sees it); not
  called for errored trials (continuing `PolicyError` path); default `None`
  unchanged; hook exception propagates (not swallowed); `eval_set` forwards it.
- **log schema**: old-format log JSON (without the new fields) still reads via
  `read_eval_log` (defaults fill in); new fields serialize; goldens updated;
  mixed-status scene (epoch 0 errors, epoch 1 scored and judged) pins
  `operator_judgements == [None, "yes"]` parallel to `epochs == [{}, {...}]`.
- **config validation**: `[defaults] max_steps = lots` (and `= 0`) →
  `SystemExit` naming file/key, no traceback; unknown scorer/policy/embodiment
  name from any source → `SystemExit` listing available names.
- **Test-quality gate**: after implementation, a reviewer subagent audits the
  new tests for vacuousness (assertions that cannot fail, mocks asserting on
  themselves, coverage-only tests) and findings are fixed before the PR.

## 6. Milestones (each = focused commit, gates green)

1. `log.py` additive `SceneResult` fields + `eval()` populates `instruction` +
   read-back tests.
2. `eval()`/`eval_set()` `before_scoring` hook (+ `operator_judgements`
   persistence) + orchestration tests.
3. `_defaults.py` (load/merge/precedence) + `test_defaults.py`.
4. `cli.py`: run-path rework (instruction XOR task, defaults chain, ad-hoc task,
   operator prompt, resolved-defaults header) + argv sugar + CLI tests.
5. Docs: `docs/guide/cli.md` zero-config section, README one-liner,
   `src/inspect_robots/CLAUDE.md` module map.

## 7. Out of scope / follow-ups

- A first-party MolmoAct2/YAM plugin (separate package; this plan only makes the
  configured-defaults UX possible for it).
- `inspect-robots configure` interactive setup; VLM scoring of ad-hoc runs
  (R10 reserved); TOML config once py3.10 support is dropped; project-local
  config with a trust mechanism.
