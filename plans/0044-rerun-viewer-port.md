# Rerun viewer port Implementation Plan

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** A host driving two rigs gets one live Rerun viewer per rig instead
of the second run's stream landing in the first run's window. `RerunSink`
gains a `spawn_port`, the config file gains a `rerun_port` key (so a per-rig
config carries its own viewer port), and `run` gains `--rerun-port`. Defaults
unchanged: single-rig behavior is byte-for-byte today's. Closes #280.

**Architecture:** `rr.spawn()` already accepts a keyword-only `port` (default
9876) and then connects to that same port, so the whole feature is plumbing:
sink ctor kwarg `spawn_port` forwarded to `rr.spawn(port=...)`, consulted only
when `spawn=True` (the `spawn_memory_limit` precedent, plan 0014); a
`rerun_port` `[defaults]` key read like `max_steps` with a 1-65535 range; and
a `--rerun-port` flag that implies the live viewer (the way `--rerun-connect`
implies connecting) and beats the config key. Flag contradictions error loudly and
immediately: `--rerun-connect --rerun-port` (remote viewer vs local spawn)
and `--no-rerun --rerun-port` (no viewer vs a viewer port) both die at the
top of `_cmd_run`, before any component resolves. `application_id` is
already a ctor parameter and needs no work.

**Tech stack:** stdlib only. The rerun-sdk is NOT installed in this worktree's
venv; every sink test runs against the existing `_StartupRR`-style mocks, and
that is sufficient: the keyword-only `port: int = 9876` on `spawn()` was
verified in rerun_sdk 0.23.1, 0.34.x, and 0.35.0, CI's locked version is
0.34.0 (`uv.lock`), and the `test (rerun extra)` CI job never calls spawn
against the real SDK (its `spawn=True` tests all use fake `_rr` objects).

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src and tests), `uv run pytest --cov` at **100%**
  (branch coverage).
- D1 docstrings on every public module/class/function.
- Repo root is the `ir-wt-rerun-port` worktree at
  `~/robocurve/ir-wt-rerun-port`; run everything via `uv run ...` there.
- Deliberately updated existing tests, and ONLY these:
  - `tests/test_rerun_sink.py:97-108` and `:111-119` assert the exact
    `rr.spawn` kwargs dict; the expected dict gains `"port": 9876` (and the
    custom value in the second). Every other startup-mock test is untouched.
  - `tests/test_registry_cli.py:4357-4363` `_FakeRerunSink.__init__` gains a
    `spawn_port` kwarg (recorded like the others) or the CLI construction
    TypeErrors.
  Any other existing test that fails is a bug in the new code.
- Docs writing rules (no em dashes in prose).
- Commit messages: imperative, scoped; reference #280.

## Reference: current wiring (main @ 113aa906)

- `src/inspect_robots/logging/rerun_sink.py`: `RerunSink` :163; `__init__`
  :169-184 (`application_id="inspect_robots"`, `spawn=False`,
  `spawn_memory_limit="2GiB"`, `connect_url`, `jpeg_quality`, `queue_size`,
  `flush_timeout`); mutual-exclusion validation :185-192; `queue_size >= 1`
  ValueError :193-194; startup in `on_eval_start` :595-621 with
  `rr.init(self.application_id)` :603, `rr.spawn(memory_limit=self.spawn_memory_limit)`
  :605, `rr.connect_grpc(self.connect_url)` :607, blanket except → disabled
  warning :610-621.
- rerun-sdk 0.35.0 `spawn()` (verified in a sibling venv): keyword-only
  `port: int = 9876`, then `connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")`.
- `src/inspect_robots/cli.py`: `DEFAULT_RERUN_CONNECT_URL` :151; `--rerun`
  :275-282 (tri-state `BooleanOptionalAction, default=None`);
  `--rerun-connect` :283-292 (`nargs="?"`, const default URL); a
  `--rerun-port` flag slots right after :292. The ONE `RerunSink`
  construction site: `_cmd_run` :1352-1366 — connect branch :1355-1359,
  spawn branch :1360-1366 with the tri-state decision
  `elif args.rerun if args.rerun is not None else defaults.rerun:` :1360 and
  `RerunSink(spawn=True)` :1365. eval-set has no rerun wiring (deliberate,
  `docs/guide/cli.md:286-288`) — do not add any.
- `src/inspect_robots/defaults.py`: `Defaults` :72-99 (`rerun: bool = False`
  :88); the int-key pattern to copy is `max_steps` in `_read_config`
  :151-156; `_CONFIG_KEYS` :197-205; `_set_default` validation :208-222.
- Tests: `tests/test_rerun_sink.py` `_StartupRR` mock :78-94 (records
  `("init", (app_id, kwargs))`, `("spawn", kwargs)`, `("connect_grpc", url)`);
  exact-equality assertions :97-108, :111-119, :122-130, :133-146;
  mutual-exclusion :60-75. `tests/test_registry_cli.py` `_FakeRerunSink`
  :4352-4378, `_fake_rerun` fixture :4381-4387 (monkeypatches
  `inspect_robots.logging.rerun_sink.RerunSink`), `_run_adhoc` helper
  :4390-4396 (writes `rerun = true` config), the six rerun CLI tests
  :4399-4454. `tests/test_defaults.py` bool tests :230-245, int idiom
  :150-152. `config set` validation tests `tests/test_registry_cli.py:5099-5111`.
- Docs: `docs/guide/logging-and-rerun.md` ctor cheat-sheet :76-81,
  headless/tunnel paragraph :104-109; `docs/guide/cli.md:203` config example
  line; CHANGELOG `### Added` under `## [Unreleased]` (newest bullet on top,
  plan link + issue number); `src/inspect_robots/CLAUDE.md:28` `logging/` row.

---

### Task 1: `RerunSink(spawn_port=...)`

**Files:**
- Modify: `src/inspect_robots/logging/rerun_sink.py`
- Test: `tests/test_rerun_sink.py`

- [ ] **Step 1: Write the failing tests**

- Update the two exact-equality spawn assertions (deliberate, listed in
  Global Constraints): `:97-108` expects
  `("spawn", {"memory_limit": "2GiB", "port": 9876})`; `:111-119` keeps its
  custom memory limit and gains the default port.
- `test_custom_spawn_port_is_forwarded_verbatim`: `RerunSink(spawn=True,
  spawn_port=9877)` → last call `("spawn", {"memory_limit": "2GiB", "port": 9877})`.
- `test_spawn_port_out_of_range_raises`: `RerunSink(spawn=True, spawn_port=0)`
  and `spawn_port=65536` raise ValueError naming `spawn_port` (mirror the
  `queue_size` test idiom near :60-75).
- Confirm `test_default_startup_never_spawns_a_viewer` (:122-130) and the
  connect test (:133-146) pass untouched: `spawn_port` must not affect the
  no-spawn or connect paths.

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_rerun_sink.py -k "spawn" -v` — TypeError on the
unknown kwarg, plus the two updated exact-equality tests failing on the
missing `port`.

- [ ] **Step 3: Implement**

`__init__` gains `spawn_port: int = 9876` (keyword-only, next to
`spawn_memory_limit`); docstring sentence mirrors the existing one:
consulted only when `spawn` is true. Validate
`if not 1 <= spawn_port <= 65535: raise ValueError(...)` next to the
`queue_size` check. In `on_eval_start`, `rr.spawn(memory_limit=...,
port=self.spawn_port)`. Update the MODULE docstring line about "a viewer
already running on the default port" (rerun_sink.py:35-36) to say "on the
same port".

- [ ] **Step 4: Run tests, then the full gate set**

`uv run pytest tests/test_rerun_sink.py -v`, then
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/logging/rerun_sink.py tests/test_rerun_sink.py
git commit -m "rerun: spawn_port selects the live viewer port (#280)"
```

---

### Task 2: `rerun_port` config key

**Files:**
- Modify: `src/inspect_robots/defaults.py`
- Test: `tests/test_defaults.py`, `tests/test_registry_cli.py` (config set
  validation only)

- [ ] **Step 1: Write the failing tests**

In `tests/test_defaults.py`, mirroring the `max_steps`/`rerun` idioms:

- `test_config_rerun_port_parses_int`: `rerun_port = 9877` →
  `defaults.rerun_port == 9877`.
- `test_config_rerun_port_defaults_none`: absent key → `None`.
- `test_config_rerun_port_rejects_invalid`: parametrized over `"true"`,
  `"0"`, `"65536"`, `"9876.5"` → SystemExit naming the file and the 1-65535
  range.

In `tests/test_registry_cli.py` next to :5099-5111:

- `test_cli_config_set_rejects_bad_rerun_port`: `main(["config", "set",
  "rerun_port", "sometimes"])` exits with the range message;
  `main(["config", "set", "rerun_port", "9877"])` succeeds and `config show`
  prints it.

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_defaults.py -k rerun_port -v` — AttributeError on
the missing field; the `config set` test fails on the argparse choices list.

- [ ] **Step 3: Implement**

`Defaults` gains `rerun_port: int | None = None` (next to `rerun`).
`_read_config` gains a block copying the `max_steps` shape with the range
check `1 <= parsed <= 65535` and message
`"[defaults] rerun_port must be an integer in 1-65535, got {raw!r}"`.
`_CONFIG_KEYS` gains `"rerun_port"`. `_set_default` gains its own branch
(`if key == "rerun_port":`) with the same range validation and a
`SystemExit` message mirroring the `max_steps` one. `config show` in
`cli.py` gains the row `("rerun_port", defaults.rerun_port, None)` next to
the `rerun` row (cli.py:2200).

- [ ] **Step 4: Run tests, then the full gate set**

As Task 1.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/defaults.py src/inspect_robots/cli.py tests/test_defaults.py tests/test_registry_cli.py
git commit -m "config: rerun_port picks a per-rig live viewer port (#280)"
```

---

### Task 3: `--rerun-port` flag and the decision wiring

**Files:**
- Modify: `src/inspect_robots/cli.py`
- Test: `tests/test_registry_cli.py`

- [ ] **Step 1: Write the failing tests**

First extend `_FakeRerunSink.__init__` (:4357-4363) with
`spawn_port: int = 9876`, recorded like the other fields (deliberate,
listed in Global Constraints). Then, using the `_fake_rerun` + `_run_adhoc`
harness:

- `test_rerun_port_flag_spawns_viewer_on_that_port`: `--rerun-port 9877`
  WITHOUT `--rerun` → one sink, `spawn=True`, `spawn_port == 9877` (the flag
  implies the viewer, like `--rerun-connect` implies connecting).
- `test_config_rerun_port_reaches_spawned_viewer`: config with
  `rerun = true` and `rerun_port = 9878`, no flags → `spawn_port == 9878`.
- `test_rerun_port_flag_beats_config`: config `rerun_port = 9878`, flag
  `--rerun-port 9879` → `spawn_port == 9879`.
- `test_config_rerun_port_alone_does_not_enable_viewer`: config with ONLY
  `rerun_port = 9878` (no `rerun = true`), no flags → no RerunSink attached
  (the port key customizes, never switches on).
- `test_rerun_port_conflicts_with_rerun_connect`: both flags → SystemExit
  with a message naming both flags; no sink constructed.
- `test_rerun_port_conflicts_with_no_rerun`: `--no-rerun --rerun-port 9877`
  → SystemExit naming both flags; no sink constructed. (An explicit off
  switch silently losing to the port flag would ship unpinned: branch
  coverage cannot see `or`-operand arcs, so this contradiction is an error,
  matching the connect conflict.)
- `test_rerun_port_rejects_out_of_range`: `--rerun-port 0` → argparse error
  (SystemExit 2).
- Default-port arc: assert in one existing-style test (e.g. extend
  `test_config_rerun_attaches_live_viewer_sink` :4399-4405 with
  `spawn_port == 9876`) that no flag and no config key yields the default.

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_registry_cli.py -k rerun_port -v` — argparse
rejects the unknown flag.

- [ ] **Step 3: Implement**

- Flag on `p_run` after `--rerun-connect` (:292): `--rerun-port`,
  `type=_port_number, default=None, metavar="PORT"`, help: spawn the live
  viewer on this port (implies `--rerun`; a per-rig `rerun_port` config key
  sets the default). `_port_number(text: str) -> int` raises
  `argparse.ArgumentTypeError("port must be in 1-65535")` outside the range
  (module-level helper next to `_add_config_arg`, with a docstring).
- Both contradiction checks live at the TOP of `_cmd_run`, before any
  component resolves (a pure flag error must not touch hardware):
  `if args.rerun_port is not None and args.rerun_connect: raise
  SystemExit("--rerun-port spawns a local viewer and --rerun-connect "
  "streams to a remote one: pass only one")` and
  `if args.rerun_port is not None and args.rerun is False: raise
  SystemExit("--no-rerun disables the live viewer and --rerun-port "
  "requests one: pass only one")` (`args.rerun` is tri-state; `is False`
  fires only on an explicit `--no-rerun`).
- The spawn decision at :1360 becomes: connect branch unchanged; else
  `spawn_wanted = args.rerun_port is not None or (args.rerun if args.rerun
  is not None else defaults.rerun)` (safe now that the `--no-rerun`
  contradiction died earlier); when spawning,
  `port = args.rerun_port if args.rerun_port is not None else
  defaults.rerun_port`, and construct `RerunSink(spawn=True)` when `port is
  None` else `RerunSink(spawn=True, spawn_port=port)` (both arms are
  covered by the tests above).

- [ ] **Step 4: Run tests, then the full gate set**

As Task 1. All six pre-existing rerun CLI tests (:4399-4454) must pass
untouched.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/cli.py tests/test_registry_cli.py
git commit -m "cli: --rerun-port spawns the live viewer on a chosen port (#280)"
```

---

### Task 4: docs

**Files:**
- Modify: `docs/guide/logging-and-rerun.md` (cheat-sheet :76-81, headless
  paragraph :104-109)
- Modify: `docs/guide/cli.md` (config example block :198-204)
- Modify: `CHANGELOG.md` (`## [Unreleased]` → `### Added`, newest on top)
- Modify: `src/inspect_robots/CLAUDE.md` (`logging/` row :28)

- [ ] **Step 1: Guide edits**

Cheat-sheet gains a `spawn_port=9877` line comment mentioning
`--rerun-port`. Headless paragraph gains one sentence: hosts driving two
rigs give each config its own `rerun_port` so each run spawns its own
viewer window. `cli.md` config example gains
`rerun_port = 9877          # viewer port for this rig (default 9876)`.
No em dashes in prose.

- [ ] **Step 2: CHANGELOG**

`### Added` bullet: `RerunSink` `spawn_port`, the `rerun_port` config key,
and `--rerun-port`, giving each rig its own live viewer on multi-rig hosts
([plan 0044](plans/0044-rerun-viewer-port.md), #280).

- [ ] **Step 3: Module map + gates + commit**

Extend the `logging/` row with the spawn-port phrase at matching density.
`uv run pytest -q` green, then:

```bash
git add docs/guide/logging-and-rerun.md docs/guide/cli.md CHANGELOG.md src/inspect_robots/CLAUDE.md
git commit -m "docs: per-rig Rerun viewer ports (#280)"
```

---

## Out of scope

- An `--rerun-app-id` flag: `application_id` is already a `RerunSink`
  constructor parameter, and with one viewer per port the id no longer
  collides in practice.
- eval-set rerun wiring (deliberately absent, `docs/guide/cli.md:286-288`).
- `--rerun-connect` port sugar (the URL form already carries any port).
- Registry-level sink kwargs plumbing (callers constructing `RerunSink`
  directly can already pass `spawn_port`, the plan 0012 position).
