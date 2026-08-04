# Config file selection Implementation Plan

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** Let one host hold several rig configs. A new `INSPECT_ROBOTS_CONFIG`
environment variable and a `--config PATH` flag select which `config.ini` a
run, the setup wizard, and `config set|show` use, so a second rig no longer
requires hijacking `XDG_CONFIG_HOME` for the whole process, and running
`inspect-robots setup` for rig B no longer rotates rig A's config to
`config.ini.bak`. Closes #274.

**Architecture:** `config_path(env)` gains one highest-precedence rung: a
non-empty `INSPECT_ROBOTS_CONFIG` in the injected env mapping is returned as
the config *file* path verbatim. Every consumer (`load_defaults`,
`_set_default`, `run_setup`, and plugin default loaders that re-read the
config from `os.environ`) resolves the file through `config_path(env)`, so
this single rung covers eval runs, the wizard, and `config set|show` alike.
The `--config` flag is declared on the five subcommands that touch the config
and is applied in `main()` by writing the expanded path into `os.environ`
before dispatch — deliberately process-global, because plugins re-read the
config from `os.environ` during component construction and a threaded local
mapping would never reach them.

**Tech stack:** stdlib only (no new deps). pytest with injected env mappings;
`monkeypatch.setenv`/`delenv` wherever a test touches `os.environ`.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, covers `src` and `tests`), `uv run pytest --cov` at
  **100% coverage**.
- Every public module/class/function needs a docstring stating the contract
  (ruff D1).
- Repo root is the `ir-wt-config-select` worktree at
  `~/robocurve/ir-wt-config-select`; run everything via `uv run ...` there.
- Every existing test passes byte-for-byte untouched. If one fails, treat it
  as a bug in the new code.
- Docs follow the repo writing rules (`CLAUDE.md` "Writing style"): no em
  dashes in prose, headers use colons, no mid-sentence bold.
- Commit messages: imperative, scoped; reference #274.

## Reference: current wiring (main @ ba873203)

- `defaults.py:98-111` — `config_path(env)`: XDG_CONFIG_HOME else HOME/.config,
  else None; empty string counts as unset. The single chokepoint: `load_defaults`
  calls it at `defaults.py:265`, `_set_default` at `defaults.py:216`,
  `run_setup` at `_setup.py:1367`.
- `defaults.py:1-22` — module docstring describing the resolution model;
  extend it when the rung lands.
- `cli.py:222-229` — `build_parser()`; subparsers: `run` 239, `eval-set` 280,
  `doctor` 442, `setup` 449 (bare `sub.add_parser(...)`, no variable),
  `config` 455 with children `set` 457 and `show` 460 (`show` has no variable).
- `cli.py:2161-2193` — `main()`: `init_dotenv(os.environ)` at 2163,
  `parse_args` at 2166, then dispatch. The override must land in `os.environ`
  after parse and before any dispatch.
- Consumers that read defaults: `_cmd_run` (`cli.py:1245`), `_cmd_eval_set`
  (`cli.py:1391`), `_cmd_doctor` (`cli.py:2089`), `_cmd_config`
  (`cli.py:2123-2128`), `_cmd_setup` (`cli.py:2112-2120`, passes `os.environ`
  to `run_setup`).
- `_setup.py` `run_setup`: computes `config_path(env)` (~1367), prints the
  `— writes {path}` banner, `path.parent.mkdir(parents=True, exist_ok=True)`
  before the atomic write, rotates an existing file to `.bak`.
- `_dotenv.py:45-49` — `init_dotenv` uses `environ.setdefault`, so a `.env`
  can seed `INSPECT_ROBOTS_CONFIG` but a real env var wins. This is a feature
  (per-directory rig selection) and is documented in Task 4, not prevented.
- Tests: `tests/test_defaults_public.py:29-78` — the `config_path` test
  idiom (plain dict envs, tmp_path). `tests/test_registry_cli.py:88-94` —
  `main([...])` invocation harness; `:39` shows the
  `monkeypatch.setenv("XDG_CONFIG_HOME", ...)` pattern. `tests/test_setup.py`
  ~553 — the golden-config run_setup harness (env dict, scripted `input_fn`,
  `io.StringIO()` out).
- Public API: `config_path` is already in `defaults.__all__`
  (`tests/test_defaults_public.py:19`); its signature does not change, so
  `tests/test_api_snapshot.py` stays untouched.

---

### Task 1: `config_path` honors `INSPECT_ROBOTS_CONFIG`

**Files:**
- Modify: `src/inspect_robots/defaults.py`
- Test: `tests/test_defaults_public.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_defaults_public.py`, following the existing idiom:

- `test_config_path_env_override_wins_over_xdg_and_home`: env with all three
  of `INSPECT_ROBOTS_CONFIG=str(tmp_path / "rig-b.ini")`, `XDG_CONFIG_HOME`,
  `HOME` set; `config_path` returns exactly `Path(tmp_path / "rig-b.ini")`
  (the file path verbatim, no `inspect-robots/config.ini` suffix, whether or
  not the file exists).
- `test_config_path_env_override_empty_string_counts_as_unset`: override set
  to `""` with `XDG_CONFIG_HOME` set falls through to the XDG-derived path
  (matches the existing empty-string semantics at
  `test_defaults_public.py:76-79`).
- `test_load_defaults_reads_the_override_file`: write two config files, a
  decoy under `XDG_CONFIG_HOME/inspect-robots/config.ini` with
  `policy = decoy` and an override at `tmp_path / "rig-b.ini"` with
  `policy = rig-b`; `load_defaults` on an env naming both returns
  `policy == "rig-b"` with `policy_source == str(override_path)`.
- `test_set_default_writes_the_override_file`: `_set_default(env, "policy",
  "x")` with the override in env writes the override path (and returns it),
  leaving the XDG location untouched. Import `_set_default` the way
  `tests/test_defaults.py` does if it already imports it; otherwise a private
  import in this file is fine (mirror whichever exists).

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_defaults_public.py -k override -v` — the first two
fail on path mismatch; the last two fail reading/writing the XDG location.

- [ ] **Step 3: Implement**

In `config_path`, before the XDG rung:

```python
    if override := env.get("INSPECT_ROBOTS_CONFIG"):
        return Path(override)
```

Add a module constant `_ENV_CONFIG = "INSPECT_ROBOTS_CONFIG"` next to the
other `_ENV_*` names (`defaults.py:36-38`) and use it in both the code and
the docstring. Update the `config_path` docstring: the override names the
config *file* itself (not a directory), takes precedence over
`XDG_CONFIG_HOME`/`HOME`, and an empty value counts as unset. Add one
sentence to the module docstring's resolution description.

- [ ] **Step 4: Run tests, then the full gate set**

`uv run pytest tests/test_defaults_public.py -v`, then
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/defaults.py tests/test_defaults_public.py
git commit -m "config: honor INSPECT_ROBOTS_CONFIG as the config file override (#274)"
```

---

### Task 2: `--config PATH` flag on the config-touching subcommands

**Files:**
- Modify: `src/inspect_robots/cli.py`
- Test: `tests/test_registry_cli.py`

- [ ] **Step 1: Write the failing tests**

Every test that makes `main()` see `--config` MUST first
`monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", "placeholder")` so pytest
restores `os.environ` at teardown (main() overwrites the key in place).

- `test_config_show_honors_config_flag`: write a config at
  `tmp_path / "rig-b.ini"` with `[defaults]\npolicy = rig-b\n`;
  `main(["config", "show", "--config", str(path)])` output (capsys) contains
  `policy: rig-b` and the override path as its source.
- `test_config_set_honors_config_flag`: `main(["config", "set", "policy",
  "x", "--config", str(path)])` writes `path`, not the XDG location (set
  `XDG_CONFIG_HOME` to a tmp decoy dir and assert nothing appears under it).
- `test_config_flag_beats_env_var`: `monkeypatch.setenv` the env var to file
  A, pass `--config` file B; `config show` reads B.
- `test_config_flag_expands_tilde`: `monkeypatch.setenv("HOME",
  str(tmp_path))`, write `tmp_path / "rig.ini"`, pass `--config "~/rig.ini"`;
  `config show` reads it (assert on a distinctive policy value).
- `test_no_config_flag_leaves_environ_untouched`:
  `monkeypatch.delenv("INSPECT_ROBOTS_CONFIG", raising=False)`, run
  `main(["list"])`, assert the key is still absent from `os.environ`.

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_registry_cli.py -k config_flag -v` — argparse
errors on the unknown `--config` flag (SystemExit 2).

- [ ] **Step 3: Implement**

A tiny helper next to `_add_shared_eval_args`:

```python
def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the --config override to a subcommand that reads the config file."""
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="use this config file instead of "
        "<config-home>/inspect-robots/config.ini (sets $INSPECT_ROBOTS_CONFIG "
        "for the invocation; useful for hosts driving more than one rig)",
    )
```

Attach it to `p_run`, `p_eval_set`, `p_doctor`, the `setup` parser (assign
`sub.add_parser("setup", ...)` at `cli.py:449` to a variable), and the
`config` children `p_set` and `show` (assign the `show` parser at
`cli.py:460` to a variable). Do not attach it to the parent `config` parser;
the flag reads naturally after the child token
(`inspect-robots config show --config ...`).

In `main()` immediately after `parse_args`:

```python
    if config_override := getattr(args, "config", None):
        # os.environ, not a local mapping: plugins re-read the config from
        # os.environ while constructing components, so the flag must be
        # visible process-wide.
        os.environ["INSPECT_ROBOTS_CONFIG"] = str(Path(config_override).expanduser())
```

`Path.expanduser` belongs here at the CLI boundary (it consults the real
environment); `config_path` itself stays literal. `getattr` with a default
covers subcommands without the flag (`list`, `view`, ...). An empty
`--config ""` is falsy and behaves as unset, matching the env-var semantics.
Check whether `Path` is already imported in `cli.py` (it is, line ~50s) and
that the walrus target does not shadow anything.

- [ ] **Step 4: Run tests, then the full gate set**

`uv run pytest tests/test_registry_cli.py -v`, then all four gates as in
Task 1.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/cli.py tests/test_registry_cli.py
git commit -m "cli: --config selects the config file per invocation (#274)"
```

---

### Task 3: pin the wizard to the selected config file (test-only)

**Files:**
- Test: `tests/test_setup.py`

No production code is expected: `run_setup` resolves its target through
`config_path(env)`, so Task 1 already redirects it. This task pins that
behavior end to end so a regression in the chokepoint cannot silently send
rig B's wizard back to rig A's file.

- [ ] **Step 1: Write the (initially passing) test**

Copy the golden-config harness scaffold (`tests/test_setup.py` ~553). Build
an env dict with BOTH `XDG_CONFIG_HOME` (pointing at a decoy tree that
already contains an `inspect-robots/config.ini` with recognizable content)
AND `INSPECT_ROBOTS_CONFIG=str(tmp_path / "rig-b" / "config.ini")` (parent
directory deliberately nonexistent: `run_setup` mkdirs it). Drive the minimal
Enter-accepting wizard flow the neighboring tests use. Assert:

- the override file exists and contains the written `[defaults]`;
- the banner line in `out` names the override path (`— writes ...rig-b...`);
- the decoy config's bytes are unchanged and no `.bak` appeared next to
  either file.

Run it against Task 1's code: it must pass immediately. Then temporarily
revert the `config_path` rung (`git stash` on defaults.py) and confirm the
test FAILS, proving it guards the chokepoint; unstash.

- [ ] **Step 2: Full gate set, then commit**

```bash
git add tests/test_setup.py
git commit -m "setup: cover the wizard writing an INSPECT_ROBOTS_CONFIG override (#274)"
```

---

### Task 4: docs

**Files:**
- Modify: `docs/guide/cli.md` (the "Default policy and embodiment" section,
  lines 29-65)
- Modify: `CHANGELOG.md` (`## [Unreleased]` → `### Added`; match the existing
  heading structure and entry format)
- Modify: `src/inspect_robots/CLAUDE.md` (module map rows for `cli.py` line
  29 and `defaults.py` line 34)

- [ ] **Step 1: cli guide**

After the existing resolution list, add a short "Several rigs on one host"
passage: the config file itself is chosen by `--config PATH` first,
`$INSPECT_ROBOTS_CONFIG` second, then the `XDG_CONFIG_HOME`/`HOME` derivation.
Show the two-command example:

```bash
inspect-robots setup --config ~/.config/inspect-robots/rig-b.ini
inspect-robots run --task my-benchmark --config ~/.config/inspect-robots/rig-b.ini
```

Keep the "There is deliberately no project-local config file" sentence, and
extend the thought: because `.env` values load into the environment, a
directory's `.env` can pin `INSPECT_ROBOTS_CONFIG` for everything run there;
treat a checked-in `.env` that selects hardware with the same suspicion as a
checked-in config. Follow the writing rules (no em dashes in prose).

- [ ] **Step 2: CHANGELOG**

One `### Added` entry under `## [Unreleased]` (#274, plan 0042): the
`INSPECT_ROBOTS_CONFIG` env var and per-subcommand `--config` flag select the
config file, enabling per-rig configs on multi-rig hosts; the wizard writes
the selected file.

- [ ] **Step 3: Module map**

`defaults.py` row: note the file-location override
(`INSPECT_ROBOTS_CONFIG`). `cli.py` row: add `--config` to the subcommand
summary. Match each row's phrasing density; mention plan 0042 alongside the
existing plan references if the row carries any.

- [ ] **Step 4: Gates + commit**

`uv run pytest -q` green, then:

```bash
git add docs/guide/cli.md CHANGELOG.md src/inspect_robots/CLAUDE.md
git commit -m "docs: document config file selection for multi-rig hosts (#274)"
```

---

## Out of scope

- Named rig profiles inside one config file (`[rig:NAME]` sections plus a
  `--rig` selector): heavier schema surgery, superseded by file-level
  selection for the observed need; revisit if two files prove clumsy.
- A `doctor` line printing the active config path: `config show` sources
  already name the file.
- Plugin loader changes (`inspect-robots-yam` and friends): they resolve
  through `config_path(os.environ)` and inherit the override with no change.
- Blocking `.env` from seeding `INSPECT_ROBOTS_CONFIG`: documented instead;
  the setdefault semantics already let a real env var win.
- A rig lock preventing two processes from driving one rig (tracked
  separately; different mechanism entirely).
