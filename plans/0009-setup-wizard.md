# 0009 — `inspect-robots setup`: first-run wizard for defaults and camera discovery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the README's error-prone heredoc quickstart with an
interactive `inspect-robots setup` wizard that discovers V4L2 camera device
paths, offers the documented defaults (`molmoact2`, `yam_arms`, ...), and
writes `~/.config/inspect-robots/config.ini`.

**Architecture:** A new stdlib-only module `src/inspect_robots/_setup.py`
holds the wizard with all IO injected (env mapping, `input_fn`, output
stream, V4L2 directory paths) so every branch is unit-testable to the 100%
coverage gate. `cli.py` grows a thin `setup` subcommand that wires real IO.
The README and docs point first-time users at the wizard; the manual heredoc
stays as a fallback.

**Tech stack:** Python stdlib only (`configparser`, `pathlib`); no new deps
(core stays NumPy-only, and the wizard itself imports no NumPy).

## 1. Motivation

The README quickstart asks users to paste a heredoc and hand-edit three
placeholder paths (`/dev/v4l/by-id/YOUR-TOP-CAM`, ...). First-time users have
no idea which of the `/dev/v4l/by-id/` entries is which physical camera, or
that the `-video-index0` node is the color stream while `-video-index1` is
UVC metadata. That is the highest-friction step of onboarding and the wizard
removes it:

```text
$ uv run inspect-robots setup
inspect-robots setup — writes ~/.config/inspect-robots/config.ini

policy [molmoact2]:                      # Enter accepts the suggestion
embodiment [yam_arms]:
scorer [success_at_end]:
max steps [1200]:
live rerun viewer [true]:
store camera frames [true]:

Found 3 camera device(s) under /dev/v4l/by-id:
  1. usb-Global_Shutter_Camera_01-video-index0
  2. usb-Global_Shutter_Camera_02-video-index0
  3. usb-Global_Shutter_Camera_03-video-index0
top camera — number, 'u' to identify by unplugging, 's' to skip: u
  Unplug the top camera now, then press Enter...
  That was: usb-Global_Shutter_Camera_02-video-index0
  Plug it back in, then press Enter... ok
left camera — number, 'u' to identify by unplugging, 's' to skip: 1
right camera — number, 'u' to identify by unplugging, 's' to skip: 3

Wrote /home/user/.config/inspect-robots/config.ini
Next: uv run inspect-robots "place the fork on the plate"
```

## 2. What exists today (grounding)

- `_defaults.py`: `_config_path(env)` resolves the config file location from
  `$XDG_CONFIG_HOME`/`$HOME`; `_read_config(path) -> Defaults` parses and
  validates it (raising `SystemExit` on malformed input); `set_default(env,
  key, value)` atomically writes one `[defaults]` key and round-trips
  unknown sections; `parse_value` / `CONFIG_KEYS` back validation. Note
  `_parse_args_section` expands `~` on read, so `Defaults` is **not** a
  faithful representation of the file text.
- `cli.py`: argparse subcommands `list` / `run` / `inspect` / `config` /
  `doctor`, dispatched from `main()` via `args.command`; the
  `_SUBCOMMANDS` tuple (cli.py:80) backs the zero-config sugar guard.
  Registry imports are deliberately lazy (function-local). `_pick_component`
  prints "fix: ... or run 'inspect-robots config set ...'" when no default
  exists; `tests/test_registry_cli.py:918` asserts on that message.
- CLI tests live in `tests/test_registry_cli.py` (there is no
  `tests/test_cli.py`).
- README "Quickstart" (lines ~70-95): the heredoc this plan replaces.
  `docs/guide/quickstart.md` ("From the command line", ~lines 55-63) and
  `docs/guide/cli.md` (one `##` section per subcommand; config resolution
  order at line ~34) are the docs-site counterparts.
- The yam rig (`inspect-robots-yam` plugin, separate repo) reads
  `[embodiment.args]` keys `top_cam_device` / `left_cam_device` /
  `right_cam_device` as plain strings and **raises `ValueError` unless the
  three are set all-or-none** (`YamConfig.__post_init__`). Users also keep
  non-camera keys in `[embodiment.args]` (e.g. `cameras = wrist,front` in
  docs/guide/cli.md, `left_channel` in the `config set` round-trip test).
- Gates: `ruff check`, `ruff format --check`, `mypy --strict` (src **and**
  tests), `pytest --cov` with branch coverage `fail_under = 100`. Core
  stays NumPy-only.

## 3. Design

### 3.1 UX decisions (binding)

1. **Interactive only.** `setup` on a non-TTY stdin exits with
   `SystemExit("setup is interactive; see the README for manual config")`.
   No `--yes` mode (YAGNI; `config set` already covers scripting).
2. **Suggested defaults are prompt placeholders, not hard requirements.**
   Constants mirror the README: `policy=molmoact2`, `embodiment=yam_arms`,
   `scorer=success_at_end`, `max_steps=1200`, `rerun=true`,
   `store_frames=true`. Pressing Enter accepts; typing overrides. If an
   entered (or accepted) policy/embodiment name is not in the registry, the
   wizard **warns** ("'molmoact2' is not registered here — install its
   plugin, e.g. `uv pip install inspect-robots-yam`") but accepts it, since
   users often configure before installing plugins. No `sim_embodiment`
   prompt (YAGNI; `config set sim_embodiment` exists).
3. **Existing config wins over built-in suggestions.** If the file exists
   and parses, its current values become the prompt defaults, and the old
   file is backed up to `config.ini.bak` before the new one replaces it
   (atomic tmp+rename, same pattern as `set_default`). If the existing file
   is **malformed**, the wizard must not die (it is the
   obvious repair tool): it prints the parse error and asks "Back up the
   broken file and start fresh? [Y/n]" — yes proceeds with built-in
   suggestions, no aborts without writing. This repair prompt covers only
   files `configparser` cannot parse; a parseable file with a
   **type-invalid value** (e.g. `max_steps = abc`) is handled per prompt:
   the raw value is used as the prompt default only when it passes that
   prompt's validation, otherwise the built-in suggestion is shown with a
   one-line note ("ignoring invalid max_steps 'abc' from config.ini") and
   the rest of the file is untouched.
4. **Validated re-prompt.** `max_steps` must parse as int >= 1; booleans
   must parse as bool (via the existing `parse_value`). Invalid input prints
   the constraint and re-asks (no crash, no silent acceptance).
5. **Camera discovery scans `/dev/v4l/by-id` first, `/dev/v4l/by-path` as
   fallback.** Prefer entries ending in `-video-index0` (the color node;
   `-index1` is UVC metadata), falling back to all entries when none match
   that suffix. Identical serial-less cameras **collide** in by-id (udev
   derives the name from vendor/model/serial, last writer wins), so: when
   the index0-filtered by-path listing has more entries than by-id, the
   wizard prints one explanatory line ("only N by-id entries for M detected
   cameras — identical cameras without serials collide there; by-path names
   are stable per physical USB port") and role prompts accept `p` to switch
   the listing to by-path. If neither directory yields devices, print why
   ("no /dev/v4l devices found (not Linux, or no cameras attached)") and
   offer manual path entry or skip. The camera section is offered with
   default **yes** when devices were found or the existing config already
   has `*_cam_device` keys, default **no** otherwise.
6. **Role assignment.** For each role (top/left/right) the prompt accepts:
   a device number; `u` (unplug-to-identify: rescan, the disappeared entry
   is the answer; if zero or 2+ entries disappeared, explain and re-ask;
   then ask to replug and rescan — if the device has not returned, offer
   one "press Enter to rescan" retry before warning and keeping the
   assignment); `p` (toggle by-id/by-path listing — always accepted, but
   advertised in the prompt text only when §3.1.5's collision heuristic
   fired, i.e. the index0-filtered by-path listing has more entries than
   by-id, which is why the §1 transcript shows the shorter prompt); `s`
   (skip); or a
   literal absolute path (accepted with only an advisory warning when the
   path does not exist locally, since configs are often written on a dev
   box). A pre-existing assignment for the role is the Enter-accept
   default, shown as "(current)" — or "(current, not detected)" when it is
   absent from the scan. Assigning one device to two roles triggers a
   warning plus confirm; declining re-asks.
7. **All-three-or-none cameras.** The yam plugin rejects partial camera
   sets, so if the user ends the section with 1 or 2 roles assigned the
   wizard says so ("yam_arms needs all three cameras or none; writing
   none") and asks: go back to the camera section, or write the config
   without any `*_cam_device` keys. It never writes a partial set.
8. **The wizard renders the file itself and carries unmanaged content
   through raw.** The final INI is rendered from a template with the same
   explanatory comments as the README block (comments survive because we
   render text, not `configparser.write`). Managed content is the
   `[defaults]` keys the wizard prompts for and the three `*_cam_device`
   keys in `[embodiment.args]`; skipped managed keys are omitted.
   Everything else — other sections (`[policy.args]`, ...), non-camera
   keys inside `[embodiment.args]`, **and unmanaged keys inside
   `[defaults]` itself** (e.g. a `sim_embodiment` set earlier via
   `config set`) — is carried through verbatim from a **separate raw
   read** of the existing file:
   `ConfigParser(interpolation=None, inline_comment_prefixes=(";", "#"))`,
   raw string values, no `~` expansion, no `parse_value` (reusing
   `_read_config`/`_parse_args_section` here would bake in expanded home
   paths and re-case booleans). `%` in values must not crash the wizard.
   Comment loss in carried-through sections is the documented trade-off;
   `.bak` is the recovery hatch.
9. **Ctrl-C / Ctrl-D abort cleanly**: `EOFError`/`KeyboardInterrupt` from
   `input_fn` exit with "setup aborted; nothing written" (exit code 1) and
   never leave a partial file (the write happens once, at the end).

### 3.2 Module: `src/inspect_robots/_setup.py`

Everything the wizard touches is injected, mirroring how `_defaults.py`
takes `env`:

```python
SUGGESTED = {
    "policy": "molmoact2",
    "embodiment": "yam_arms",
    "scorer": "success_at_end",
    "max_steps": "1200",
    "rerun": "true",
    "store_frames": "true",
}
CAM_ROLES = ("top", "left", "right")  # -> {role}_cam_device in [embodiment.args]
V4L_BY_ID = Path("/dev/v4l/by-id")
V4L_BY_PATH = Path("/dev/v4l/by-path")

def run_setup(
    env: Mapping[str, str],
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
    interactive: bool,
    by_id_dir: Path = V4L_BY_ID,
    by_path_dir: Path = V4L_BY_PATH,
) -> int:
    """Drive the wizard; returns process exit code."""

def _scan_cameras(v4l_dir: Path) -> list[str]:
    """Sorted absolute device paths; -video-index0 entries preferred."""

def _identify_by_replug(role, devices, *, input_fn, out, rescan) -> str | None:
    """The unplug/replug diff; rescan is a zero-arg callable for testability."""

def _read_raw_config(path: Path) -> dict[str, dict[str, str]] | str:
    """Raw section->key->string map (interpolation off).

    Unparseable file -> the parse-error text (so the repair prompt of
    UX decision 3 can show it), never an exception.
    """

def _render_config(defaults: dict[str, str], embodiment_args: dict[str, str],
                   carried: dict[str, dict[str, str]]) -> str:
    """Full config.ini text, commented like the README block."""
```

Internal prompt helpers (`_ask(prompt, default, validate)`) loop until valid.
`registered` from `inspect_robots.registry` is imported lazily inside the
one function that warns about unregistered names, keeping import cost and
the NumPy-only-core CI job unaffected.

### 3.3 CLI integration (`cli.py`)

- `sub.add_parser("setup", help="interactive first-run wizard: pick defaults "
  "and discover camera devices, then write config.ini")` — no arguments.
- `main()` dispatch: `if args.command == "setup": return _cmd_setup()` where
  `_cmd_setup` calls `run_setup(os.environ, input_fn=input, out=sys.stdout,
  interactive=sys.stdin.isatty())`.
- Add `"setup"` to the `_SUBCOMMANDS` tuple (cli.py:80).
- `_pick_component`'s guidance string gains the wizard:
  `"fix: pass --{kind} NAME, set ${...}, run 'inspect-robots setup', or
  'inspect-robots config set {kind} NAME'"` (keeps the exact substring
  `inspect-robots config set` that `tests/test_registry_cli.py:918`
  asserts on; extend that test to also expect `inspect-robots setup`).
  `_pick_sim_embodiment`'s message stays unchanged (the wizard does not
  configure `sim_embodiment`).
- Module docstring subcommand list gains one line for `setup`.

### 3.4 README and docs

- README Quickstart: lead with
  ```bash
  uv run inspect-robots setup
  ```
  and one short paragraph: walks you through defaults, lists the cameras
  under `/dev/v4l/by-id`, and can identify which is which when you unplug
  one; writes `~/.config/inspect-robots/config.ini`. One sentence notes
  that later `inspect-robots config set` edits drop comments from the file
  (already the documented `set_default` trade-off). The existing heredoc
  moves under a "Prefer to write the file yourself?" line, content
  unchanged. Style rules apply (no em dashes in prose, no mid-sentence
  bold, headers use colons).
- `docs/guide/cli.md`: add a new `## inspect-robots setup` section (the
  page is one `##` per subcommand; there is no single list to append to).
- `docs/guide/quickstart.md`: mention `inspect-robots setup` in the "From
  the command line" section.
- `CHANGELOG.md`: one "Added" entry.
- `src/inspect_robots/CLAUDE.md` module map: add `_setup.py` row; extend the
  `cli.py` row's subcommand list.

## 4. Non-goals (YAGNI)

- No camera **preview** (needs OpenCV; core is NumPy-only). Unplug-diff
  identification is dependency-free and unambiguous.
- No plugin-provided setup hooks / entry points. The three yam camera roles
  are constants here; a second embodiment plugin with different args is the
  trigger to generalize.
- No `--non-interactive` / `--yes` flags; `config set` is the scripting API.
- No `sim_embodiment` prompt; `config set sim_embodiment` covers it.
- No Windows camera support (`/dev/v4l` is Linux; on other OSes the wizard
  still configures `[defaults]` and offers manual path entry).

## 5. Tasks

### Task 1: scan + raw-read + render (pure helpers)

**Files:** create `src/inspect_robots/_setup.py`, `tests/test_setup.py`.

- [ ] Failing tests — `_scan_cameras`: index0 filtering (mixed
  index0/index1 dir → only index0, sorted); fallback (no index0 suffix →
  all entries); missing dir → `[]`. `_read_raw_config`: raw `%` value
  survives (no interpolation error); `~` paths stay literal; malformed file
  → the parse-error string. `_render_config`: golden test (full defaults +
  3 cams → exact INI text, comments included); omits skipped managed keys
  and empty sections; carries through an unknown `[policy.args]` section,
  a non-camera `[embodiment.args]` key (e.g. `left_channel = can2`)
  alongside newly assigned cam keys, **and** an unmanaged `[defaults]`
  key (`sim_embodiment = cubepick`).
- [ ] Implement; `uv run pytest tests/test_setup.py -v` passes.
- [ ] Gates: `uv run ruff check . && uv run mypy`.
- [ ] Commit `feat(setup): camera scan + config rendering helpers`.

### Task 2: prompt loop + `run_setup` happy path

**Files:** modify `src/inspect_robots/_setup.py`, `tests/test_setup.py`.

- [ ] Failing tests drive `run_setup` with a scripted `input_fn` (pop from a
  list) and `io.StringIO` out, `by_id_dir`/`by_path_dir` pointed at
  `tmp_path` dirs, `XDG_CONFIG_HOME` in `env` pointed at `tmp_path`:
  all-Enter run writes the README-equivalent config; typed overrides land
  in the file; invalid `max_steps` ("abc", "0") re-prompts;
  non-interactive → `SystemExit`; EOF mid-wizard → exit 1, no file
  written; existing valid config: values become prompt defaults and `.bak`
  is created; existing **malformed** config: repair prompt shows the parse
  error (yes → fresh suggestions + `.bak`, no → abort, nothing written);
  parseable config with type-invalid `max_steps = abc` → built-in
  suggestion offered with the "ignoring invalid" note, other keys still
  used; unregistered-name warning text appears (registry check
  monkeypatched); registered name → no warning; non-camera
  `[embodiment.args]` keys **and** an unmanaged `sim_embodiment` in
  `[defaults]` survive the rewrite; all-three-or-none guard: ending the
  camera section with 1 or 2 roles assigned → the message plus both
  branches ("go back" re-enters the section; "write none" writes a config
  with zero `*_cam_device` keys).
- [ ] Implement prompt helpers + `run_setup` minus camera interactivity
  (camera section: number pick + manual path with advisory-only existence
  warning + skip + the all-three-or-none guard of §3.1.7).
- [ ] Gates + commit `feat(setup): interactive wizard core`.

### Task 3: unplug-to-identify, by-path toggle, duplicate guard

**Files:** modify `src/inspect_robots/_setup.py`, `tests/test_setup.py`.

- [ ] Failing tests: `u` flow where rescan (injected callable returning
  shrinking then restored lists) identifies the missing device; zero or 2+
  devices missing → explanatory message, re-prompt; replug not detected →
  one extra Enter-to-rescan retry, then warning but assignment kept; `p`
  toggles the listing to by-path and back, and the prompt text mentions
  `p` only when the index0-filtered by-path listing has more entries than
  by-id (it is accepted regardless); by-id shorter than by-path → the
  collision explainer line is printed; same device picked twice →
  confirm prompt, 'n' re-asks; pre-existing assignment shown as
  "(current)" Enter-accept default, and "(current, not detected)" when
  absent from the scan.
- [ ] Implement `_identify_by_replug`, the `p` toggle, and wire into the
  role prompt.
- [ ] Gates + commit `feat(setup): unplug-to-identify cameras`.

### Task 4: CLI wiring

**Files:** modify `src/inspect_robots/cli.py`,
`tests/test_registry_cli.py` (the existing CLI test module; follow its
patterns).

- [ ] Failing tests: `inspect-robots setup` with patched stdin-isatty=False
  → the interactive-only SystemExit; dispatch reaches `run_setup` (monkey-
  patched) and returns its code; `--help` lists setup; the
  `_pick_component` error-message test now also expects
  `inspect-robots setup`; `_SUBCOMMANDS` contains `"setup"`.
- [ ] Implement subparser, `_cmd_setup`, `_SUBCOMMANDS` entry, docstring
  line, guidance string.
- [ ] Gates incl. full `uv run pytest --cov` (100%) + commit
  `feat(cli): setup subcommand`.

### Task 5: README, docs, changelog, module map

**Files:** modify `README.md`, `docs/guide/cli.md`,
`docs/guide/quickstart.md`, `CHANGELOG.md`, `src/inspect_robots/CLAUDE.md`.

- [ ] Rewrite Quickstart per §3.4 (setup first, heredoc as fallback,
  comment-loss sentence), add the `## inspect-robots setup` docs section,
  quickstart-guide mention, changelog entry, module-map row.
- [ ] Check the README writing-style rules (no em dashes in prose, header
  style) and that the heredoc block content is unchanged.
- [ ] Commit `docs: point quickstart at inspect-robots setup`.

### Task 6: end-to-end sanity + PR

- [ ] `uv sync --all-packages --extra dev` in the worktree, then full gates:
  `ruff check .`, `ruff format --check .`, `mypy`, `pytest --cov`
  (must report 100% with branch coverage).
- [ ] Manual smoke: `uv run inspect-robots setup` with `XDG_CONFIG_HOME`
  pointed at a temp dir (macOS: camera section reports no devices and
  offers manual entry — expected).
- [ ] Push branch `feat/setup-wizard`, open PR; `ci-ok` is the required
  check.
