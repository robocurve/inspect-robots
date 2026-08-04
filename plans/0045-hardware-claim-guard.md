# Hardware claim guard Implementation Plan

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** Two eval processes must not silently drive the same hardware.
Before constructing a hardware embodiment, the CLI takes an advisory `flock`
on a lockfile per configured device-slot value (CAN channel names, camera and
serial node paths) and fails loudly ("can0 is already claimed by PID N")
when another live process holds one; claims release on teardown and vanish
with the process (flock is kernel-scoped, so no stale-lock handling exists
or is needed). Sim embodiments declare no device slots and are untouched.
POSIX-only; where `fcntl` is unavailable the guard is a silent no-op.
Closes #281.

**Architecture:** A new private module `_claims.py` (stdlib-only, lazy
`fcntl` import) exposes `claim_devices(slots, kvs, env) -> DeviceClaim`. It
selects the kvs values whose keys match a `DeviceSlot.arg`, normalizes each
(paths resolve through symlinks so a by-id and a by-path spelling of one
camera collide; plain strings like `can0` are taken verbatim), and flocks
`<runtime-dir>/inspect-robots/locks/<sha256[:16]>.lock` per value,
`LOCK_EX | LOCK_NB`. On conflict it raises `SystemExit` with the device
value and the holder's PID (read from the lockfile's advisory content); on
success it writes its own PID+value into the file and keeps the fd open.
The CLI acquires in `_resolve_components` between policy resolution and
embodiment construction (fail before touching hardware), carries the claim
in `_ResolvedComponents`, and releases it in the existing
`finally: embodiment.close()` blocks of `run` and `eval-set`. The guard
never blocks an eval for environmental reasons: unusable lock directory,
missing `fcntl`, or an unregistered embodiment name all degrade to a
warning or silent no-op.

**Tech stack:** stdlib only (`fcntl` lazily imported, `hashlib`, `os`,
`pathlib`). pytest; flock contention is testable in-process because flock
conflicts across distinct file descriptors within one process.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src and tests), `uv run pytest --cov` at **100%**
  (branch coverage: BOTH sides of the lazy-import try and every degradation
  branch need tests).
- D1 docstrings on every public module/class/function.
- Repo root is the `ir-wt-claim-guard` worktree at
  `~/robocurve/ir-wt-claim-guard`; run everything via `uv run ...` there.
- **Core stays NumPy-only**: no `filelock` dependency; `fcntl` is imported
  inside the function, never at module top (the `_v4l2_color_capture`
  precedent, `_setup.py:966-969`).
- Every existing test passes byte-for-byte untouched. If one fails, treat it
  as a bug in the new code.
- POSIX-only tests that monkeypatch attributes ON the real `fcntl` module
  carry the `_needs_fcntl` skipif (`tests/test_setup.py:621` precedent); the
  no-fcntl arc is covered on Linux via
  `monkeypatch.setitem(sys.modules, "fcntl", None)`
  (`tests/test_setup.py:709-714` precedent).
- `_claims` is private: no `inspect_robots.__all__` change, no
  `test_api_snapshot.py` change.
- Docs writing rules (no em dashes in prose).
- Commit messages: imperative, scoped; reference #281.

## Reference: current wiring (main @ 113aa906)

- `src/inspect_robots/conformance.py`: `DEVICE_KINDS = ("v4l2", "can",
  "serial")` :32; frozen `DeviceSlot(arg, kind, label, group=None)` :35-50;
  `device_slots(factory)` :53-72 (reads `DEVICE_SLOTS` off the FACTORY,
  never crashes, filters to known kinds).
- Registry: `registered("embodiment") -> dict[str, Callable]`
  (`registry.py:103-108`); the name→factory→slots idiom is
  `_setup.py:1443-1450` (`slots = device_slots(factories[name]) if name in
  factories else ()`).
- `src/inspect_robots/cli.py`: `_ResolvedComponents` NamedTuple :1161-1169
  (policy, policy_name, policy_source, embodiment, embodiment_name,
  embodiment_source); `_resolve_components` :1194-1236 — embodiment args
  dict built at :1225 (`embodiment_kvs = {**embodiment_defaults,
  **_parse_kvs(args.embodiment_args)}`), embodiment constructed LAST at
  :1227-1233 (docstring :1197-1199 says this ordering exists so callers can
  open their try/finally right after). `_cmd_run` :1275: resolve :1322-1323,
  `try:` :1324, `finally: embodiment.close()` :1390-1396. `_cmd_eval_set`
  :1437: resolve :1466-1467, `finally:` :1518-1522. `_cmd_doctor` :2144
  constructs too but is a short-lived diagnostic (out of scope, see below).
- Programmatic `eval()` (`eval.py:136-231`): `owns_embodiment =
  isinstance(embodiment, str)` :203, closes only what it resolved :227-231.
  Out of scope here (see below).
- fcntl precedent: `_setup.py:965-1000` (`try: import fcntl / except
  ImportError: return None`); plan 0013 documented the same shape.
- No lockfile/XDG_RUNTIME_DIR/pid conventions exist anywhere in `src/` yet;
  env-injected path derivation precedent is `defaults.config_path(env)`
  (`defaults.py:106-121`).
- Coverage config: `pyproject.toml:125-141` (`branch = true`,
  `fail_under = 100`, `pragma: no cover` honored).
- Module map: `src/inspect_robots/CLAUDE.md` table :8-38 (underscore
  modules listed; add a `_claims.py` row), "Key invariants" :40+ with the
  "close what we open" invariant :55.
- Docs: `docs/guide/adapters.md:35-83` "Declare device slots" (the natural
  home for "slots are also claimed at run time"); CHANGELOG `### Added`
  under `## [Unreleased]`.

---

### Task 1: `_claims.py` — normalization, lock dir, `DeviceClaim`, `claim_devices`

**Files:**
- Create: `src/inspect_robots/_claims.py`
- Create: `tests/test_claims.py`

**Interfaces (Task 2 consumes):**

```python
class DeviceClaim:
    """Held advisory locks on a rig's devices; release() is idempotent."""
    def release(self) -> None: ...

def claim_devices(
    slots: tuple[DeviceSlot, ...],
    kvs: Mapping[str, Any],
    env: Mapping[str, str],
) -> DeviceClaim: ...
```

Behavior contract:

- Claimed values: for each slot, `kvs.get(slot.arg)` when it is a non-empty
  `str` (None/absent/non-string skipped). `v4l2`/`serial` values are
  normalized `str(Path(value).resolve())` (a dangling or non-path value
  resolves to itself harmlessly); `can` values are taken verbatim.
  Duplicates after normalization are claimed once.
- Lock directory: `<runtime>/inspect-robots/locks` where `<runtime>` is
  `env["XDG_RUNTIME_DIR"]` when set and non-empty, else
  `Path(tempfile.gettempdir()) / f"inspect-robots-{os.getuid()}"` (getuid
  is only reached on POSIX because the fcntl import gates everything;
  document this in the docstring). Created with `parents=True,
  exist_ok=True`.
- Lock file per value: `sha256(normalized.encode())[:16] + ".lock"`, opened
  `O_CREAT | O_RDWR`, `flock(LOCK_EX | LOCK_NB)`. On success, truncate and
  write `f"{os.getpid()} {normalized}\n"` (advisory diagnostics), keep the
  fd. Lockfiles are deliberately never unlinked (unlink-while-locked races
  a concurrent opener onto a dead inode; the runtime dir is per-user tmpfs).
- On `BlockingIOError`/`OSError` from flock: release everything acquired so
  far, read the holder line (best effort, empty on any error), raise
  `SystemExit(f"device {value!r} is already claimed by another inspect-robots"
  f" process{holder_suffix}: two evals must not drive one rig")` where
  `holder_suffix` is `f" (PID {pid})"` when the line parsed.
- Degradations (each a covered branch): `import fcntl` fails → return an
  empty claim silently; lock dir uncreatable or a lock file unopenable
  (OSError from mkdir/open) → one `warnings.warn(RuntimeWarning)` naming the
  path, return an empty claim (never block an eval on environment trouble);
  no slots or no matching kvs → empty claim.
- `release()`: `flock(LOCK_UN)` + close each fd, tolerate OSError per fd,
  idempotent (second call no-op).

- [ ] **Step 1: Write the failing tests**

`tests/test_claims.py`, driving everything through a tmp_path
`XDG_RUNTIME_DIR` env dict and slots built with
`DeviceSlot(arg="left_channel", kind="can", label="left arm CAN")` etc.:

- `test_claim_then_conflict_reports_holder_pid`: claim `can0`; a second
  `claim_devices` on the same env/slots/kvs raises SystemExit whose message
  contains `can0` and `f"PID {os.getpid()}"`. (flock conflicts across two
  fds in one process, so no subprocess is needed.)
- `test_release_frees_the_device`: claim, release, claim again succeeds;
  second release call is a no-op.
- `test_conflict_releases_partial_acquisitions`: two slots, first value
  free, second value pre-claimed → SystemExit, AND the first value is
  claimable afterwards (the partial claim was rolled back).
- `test_symlink_spellings_collide`: a v4l2 slot claimed via a tmp_path
  symlink and a second claim via the target path conflict (normalization).
- `test_can_names_taken_verbatim`: `can0` and `can1` coexist.
- `test_none_and_missing_and_nonstring_values_skipped`: kvs with `None`,
  absent key, and an `int` value yield an empty claim (no lock files
  created).
- `test_duplicate_values_claimed_once`: two slots naming one value → one
  lock file, release works.
- `test_without_fcntl_is_a_noop`: `monkeypatch.setitem(sys.modules,
  "fcntl", None)` → empty claim, no lock dir created, and a concurrent
  "claim" also succeeds.
- `test_unusable_lock_dir_warns_and_noops`: point `XDG_RUNTIME_DIR` at a
  path whose parent is a FILE → `pytest.warns(RuntimeWarning)`, empty
  claim, eval not blocked.
- `test_gettempdir_fallback_used_without_runtime_dir`: env without
  `XDG_RUNTIME_DIR` → lock file appears under
  `tempfile.gettempdir()/inspect-robots-<uid>/...` (monkeypatch
  `tempfile.gettempdir` to tmp_path to stay hermetic).

Mark any test that monkeypatches attributes ON the real fcntl module with
`_needs_fcntl`-style skipif; plain flock-using tests need no mark on the
blocking CI (Linux/macOS) but MUST be skipped on Windows — add a
module-level `pytestmark = pytest.mark.skipif(sys.platform == "win32",
reason="fcntl is POSIX-only")` EXCEPT for `test_without_fcntl_is_a_noop`,
which must run everywhere; structure the file so that test sits outside the
marked class or carries no mark (e.g. put the POSIX tests in a
`@pytest.mark.skipif`-decorated class and leave the no-fcntl test at module
level).

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_claims.py -v` — ModuleNotFoundError.

- [ ] **Step 3: Implement**

Follow the contract above. Structure hint: a module-level
`_lock_dir(env) -> Path | None`, `_normalize(slot_kind, value) -> str`, and
`claim_devices` building a `DeviceClaim(fds: list[int])`. Keep the fcntl
import at the top of `claim_devices` so the no-op arc is one early return.
Use `os.open`/`os.write`/`os.close` (fd-based, matching flock) rather than
file objects. Module docstring states the advisory nature and the flock
lifetime guarantee (kernel releases on process death; stale locks cannot
exist).

- [ ] **Step 4: Run tests, then the full gate set**

`uv run pytest tests/test_claims.py -v`, then
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/_claims.py tests/test_claims.py
git commit -m "claims: advisory flock guard over device-slot values (#281)"
```

---

### Task 2: acquire in `_resolve_components`, release in the CLI finallys

**Files:**
- Modify: `src/inspect_robots/cli.py`
- Test: `tests/test_registry_cli.py`

- [ ] **Step 1: Write the failing tests**

Harness: register a fake embodiment whose factory carries `DEVICE_SLOTS =
(DeviceSlot(arg="channel", kind="can", label="bus"),)` and accepts
`channel` as a constructor kwarg (mirror the existing fake-registry idiom
in `tests/test_registry_cli.py` — find the `_register`/fixture pattern the
run-path tests use). Point `XDG_RUNTIME_DIR` at tmp_path via monkeypatch
(the CLI passes `os.environ`).

- `test_run_claims_and_releases_device`: pre-claim `channel=can9` with
  `claim_devices` directly, run `main(["run", ...])` with `-E channel=can9`
  → SystemExit message contains `can9` and "already claimed"; then release
  the pre-claim, run again → eval executes (exit 0) AND after main returns
  the value is claimable again (release happened in the finally).
- `test_run_without_device_slots_untouched`: the standard sim/mock
  embodiment path constructs no claim (no lock dir appears under tmp_path).
- `test_claim_released_when_embodiment_construction_fails`: fake factory
  raises TypeError on construction → `main` exits via the existing
  `_resolve_or_exit` SystemExit, and the device is claimable afterwards
  (the claim rolled back on construction failure).
- `test_eval_set_claims_once_for_the_set`: eval-set over two tasks with the
  slotted embodiment claims before and releases after (claimable again
  post-run; use the pre-claim trick to assert the conflict message also
  fires for eval-set).
- Windows: these tests use real flock — give the file/section the same
  skipif treatment as Task 1 (module already imports `_claims`; a plain
  platform skipif on these specific tests is fine).

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/test_registry_cli.py -k claim -v` — the pre-claimed
run currently succeeds (no guard), so the conflict assertions fail.

- [ ] **Step 3: Implement**

- `_ResolvedComponents` gains `claim: DeviceClaim` (import `_claims` at the
  top of cli.py; it is stdlib-only and cheap).
- In `_resolve_components`, after the policy resolves and `embodiment_kvs`
  is built (:1225), and BEFORE the embodiment constructs (:1227-1233):

```python
    factories = registered("embodiment")
    slots = (
        device_slots(factories[embodiment_name]) if embodiment_name in factories else ()
    )
    claim = claim_devices(slots, embodiment_kvs, os.environ)
    try:
        embodiment = _resolve_or_exit("embodiment", embodiment_name, ..., **embodiment_kvs)
    except BaseException:
        claim.release()
        raise
```

  (`BaseException` because `_resolve_or_exit` raises SystemExit; a comment
  says so. `registered` and `device_slots` are already imported or need
  imports — check the top of cli.py.)
- `_cmd_run` finally (:1390-1396) becomes `finally: embodiment.close();
  resolved.claim.release()` — close first, release second, and the release
  must run even if `close()` raises (nested try/finally). Same in
  `_cmd_eval_set` (:1518-1522).
- Do NOT touch `_cmd_doctor` or `eval.py` (out of scope below).

- [ ] **Step 4: Run tests, then the full gate set**

`uv run pytest tests/test_registry_cli.py -v && uv run pytest tests/test_claims.py -v`,
then all four gates. Every pre-existing run/eval-set test must pass
untouched (sim embodiments declare no slots, so the guard is invisible to
them).

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/cli.py tests/test_registry_cli.py
git commit -m "cli: claim device slots before constructing hardware embodiments (#281)"
```

---

### Task 3: docs

**Files:**
- Modify: `docs/guide/adapters.md` ("Declare device slots" section, :35-83)
- Modify: `docs/guide/cli.md` (a short note near the run reference)
- Modify: `CHANGELOG.md` (`### Added`, newest on top)
- Modify: `src/inspect_robots/CLAUDE.md` (new `_claims.py` row; extend the
  "Key invariants" list with the claim-release-mirrors-close point)

- [ ] **Step 1: Guide edits**

adapters.md: one paragraph after the slot declaration example: declared
slots also feed a run-time advisory claim; when two evals name the same
device the second fails at startup instead of double-driving the hardware;
the claim is flock-based, per-user, and vanishes with the process. cli.md:
one sentence in the run section with the conflict error's shape. No em
dashes in prose.

- [ ] **Step 2: CHANGELOG + module map + gates + commit**

CHANGELOG `### Added` bullet ([plan 0045](plans/0045-hardware-claim-guard.md),
#281). Module map row for `_claims.py` at matching density; invariants list
gains "the CLI releases device claims in the same finally that closes the
embodiment". `uv run pytest -q` green, then:

```bash
git add docs/guide/adapters.md docs/guide/cli.md CHANGELOG.md src/inspect_robots/CLAUDE.md
git commit -m "docs: advisory device claims for multi-rig hosts (#281)"
```

---

## Out of scope

- Guarding the programmatic `eval()` path: callers passing a built
  embodiment own its lifecycle; a claim inside `eval()` would only cover
  the registry-name path and would acquire/release per task under
  `eval_set`. The CLI is where concurrent-rig collisions actually happen.
- `_cmd_doctor`: short-lived diagnostic; claiming there would make "why is
  my eval failing" spot checks impossible during a run. Revisit if it bites.
- Cross-host claims (locks are per-host by nature; two hosts cannot share a
  CAN bus anyway).
- An opt-out flag: flock cannot go stale (kernel-released), so the only
  reason to bypass is deliberately driving one rig from two processes,
  which is exactly what the guard exists to stop.
- Claiming RealSense depth serials (`*_depth_serial` args are not
  device-slot declared today; the capture child claims per-serial at the
  librealsense layer already).
