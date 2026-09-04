# 0082: Collision-proof run identity and final log names

Closes #355. Branch: `fix/collision-proof-run-identity`.

## Problem

Two eval processes may share a `log_dir`. Their current names include an
eight-hex UUID suffix, but neither name is claimed before use:

- `_run_eval()` constructs a run stamp in memory, then `FrameStore` creates
  the corresponding `frames/<stamp>` directory with `exist_ok=True`. A rare
  collision therefore lets both runs write frames, action sidecars, and
  policy artifacts under one identity.
- `JsonLogSink.on_eval_end()` writes through a predictable
  `<final>.json.tmp` path and publishes with `os.replace()`. A rare final-name
  collision therefore overwrites the first complete log, while concurrent
  writers to the same temp path can corrupt the file before publication.

The MP4 encoder also omits ffmpeg's `+faststart` layout flag, so browsers must
download the complete rendered video before finding metadata stored at its
end. These are the surviving independent changes from closed PR #272 and plan
0039, re-planned against current `eval.py`, default-on action sidecars, live
logs, and the composite report encoder.

## Design

### Run-stamp claim

Add a private `_claim_run_stamp(log_dir) -> tuple[str, Path, bool]` in
`eval.py`. It atomically creates `log_dir` when absent and reports whether it
owns that directory, computes the timestamp prefix once, then attempts up to
16 short stamps:

1. draw a UUID and use its first eight hex characters;
2. atomically claim `log_dir/frames/<timestamp>_<suffix>` with
   `Path.mkdir(parents=True, exist_ok=False)`;
3. retry only `FileExistsError`; propagate every other filesystem failure.

After 16 short collisions, make one final claim with the complete UUID hex.
If that claim also collides, propagate `FileExistsError`; an existing run is
never joined or overwritten.

`eval()` owns the claim around `_run_eval()` and passes the claimed stamp into
it instead of allowing `_run_eval()` to generate another value. This places
the claim before every current consumer: `FrameStore`, policy
`on_trial_start`/`on_trial_end` hooks, and `actions/<stamp>` sidecars.
`FrameStore` may call `mkdir(..., exist_ok=True)` on the already claimed
directory, but never chooses its identity.

When `store_frames=False`, `eval()` attempts to remove the claimed stamp
directory in the same `finally` that protects all `_run_eval()` exits, then
attempts to remove the now-empty `frames/` parent. Both removals suppress
`OSError`: a policy or concurrent process may have placed an artifact under
that namespace, or another run may still own a sibling claim. When
`store_frames=True`, the directory remains even if the run stored no camera
frames, preserving the existing `EvalStats.frames_dir` contract. Embodiment
ownership cleanup remains the outer `finally`, so a claim or cleanup failure
cannot leak a registry-resolved hardware connection. If the eval created
`log_dir` and all output directories are empty after cleanup, it also removes
that owned directory; a directory that predated the eval is never removed.

### Final JSON-log claim

Keep the existing complete-temp-file publication contract in
`JsonLogSink.on_eval_end()`, but make both names collision-safe:

- Each attempt gets a unique hidden temp path in `log_dir`,
  `.<task-slug>_<first-eight-final-id>.<full-temp-uuid>.tmp`, opened
  exclusively. Using eight final-ID characters in this internal name also
  keeps the full-UUID fallback attempt under filesystem component limits. The
  file is serialized, flushed, and fsynced before publication.
- Publish with `os.link(temp, final)`. A same-directory hard link is an atomic
  no-clobber claim: `FileExistsError` leaves the existing final log untouched.
- On a final-name collision, remove the temp file, redraw the final suffix,
  and retry. Use 16 short UUID suffixes, then one complete UUID suffix. A
  collision on the complete UUID propagates.
- If `os.link` fails with another `OSError` because the filesystem does not
  support the operation, fall back to the prior atomic `os.replace` behavior.
  This preserves portability and no-half-written logs, although that fallback
  cannot provide no-clobber semantics on such filesystems.
- A `finally` removes the per-attempt temp path after successful linking,
  collision, serialization failure, or publication failure. `path` is set
  only after a final file has been published.

The task slug cap remains 200 bytes. With a real 32-hex UUID, the hidden temp
filename is at most 247 bytes and the complete-UUID final filename is 238
bytes, both within the intended 255-byte component limit. The full-UUID test
uses a 300-character task name so this bound is exercised, not just computed.

### Video layout

Add `-movflags +faststart` to `_ffmpeg_argv()`. Both `inspect-robots video`
and the composite HTML-report encoder use this one argv builder, so filesystem
and temporary in-memory MP4s receive the same browser-friendly layout.

## Tests

`tests/test_eval_orchestration.py`:

- simulate one `mkdir` collision, assert the short suffix is redrawn, and
  assert the action-sidecar pointer/header use the successfully claimed stamp;
- simulate 16 short collisions and assert the seventeenth claim uses a full
  UUID;
- assert a non-frame run removes its empty stamp and `frames/` parent after
  success and cancellation, and removes `log_dir` only when the eval created
  that directory and it remains empty;
- assert a frame-storing run keeps its claimed directory;
- assert non-frame cleanup preserves a claimed directory that gained a policy
  artifact, and tolerates parent removal while a sibling run directory exists.

`tests/test_strict_json.py`:

- simulate one final-name collision and assert both final and temp UUIDs are
  redrawn, the existing file is not overwritten, and no temp remains;
- simulate 16 short collisions and assert the final attempt uses a full UUID;
- assert a collision on the full UUID propagates, leaves `path` unset, and
  leaks no temp file;
- make hard links unavailable and assert the `os.replace` fallback publishes
  a complete log;
- make serialization and fallback publication fail and assert temp files are
  still removed.

`tests/test_video.py` pins `-movflags +faststart` in the shared ffmpeg argv.

Run the repository gates: `ruff check .`, `ruff format --check .`, strict
`mypy`, and the full pytest suite with 100% line and branch coverage.

## Docs

- Add a `CHANGELOG.md` entry under `## [Unreleased]` / `### Fixed`, scoped to
  Core and linking this plan and #355.
- Update `src/inspect_robots/CLAUDE.md` module-map rows for `eval.py`,
  `logging/`, and `_video.py` so future changes retain the claim and faststart
  invariants.

No public API or schema changes, so `__all__`, the API snapshot, and generated
reference docs stay unchanged.

## Out of scope

- Changing the eight-hex normal-case filename or run-stamp format.
- Adding a run ID to `EvalSpec` or changing action/frame sidecar schemas.
- Collision hardening for `LiveLogSink` or `RerunSink`; their artifacts are
  observational and are not the canonical completed eval log addressed by
  #355.
- Replacing the portable `os.replace` fallback with platform-specific
  no-clobber rename syscalls.
