# 0016 — Render stored frames to video (`inspect-robots video`)

## Problem

`--store-frames` streams every camera frame to disk as raw `.npy` arrays
(`FrameStore`, R5), and the log records the directory in `stats.frames_dir`.
Nothing can play them back: there is no video export, and `inspect-robots
inspect` never even mentions that frames exist. A 2-minute 3-camera run
leaves 3,600 opaque files (~500 MB) with no viewing path.

## Goals

1. `inspect-robots video <log.json>` renders the run's stored frames into one
   MP4 per (trial, camera) stream — a 3-scene, 2-epoch, 3-camera run emits
   18 files — with no new Python dependencies.
2. `inspect-robots inspect <log.json>` surfaces the frames directory and a
   hint pointing at the `video` command, so the feature is discoverable from
   the place people already look.

Non-goals: thumbnails, GIFs, in-terminal playback, Rerun re-ingestion,
transcoding options beyond fps, and deleting/compacting frame directories.

## Design

### Module layout

The encoder and stream-discovery logic (~150 lines) live in a new module
`src/inspect_robots/_video.py` (stdlib + numpy only); `cli.py` keeps just the
thin `_cmd_video` argument wiring. Registration touchpoints in `cli.py`: the
`video` subparser, the dispatch branch in `main()`, and `"video"` added to
`_SUBCOMMANDS` for consistency with every other subcommand (the sugar only
fires on a first token with interior whitespace, so this is hygiene, not a
live bug guard). `cli.py`'s module docstring
subcommand list and `src/inspect_robots/CLAUDE.md`'s module map both gain
the new entry.

### Encoding: the `ffmpeg` binary over a pipe

Core stays NumPy-only, so video encoding cannot pull in imageio/av/opencv.
Instead we shell out to the `ffmpeg` executable and stream raw frames to its
stdin:

```
ffmpeg -hide_banner -nostats -loglevel error -y \
       -f rawvideo -pix_fmt rgb24 -s {W}x{H} -r {fps} -i - \
       -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {out}.mp4
```

The codec is pinned to `-c:v libx264` rather than trusting the build's mp4
default: an LGPL ffmpeg without libx264 silently falls back to mpeg4, which
browsers won't play — pinning turns that into a loud per-stream failure the
stderr-tail machinery already reports well. ffmpeg's stdout is
`subprocess.DEVNULL`, keeping the CLI's results-on-stdout guarantee
independent of ffmpeg's behavior.

- Frames are loaded one at a time (`np.load` per file) and written to the
  pipe, so peak memory is one frame regardless of episode length.
- **stderr goes to a temp file, never a pipe.** ffmpeg writes progress and
  error text to stderr; with `stderr=PIPE` unread while we stream thousands
  of frames into stdin, the ~64 KB pipe buffer fills, ffmpeg blocks on
  stderr, stops draining stdin, and both processes deadlock. A temp file
  (plus `-hide_banner -nostats -loglevel error` to bound its size) sidesteps
  the classic `Popen` deadlock and is trivially readable afterward for error
  reporting — and trivially fakeable in tests. Lifecycle: created with
  `tempfile.NamedTemporaryFile(delete=False)` and passed as the `stderr`
  handle, reopened by name after `wait()` to read the tail (reading only
  after the child exits also guarantees the tail is complete, and stays
  clear of Windows sharing quirks), and unlinked on **every** path,
  success included, after all handles on it — the parent's original handle
  and the reopened read handle — are closed (Windows again).
- Streams are processed in sorted prefix order — `glob` order is
  OS-dependent, and deterministic output keeps tests stable.
- The `pad` filter keeps odd frame dimensions legal for yuv420p.
- `yuv420p` makes the output playable everywhere (QuickTime, browsers).
- If `ffmpeg` is not on `PATH` (`shutil.which`), exit via
  `SystemExit("ffmpeg not found on PATH; install it (e.g. apt install ffmpeg) or pass --ffmpeg PATH")` —
  bare-message style, matching the CLI's existing `SystemExit` errors (no
  `error:` prefix convention exists and this plan does not invent one).

### Failure paths (per stream)

A stream failure never aborts the whole command: the failure is reported,
the remaining streams still encode, and the final exit code is 1 if any
stream failed. Three failure shapes, one cleanup rule:

- **ffmpeg exits nonzero** after a clean stdin close → report the stderr
  tail (last ~5 lines of the temp file).
- **ffmpeg dies mid-stream** (bad output path, disk full): a subsequent
  stdin operation raises `BrokenPipeError`/`OSError`. The write loop **and
  the final `stdin.close()`** live inside the same catch: `proc.stdin` is a
  `BufferedWriter`, so a buffered tail (or a failure while ffmpeg finalizes
  the moov atom) surfaces the broken pipe at close/flush, not at a write —
  and on Windows the same event can arrive as `OSError(EINVAL)`. Catch,
  `wait()`, report the stderr tail — the user gets ffmpeg's actual
  complaint, not a Python traceback (the CLI's established clean-error
  philosophy).
- Trade-off, acknowledged: with stable output names and `-y`, a failed
  re-run truncates-then-unlinks a good video from an earlier run. Accepted:
  a stale output masquerading as current is worse than an absent one, and
  re-running after fixing the reported file recovers it.
- Trade-off, acknowledged: failing a stream on a truncated tail frame
  discards the good frames piped before it, in the very interrupted-run
  scenario the tool serves. The alternative (close stdin early, let ffmpeg
  finalize what it got) was rejected as silent truncation: the failure
  names the offending file, so deleting it and re-running recovers the
  stream deliberately rather than accidentally.

- **A frame is invalid** — `np.load` itself fails (caught as broad
  `Exception` per file: truncated files raise `ValueError`, `EOFError`, or
  `OSError` depending on where the chop landed, and a truncated file from
  an interrupted run is the *expected* corrupt artifact here since frames
  stream to disk mid-episode; object arrays under `allow_pickle=False`
  raise too), or the shape/dtype is unsupported (checked after the
  empty-skip: emptiness makes dtype meaningless, so an empty float64 array
  skips rather than fails), or the shape
  differs from the stream's first frame → we kill ffmpeg ourselves and
  report the offending file name. Closing the buffered stdin of a killed
  process can itself raise `BrokenPipeError`, so this path's
  `stdin.close()` sits inside the same catch as the mid-stream-death one.
- **The pre-spawn probe fails** — a distinct shape: the ffmpeg argv needs
  `-s {W}x{H}`, so frames are loaded and validated *before* `Popen`,
  scanning forward past empty frames until the first usable one. Any
  invalid frame encountered during that scan (frame 0 truncated, or frame
  0 empty and frame 1 truncated) is this shape: there is no process to
  kill, no partial `.mp4`, and no stderr temp file yet (it is created
  immediately before `Popen`) — report the offending file, count the
  stream as failed, move on. Nothing to clean.
- **`Popen` itself raises** (`OSError`: broken wrapper-script shebang,
  `ETXTBSY`, permission oddities). Unlike the per-stream shapes, this
  failure is identical for every stream, so it is a hard bare-message
  `SystemExit` naming the executable — after unlinking the just-created
  stderr temp file, keeping the unlink-on-every-path rule intact.

Cleanup rule: any failed stream's partial `.mp4` is unlinked
(`missing_ok=True` — ffmpeg may die before creating the file at all), in
strict kill/close → `wait()` → unlink order — unlinking a file the child still
holds open fails on Windows. (CI cannot validate that OS behavior — no test
runs a real ffmpeg — so the ordering is a design rule, asserted as call
ordering against the fake.)

Every path ends in an untimed `proc.wait()` — deliberately: after `kill()`
it returns promptly and after a clean stdin close ffmpeg reliably exits;
a wedged output target (stuck network mount) hanging the command is
accepted rather than feeding timeout branches through the coverage gate.

Failure reports and skipped-file warnings print to **stderr**, matching the
CLI's existing convention (results on stdout, notes on stderr), so
`inspect-robots video … | xargs` stays clean; the `wrote …` lines and the
final summary are stdout.

### Stream discovery: group by filename prefix

`FrameStore` writes `{safe(trial_id)}_{safe(camera)}_{t:06d}.npy`. Both the
trial id and the camera name may themselves contain `_`, so splitting a
filename back into (trial, camera) is ambiguous. We never need to: enumerate
**`glob("*.npy")` only** (the default `--out` is the frames dir itself, so
iterating all entries would make run 2 warn about run 1's `.mp4` outputs),
group by everything before the trailing `_NNNNNN.npy` (regex
`^(.+)_(\d{6,})\.npy$`), sort numerically by the step suffix, and emit one
video named `<prefix>.mp4` (e.g. `scene-0-e0_left_cam.mp4`). A `.npy` file
that does not match the pattern is skipped with a warning line. Zero
pattern-matching `.npy` files is `SystemExit("no frames found in <dir>")` —
even when nonmatching `.npy` files were warned about first; the warnings
explain the error rather than reading oddly against an exit-0 empty
summary.

One residual ambiguity is knowingly waived: trial `a` + camera `b_c` and
trial `a_b` + camera `c` share the prefix `a_b_c` and would merge into one
stream with duplicate step suffixes. The `-e{epoch}` suffix on every trial
id makes this contrived in practice, and FrameStore itself already collides
on such names, so the grouping cannot do better than the store.

### Frame shapes

Accept `(H, W, 3)` uint8 — assumed RGB, matching the precedent set by
`RerunSink`, which already renders these images as RGB; the `Observation`
contract itself promises only `(H, W, C)` uint8 with channel order left to
policy-side preprocessing. (Precisely: RerunSink renders *live* observation
images via `rr.Image`, which assumes RGB — frames on disk are the same
arrays, so the RGB assumption is the same one, made earlier in the pipe.) Also accept `(H, W)` and `(H, W, 1)` grayscale,
expanded to 3 channels before piping, and `(H, W, 4)` RGBA with the alpha
channel dropped — the in-repo isaacsim adapter deliberately preserves
4-channel cameras, so RGBA is in-contract data, not corruption. **Empty
arrays are skipped with a warning**, not failed: the same isaacsim adapter
deliberately passes empty arrays through during render warm-up, so an empty
frame at t=0 is expected first-party data and must not kill the stream
(the `-s` probe scans forward to the first non-empty frame, and skipped
frames are simply absent from the piped sequence). Empty-frame skips are
reported **once per stream with a count** ("skipped N empty frames"), not
per frame — a warm-up-heavy adapter could otherwise emit hundreds of
identical lines. Note the timing consequence: cameras that skip different
warm-up counts produce videos whose t=0 differ — acceptable while
composites are out of scope, but worth knowing when eyeballing two videos
side by side. A stream whose frames are *all* empty pipes nothing: it is
reported as failed ("no usable frames") like any other per-stream failure.
Dtype is strictly `uint8` — a deliberate divergence from `FrameRef.load`,
which coerces via `np.asarray(..., dtype=np.uint8)`; coercing 0–1 floats
would produce silently black video, so the video path validates instead of
reusing that helper. Any other shape/dtype fails the
stream with the offending file named; shape consistency within a stream is
checked on the **post-expansion** `(H, W, 3)` shape, so mixing `(H, W)` and
`(H, W, 1)` grayscale (the same pixels) is legal while a resolution change
mid-stream is not (see failure paths).

### fps

Default to the log's `eval.embodiment_info["control_hz"]` using the same
defensive guards as `_print_step_limit_notice` (numeric, not bool, > 0)
**plus finite** — plain `json.load` accepts the `Infinity` literal, and the
sink's sanitizer only protects logs it wrote, so a hand-edited
`control_hz: Infinity` must fall back to 10 rather than become `-r inf`;
otherwise 10. This is the embodiment's nominal rate, not a measured one —
frames are stored once per rollout step, so it is the best proxy the log
offers. (R1's effective rate — chunk → task → embodiment — partially exists
at rollout time already; it just never reaches the log. Recording the
effective rate in `EvalSpec` is the future fix, out of scope here.) `--fps N` overrides, parsed
as **float** (`control_hz` is a float; a 12.5 Hz rig must be expressible)
and validated finite and `> 0` with a bare-message `SystemExit` (`inf > 0`
is true, and an infinite rate reaching ffmpeg fails as a confusing encode
error; `nan` already fails the `> 0` check). The chosen rate and its source
are printed once per run, on stdout (it parallels `run`'s
`policy: X (source)` lines).

### CLI surface

```
inspect-robots video <log.json> [--out DIR] [--fps N] [--ffmpeg PATH]
```

- Reads the log, resolves `stats.frames_dir`. `None` → exit via
  `SystemExit("this log has no stored frames (re-run with --store-frames)")`.
- `frames_dir` is stored as configured at run time (typically relative to
  the run's CWD, e.g. `logs/frames/<run>`). Resolution order: (1) as-is,
  then (2) `log.parent / "frames" / Path(frames_dir).name` — by
  construction the log lives directly in `<log-dir>/` and frames in
  `<log-dir>/frames/<stamp>`, so the log's parent *is* the log dir; this
  holds for any `--log-dir` depth and for absolute paths after a machine
  move (a naive "relative to the log's parent's parent" rule breaks on
  multi-component log dirs like `out/logs`). If neither exists,
  `SystemExit` listing both tried paths.
- `--ffmpeg PATH` bypasses the `shutil.which` lookup and is used verbatim;
  if it is not an existing regular **file** (`isfile`, not just `X_OK` —
  directories pass `os.access(..., os.X_OK)`) or not executable, the same
  bare-message `SystemExit` names the path that was tried.
- `--out DIR` (default: the frames directory itself) is created if needed;
  if the path exists and is not a directory, a bare-message `SystemExit`
  (not a `mkdir` traceback); videos land there as `<prefix>.mp4`.
- Cross-OS note: a log written on Windows stores `frames_dir` with
  backslashes, so the fallback derives the stamp with
  `PureWindowsPath(frames_dir).name` when the string contains `\` (POSIX
  `Path.name` would return the whole string).
- Prints one line per stream: `wrote <out>/<prefix>.mp4 (<n> frames)`,
  where `n` counts **piped** frames (empty skips excluded), and a final
  summary line of the shape `wrote 17/18 streams, 1 failed` (`wrote 18/18
  streams` when clean) — the line the exit code mirrors. Exit 0 only if
  every discovered stream encoded.
- When a failed stream's stderr tail is empty (a SIGKILLed ffmpeg — OOM
  killer — writes nothing), the report falls back to the return/signal
  code so the user never sees a blank-reason failure.

### `inspect` integration

When `stats.frames_dir` is set, print after the `scenes:` count line:

```
frames:      logs/frames/20260715_184213_30ff086f (3600 frames)
hint: render videos with: inspect-robots video logs/adhoc_ff057c79.json
```

- The frame count enumerates pattern-matching `.npy` files in the resolved
  directory — the same enumeration `video` errors on, so the hint gate and
  the error gate cannot disagree over a directory of stray non-matching
  `.npy` files (same two-step resolution too), and the `frames:` line
  prints the **resolved** path — after a machine move the stored string is
  exactly the path that does not work. If the directory does not resolve,
  print `frames:      <dir> (not found from this directory)` and skip the
  hint. If it resolves but holds 0 frames (a camera-less embodiment still
  mkdirs the directory), print the count and skip the hint — pointing at a
  command that would exit "no frames found" helps nobody.
- The hint reuses the existing `_styled(..., _DIM)` hint style.
- Logs without `frames_dir` print nothing new (byte-identical output).

### Run-summary integration

`_print_run_summary` already prints `log:` and `hint: view it with ...`
lines after a run. When the eval stored frames, add the same
`hint: render videos with: inspect-robots video <log>` line there too —
that is the moment the user is looking for "how do I see what happened".
Gated on the same pattern-matching frame count as `inspect` (the directory
is fresh and local at run end): a camera-less `--store-frames` run records a
`frames_dir` but writes no frames, and the hint must not point at a command
that would exit "no frames found".

## Testing (100% coverage, no ffmpeg in CI)

- Unit-test stream grouping, numeric sort, sorted-prefix processing order,
  `.npy`-scoped enumeration (an `.mp4` in the dir is silently ignored, a
  stray `notes.npy` warns), and shape validation as pure functions over a
  tmp_path of tiny synthetic `.npy` files. A truncated `.npy` (write real
  bytes, chop the file) exercises the load-failure path; RGBA input pins
  the alpha drop; a resolvable directory with zero `.npy` files exercises
  the `SystemExit("no frames found in <dir>")` case.
- Encode path: monkeypatch `subprocess.Popen` with a fake that records argv
  and consumes stdin into a buffer; fixture frames are **non-square**
  (e.g. 4×6) so a swapped `-s WxH` cannot pass; assert the rawvideo header
  args (`-s WxH -r fps`, `-loglevel error`), the exact bytes piped for a
  2-frame
  stream, grayscale expansion, and every failure path: nonzero returncode →
  stderr-tail surfaced from the temp file, `BrokenPipeError` raised on
  write and (separately) on `stdin.close()` → wait + stderr tail,
  mismatched shape and unreadable file → kill + wait, and partial-output
  unlink on all of them.
- `shutil.which` monkeypatched for the missing-ffmpeg error; an explicit
  `--ffmpeg` pointing at a nonexistent path exercises the bypass branch's
  `SystemExit`, and one encode test passes `--ffmpeg` at an executable tmp
  stub (chmod +x) with the fake Popen — the validation's happy-path exit is
  a branch the 100% branch-coverage gate requires on its own. A `--ffmpeg`
  pointing at a directory pins the `isfile` requirement. A fake Popen that
  raises `OSError` pins the hard-`SystemExit`-plus-temp-file-cleanup path.
- The default `shutil.which` success branch (encode tests that pass no
  `--ffmpeg` and monkeypatch `shutil.which` to return a fake path — the
  real lookup is environment-dependent and macOS CI, a blocking leg, does
  not guarantee ffmpeg), a non-default `--out DIR`, and empty-array frames
  (skipped mid-stream; all-empty stream reported as failed) each get a
  test. One encode test asserts the full pinned argv — `-c:v libx264`,
  `-pix_fmt yuv420p`, the pad filter — so a refactor cannot silently drop
  the codec-pinning decision.
- Truncated-`.npy` coverage distinguishes **frame 0** (pre-spawn failure:
  no Popen call recorded, nothing to clean) from a **mid-stream** frame
  (kill + wait + unlink) — two distinct branches.
- **Failure isolation gets its own test**: two streams where the first
  fails (truncated frame) and the second succeeds — assert the second's
  `wrote` line on stdout, the first's failure report on stderr, and exit
  code 1. Loop-continue and the exit-code aggregation are behaviors branch
  coverage alone cannot force.
- fps: control_hz present / absent / non-numeric / bool / ≤ 0 / `Infinity`
  (→ default 10), `--fps 12.5` override accepted as float, `--fps 0` and
  negative → `SystemExit`. The non-executable-`--ffmpeg` sub-branch is
  pinned by monkeypatching `os.access` to return False — not by
  `chmod 644`, which is meaningless on Windows (`X_OK` is true for any
  existing file there) and would permanently redden the advisory tier.
  An `--out` colliding with an existing file pins the
  clean-`SystemExit`-not-traceback rule.
- frames_dir resolution: as-is hit, fallback hit (log in `out/logs/`,
  invoked from elsewhere), double miss, and a backslashed
  `logs\frames\stamp` string (hand-written fixture — no Windows needed)
  resolving through the `PureWindowsPath` branch on POSIX.
- `inspect` and run-summary lines via the existing `EvalLog` fixture style
  in `tests/test_registry_cli.py` (frames_dir set / unset / unresolvable /
  resolvable-but-empty, the last asserting the hint is suppressed in both
  places).
- No test invokes a real ffmpeg, so CI needs no new system packages and the
  Windows/py3.10 advisory tier is unaffected.

## Docs

- `docs/guide/cli.md`: new `video` section.
- `docs/guide/logging-and-rerun.md`: connect `--store-frames` to playback.
- README: one line in the CLI overview.
- `src/inspect_robots/CLAUDE.md`: `_video.py` module-map row.

## Out of scope / future

- Side-by-side multi-camera composites and step/instruction overlays.
- `inspect view`-style HTML viewer.
- Frame directory GC (`inspect-robots frames prune`).
