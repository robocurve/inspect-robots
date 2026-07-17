# Command-line interface

The `inspect_robots` CLI wraps the registry and [`eval`][inspect_robots.eval.eval].
The command is installed as `inspect-robots`, with `inspect-robot` as an alias
for the common typo; both run the same CLI.

## Zero-config: `inspect-robots "<instruction>"`

Once you have configured a default policy and embodiment (run
`inspect-robots setup`, or see below), giving the robot a command is a
one-liner:

```bash
inspect-robots "place the spoon on the plate"
```

This runs a single ad-hoc scene built from that language instruction on
your default policy/embodiment: sugar for
`inspect-robots run --instruction "..."`. The resolved components and where
they came from are printed before the robot moves. Two flags exist only for
instruction runs: `--max-steps N` (horizon, default 300) and `--scorer NAME`
(default `operator`).

The sugar only fires when the first argument contains whitespace, so a
mistyped subcommand (`inspect-robots isnpect`) errors out instead of starting
a rollout; a single-word instruction needs the explicit
`run --instruction "wipe"` form.

### Default policy and embodiment

Resolved in order (first hit wins):

1. explicit flags: `--policy` / `--embodiment`
2. environment: `INSPECT_ROBOTS_POLICY` / `INSPECT_ROBOTS_EMBODIMENT`
3. the user config file `~/.config/inspect-robots/config.ini`
   (`$XDG_CONFIG_HOME` is honored):

```ini
[defaults]
policy = molmoact2-yam
embodiment = yam-bimanual            ; the default: real hardware
sim_embodiment = yam-bimanual-isaac  ; what --sim swaps in
scorer = operator      ; optional, ad-hoc runs only
max_steps = 300        ; optional, ad-hoc runs only
store_frames = true    ; optional, capture frames on every run

[policy.args]          ; default -P key=value pairs
checkpoint = ~/ckpts/molmoact2-yam.pt

[embodiment.args]      ; default -E key=value pairs
cameras = wrist,front

[sim_embodiment.args]  ; -E pairs used only under --sim
headless = true
```

Values parse like `-P/-E` args (bool/int/float/None/str), `~` expands in
`[*.args]` values, and an explicit `-P/-E key=value` flag overrides the
same-named config key. An `[*.args]` section belongs to the component named
in `[defaults]`: it applies whenever that same component is the one selected
(by default, by flag, or by env var), and is ignored with a stderr note when
a *different* component is selected. Your YAM rig's `rest_pose` never reaches
`--embodiment kitchen`. There is deliberately no project-local config file:
a checked-in file choosing which policy drives your hardware would be a trust
footgun.

### Running in simulation: `--sim`

Real hardware is the default (it is whatever you configured as `embodiment`).
`--sim` swaps in your configured sim counterpart for one invocation:

```bash
inspect-robots "place the spoon on the plate" --sim
inspect-robots run --task my-benchmark --policy molmoact2-yam --sim
```

The sim embodiment resolves as `$INSPECT_ROBOTS_SIM_EMBODIMENT` > config
`sim_embodiment`, with constructor args from `[sim_embodiment.args]` only:
real-rig args (`[embodiment.args]`: serial ports, camera IDs) never leak into
a sim run, and vice versa. `--sim` together with an explicit `--embodiment`
is an error (they both pick the embodiment); an exported
`$INSPECT_ROBOTS_EMBODIMENT` is simply not consulted under `--sim`: it's a
persistent default for real runs, not a per-invocation intent. The mapping is
explicit configuration: the framework never guesses which sim matches your
robot.

### Operator scoring

An arbitrary instruction has no success oracle, so ad-hoc runs default to the
`operator` scorer. When run on an interactive terminal, the CLI asks after each
trial unless the embodiment already terminated the episode with a definitive
`success` or `failure` verdict. In that case, the CLI records the embodiment's
verdict as the operator judgement instead of asking the operator a second time,
and prints `operator verdict adopted from embodiment: success` (or `failure`) so
the operator can catch a mistaken adoption live.

```text
did the robot succeed? [y/n/partial/skip] (partial scores as failure)
```

Prompted verdicts are recorded in the log (`skip` records nothing). Piped/CI
stdin, `--no-prompt`, or a registered `--task` run never prompt or adopt an
embodiment verdict: unattended runs stay non-blocking, and an unjudged trial
honestly scores as failure with "no operator judgement recorded".

## `inspect-robots setup`

The interactive first-run wizard: it prompts for each `[defaults]` key with
a suggested value (Enter accepts, typing overrides), warns when a chosen
policy or embodiment is not registered in the current environment, and then
helps assign camera devices by listing `/dev/v4l/by-id`. If you do not know
which physical camera a device path belongs to, answer `u` and unplug that
camera when asked: the wizard rescans and identifies it from the entry that
disappeared. When identical cameras without serial numbers collide in the
by-id listing, `p` switches to `/dev/v4l/by-path` names, which are stable
per physical USB port.

When the selected registered embodiment declares device slots, those slots
drive one device interview for cameras, CAN interfaces, and serial devices.
CAN slots list SocketCAN interfaces and support unplug-to-identify; rigs with
multiple USB adapters named `can0`, `can1`, and so on also receive a udev
pinning suggestion so replug order cannot swap physical devices.

```bash
inspect-robots setup
```

The result is written to `~/.config/inspect-robots/config.ini`
(`$XDG_CONFIG_HOME` honored); an existing file is backed up to
`config.ini.bak` first, and settings the wizard does not manage (such as
`[policy.args]` or `sim_embodiment`) are carried through unchanged. Note
that later `inspect-robots config set` edits drop comments from the file.
The setup command requires an interactive terminal; for scripted
configuration use `inspect-robots config set`.
After writing the config, setup lists missing runtime requirements declared by
the selected registered policy and embodiment, together with their remediation
commands.

Prefer to write the file yourself? This is the wizard's output for a YAM
rig; replace the three camera paths with your rig's V4L2 color nodes
(stable `/dev/v4l/by-id/...` or udev-symlink paths):

```bash
mkdir -p ~/.config/inspect-robots && cat > ~/.config/inspect-robots/config.ini <<'EOF'
[defaults]
policy = molmoact2        # from the inspect-robots-yam plugin
embodiment = yam_arms     # same plugin; cameras configured below
scorer = success_at_end
max_steps = 1200          # 120 s at 10 Hz
rerun = true              # live viewer of cameras/state/actions each run
store_frames = true       # save each run's camera frames under logs/frames/

[embodiment.args]
top_cam_device = /dev/v4l/by-id/YOUR-TOP-CAM
left_cam_device = /dev/v4l/by-id/YOUR-LEFT-CAM
right_cam_device = /dev/v4l/by-id/YOUR-RIGHT-CAM
EOF
```

## `inspect-robots list`

Show registered components (builtins + installed plugins):

```bash
inspect-robots list                 # all kinds
inspect-robots list policies        # just one kind
inspect-robots list embodiments
```

## `inspect-robots run`

Resolve a task/policy/embodiment from the registry and run an eval. Pass
constructor arguments with `-T` (task), `-P` (policy), and `-E` (embodiment) as
`key=value` (parsed as bool/int/float/None/str):

```bash
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick
inspect-robots run --task cubepick-reach -T num_scenes=10 --policy scripted -P chunk_size=8 \
             --embodiment cubepick --log-dir logs --seed 0
```

`--epochs N` overrides the task's epoch count, `--fail-on-error X` halts on
`PolicyError`s (`1` = first error, `0<X<1` = proportion, `X>1` = count), and
`--store-frames` streams camera frames to a per-run subdirectory of
`<log-dir>/frames` (trial ids repeat across runs, so each run gets its own
directory; the log's `stats.frames_dir` records the exact path). A
`store_frames = true` config default enables capture on every run;
`--no-store-frames` disables it for one invocation. When the run finishes,
the path of the written log is printed.

`--policy`/`--embodiment` may be omitted when defaults are configured (see
the zero-config section above); `--instruction "..."` replaces `--task` to
run a single ad-hoc scene.

The exit code is `0` on a successful eval, `1` otherwise. When trials errored,
the summary shows the count (`trials: 4 (2 errored)`) and lists each errored
scene; a run in which every trial errored reports `run status: error` and exits `1`.

## `inspect-robots eval-set`

Run several registered tasks against one resolved policy/embodiment pair in a
single invocation — the CLI counterpart of
[`eval_set`][inspect_robots.eval.eval_set]. Task names are matched exactly, or
by shell-quoted `fnmatch` glob (entry-point discovery namespaces tasks as
`<benchmark>/<key>`, so a benchmark name is a ready-made prefix):

```bash
inspect-robots eval-set 'kitchenbench/*' --policy xpolicylab -P url=ws://host:19000 \
             --embodiment yam_arms
inspect-robots eval-set cubepick-reach my-other-task --policy scripted --embodiment cubepick
```

Multiple patterns may match the same task; it still runs once. A pattern that
matches nothing is an error listing every registered task. `--policy` and
`--embodiment` (and `-P`/`-E`, `--sim`, `--epochs`, `--fail-on-error`,
`--store-frames`, `--disable-guardrails`, `--max-action-delta`) apply exactly
as they do for `run`, to every matched task — there is no per-task `-T` in
this release. The embodiment is resolved once for the whole set, not once per
task, so a real robot is not reconnected between tasks.

Rather than one full summary per task, the CLI prints the resolved
policy/embodiment, one status line for the whole set, a compact `[status]
task_name  metrics-or-error` row per task, and the shared log directory once
(`eval_set` still writes one `EvalLog` per task inside it). The exit code is
`0` iff every task's log has `status == "success"`.

`--retry-attempts` is accepted and threaded through to `eval_set()`, whose
resumption-of-a-partial-run behavior is reserved for a follow-up: passing it
today does not yet skip already-finished scenes. `--rerun`'s live viewer
is not offered for `eval-set`: streaming several back-to-back tasks into one
viewer window is a separate design question from running the set at all.

## `inspect-robots doctor`

`doctor` reports a registered embodiment's missing declared runtime modules
before constructing it, then checks its spaces for adapter conformance.

```bash
inspect-robots doctor --embodiment my_arms
```

## `inspect-robots inspect`

Print a summary of a saved [`EvalLog`][inspect_robots.log.EvalLog]:

```bash
inspect-robots inspect logs/cubepick-reach_xxxx.json
```

```text
task:        cubepick-reach
policy:      scripted
embodiment:  cubepick
run status:  completed
outcome:     5 succeeded
scenes:      5   trials: 5
metrics:
  success_at_end: 1
scenes:
  [success] scene-0: success_at_end=1
  ...
```

`completed` is the display form of the log's `success` status value; the
on-disk field and Python API keep `success`.

## `inspect-robots view`

Render a saved [`EvalLog`][inspect_robots.log.EvalLog] as a self-contained HTML
report:

```bash
inspect-robots view logs/cubepick-reach_xxxx.json
```

The report puts the run status, configuration, metrics, scene results, and
recorded policy conversations on one page. Agent notes from tool calls are
highlighted above their call lines. For runs captured with `--store-frames`,
the report also embeds the stored camera frames at the exact observation turns
where the model saw them. The file contains its stylesheet and frame data
inline and uses native browser controls to collapse transcripts, so it has no
network or JavaScript dependency.

By default, `view` replaces the log path's suffix with `.html` and prints the
written path. Use `-o REPORT.html` to choose another file, `-o -` to write only
the HTML document to stdout, or `--open` to launch the written file in the
default browser. Missing output directories are created. The command returns
0 whenever it produces the report, even when the evaluation recorded a failed
or cancelled run.

Frame embedding is on by default when the log's frame directory can be found.
Use `--no-frames` to keep the transcript placeholders, or
`--frames-budget MB` to change the default 50 MB inline-frame payload limit.
`--frames-budget 0` removes the limit. Inlined frames make the HTML document
larger, so use a smaller budget or `--no-frames` when page size matters.

## `inspect-robots video`

Render a `--store-frames` run's stored camera frames into one MP4 per
(trial, camera) stream:

```bash
inspect-robots video logs/adhoc_xxxx.json
```

```text
fps: 10 (control_hz from log)
wrote logs/frames/20260715_184213/scene-0-e0_left_cam.mp4 (1200 frames)
wrote logs/frames/20260715_184213/scene-0-e0_right_cam.mp4 (1200 frames)
wrote 2/2 streams
```

Encoding is done by the `ffmpeg` binary (no Python dependencies are added);
install it from your package manager, or point at a specific build with
`--ffmpeg PATH`. Videos land in the frames directory by default (`--out DIR`
overrides). The playback rate defaults to the log's `control_hz` and can be
overridden with `--fps N`. A stream that fails to encode is reported on
stderr and the remaining streams still encode; the exit code is 1 if any
stream failed.

## `inspect-robots --version`

```bash
inspect-robots --version
```
