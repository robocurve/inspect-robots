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
`operator` scorer: when run on an interactive terminal, the CLI asks after
each trial,

```text
did the robot succeed? [y/n/partial/skip] (partial scores as failure)
```

and records the verdict in the log (`skip` records nothing). Piped/CI
stdin, `--no-prompt`, or a registered `--task` run never prompt: unattended
runs stay non-blocking, and an unjudged trial honestly scores as failure with
"no operator judgement recorded".

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

```bash
inspect-robots setup
```

The result is written to `~/.config/inspect-robots/config.ini`
(`$XDG_CONFIG_HOME` honored); an existing file is backed up to
`config.ini.bak` first, and settings the wizard does not manage (such as
`[policy.args]` or `sim_embodiment`) are carried through unchanged. Note
that later `inspect-robots config set` edits drop comments from the file.
The command requires an interactive terminal; for scripted configuration
use `inspect-robots config set`.

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

The exit code is `0` on a successful eval, `1` otherwise.

## `inspect-robots inspect`

Print a summary of a saved [`EvalLog`][inspect_robots.log.EvalLog]:

```bash
inspect-robots inspect logs/cubepick-reach_xxxx.json
```

```text
task:        cubepick-reach
policy:      scripted
embodiment:  cubepick
status:      success
scenes:      5   trials: 5
metrics:
  success_at_end: 1
scenes:
  [success] scene-0: success_at_end=1
  ...
```

## `inspect-robots --version`

```bash
inspect-robots --version
```
