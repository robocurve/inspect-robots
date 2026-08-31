# Command-line interface

The `inspect_robots` CLI wraps the registry and [`eval`](/api/#inspect_robots.eval.eval).
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

If another eval holds a declared device, startup exits with
`device 'can0' is already claimed by another inspect-robots process (PID 123):
two evals must not drive one rig`.

The sugar only fires when the first argument contains whitespace, so a
mistyped subcommand (`inspect-robots isnpect`) errors out instead of starting
a rollout; a single-word instruction needs the explicit
`run --instruction "wipe"` form.

### Default policy and embodiment

Resolved in order (first hit wins):

1. explicit flags: `--policy` / `--embodiment`
2. environment: `INSPECT_ROBOTS_POLICY` / `INSPECT_ROBOTS_EMBODIMENT`
3. the user config file `~/.config/inspect-robots/config.ini`
   (`$XDG_CONFIG_HOME` is honored).

#### Several rigs on one host

The config file itself is selected in this order: `--config PATH`,
`$INSPECT_ROBOTS_CONFIG`, then the path derived from `XDG_CONFIG_HOME` or
`HOME`. Use a separate file for each rig without changing the config home for
the whole process:

```bash
inspect-robots setup --config ~/.config/inspect-robots/rig-b.ini
inspect-robots run --task my-benchmark --config ~/.config/inspect-robots/rig-b.ini
```

The selected file uses INI:

```ini
[defaults]
policy = molmoact2
embodiment = yam_arms     ; the default: real hardware
sim_embodiment = my-sim   ; what --sim swaps in (any registered sim embodiment)
scorer = operator         ; optional, ad-hoc runs only
max_steps = 300           ; optional, ad-hoc runs only
store_frames = true       ; optional, capture frames on every run

[policy.args]          ; default -P key=value pairs
server_url = http://gpu-box:8202

[embodiment.args]      ; default -E key=value pairs
top_cam_device = /dev/v4l/by-id/usb-CAM123-video-index0

[sim_embodiment.args]  ; -E pairs used only under --sim
headless = true

[taskgen.args]         ; default -A pairs for --auto-task (no owner; see below)
model = gpt-5.2
base_url = https://api.openai.com/v1
api_key_env = OPENAI_API_KEY
```

Values parse like `-P/-E` args (bool/int/float/None/str), `~` expands in
`[*.args]` values, and an explicit `-P/-E key=value` flag overrides the
same-named config key. An `[*.args]` section belongs to the component named
in `[defaults]`: it applies whenever that same component is the one selected
(by default, by flag, or by env var), and is ignored with a stderr note when
a *different* component is selected. Your YAM rig's `rest_pose` never reaches
`--embodiment kitchen`. The one exception is `[taskgen.args]`: automatic
task generation is a single fixed function rather than a component named in
`[defaults]`, so the section has no owner and applies to every `--auto-task`
run. There is deliberately no project-local config file.
Because `.env` values load into the environment, a directory's `.env` can pin
`INSPECT_ROBOTS_CONFIG` for everything run there. Treat a checked-in `.env`
that selects hardware with the same suspicion as a checked-in config: either
can choose which policy drives your hardware. Use an absolute path in `.env`:
the value is literal, so `~` is not expanded and a config write would create a
real `~` directory.

### Running in simulation: `--sim`

Real hardware is the default (it is whatever you configured as `embodiment`).
`--sim` swaps in your configured sim counterpart for one invocation:

```bash
inspect-robots "place the spoon on the plate" --sim
inspect-robots run --task my-benchmark --policy molmoact2 --sim
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

### Operator grading

Judgement capture is a grader component, selected per run: `--grader NAME`
overrides the `grader` config key, and with neither set an attended run
(interactive terminal, no `--no-prompt`) uses the builtin `operator` grader
while an unattended run grades nothing. `--grader none` disables grading
outright. The grader runs after every scored trial, whatever the task or
scorer, so a trial the policy ended with its `done()` or `give_up()` tool is
graded like any other.

An arbitrary instruction has no success oracle, so ad-hoc runs also default to
the `operator` scorer, which reads the captured judgement. The operator grader
asks after each trial unless the embodiment already terminated the episode
with a definitive `success` or `failure` verdict. In that case, it records the
embodiment's verdict as the operator judgement instead of asking the operator
a second time, and prints `operator verdict adopted from embodiment: success`
(or `failure`) so the operator can catch a mistaken adoption live. A trial
that ended early for any other reason is announced first, for example
`note: this trial ended early ('done')`.

```text
did the robot succeed? [y/n/partial/skip] (partial scores as failure) n
grader notes (Enter for none): gripper closed early, cube still in frame
```

Prompted verdicts are recorded in the log. The CLI then asks for one optional
line of grader notes. Bare Enter or whitespace-only input records no note. An
adopted embodiment verdict is not followed by a notes prompt, so a self-scoring
embodiment still costs no keypresses per trial.
`skip` records no judgement, but a grader note entered for that trial is still
recorded. Notes never affect the score. Piped/CI stdin or `--no-prompt` never
prompt: without a TTY no grader is selected by default, and `--no-prompt`
suppresses the operator grader specifically (combining it with an explicit
`--grader operator` is an error, and a config-set `grader = operator` is
downgraded with a stderr note whenever the run cannot actually be attended).
A custom grader named in config or `--grader` runs regardless of TTY-ness,
which is what the builtin `vlm` autograder relies on. An unjudged trial
honestly scores as failure with "no operator judgement recorded".

### Automated grading: `--grader vlm`

The builtin `vlm` grader replaces the human verdict with a vision model. It
sends the trial's first and last camera frames, the task instruction, and a
rubric to an OpenAI-compatible chat endpoint, then records the model's
success or failure verdict exactly where the operator grader would have
written one, so the `operator` scorer reads it unchanged:

```bash
export ANTHROPIC_API_KEY=...
inspect-robots "stack the red cube on the blue cube" \
  --grader vlm -G model=claude-sonnet-5
```

Grader arguments ride the repeatable `-G k=v` flag (the grader counterpart of
`-P`): `model` (required), `rubric` (inline text) or `rubric_file` (a path,
mutually exclusive with `rubric`), `base_url` (default
`https://api.anthropic.com/v1`), `api_key_env` (default `ANTHROPIC_API_KEY`),
`max_cameras` (frames per phase, default 4), and `effort` (sent to the
endpoint as `reasoning_effort`: leave it out for the provider default;
`effort=none` requests the minimum, it does not mean unset; a value the
endpoint rejects leaves trials ungraded with a stderr note, like any grader
wire failure). Without a rubric the grader
uses a strict default: success only if the frames show the instruction
completed, failure when the outcome is ambiguous or not visible. A scene that
carries its own rubric at `scene.metadata["rubric"]` (what `--auto-task`
generates) wins over all of these for that trial. `-G` without any selected
grader is an error rather than a silent no-op.

The log records what actually graded the run. `EvalSpec.grader` holds the
grader's name and `EvalSpec.grader_config` holds its effective configuration,
so a saved log says which model judged, against which rubric, at which effort.
The values are the resolved ones: a `rubric_file` is recorded as the text that
was read from it, an omitted rubric as the default that replaced it, and
`effort` as the value sent on the wire (`null` when the field was omitted and
the provider default applied, `"none"` when the minimum was requested). The
rubric recorded there is the run-level one, since a scene carrying its own is
already persisted with that scene. The API key is never recorded.

Both pieces persist in config, and the args section is owned by the grader it
was written for (the same rule as `[policy.args]`):

```ini
[defaults]
grader = vlm

[grader.args]
model = claude-sonnet-5
rubric_file = ~/rigs/stacking-rubric.md
```

Configuration problems (a missing model or API key, an unreadable rubric
file) stop the run before the robot moves. After a rollout the grader never
crashes the run: transport failures or an unparseable reply leave the trial
ungraded with a stderr note. A trial the embodiment already terminated with a
definitive `success` or `failure`, or one the operator already judged from
the console, is adopted without spending a model call. The log records which
path produced each verdict in `judgement_sources`.

### Live operator feedback

On an attended run, an opted-in policy can also receive feedback while the
episode is running. The CLI prints the operator console usage hint when this
channel is active. Type a normal line and press Enter to deliver it at the
policy's next inference. Esc ends the episode, and so does `/stop`; trailing
text after `/stop` is recorded to the log before the episode ends. In the
plain fallback without the footer, stdin is line-buffered, so a lone Esc needs
an Enter after it; `/stop` works the same everywhere. `/y`, `/n`,
or `/p` plus an optional note ends it and records the verdict immediately,
without a second post-trial prompt. Bare Enter never ends the run: it prints
the usage reminder instead, since Enter is also the key you press right after
typing feedback. Cmd+Enter is not offered because terminal emulators do not
forward the Cmd modifier to stdin, so it is indistinguishable from plain
Enter. Feedback is saved per trial and appears in summaries and HTML reports.
Piped stdin and `--no-prompt` disable the channel.

On an attended run with the console enabled and a real POSIX TTY, the session
renders a two-row footer. The timer and controls repaint in place above a stable
`> ` line that the session owns, so ticker updates never tear the operator's
typing. The framework appends `Esc ends the episode` to the footer status line
and replaces any older trailing end-gesture clause supplied by an embodiment.
The hint
is dropped only when the line is width-clipped, so embodiment status text never
goes stale when the framework gesture changes:

```text
  [sent] you might wanna move the right arm out of the way
  t = 61s / 120s | Esc ends the episode
  > is there anything I can hand you█
```

After Enter, feedback moves into scrollback as `[sent] ...`. End-only rows use
`[noted] ...`, and text that ends the episode (such as a `/stop` note) is
confirmed as `[noted]` even in a sent-labeled session, because the policy never
receives it. Keystroke echo runs on its own background cadence, so typing shows up
immediately even while the policy is still thinking. Third-party prints can
smudge one frame; the next repaint heals it. Off-TTY, Windows, and piped-stdin
rendering is unchanged.

Install `inspect-robots-voice` and pass `--voice` to add local microphone
transcription to an attended run. Repeat `-V key=value` for voice settings such
as the model, microphone device, language, and compute type. Voice input is
feedback-only, so episode end and verdicts remain keyboard actions. See
[Voice operator input](voice-mode.md) for setup and filtering details.

The plugin also provides `run --speak` for local narration of streamed policy
notes and terminal summaries. Repeat `-S key=value` to select the speech mode,
output voice, speed, volume, device, language, or offline model paths. Speaking
works without a TTY. See [Speaking policy notes](voice-mode.md#speaking-policy-notes---speak)
for model setup and the microphone echo caveat.

A session-aware embodiment offers `connect_operator_session(session)`. On
POSIX, the CLI calls that hook once before evaluation and the session becomes
the only owner of terminal input and status output. The console stays active
for every policy because it must own the end-of-episode input. A policy that
accepts messages gets the full feedback usage line. Other policies get the
end-only mode:

```text
operator console: Esc (or /stop [note]) ends the episode; /y /n /p [note] records a verdict; typed notes are saved to the log
```

Without that hook, the compatibility path preserves the previous gating. A
policy that does not accept operator messages leaves the console off silently.
For an accepting policy, a simulator enables the console directly and a
real-hardware embodiment can enable it with the supported legacy
`defer_operator_end()` hook. Older hardware keeps its existing keypress
behavior, prints a notice, and leaves feedback typing off. Windows cannot poll
stdin with this console, so both paths print the Windows notice there: a
session-aware embodiment is never connected, regardless of policy, and the
legacy path prints it for an accepting policy.

## `inspect-robots setup`

The interactive first-run wizard: it prompts for each `[defaults]` key with
a suggested value (Enter accepts, typing overrides), warns when a chosen
policy or embodiment is not registered in the current environment, offers
the `agent` policy's on-demand camera mode (`images = on_demand`), and then
helps assign camera devices. It lists every color-capable camera that udev
names under `/dev/v4l`, preferring
`/dev/v4l/by-id` names and falling back to port-stable `/dev/v4l/by-path`
names when a by-id link is missing or when multiple physical cameras can claim
the same by-id identity. This includes cameras with duplicate serials and
same-model cameras with missing serials. Multi-interface cameras such as the
RealSense D435 can lose udev's name race between their depth and RGB interfaces.

Answer `u` and unplug the camera when asked to identify the physical USB
device that disappeared, including cameras the by-id listing cannot name. The
wizard chooses the stored path after replug because udev reassigns links on
every plug. Answer `p` to switch the listing to port names.

When the selected registered embodiment declares device slots, those slots
drive one device interview for cameras, CAN interfaces, and serial devices.
CAN slots list SocketCAN interfaces and support unplug-to-identify; rigs with
multiple USB adapters named `can0`, `can1`, and so on also receive a udev
pinning suggestion so replug order cannot swap physical devices. The suggestion
pins by adapter serial when serials are present and distinct, and otherwise
emits USB-port `KERNELS` rules. The fallback is needed for rigs with several
identical `gs_usb` adapters such as Innomaker's, because every unit reports
`SN0001`. A port-pinned name stays valid only while the adapter keeps the same
physical USB port.

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
rerun_save = true         # save the live stream as a replayable .rrd (default true)
rerun_port = 9877         # viewer port for this rig (default 9876)
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
inspect-robots list operator_inputs
```

## `inspect-robots run`

Resolve a task/policy/embodiment from the registry and run an eval. Pass
constructor arguments with `-T` (task), `-P` (policy), and `-E` (embodiment) as
`key=value` (parsed as bool/int/float/None/str):

```bash
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick
inspect-robots run --task cubepick-reach -T num_scenes=10 --policy scripted -P chunk_size=8 \
             --embodiment cubepick --log-dir logs --seed 0
inspect-robots run --task my-task --policy agent --embodiment my-robot \
             --voice -V model=small -V device="USB Microphone"
inspect-robots run --task my-task --policy agent --embodiment my-robot \
             --speak -S voice=af_sarah -S volume=0.8
```

`--epochs N` overrides the task's epoch count, `--fail-on-error X` halts on
`PolicyError`s (`1` = first error, `0<X<1` = proportion, `X>1` = count), and
`--store-frames` streams camera frames to a per-run subdirectory of
`<log-dir>/frames` (trial ids repeat across runs, so each run gets its own
directory; the log's `stats.frames_dir` records the exact path). A
`store_frames = true` config default enables capture on every run;
`--no-store-frames` disables it for one invocation. When the run finishes,
the path of the written log is printed.

`--rerun` and `--rerun-connect` also save the viewed stream as a `.rrd` under
`<log-dir>` by default. Replay it with `rerun <file>`. Pass `--no-rerun-save`,
or set `rerun_save = false`, to keep the live stream only. Explicit
`--rerun-save` without an active viewer records to a `.rrd` only. This flag is
run-only; `eval-set` does not add Rerun sinks.

`--policy`/`--embodiment` may be omitted when defaults are configured (see
the zero-config section above); `--instruction "..."` replaces `--task` to
run a single ad-hoc scene. `--voice` adds local spoken feedback on attended
runs; repeat `-V key=value` to configure the installed voice plugin. `--speak`
narrates streamed policy notes on attended or unattended runs; repeat `-S key=value`
to select the speech mode, output voice, speed, volume, device, language, or offline model paths.

The exit code is `0` on a successful eval, `1` otherwise. When trials errored,
the summary shows the count (`trials: 4 (2 errored)`) and lists each errored
scene; a run in which every trial errored reports `run status: error` and exits `1`.

### Automatic task generation: `--auto-task`

Use `--auto-task` in place of `--task` or `--instruction` to have a
vision-capable model inspect the embodiment's initial camera frames and write
both the task and its grading rubric:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
inspect-robots run --auto-task -A model=claude-sonnet-4-5 \
             --policy agent --embodiment my-robot
```

The command prints the generated instruction and rubric before rollout. The
instruction is sent to the policy, and the rubric is shown to the operator
grader when a verdict is needed. Both are persisted in the eval log.

Repeat `-A key=value` to pass generator arguments. Common arguments are
`model`, `instructions`, `instructions_file`, `base_url`, `api_key_env`,
`max_cameras`, `scene_id`, and `effort` (sent as `reasoning_effort`: leave
it out for the provider default; `effort=none` requests the minimum, it
does not mean unset; an invalid value fails before rollout). Values use the
same bool/int/float/None/string parsing as the component argument flags:

```bash
inspect-robots run --auto-task \
             -A model=vision-model \
             -A instructions_file=task-designer.txt \
             -A max_cameras=2 \
             --policy agent --embodiment my-robot
```

Generator arguments persist in the `[taskgen.args]` config section, which
applies to every `--auto-task` run (it has no owner; see the config-file
section above). An explicit `-A key=value` overrides the same-named config
key. Set the seed with `--seed`, never a `[taskgen.args] seed` key: the seed
must reach generation and evaluation together, and a persisted `seed` key
exits every auto run with a guided error.

Exactly one of `--task`, `--instruction`, and `--auto-task` is required.
`-A` requires `--auto-task`, and `-T` cannot be combined with it. Automatic
tasks use the same `--max-steps` and `--scorer` defaults as instruction runs.
`--epochs N` repeats the generated scene without regenerating the task. Keep
the default integer seed, or pass an explicit `--seed N`; task generation and
evaluation use the same value so the initial peek matches epoch zero.

## `inspect-robots eval-set`

Run several registered tasks against one resolved policy/embodiment pair in a
single invocation — the CLI counterpart of
[`eval_set`](/api/#inspect_robots.eval.eval_set). Task names are matched exactly, or
by shell-quoted `fnmatch` glob (entry-point discovery namespaces tasks as
`<benchmark>/<key>`, so a benchmark name is a ready-made prefix):

```bash
inspect-robots eval-set 'kitchenbench/*' --policy xpolicylab -P url=ws://host:19000 \
             --embodiment yam_arms
inspect-robots eval-set cubepick-reach my-other-task --policy scripted --embodiment cubepick
```

For a task declared with `max_seconds`, its summary row includes both the
physical-time budget and the integer step limit resolved from the selected
embodiment, for example `[120s -> 1200 steps at 10 Hz]`.

Multiple patterns may match the same task; it still runs once. A pattern that
matches nothing is an error listing every registered task. `--policy` and
`--embodiment` (and `-P`/`-E`, `--sim`, `--epochs`, `--fail-on-error`,
`--store-frames`, `--disable-guardrails`, `--max-action-delta`, `--voice`, and
`-V`) apply exactly
as they do for `run`, to every matched task — there is no per-task `-T` in
this release. The embodiment is resolved once for the whole set, not once per
task, so a real robot is not reconnected between tasks.

`--speak` and `-S` are run-only options and are not accepted by `eval-set`.

Rather than one full summary per task, the CLI prints the resolved
policy/embodiment, one status line for the whole set, a compact `[status]
task_name  metrics-or-error` row per task, and the shared log directory once
(`eval_set` still writes one `EvalLog` per task inside it). The exit code is
`0` iff every task's log has `status == "success"`. A task that raises before
producing a log contributes an in-memory error row and the remaining tasks
still run. A `SafetyAbort` or `EmbodimentFault` that escapes `eval()` (raised
outside a trial) and `KeyboardInterrupt` still propagate. A halt inside a
trial ends that task with an error log and the set continues to the next task.

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

Print a summary of a saved [`EvalLog`](/api/#inspect_robots.log.EvalLog):

```bash
inspect-robots inspect logs/cubepick-reach_xxxx.json
```

```text
task:        cubepick-reach
policy:      scripted
embodiment:  cubepick
run status:  completed
outcome:     5 succeeded
horizon:     120s -> 1200 steps at 10 Hz
scenes:      5   trials: 5
metrics:
  success_at_end: 1
scenes:
  [success] scene-0: success_at_end=1
  ...
```

The `horizon` line appears for seconds-based tasks; step-only logs retain the
existing output. The HTML viewer likewise separates declared seconds from the
resolved step limit.

`completed` is the display form of the log's `success` status value; the
on-disk field and Python API keep `success`.

For runs whose policy recorded conversations (such as `--policy agent`),
`--transcript` appends each trial's recorded transcript after the summary:

```bash
inspect-robots inspect logs/cubepick-reach_xxxx.json --transcript
```

## `inspect-robots summarize`

Distill a saved [`EvalLog`](/api/#inspect_robots.log.EvalLog) into a markdown
learnings file:

```bash
inspect-robots summarize logs/cubepick-reach_xxxx.json
```

Without `--model`, the command works offline and writes a deterministic digest
of run identity, trial outcomes, operator feedback, errors, and transcript
statistics. The default output is
`logs/learnings/cubepick-reach_xxxx.md`. Use `-o FILE` to select another path
or `-o -` to write only the document to stdout.

With `--model`, the digest and the tail of each recorded policy transcript are
sent to an OpenAI-compatible chat-completions endpoint:

```bash
inspect-robots summarize logs/cubepick-reach_xxxx.json \
  --model claude-sonnet-4-5
```

The default endpoint is `https://api.anthropic.com/v1`, and the default API key
variable is `ANTHROPIC_API_KEY`. Override them with `--base-url URL` and
`--api-key-env VAR` for another compatible provider.

### Retry with learning

The learnings file exists to be fed back in. The
[`agent`](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent)
and
[`capx`](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-capx)
policies accept a `prior_learnings` path and append the file's text to the
system prompt after any embodiment notes, framed as hints that may be stale
(the current observation always wins):

```bash
inspect-robots summarize logs/cubepick-reach_xxxx.json --model claude-sonnet-4-5
inspect-robots "place the fork on the plate" --policy agent \
    -P prior_learnings=logs/learnings/cubepick-reach_xxxx.md
```

The policy reads the file once when it is constructed and records its resolved
path and content hash in the log's policy configuration, so runs that saw
prior notes are never mistaken for cold-start runs when comparing results. Any
hand-written markdown file works in place of a generated one. Validation
details and limits live in the plugin READMEs linked above.

## `inspect-robots view`

Render a saved [`EvalLog`](/api/#inspect_robots.log.EvalLog) as a self-contained HTML
report:

```bash
inspect-robots view logs/cubepick-reach_xxxx.json
```

The report puts the run status, configuration, metrics, scene results, and
recorded policy conversations on one page. Chat transcripts are grouped into
observation turns. A turn's default view shows its step, camera frames,
structured operator feedback, assistant prose, agent-note headlines, and
readable tool argument chips. Scene cards in auto-task logs also include a
collapsed rubric dropdown. A collapsed Raw transcript section preserves the
raw observation, state dumps, calls, and tool results. Non-chat transcripts
remain available as bounded JSON.

For runs captured with `--store-frames`, the report embeds the stored camera
frames at the exact observation turns where the model saw them. When ffmpeg is
available, a completed report rendered outside `--serve` also embeds one
side-by-side composite MP4 above each trial transcript at the recorded control
rate. Its caption names the cameras in left-to-right order, and one playhead
keeps every view aligned. Otherwise the player provides per-camera tabs, play
or pause, and step scrubbing over the existing frame images as a lightweight
flipbook. The file contains its stylesheet and media inline, so it has no
network dependency.

On composite-video pages, the transcript is also a timeline rail: the active
turn highlights as the video plays, clicking a turn's step header seeks to its step, and the
opt-in Follow button keeps the active turn in view.

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

Embedded MP4 data has a separate 30 MB per-page budget. A trial composite that
exceeds it falls back to the per-camera flipbook, and later trials skip their
composite encode. `--no-video` skips MP4 encoding without removing frames or
the flipbook. Served and running pages also use the flipbook so the two-second
live render loop never waits for ffmpeg. In directory mode these suppressed
pages remain upgradeable: the next eligible plain `view` pass re-renders them
with the composite MP4. Reports created before this behavior need `--force`
once to gain embedded video.

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
