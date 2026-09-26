# Actuate conference demo

A live eval show for the Actuate conference. Seven LLMs compete as
test-takers of an automatic-task eval, a browser screen shows who is who, and
a leaderboard accumulates across evals. This lives on the
`demo/actuate-conference` branch only (draft PR #397) and is never merged:
run it from the branch, iterate freely.

## How it works

Every eval draws each role from that role's eligible pool in `_roster.py`.
The tasker and grader pools have one member each, so those roles are pinned:
GPT-5.6 Sol always authors tasks and Gemini 3.7 Flash always grades. The
test-taker rotates over all seven models, so Sol can be drawn against its
own task and Flash can grade itself, which is intended and part of the show:

- **tasker:** writes the task and rubric from the initial camera frame
  (`--auto-task`, `-A` args)
- **test-taker:** the policy driving the robot (`agent` policy, `-P` args)
- **grader:** judges first and last frames against the rubric (`vlm` grader,
  `-G` args)

All three run at effort high. Each model uses its provider's
OpenAI-compatible endpoint, except that a GPT test-taker speaks the Responses
API (OpenAI rejects function tools with `reasoning_effort` on
`/chat/completions`).

`run.py` draws the roles, launches one `inspect-robots run` per eval, and
records each outcome. `serve.py` renders one dark full-screen page: the
current draw and generated task with the live eval number, a horizontal
leaderboard (mean score over evals where the model was test-taker, top bar
scaled to 80% of the track), and cards for the last three completed evals so
the audience can see every eval re-rolls. Tile that page beside the Rerun
viewer window the run spawns.

## Running it

Needs the rig's existing inspect-robots env: core 0.56.0 or newer (the
`-A`/`-G` effort channel), the `inspect-robots-agent` plugin, and the rig's
embodiment plugin.

```bash
git clone -b demo/actuate-conference https://github.com/robocurve/inspect-robots actuate-demo
cd actuate-demo

# the CLI auto-loads .env from the working directory
printf 'OPENAI_API_KEY=...\nANTHROPIC_API_KEY=...\nGEMINI_API_KEY=...\nOPENROUTER_API_KEY=...\n' > .env

# with the rig's venv active:
./examples/actuate/start.sh

# arguments after the script name reach inspect-robots run verbatim:
./examples/actuate/start.sh --config /path/to/rig-folder/config.ini
```

The launcher starts or reuses the detached display server, then starts the
eval loop in an attached tmux session. These are the equivalent manual
commands:

```bash
tmux new -d -s actuate-serve 'python examples/actuate/serve.py'
tmux new -s actuate 'python examples/actuate/run.py -- --config /path/to/rig-folder/config.ini'
```

The launcher does not forward arguments to `serve.py`. To pass `--port`, use
the manual recipe and add it to the `serve.py` command.

Open http://localhost:8377/ (or `http://<rig-host>:8377/` from another
machine). The loop is autonomous: each eval ends (Esc in the `actuate` tmux
ends an episode early and triggers grading) and the next one starts with a
fresh draw after a short pause (`PAUSE_S` in `run.py`, longer after a failed
eval). Ctrl-C in the tmux stops the whole loop.

Arguments after the launcher name are passed to `inspect-robots run` verbatim.
To watch remotely, add `--rerun-connect rerun+http://<your-machine>:9888/proxy`;
at the booth the rig config's `rerun = true` spawns a local viewer instead.

## Fixed-trials campaign

`run_trials.py` runs every test-taker through exactly ten scored evals, then
prints a per-model summary and exits. Run it manually in the same tmux
arrangement because `start.sh` stays wired to the rotating demo:

```bash
tmux new -d -s actuate-serve 'python examples/actuate/serve.py'
tmux new -s actuate 'python examples/actuate/run_trials.py -- --config /path/to/rig-folder/config.ini'
```

Arguments after `--` reach `inspect-robots run` verbatim, exactly as with
`run.py`. Trials interleave as randomized blocks: every model runs once per
round of the roster, in a fresh random order each round, so drift over the
campaign (motor heat, lighting, scene wear) cannot align with any one model's
slot. Start from a fresh leaderboard for a clean campaign because existing
scored evals in `state/results.jsonl` count toward the ten. The runner resumes
where it left off after a restart and exits once every test-taker has ten scored
evals.

For the dual-rig campaign, each test-taker runs five scored trials on rig-1
and five on rig-2. That is 7 models x 5 trials x 2 rigs = 70 evals. Measuring
every model on both rigs keeps rig differences out of the model comparison.
The rigs run concurrently, so each uses its own `state-rigN/` and `logs-rigN/`
directories. Shared directories would race newest-log attribution and
interleave the two campaigns in one `results.jsonl`.

Start the overnight page alongside the rig halves. Each rig campaign runs in
its own attended terminal so an operator can press Esc to end its current
episode early:

```bash
tmux new -d -s actuate-overnight 'python examples/actuate/overnight.py'
tmux new -s actuate-rig1 'python examples/actuate/run_trials_rig1.py'
tmux new -s actuate-rig2 'python examples/actuate/run_trials_rig2.py'
```

The combined overnight page is available on the tailnet at
`http://<rig-host>:8380/`. For attended watching, the individual monitor pages
remain optional. Serve rig-1 on the default port 8377 and rig-2 on port 8378:

```bash
python examples/actuate/serve.py --state-dir examples/actuate/state-rig1 --logs-dir examples/actuate/logs-rig1
python examples/actuate/serve.py --port 8378 --state-dir examples/actuate/state-rig2 --logs-dir examples/actuate/logs-rig2
```

Both halves run from the clone root and share its single `.env` (the per-rig
`~/robocurve/rig-N/.env` pins are not loaded). Each campaign holds a lock
keyed to its rig config, so a second campaign against the same rig refuses
to start, and the flock claim guard still protects the devices underneath.
The rig scripts pin `--max-steps 1200` (120 seconds at 10 Hz) on both rigs
so the two configs cannot give the halves different episode ceilings. They
also run without live viewer windows and record everything needed for morning
review: a per-eval rrd with camera and joint-state timelines, stored frames,
actions logs, and LLM transcripts. After both halves finish, print the per-rig
breakdowns and pooled leaderboard with:

```bash
python examples/actuate/combine_results.py examples/actuate/state-rig1/results.jsonl examples/actuate/state-rig2/results.jsonl
```

For a fresh dual-rig campaign, clear `state-rig1/`, `state-rig2/`,
`logs-rig1/`, and `logs-rig2/` instead of `state/`. Each half resumes from its
own `results.jsonl` after a restart and stops when every test-taker has five
finite scored evals on that rig.

## Booth checklist

- **Rig config:** `RIG_CONFIG` in `run.py` defaults to
  `~/robocurve/rig-1/config.ini`. Edit it, or override with `-- --config
  <path>` (the later flag wins). Always use the ini in the rig folder, the
  file the setup wizard writes: the XDG-global config drifts stale (it once
  pinned the top camera to the D435's mono IR node, giving a black-and-white
  stream).
- **Motor thermal gate:** the loop checks motor temperatures before every
  eval. At or above 60 C it suspends evals and shows the cooling banner, then
  resumes below 50 C. Thresholds live in `_thermal.py`, and channel names come
  from the rig config. The gate only exists when `run.py` is launched with a
  Python that has i2rt installed, such as the rig venv. On machines without
  i2rt or CAN it disables itself with a console warning and the demo runs as
  before. Reading temperatures clears any latched motor fault codes.
- **Model IDs:** confirm the seven IDs at the top of `_roster.py` with their
  providers. Validated on the rig: `claude-opus-5`, `gpt-5.6-sol`
  (test-taker over the Responses wire, plus its pinned tasker role),
  `gemini-3.7-flash` (test-taker plus its pinned grader role), and
  `gemini-3.1-pro-preview` (2026-08-19 forced eval). `moonshotai/kimi-k3`
  (OpenRouter, needs `OPENROUTER_API_KEY` in the demo `.env`, copy the
  existing key from `~/robocurve/test-dir/.env`) is validated on the rig
  but its per-turn latency depends on OpenRouter's provider routing: 4 to
  90 seconds on a good route, 100 to 230 seconds with one observed
  indefinite wedge on a bad one; the eval watchdog contains the damage. Kimi K2 Thinking was
  removed 2026-08-19: it is text-only, and OpenRouter rejects the
  test-taker's camera frames with "No endpoints found that support image
  input", which would burn the unscored streak and halt the campaign.
- **Fresh leaderboard:** eval numbering and scores persist in
  `examples/actuate/state/` and survive restarts. To start the show clean at
  eval 1: `rm examples/actuate/state/status.json
  examples/actuate/state/results.jsonl`. The dual-rig campaign keeps its state
  in the per-rig dirs and is reset by clearing `state-rig1/`, `state-rig2/`,
  `logs-rig1/`, and `logs-rig2/` instead.
- An eval that errors (bad model ID, rejected request) shows as a score-less
  card and costs nothing else; fix `_roster.py` and start the next eval.
- If every camera and CAN interface vanishes at once, the USB host controller
  died (seen live on 2026-08-17: kernel log "xHCI host controller not
  responding, assume dead"). Recover without a reboot, using the controller's
  PCI address from the kernel log:
  `echo 1 | sudo tee /sys/bus/pci/devices/<addr>/remove && sleep 3 && echo 1 | sudo tee /sys/bus/pci/rescan`

## Files

- `run.py`: orchestrator (draws roles, launches evals, records results)
- `run_trials.py`: deterministic fixed-trials campaign built on run.py
- `run_trials_rig1.py`, `run_trials_rig2.py`: per-rig dual-campaign entry points
- `overnight.py`: stdlib server for the combined overnight campaign view
- `overnight.html`: phone-friendly dual-rig progress page
- `combine_results.py`: per-rig and pooled fixed-trials summaries
- `start.sh`: booth launcher for the display server plus the rotating demo
- `serve.py`: stdlib HTTP server with `--state-dir` and `--logs-dir` overrides
- `monitor.html`: the combined conference screen
- `_roster.py`: booth-editable models, endpoints, key env vars, accents
- `_thermal.py`: motor-temperature probe and thermal-gate thresholds
- `state/`, `logs/`: gitignored run state and eval logs
- `state-rig*/`, `logs-rig*/`: gitignored per-rig campaign state and eval logs
