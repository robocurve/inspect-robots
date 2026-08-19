# Actuate conference demo

A live eval show for the Actuate conference. Four LLMs rotate through the
three roles of an automatic-task eval, a browser screen shows who is who, and
a leaderboard accumulates across evals. This lives on the
`demo/actuate-conference` branch only (draft PR #397) and is never merged:
run it from the branch, iterate freely.

## How it works

Every eval independently draws a roster model for each role from that role's
eligible pool: GPT, Opus, and Gemini can hold any role, Kimi K3 competes as
test-taker only. The same model can hold two or three roles, that is intended
and part of the show:

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
- **Model IDs:** confirm the four IDs at the top of `_roster.py` with their
  providers. Validated live so far: `claude-opus-5` (tasker and test-taker),
  `gpt-5.6-sol` (tasker and grader, test-taker over the Responses wire).
  `gemini-3.7-flash` has not been drawn yet; force one eval with it in each
  role during setup. Kimi K3 is test-taker only and is served via OpenRouter
  as `moonshotai/kimi-k3`. `OPENROUTER_API_KEY` must be added to the demo
  `.env` by copying the existing key from `~/robocurve/test-dir/.env`. It has
  not yet been validated live, so force one eval with it as test-taker
  during setup.
- **Fresh leaderboard:** eval numbering and scores persist in
  `examples/actuate/state/` and survive restarts. To start the show clean at
  eval 1: `rm examples/actuate/state/status.json
  examples/actuate/state/results.jsonl`.
- An eval that errors (bad model ID, rejected request) shows as a score-less
  card and costs nothing else; fix `_roster.py` and start the next eval.
- If every camera and CAN interface vanishes at once, the USB host controller
  died (seen live on 2026-08-17: kernel log "xHCI host controller not
  responding, assume dead"). Recover without a reboot, using the controller's
  PCI address from the kernel log:
  `echo 1 | sudo tee /sys/bus/pci/devices/<addr>/remove && sleep 3 && echo 1 | sudo tee /sys/bus/pci/rescan`

## Files

- `run.py`: orchestrator (draws roles, launches evals, records results)
- `serve.py`: stdlib HTTP server for the display page
- `monitor.html`: the combined conference screen
- `_roster.py`: booth-editable models, endpoints, key env vars, accents
- `state/`, `logs/`: gitignored run state and eval logs
