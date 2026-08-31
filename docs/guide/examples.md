# Example commands

Copy-paste recipes for the run shapes people ask about most: picking a model,
scaling reasoning effort, driving VLA policies, switching control interfaces,
and varying the instruction. Swap in your own tasks, embodiments, and hosts.
Values chosen during `inspect-robots setup` act as defaults, so the flags shown
here override your config; the [CLI guide](cli.md) documents every flag.

## Choosing a model

LLM policies take the model as a policy arg rather than a top-level flag. The
`agent` policy routes by provider prefix and reads the matching API key from
the environment. `$INSPECT_ROBOTS_MODEL` supplies the model when `-P model` is
omitted.

Claude Fable 5 on the default chat wire:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
inspect-robots "pick up the cube" --policy agent \
    -P model=anthropic/claude-fable-5 --embodiment cubepick
```

Claude Opus 5 on the native messages wire, with fast output enabled:

```bash
inspect-robots run --policy agent --rerun-connect \
    -P model=anthropic/claude-opus-5 -P wire=messages -P speed=fast \
    -P effort=high \
    --instruction "place the fork on the plate"
```

Gemini 3.7 Flash requires the interactions wire:

```bash
export GEMINI_API_KEY=...
inspect-robots "pick up the cube" --policy agent \
    -P model=google/gemini-3.7-flash -P wire=interactions \
    -P effort=low --embodiment cubepick
```

GPT-5.6-sol on the responses wire:

```bash
export OPENAI_API_KEY=...
inspect-robots "pick up the cube" --policy agent \
    -P model=openai/gpt-5.6-sol -P wire=responses --embodiment cubepick
```

Ids without a recognized provider prefix, and ids carrying an OpenRouter
suffix such as `:free` or `:nitro`, route to OpenRouter using
`OPENROUTER_API_KEY`. The full routing table is in the
[agent plugin README](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent).

## Scaling reasoning effort

`-P effort=` accepts the named levels `minimal`, `low`, `medium`, `high`,
`xhigh`, and `max`, the value `none` for the provider's true minimum, or a
fraction such as `0.7` on Tinker's OpenAI-compatible endpoint. Omit the arg to
keep the provider default.

A cheap smoke run against a mock rig:

```bash
inspect-robots "reach the cube" --policy agent \
    -P model=anthropic/claude-fable-5 -P effort=minimal \
    --embodiment cubepick --sim
```

A hard task at high effort:

```bash
inspect-robots run --task kitchenbench/clear-table --policy agent \
    -P model=anthropic/claude-opus-5 -P wire=messages -P effort=high \
    --embodiment yam_arms
```

:::note
The interactions wire accepts only `minimal`, `low`, `medium`, and `high`. On
Anthropic's own endpoint keep effort at `high` or below; `xhigh` and `max`
need a larger token cap plus streaming, which the messages wire does not
implement yet.
:::

Effort is a per-owner arg, so task generation and grading scale
independently of the policy: `-A effort=` for `--auto-task` and `-G effort=`
for the VLM grader.

## Running VLA policies

VLA policies are clients: the model weights live behind a server you start
first, usually on a GPU machine.

MolmoAct 2 (the `molmoact2` policy from the yam plugin):

```bash
# On the GPU machine, from the MolmoAct2 repo. Leave it running, e.g. in tmux:
python examples/yam/host_server_yam.py --host 0.0.0.0 --port 9202
curl http://127.0.0.1:9202/act      # 200 means the server is ready
```

```bash
# On the rig:
inspect-robots "place the fork on the plate" --policy molmoact2 \
    -P server_url=http://gpu-box:9202 --embodiment yam_arms
```

Pi 0 and Pi 0.5 (served from an XPolicyLab checkout, where the checkpoint
argument picks the variant):

```bash
# Terminal 1, from your XPolicyLab checkout:
cd XPolicyLab/policy/Pi_0
bash setup_eval_policy_server.sh <bench_name> <task_name> <ckpt_name> \
    <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_env> 9100 0.0.0.0
```

```bash
# Terminal 2:
inspect-robots run --task my-task --policy xpolicylab --embodiment isaacsim \
    -P url=ws://gpu-box:9100 -P cameras=cam_head:base_rgb \
    -P name=xpolicylab:pi0.5
```

`name` only tags the run in logs; keep it in sync with the checkpoint you
served. For GR00T on a real yam rig, follow the
[GR00T on yam cookbook](../cookbooks/gr00t-on-yam.md).

## Switching control interfaces

A policy and an embodiment must agree on a control mode (`joint_pos`,
`joint_delta`, `eef_delta_pose`, `eef_abs_pose`, and the rest) before anything
moves; compatibility checking rejects mismatches up front. The mode is
declared by the components, so the command line selects it through their args.

Joint-space versus end-effector actions from the same XPolicyLab server:

```bash
inspect-robots run --task my-task --policy xpolicylab --embodiment isaacsim \
    -P url=ws://gpu-box:9100 -P action_type=joint
inspect-robots run --task my-task --policy xpolicylab --embodiment isaacsim \
    -P url=ws://gpu-box:9100 -P action_type=ee
```

A ROS embodiment switching its command interface:

```bash
inspect-robots "reach forward slowly" --policy agent \
    -P model=anthropic/claude-fable-5 \
    --embodiment ros -E command_type=joint_trajectory \
    -E joints=shoulder_pan,shoulder_lift,elbow,wrist_1,wrist_2,wrist_3 \
    -E control_hz=10
```

The `agent` policy adapts its tool surface to whatever mode the embodiment
offers: `move_joints` for joint spaces, `move_to` for absolute Cartesian
poses, `move_by` for displacements. On real hardware, cap speed while you
build trust:

```bash
inspect-robots "stack the red cube on the blue cube" --policy agent \
    -P model=anthropic/claude-fable-5 -P max_speed_frac=0.1 \
    --embodiment yam_arms
```

## Varying the instruction

An argument with spaces is sugar for `run --instruction`:

```bash
inspect-robots "wipe the table with the sponge"
inspect-robots run --instruction "wipe the table with the sponge" --max-steps 500
```

Registered tasks carry their own instructions and scenes; pass constructor
args with `-T`:

```bash
inspect-robots run --task cubepick-reach -T init_seed=3
```

Generate the instruction and rubric with a model instead of writing them:

```bash
inspect-robots run --auto-task \
    -A model=gpt-5.2 -A instructions_file=task-designer.txt \
    --policy agent -P model=anthropic/claude-fable-5
```

Grade against your own rubric, or feed back learnings from earlier runs:

```bash
inspect-robots "stack the red cube on the blue cube" \
    --grader vlm -G model=claude-sonnet-5 -G rubric_file=~/rigs/stacking-rubric.md
inspect-robots summarize logs/2026-08-31T09-00-00-stack-the-red-cube.json
inspect-robots "stack the red cube on the blue cube" --policy agent \
    -P prior_learnings=logs/learnings/2026-08-31T09-00-00-stack-the-red-cube.md
```

## Operator interfaces and unattended runs

The default operator interface is the keyboard console. Voice input and
spoken policy notes are opt-in; `--no-prompt` with a VLM grader removes the
operator entirely:

```bash
inspect-robots "place the fork on the plate" --voice -V language=en
inspect-robots "place the fork on the plate" --speak -S mode=interrupt
inspect-robots "place the fork on the plate" --no-prompt \
    --grader vlm -G model=claude-sonnet-5
```

See [voice mode](voice-mode.md) for the full `-V` and `-S` tables. `--speak`
is a `run`-only flag; `eval-set` rejects it.

## Evaluation sets

`eval-set` runs several registered tasks back to back, with exact names or
shell-quoted globs:

```bash
inspect-robots eval-set 'kitchenbench/*' --policy xpolicylab \
    -P url=ws://gpu-box:9100 --embodiment yam_arms
inspect-robots eval-set cubepick-reach my-other-task \
    --policy scripted --embodiment cubepick
```

A halt raised inside a trial ends that task with an error log and the set
continues with the next task; see the [CLI guide](cli.md) for the full
`eval-set` contract.
