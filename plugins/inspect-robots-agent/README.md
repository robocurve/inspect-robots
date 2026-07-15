# inspect-robots-agent

LLM agent policy for [Inspect Robots](https://github.com/robocurve/inspect-robots):
frontier LLMs (Claude, GPT, anything behind an OpenAI-compatible API) drive any
registered embodiment through tool calls, as a first-class `Policy` named
`agent`. The same policy runs ad-hoc instructions and scores on registered
tasks next to fine-tuned VLAs.

## Install

```bash
pip install inspect-robots inspect-robots-agent
```

## Quickstart (no hardware)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

inspect-robots "pick up the cube" --policy agent \
    -P model=anthropic/claude-fable-5 --embodiment cubepick
```

Model strings are OpenRouter-style `provider/model`, resolved from
`-P model=...` or `$INSPECT_ROBOTS_MODEL`. API keys come from the environment:

1. `-P base_url=...` (with `-P api_key_env=NAME`): any OpenAI-compatible endpoint
2. `anthropic/*` model with `ANTHROPIC_API_KEY`: the Anthropic compat endpoint
3. `openai/*` model with `OPENAI_API_KEY`: OpenAI
4. `OPENROUTER_API_KEY`: OpenRouter, any model string

## How it works

Motion tool calls state where to go, not how long to move. For absolute modes,
the move tool (`move_joints` for joint spaces, `move_to` for Cartesian pose
modes) interpolates named partial targets from the observed state at a fixed
safe speed. The default `max_speed_frac=0.1` allows a tenth of each
dimension's range per second, subject to a 5%-of-range per-step ceiling that
matches the core's default delta backstop. At that default a near-full-range
move exceeds the 10 s per-call playout cap, so the agent receives a
split-the-move error and issues it as two smaller motions; raise the fraction
(up to `0.5` before the ceiling binds at 10 Hz) for faster arms. The tool
result reports the computed step count and, when the embodiment declares
`control_hz`, the corresponding playout time. `duration_s` is not part of either motion tool.

For displacement modes, `move_by` splits the requested total so every action
fits the box side in that direction. The action box is the embodiment author's
per-step speed statement, so `max_speed_frac` does not apply to displacement
modes. `done` and `give_up` end the trial through the core's policy-stop
channel.

When `control_hz` is `None`, the plugin uses a 10 Hz fallback to compute step
counts and the per-call playout cap, but leaves the emitted chunk rate unset.
The embodiment then plays the chunk at its native rate. In this case the speed
and playout caps are step-count constructs, not wall-clock guarantees, and the
tool result does not report seconds.

When the embodiment publishes operating notes via `EmbodimentInfo.docs`
(joint layout, sign conventions, gripper polarity), the policy appends them
to the system prompt as an `Embodiment notes:` section. The per-step
observation also labels the proprioceptive state vector with the action
dimension names (`left_j0=0.01 ...`) whenever the mapping is unambiguous.

Every action still passes the CLI's default safety approvers (bounds clamp plus
per-step delta limit); the plugin contains no safety-critical code path of its
own. An explicit `--max-action-delta` tighter than 5% of range can truncate
absolute interpolants. In displacement modes, a value tighter than the action
box can truncate each `move_by` step. Either setting can make the executed
motion fall short of the tool's requested total.

> [!WARNING]
> Guardrails are on by default at the CLI. **Never pass `--disable-guardrails`
> on real hardware** unless you fully trust the policy and the rig.

Configuration knobs (all `-P key=value`): `model`, `base_url`, `api_key_env`,
`max_llm_calls` (default `100`), `temperature`, `effort`, `max_speed_frac`.
The speed fraction defaults to `0.1` and applies only to absolute modes.

`LLMAgentPolicy.transcript()` returns the current conversation as a deep copy with streamed camera frames replaced by omission markers, ready for core eval-log persistence.

Reasoning effort defaults to `low`: robot control is latency-sensitive (the
arm stands still while the model thinks), safety guardrails sit below the
model either way, and frontier models at low effort remain strong at this
task shape. Raise it for hard manipulation problems (`-P effort=high`) or
pass `-P effort=none` to omit the parameter for endpoints that reject it
(the CLI reads a bare `none` as null). To send the literal wire value
`none` and disable reasoning, quote it: `-P effort="'none'"`. GPT-5.x on
chat completions requires the literal `none` when function tools are in
play (any other value, or omitting the field, is a 400). In Python,
`effort=None` omits the field and `effort="none"` sends the wire value.
