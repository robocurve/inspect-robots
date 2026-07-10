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

Each LLM tool call becomes one smooth, open-loop action chunk: named partial
joint targets are interpolated at the embodiment's control rate
(`move_joints`), displacements are split across steps (`move_by`), and
`done`/`give_up` end the trial through the core's policy-stop channel. Every
action still passes the CLI's default safety approvers (bounds clamp plus
per-step delta limit); the plugin contains no safety-critical code path of
its own.

> [!WARNING]
> Guardrails are on by default at the CLI. **Never pass `--disable-guardrails`
> on real hardware** unless you fully trust the policy and the rig.

Configuration knobs (all `-P key=value`): `model`, `base_url`, `api_key_env`,
`max_llm_calls`, `temperature`, `effort`.

Reasoning effort defaults to `low`: robot control is latency-sensitive (the
arm stands still while the model thinks), safety guardrails sit below the
model either way, and frontier models at low effort remain strong at this
task shape. Raise it for hard manipulation problems (`-P effort=high`) or
pass `-P effort=none` to omit the parameter for endpoints that reject it.
