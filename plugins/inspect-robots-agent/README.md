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
    -P model=anthropic/claude-fable-5 -P effort=low --embodiment cubepick
```

Model strings are OpenRouter-style `provider/model`, resolved from
`-P model=...` or `$INSPECT_ROBOTS_MODEL`. API keys come from the environment:

1. `-P base_url=...` (with `-P api_key_env=NAME`): any OpenAI-compatible endpoint
2. A known provider prefix with that provider's key set: the provider's own
   endpoint, prefix stripped from the model id
3. `OPENROUTER_API_KEY`: OpenRouter, any model string. Ids ending in a known
   OpenRouter variant suffix (`:free`, `:nitro`, `:floor`, `:extended`,
   `:online`, `:thinking`) always route here, since the variant means nothing
   to a provider's own API; other colons (fine-tune ids like `openai/ft:...`)
   still resolve directly.

Providers resolved directly by prefix:

| Prefix | Key | Endpoint |
|---|---|---|
| `anthropic/*` | `ANTHROPIC_API_KEY` | Anthropic (OpenAI-compat, or native with `-P wire=anthropic`) |
| `openai/*` | `OPENAI_API_KEY` | OpenAI |
| `google/*` | `GEMINI_API_KEY` | Google Gemini (OpenAI-compat) |
| `x-ai/*` or `xai/*` | `XAI_API_KEY` | xAI |
| `groq/*` | `GROQ_API_KEY` | Groq (rest of the id passed through, slashes and all) |
| `mistralai/*` | `MISTRAL_API_KEY` | Mistral |
| `deepseek/*` | `DEEPSEEK_API_KEY` | DeepSeek |

The wire format defaults to Chat Completions for broad OpenAI-compatible
endpoint support:

| `-P wire=` | Endpoint | Use it when |
|---|---|---|
| `chat` (default) | `/chat/completions` | Anything OpenAI-compatible: OpenRouter, vLLM, Ollama, the Anthropic and Gemini compat endpoints |
| `responses` | `/responses` | A direct OpenAI or compatible endpoint requires the Responses API |
| `anthropic` | `/messages` | Driving Claude natively, which is what fast mode needs |

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

Every move tool call also requires a `note` with one or two plain sentences
describing the current observation and why the agent chose that motion. The
user reads these notes live and in the saved transcript to follow what the
agent sees and decides.

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
`wire`, `speed`, `max_output_tokens`, `max_llm_calls` (default `100`),
`temperature`, `effort`, `max_speed_frac`, `transcript_echo`.
`speed` and `max_output_tokens` apply to `-P wire=anthropic` only, and passing
either on another wire is an error rather than a silent no-op.
Set `-P transcript_echo=true` to print live `[agent]` conversation lines to
stderr, including goals, observation summaries, assistant output, tool calls,
and tool results.
Move notes appear inside the echoed tool-call arguments.
The speed fraction defaults to `0.1` and applies only to absolute modes.

`LLMAgentPolicy.transcript()` returns the current conversation as a deep copy with streamed camera frames replaced by omission markers, ready for core eval-log persistence.
Camera labels such as `camera 'top_cam' (step 480):` provide the join key from a transcript observation to its stored frame.
Live Rerun transcript streaming happens automatically when a Rerun sink is attached.

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

## Fast mode on Claude

`-P wire=anthropic` drives Claude through the native Messages API instead of
the OpenAI-compat endpoint. That is the only way to reach fast mode, which
serves the same model at up to 2.5x higher output tokens per second:

```bash
inspect-robots "pick up the cube" --policy agent \
    -P model=anthropic/claude-opus-5 -P wire=anthropic -P speed=fast \
    --embodiment cubepick
```

The model id keeps the `anthropic/` prefix on this wire, the same as every
other model string here. Only Anthropic's own endpoint serves `/v1/messages`,
so anything that resolves elsewhere is refused up front with the fix named: a
bare `-P model=claude-opus-5`, another provider's prefix such as `openai/`, or
an OpenRouter `:variant` suffix. Pass `-P base_url=...` to point at a gateway
that serves the endpoint yourself.

> [!NOTE]
> With `-P base_url=...` and no `-P api_key_env=`, this wire sends
> `$ANTHROPIC_API_KEY` to that host. The other wires default to
> `$OPENROUTER_API_KEY` instead. Name the variable explicitly
> (`-P api_key_env=MYGW_KEY`) when the gateway takes its own credential, and
> point it at an unset variable to send no key at all.

Fast mode costs roughly double the standard price on both input and output
(see [Anthropic's pricing](https://www.anthropic.com/pricing)), and it draws on
a rate limit separate from standard capacity, so a fast-mode run can hit a 429
while standard quota sits idle. It is available on Claude Opus 5 and Opus 4.8,
on the Claude API only: not Bedrock, Vertex, Foundry, or Claude Platform on
AWS. A rejection that names fast mode is turned into an error naming the fix.

This wire always requests adaptive thinking, which pre-4.6 models such as
Sonnet 4.5 and Haiku 4.5 do not support. Use `-P wire=chat` for those.

The Messages API requires an output cap, so `-P max_output_tokens=` defaults to
`16000` here. Thinking bills against that same cap, and a response truncated at
the limit is an error naming the knob rather than a silently missing tool call.
Keep `-P effort=` at `high` or below on this wire: `xhigh` and `max` want a cap
of 64000 or more, which needs streaming this client does not implement yet.
The read timeout scales with the cap and tops out at 600 s per attempt, so a
large cap plus retries can sit for several minutes before failing.

## Reasoning effort on OpenAI models

Recent OpenAI reasoning models can reject function tools on the Chat
Completions wire with an error like this:

```text
Function tools with reasoning_effort are not supported. To use function
tools, use /v1/responses or set reasoning_effort to 'none'.
```

This is a Chat Completions API restriction, not an inspect-robots bug. Use a
direct OpenAI endpoint and select the Responses wire to keep reasoning enabled:

```bash
inspect-robots "pick up the cube" --policy agent \
    -P model=openai/gpt-5.6-sol -P wire=responses -P effort=medium \
    --embodiment cubepick
```
