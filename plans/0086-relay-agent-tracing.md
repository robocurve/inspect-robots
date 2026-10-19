# Optional Relay tracing for the LLM agent

## Problem and scope

Wire capture already records provider attempts, retries, timing, errors and
payloads. The missing piece for users of NVIDIA NeMo Relay is a common trace
format connecting an evaluation trial to its logical model calls and actual
tool dispatches. Add this to the agent plugin without changing core interfaces.

| Decision | Contract |
| --- | --- |
| Opt-in | `relay=True`; default execution never imports Relay |
| Dependency | Optional `relay` extra, Python 3.11+; base plugin keeps Python 3.10 |
| Capture | Native ATOF JSONL under `relay/<run>/<trial>/`, linked in trial metadata |
| Correlation | Run, scene, epoch, model-call index and tool-call ID |
| Payload | Metadata only: no prompts, images, tool arguments or exception messages |
| Boundaries | Logical model call includes retries; tool acceptance precedes motion playout |
| Failure | Trace errors produce an error receipt, never retry application callbacks |
| Isolation | Per-trial scope stack and filtered subscriber, bounded finalization |

Use Relay's manual lifecycle APIs. Execution remains in the existing synchronous
policy, so Relay execution interceptors cannot replace or repeat robot commands.
Retain native wire capture, transcripts, approvers and task scoring. External
subscribers may consume the same events; configuring a network exporter is out
of scope. This does not add physical-action spans or claim that `done` proves
task success.

## Acceptance

Run the real evaluation loop with CubePick and a scripted HTTP provider. Verify
a rejected tool followed by recovery, parent/child correlation, reported token
usage, unchanged calls/transcripts/scores, and concurrent trial isolation.
Inject provider and export failures, including cancellation. Run the optional
SDK tests in CI and preserve the base install. Provide a no-key demonstration
and measure enabled/disabled overhead without claiming model-quality evidence.
