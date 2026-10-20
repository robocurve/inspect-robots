# Optional Relay tracing for the LLM agent

Let Relay users consume Inspect LLM-policy runs through existing Relay tools.
Keep the integration in the agent plugin and leave core interfaces unchanged.

Status: failed-call reporting is fixed upstream in
[NeMo Relay #1405](https://github.com/NVIDIA/NeMo-Relay/pull/1405).
The integration can be reviewed now. Wait for a published release containing
that fix, then update the optional dependency and lockfile before merging.
The current 0.10.3 pin still fails the provider-error regression.
No local event-rewriting workaround is used.

| Decision | Contract |
| --- | --- |
| Enablement | `relay=True`; optional pinned SDK on Python 3.11+; no default import |
| Boundaries | Trial, logical model call including retries, actual tool dispatch |
| Data | Correlation IDs, reported usage and outcomes; omit payloads |
| Execution | Manual lifecycle APIs; never repeat application callbacks |
| Capture | Trial-local ATOF file and receipt; independent stacks and subscribers |
| Completion | Record end time before publication; flag missing events; bound draining |
| Consumer | One scripted CubePick run through Relay's ATIF exporter |

The local writer links traces to Inspect logs without requiring a consumer to
be configured. Keep native wire capture and scoring. Tool acceptance precedes
motion playout. Other policy adapters, physical-action spans, remote exporter
setup and performance comparisons are outside this change.

Verify correlation, reported usage, payload omission, unchanged execution,
concurrent isolation, timestamps, SDK failures and cancellation. Run the actual
ATIF consumer when Relay is installed, and the plugin suite with and without
the optional dependency.
