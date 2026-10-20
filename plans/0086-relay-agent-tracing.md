# Optional Relay tracing for the LLM agent

Let Relay users consume Inspect LLM-policy runs through existing Relay tools.
The ATIF example demonstrates this without a separate Inspect-log converter.
Keep the integration in the agent plugin and leave core interfaces unchanged.

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
