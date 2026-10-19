The LLM agent supports optional NVIDIA NeMo Relay tracing with `relay=True`
and the `inspect-robots-agent[relay]` extra (Python 3.11+). Per-trial ATOF files
correlate logical model calls and actual tool dispatches without copying their
payloads. Trial metadata links the trace and reports capture failures.
