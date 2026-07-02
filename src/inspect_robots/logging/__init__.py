"""Logging sinks for Inspect Robots runs.

``LogSink`` is the protocol; ``JsonLogSink`` is the canonical, always-on sink
that writes the immutable [`EvalLog`][inspect_robots.log.EvalLog] to disk. The optional
``RerunSink`` (added later) is lazily imported and no-ops if ``rerun-sdk`` is
absent.
"""

from __future__ import annotations

from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.logging.rerun_sink import RerunSink
from inspect_robots.logging.sink import LogSink, NullSink

__all__ = ["JsonLogSink", "LogSink", "NullSink", "RerunSink"]
