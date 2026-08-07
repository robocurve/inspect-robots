"""Logging sinks for Inspect Robots runs.

``LogSink`` is the protocol; ``JsonLogSink`` is the canonical, always-on sink
that writes the immutable [`EvalLog`][inspect_robots.log.EvalLog] to disk.
``LiveLogSink`` maintains the transient running snapshot. The optional
``RerunSink`` is lazily imported and no-ops if ``rerun-sdk`` is absent.
"""

from __future__ import annotations

from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.logging.live_log import LiveLogSink
from inspect_robots.logging.rerun_sink import RerunSink
from inspect_robots.logging.sink import LogSink, NullSink

__all__ = ["JsonLogSink", "LiveLogSink", "LogSink", "NullSink", "RerunSink"]
