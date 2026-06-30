"""Logging sinks for RoboInspect runs.

``LogSink`` is the protocol; ``JsonLogSink`` is the canonical, always-on sink
that writes the immutable [`EvalLog`][roboinspect.log.EvalLog] to disk. The optional
``RerunSink`` (added later) is lazily imported and no-ops if ``rerun-sdk`` is
absent.
"""

from __future__ import annotations

from roboinspect.logging.json_log import JsonLogSink
from roboinspect.logging.rerun_sink import RerunSink
from roboinspect.logging.sink import LogSink, NullSink

__all__ = ["JsonLogSink", "LogSink", "NullSink", "RerunSink"]
