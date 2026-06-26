"""Logging sinks for RoboLens runs.

``LogSink`` is the protocol; ``JsonLogSink`` is the canonical, always-on sink
that writes the immutable :class:`~robolens.log.EvalLog` to disk. The optional
``RerunSink`` (added later) is lazily imported and no-ops if ``rerun-sdk`` is
absent.
"""

from __future__ import annotations

from robolens.logging.json_log import JsonLogSink
from robolens.logging.sink import LogSink, NullSink

__all__ = ["JsonLogSink", "LogSink", "NullSink"]
