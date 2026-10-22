# 0086 — Persist metric depth beside RGB frames

**Issue:** #236 (frame-store half). The Rerun live-viewer half stays open.

## Problem

Embodiments publish per-camera metric depth as `observation.extra["<camera>_depth"]`:
a 2-D float array of metres, or a zero-argument callable returning one (plan 0028).
The agent plugin renders that for the policy. `FrameStore` writes only
`observation.images`, so a saved run has no depth to inspect later.

`inspect-robots video` globs `*.npy` in the run directory, non-recursively, and
treats every match as a uint8 RGB stream. A sibling `*_depth_NNNNNN.npy` would
become a phantom camera and then fail the dtype check.

## Design

Resolve depth in the rollout, and only when a frame store is on. Each callable
is invoked once, before `policy.act` sees that observation. A 2-D array passes
through. An ordinary failure, or an entry that is already failure text,
becomes a string on the policy-facing observation and is not written.
`SafetyAbort` and `EmbodimentFault` from a depth callable propagate and are
recorded on the trial. RGB for that observation is written first. If the
action already ran, the step, its step event, and those RGB frames stay on
the partial trial, and the failing callable is not called again. The agent plugin's
`resolve_depth` keeps that string instead of parsing it again.

Successful maps are stored as float32 under `frames/<run>/depth/`, using the
same `~f1~` filename as RGB. `DepthFrameRef.load()` returns float32.
`FrameRef.load()` stays uint8. `StepRecord.depth_refs` and
`result_depth_refs` hold the handles. The persisted observation drops
`{camera}_depth` keys so the maps do not stay in memory.

No new config flag. No depth key means no depth files, and RGB-only runs are
unchanged. `inspect-robots video` and the HTML viewer are unchanged: the
subdirectory keeps them on RGB. Transcripts stay image-free.

## Follow-up

Log the same resolved maps to the Rerun viewer as metric depth images. That
needs the optional `rerun-sdk` and is not part of this change.
