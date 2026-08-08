# Watching a run live

The HTML viewer can follow an evaluation while it runs. Start the viewer in one
terminal before or after starting the evaluation:

```bash
inspect-robots view logs --serve --open
```

Then start the run in another terminal:

```bash
inspect-robots run --task cubepick-reach --policy agent --embodiment cubepick
```

The index and the running report refresh every two seconds. The report shows
completed trials and the current policy transcript, including agent turns,
notes, and operator or voice feedback. When frame storage is enabled, running
pages show the most recent camera frames that fit the live frame budget. The
final report shows the full set under the normal report frame budget.

Scores appear only in the final log. Scoring happens after the live sink sees a
trial end, so score cells remain `pending` while the run is active. When the
run finishes, the browser returns to the index and the canonical final report
replaces the transient running report.

Pass `--no-live-log` to `run` or `eval-set` to disable transient snapshots. The
final JSON log is still written normally.

## Stale running pages

A force-killed process can leave a `*.live.json` snapshot behind. Its index row
continues to say `running`, and the report shows the last update time. This is a
stale snapshot, not evidence that the process is still alive. Remove it after
confirming that the evaluation process has stopped.

## Higher-rate visualization

The browser view is designed for turn-by-turn policy transcripts. It rewrites a
JSON snapshot at most once per second and refreshes the browser every two
seconds. See [Rerun visualization](logging-and-rerun.md#rerun-visualization) for
high-rate control-timeline streams and their recording options.
