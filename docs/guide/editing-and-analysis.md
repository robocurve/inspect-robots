# Editing and analyzing records

A run's log and its [trial records](logging-and-rerun.md#trial-records) are
raw data: nothing rewrites them after the run. Judgements made later, such as
a reviewer correcting a verdict or adding a safety note, are
[annotations](../../src/inspect_robots/annotation.py) kept beside the log, each
with its provenance. Results can be re-scored from those annotations, and
analyzers read logs and annotations without changing either.

## Annotations

An [`Annotation`](../../src/inspect_robots/annotation.py) is one judgement or
note about one trial:

| Field | Meaning |
|---|---|
| `record_id` | The trial. Logs from before records had ids use `"<scene_id>/e<index>"`. |
| `kind` | `judgement` (a verdict in `value`), `note`, `grading_error`, or a tool's own kind. |
| `value` | The verdict (`yes`, `no`, `partial`, ...) or the note text. |
| `rubric` | The rubric a judgement is for; empty means every rubric that reads judgements. |
| `author`, `source`, `created` | Who made it, how (`console`, `prompt`, `vlm`, `editor`), and when. |
| `supersedes` | The `annotation_id` this one corrects, which then no longer counts. |
| `note`, `metadata` | Why, and anything else a tool records. |

Judgements captured during the run, by the operator or a grader, are already
in the log.
[`read_annotations`](../../src/inspect_robots/annotation.py) returns them as
annotations (for any log schema), followed by the edits in
`<log stem>.annotations.jsonl` beside the log. That file is only ever appended
to. For each trial and rubric, the last annotation that has not been
superseded counts, and one for a specific rubric wins over one for every
rubric.

## Editing from the command line

```bash
inspect-robots edit logs/run.json --trial scene-0-e0 --judgement yes \
  --author reviewer --reason "the stack held after release"
inspect-robots edit logs/run.json --record 3f2a... --note "bowl chipped" --author reviewer
```

`--trial` takes `SCENE-eEPOCH`; `--record` takes a record id. A new judgement
automatically supersedes the one that counted before. A reference editor with
a user interface is outside the core package; any tool that writes the same
annotation lines works.

## Re-scoring

```bash
inspect-robots score logs/run.json                 # stored vs re-scored metrics
inspect-robots score logs/run.json --out rescored.json
```

[`rescore`](../../src/inspect_robots/analyzer.py) recomputes results from
annotations. The built-in `operator` scorer scores each trial from the
judgement that counts, through `score_annotations(annotations, record_id,
rubric)`, and returns exactly what it returned during the run when nothing was
edited. Scores from other scorers (for example `success_at_end`, which reads
the trajectory, or a custom scorer whose configuration the log does not record)
are kept as stored. Only values
whose inputs changed are recomputed, using the log's recorded epoch reducer, so
a log with no edits re-scores to exactly its stored results. `--out` writes a
new file in the input log's directory, with a copy of its annotations, so its
records stay reachable; the input log is never rewritten.

## Analyzers

An [`Analyzer`](../../src/inspect_robots/analyzer.py) receives read-only
[`AnnotatedLog`](../../src/inspect_robots/analyzer.py) views and returns an
[`Analysis`](../../src/inspect_robots/analyzer.py): text to print, JSON-safe
`data`, and optional `files` to save.

```bash
inspect-robots analyze scores logs/*.json          # stored vs re-scored metrics
inspect-robots analyze terminations logs/*.json    # how trials ended, with stated reasons
inspect-robots analyze digest logs/run.json        # the offline summarize digest
inspect-robots analyze html logs/run.json --out pages/
inspect-robots analyze terminations logs/*.json --json
```

The HTML analyzer writes one page per input log. When logs from different
directories share a filename, it adds a numeric suffix to keep every page.

Write your own and register it like any other component:

```python
from collections.abc import Sequence

from inspect_robots import Analysis, AnnotatedLog
from inspect_robots.registry import analyzer


class GiveUpRate:
    name = "give_up_rate"

    def analyze(self, logs: Sequence[AnnotatedLog]) -> Analysis:
        reasons = [r for item in logs for s in item.log.samples for r in s.termination_reasons]
        rate = reasons.count("give_up") / len(reasons) if reasons else 0.0
        return Analysis(name=self.name, text=f"give-up rate: {rate:.2f}\n", data={"rate": rate})


@analyzer("give_up_rate")
def give_up_rate() -> GiveUpRate:
    return GiveUpRate()
```

Plugins register analyzers through the `inspect_robots.analyzers` entry-point
group; `inspect-robots list analyzers` shows what is installed.
