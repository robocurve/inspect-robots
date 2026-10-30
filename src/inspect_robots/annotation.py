"""Annotations: judgements and notes about a recorded trial, kept beside the raw record.

The paper separates a trial's raw record, which nothing rewrites after the run,
from edits to it: a [`Record`][inspect_robots.rollout.Record] stays as it was
recorded, and every judgement or note about it is an
[`Annotation`][inspect_robots.annotation.Annotation] with provenance (who made
it, how, and when). A record editor writes annotations; analyzers and
re-scoring read them.

Annotations come from two places. Judgements captured during the run (by the
operator or a grader) are already in the log;
[`lift_annotations`][inspect_robots.annotation.lift_annotations] reads them as
annotations, for any log schema. Edits made after the run are
appended to a JSON Lines file beside the log,
``<log stem>.annotations.jsonl``, which is never rewritten: a correction is a
new annotation that names the one it ``supersedes``.
[`read_annotations`][inspect_robots.annotation.read_annotations] returns both,
lifted first.

Each trial is keyed by its ``record_id``. Logs written before records had ids
(schemas 1 and 2) key a trial as ``"<scene_id>/e<index>"``, the trial's
position in its scene.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_robots.log import EvalLog, SceneResult

JUDGEMENT = "judgement"
NOTE = "note"
GRADING_ERROR = "grading_error"

_SUFFIX = ".annotations.jsonl"


@dataclass(frozen=True)
class Annotation:
    """One judgement or note about one trial, with its provenance.

    ``kind`` is ``"judgement"`` (``value`` is a verdict such as ``"yes"``,
    ``"no"`` or ``"partial"``), ``"note"`` (free text in ``value``),
    ``"grading_error"`` (a grader tried and failed; ``value`` says why) or any
    other string a tool defines. ``rubric`` names the rubric a judgement is for;
    ``None`` applies it to every rubric that reads judgements. ``author`` says
    who made it (``"operator"``, a grader name, a person) and ``source`` how
    (``"console"``, ``"prompt"``, ``"vlm"``, ``"editor"``). ``supersedes`` is the
    ``annotation_id`` this one corrects, which then no longer counts.
    """

    record_id: str
    kind: str
    value: Any
    author: str
    source: str
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    rubric: str | None = None
    note: str | None = None
    supersedes: str | None = None
    annotation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("record_id", "kind", "author", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"annotation {name} must be a non-empty string, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        """The JSON object written to an annotations file."""
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Annotation:
        """Read one annotation back, ignoring fields this version does not know."""
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


def trial_key(sample: SceneResult, index: int) -> str:
    """The id annotations use for trial ``index`` of ``sample``."""
    if index < len(sample.record_ids):
        record_id = sample.record_ids[index]
        if record_id:
            return record_id
    return f"{sample.scene_id}/e{index}"


def annotations_path(log_path: str | Path) -> Path:
    """The annotations file that belongs to the log at ``log_path``."""
    path = Path(log_path)
    return path.with_name(path.stem + _SUFFIX)


def _parallel(values: tuple[Any, ...], index: int) -> Any:
    return values[index] if index < len(values) else None


def lift_annotations(log: EvalLog) -> tuple[Annotation, ...]:
    """The judgements, notes and grading errors captured during the run, as annotations.

    Works on every log schema. Their ``author`` is the run's grader (or
    ``"operator"`` when none was recorded), their ``source`` the recorded
    judgement source, their ``created`` the run's completion time, and their
    ids are derived from the trial, so lifting the same log twice gives equal
    annotations.
    """
    author = log.eval.grader or "operator"
    created = log.stats.completed_at
    lifted: list[Annotation] = []
    for sample in log.samples:
        for index in range(len(sample.epochs)):
            key = trial_key(sample, index)
            judgement = _parallel(sample.operator_judgements, index)
            source = _parallel(sample.judgement_sources, index) or "run"
            note = _parallel(sample.operator_notes, index)
            metadata = _parallel(sample.trial_metadata, index)
            if judgement is not None:
                lifted.append(
                    Annotation(
                        record_id=key,
                        kind=JUDGEMENT,
                        value=judgement,
                        author=author,
                        source=source,
                        created=created,
                        note=note,
                        annotation_id=f"{key}:run:{JUDGEMENT}",
                    )
                )
            elif note is not None:
                lifted.append(
                    Annotation(
                        record_id=key,
                        kind=NOTE,
                        value=note,
                        author=author,
                        source=source,
                        created=created,
                        annotation_id=f"{key}:run:{NOTE}",
                    )
                )
            grading_error = metadata.get("grading_error") if isinstance(metadata, dict) else None
            if grading_error:
                lifted.append(
                    Annotation(
                        record_id=key,
                        kind=GRADING_ERROR,
                        value=str(grading_error),
                        author=author,
                        source=source,
                        created=created,
                        annotation_id=f"{key}:run:{GRADING_ERROR}",
                    )
                )
    return tuple(lifted)


def append_annotations(log_path: str | Path, annotations: Iterable[Annotation]) -> Path:
    """Append ``annotations`` to the log's annotations file; return the file's path.

    The file is only ever appended to, one JSON object per line, and each
    write is flushed to disk before returning.
    """
    path = annotations_path(log_path)
    lines = "".join(
        json.dumps(annotation.to_dict(), allow_nan=False) + "\n" for annotation in annotations
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(lines)
        fh.flush()
        os.fsync(fh.fileno())
    return path


def read_annotations(log_path: str | Path, log: EvalLog | None = None) -> tuple[Annotation, ...]:
    """Every annotation for the log at ``log_path``: lifted from the log, then the file's.

    Pass ``log`` when it is already loaded to skip reading it again. A missing
    annotations file means no edits were made. A malformed line raises
    ``ValueError`` naming the file and line.
    """
    from inspect_robots.log import read_eval_log

    loaded = log if log is not None else read_eval_log(str(log_path))
    annotations = list(lift_annotations(loaded))
    path = annotations_path(log_path)
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                annotations.append(Annotation.from_dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: not a valid annotation: {exc}") from exc
    return tuple(annotations)


def current(
    annotations: Sequence[Annotation],
    record_id: str,
    kind: str,
    rubric: str | None = None,
) -> Annotation | None:
    """The annotation of ``kind`` that counts for one trial, or ``None``.

    Superseded annotations do not count; of the rest, the last one wins. With
    a ``rubric``, an annotation for that rubric wins over one for every rubric
    (``rubric=None``).
    """
    superseded = {a.supersedes for a in annotations if a.supersedes is not None}
    live = [
        a
        for a in annotations
        if a.record_id == record_id and a.kind == kind and a.annotation_id not in superseded
    ]
    specific = [a for a in live if rubric is not None and a.rubric == rubric]
    general = [a for a in live if a.rubric is None]
    chosen = specific or general
    return chosen[-1] if chosen else None
