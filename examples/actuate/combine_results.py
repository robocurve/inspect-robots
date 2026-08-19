"""Combine fixed-trials result files into per-file and pooled summaries.

Run with:  python examples/actuate/combine_results.py state-rig1/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _trial_scores(path: Path) -> dict[str, list[float]]:
    observations: dict[str, list[float]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

    for line in lines:
        try:
            result = json.loads(line)
            name = result["roles"]["test_taker"]
            score = result.get("score")
            numeric_score = float(score) if score is not None else None
            if numeric_score is not None and math.isfinite(numeric_score):
                observations.setdefault(name, []).append(numeric_score)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    return observations


def _print_summary(header: str, observations: dict[str, list[float]]) -> None:
    print(f"{header}:")
    for name, scores in observations.items():
        mean_score = sum(scores) / len(scores)
        print(f"{name}: n={len(scores)}, mean={mean_score:.2f}")


def main() -> None:
    """Print per-file and pooled summaries for result paths from the command line."""
    parser = argparse.ArgumentParser(description="Combine Actuate fixed-trials results.")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    pooled: dict[str, list[float]] = {}
    for index, path in enumerate(args.results):
        observations = _trial_scores(path)
        if index:
            print()
        _print_summary(str(path), observations)
        for name, scores in observations.items():
            pooled.setdefault(name, []).extend(scores)

    print()
    _print_summary("Pooled", pooled)


if __name__ == "__main__":
    main()
