"""Combine fixed-trials result files into per-file and pooled summaries.

Run with:  python examples/actuate/combine_results.py examples/actuate/state-rig*/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _roster import ROSTER


def _trial_scores(path: Path) -> dict[str, list[float]]:
    # Seeded from the roster like the leaderboard, so a model with no scored
    # evals still shows an n=0 row and stale non-roster names are ignored.
    observations: dict[str, list[float]] = {name: [] for name in ROSTER}
    lines = path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        try:
            result = json.loads(line)
            name = result["roles"]["test_taker"]
            score = result.get("score")
            numeric_score = float(score) if score is not None else None
            if name in observations and numeric_score is not None and math.isfinite(numeric_score):
                observations[name].append(numeric_score)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    return observations


def _print_summary(header: str, observations: dict[str, list[float]]) -> None:
    print(f"{header}:")
    for name, scores in observations.items():
        mean_text = f"{sum(scores) / len(scores):.2f}" if scores else "n/a"
        print(f"{name}: n={len(scores)}, mean={mean_text}")


def main() -> None:
    """Print per-file and pooled summaries for result paths from the command line."""
    parser = argparse.ArgumentParser(description="Combine Actuate fixed-trials results.")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    # A typo'd path must not silently print a half-campaign leaderboard.
    missing = [path for path in args.results if not path.is_file()]
    if missing:
        parser.error("results file not found: " + ", ".join(str(path) for path in missing))

    pooled: dict[str, list[float]] = {name: [] for name in ROSTER}
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
