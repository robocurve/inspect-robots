"""Run the rig-2 half of the dual-rig Actuate fixed-trials campaign.

Run with:  python examples/actuate/run_trials_rig2.py [-- extra inspect-robots run args]
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_trials import run_campaign

TRIALS_PER_RIG = 5  # booth-editable; the combined two-rig target is 2x this
# Pinned on both rigs so the campaign halves share one episode ceiling: the
# rig configs disagree today (12000 vs 1200). Later user flags still win.
MAX_STEPS = 12000
RIG_CONFIG = Path.home() / "robocurve" / "rig-2" / "config.ini"
HERE = Path(__file__).parent

if __name__ == "__main__":
    user_args = sys.argv[1:]
    if user_args and user_args[0] == "--":
        user_args = user_args[1:]
    run_campaign(
        RIG_CONFIG,
        HERE / "state-rig2",
        HERE / "logs-rig2",
        TRIALS_PER_RIG,
        ["--", "--max-steps", str(MAX_STEPS), *user_args],
    )
