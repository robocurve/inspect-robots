"""Run the rig-1 half of the dual-rig Actuate fixed-trials campaign.

Run with:  python examples/actuate/run_trials_rig1.py [-- extra inspect-robots run args]
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_trials import run_campaign

TRIALS_PER_RIG = 5  # booth-editable; the combined two-rig target is 2x this
# Recording pins prevent overnight viewer windows, save per-eval .rrd camera,
# joint, and action timelines, and drift-proof frame storage while sharing a
# 120-second ceiling at the rigs' 10 Hz control rate. Later user flags win.
MAX_STEPS = 1200
RIG_CONFIG = Path.home() / "robocurve" / "rig-1" / "config.ini"
HERE = Path(__file__).parent

if __name__ == "__main__":
    user_args = sys.argv[1:]
    if user_args and user_args[0] == "--":
        user_args = user_args[1:]
    run_campaign(
        RIG_CONFIG,
        HERE / "state-rig1",
        HERE / "logs-rig1",
        TRIALS_PER_RIG,
        [
            "--",
            "--max-steps",
            str(MAX_STEPS),
            "--store-frames",
            "--no-rerun",
            "--rerun-save",
            *user_args,
        ],
    )
