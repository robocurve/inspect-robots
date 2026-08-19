"""Run the rig-2 half of the dual-rig Actuate fixed-trials campaign.

Run with:  python examples/actuate/run_trials_rig2.py [-- extra inspect-robots run args]
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_trials import run_campaign

TRIALS_PER_RIG = 5  # booth-editable; the combined two-rig target is 2x this
RIG_CONFIG = Path.home() / "robocurve" / "rig-2" / "config.ini"
HERE = Path(__file__).parent

if __name__ == "__main__":
    run_campaign(
        RIG_CONFIG,
        HERE / "state-rig2",
        HERE / "logs-rig2",
        TRIALS_PER_RIG,
        sys.argv[1:],
    )
