#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/pred_name_normalization.py

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# noinspection PyPep8
from name_normalization_common import run_name_normalization


PREDICTIONS_DIR = Path(
    "docs/win/hockey/nhl/00_intake/predictions"
)
MAP_FILE = Path(
    "docs/win/hockey/nhl/config/mapping/team_map_nhl.csv"
)
NO_MAP_FILE = Path(
    "docs/win/hockey/nhl/config/mapping/no_map_nhl_pred.csv"
)
LOG_FILE = Path(
    "docs/win/hockey/nhl/errors/00_intake/"
    "pred_name_normalization.txt"
)


run_name_normalization(
    run_name="pred_name_normalization",
    source="dratings",
    map_file=MAP_FILE,
    no_map_file=NO_MAP_FILE,
    log_file=LOG_FILE,
    target_files=sorted(
        PREDICTIONS_DIR.glob("hockey_*.csv")
    ),
)

print("NHL prediction name normalization complete.")
