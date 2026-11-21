#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/tests/test_secondary_model_signals.py

from __future__ import annotations

import importlib.util
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def find_repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "docs" / "win" / "hockey" / "nhl").is_dir():
            return candidate
    raise RuntimeError("Unable to locate NHL repository root")


REPO_ROOT = find_repo_root()
NHL_ROOT = REPO_ROOT / "docs" / "win" / "hockey" / "nhl"


def load_module(relative_path: str):
    path = REPO_ROOT / relative_path
    name = f"nhl_secondary_test_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_merge_intake_loads_current_sdv_predictions_by_game_id(tmp_path: Path) -> None:
    module = load_module("docs/win/hockey/nhl/scripts/01_merge/merge_intake.py")
    module.SDV_PREDICTIONS_PATH = tmp_path / "latest_predictions.csv"
    module.LOG_FILE = tmp_path / "merge.log"

    pd.DataFrame(
        [
            {
                "game_id": "2026020001",
                "home_win_prob": 0.61,
                "exp_margin": 0.72,
                "exp_total": 6.15,
            }
        ]
    ).to_csv(module.SDV_PREDICTIONS_PATH, index=False)

    index = module.load_sdv_prediction_index()
    assert index["2026020001"] == {
        "game_id": "2026020001",
        "sdv_home_win_prob": "0.61",
        "sdv_exp_margin": "0.72",
        "sdv_exp_total": "6.15",
    }


def test_merge_intake_rejects_duplicate_sdv_game_id(tmp_path: Path) -> None:
    module = load_module("docs/win/hockey/nhl/scripts/01_merge/merge_intake.py")
    module.SDV_PREDICTIONS_PATH = tmp_path / "latest_predictions.csv"
    module.LOG_FILE = tmp_path / "merge.log"

    pd.DataFrame(
        [
            {"game_id": "2026020001", "home_win_prob": 0.61, "exp_margin": 0.72, "exp_total": 6.15},
            {"game_id": "2026020001", "home_win_prob": 0.62, "exp_margin": 0.80, "exp_total": 6.20},
        ]
    ).to_csv(module.SDV_PREDICTIONS_PATH, index=False)

    with pytest.raises(SystemExit):
        module.load_sdv_prediction_index()


def test_build_juice_files_preserves_sdv_prediction_columns() -> None:
    module = load_module("docs/win/hockey/nhl/scripts/01_merge/build_juice_files.py")
    expected = {"sdv_home_win_prob", "sdv_exp_margin", "sdv_exp_total"}
    assert expected.issubset(module.MERGED_REQUIRED_COLUMNS)
    assert expected.issubset(module.MONEYLINE_COLUMNS)
    assert expected.issubset(module.PUCK_LINE_COLUMNS)
    assert expected.issubset(module.TOTAL_COLUMNS)


def make_history(prior_rows: int = 110, same_day_rows: int = 10) -> tuple[pd.DataFrame, date]:
    target_day = date(2026, 5, 1)
    rows = []

    for i in range(prior_rows):
        game_day = target_day - timedelta(days=prior_rows - i)
        drat_prob = 0.40 + 0.002 * (i % 50)
        sdv_prob = 0.42 + 0.002 * ((i * 3) % 45)
        drat_margin = -1.0 + 0.05 * (i % 40)
        sdv_margin = -0.8 + 0.04 * ((i * 2) % 40)
        drat_total = 5.2 + 0.03 * (i % 40)
        sdv_total = 5.4 + 0.025 * ((i * 3) % 40)
        rows.append(
            {
                "game_id": f"202602{i + 1:04d}",
                "game_date": game_day.isoformat(),
                "_date": game_day,
                "drat_home_win_prob": drat_prob,
                "sdv_home_win_prob": sdv_prob,
                "drat_exp_margin": drat_margin,
                "sdv_exp_margin": sdv_margin,
                "drat_exp_total": drat_total,
                "sdv_exp_total": sdv_total,
                "actual_home_win": i % 2,
                "actual_margin": (-1 if i % 2 == 0 else 1) * (1 + (i % 4)),
                "actual_total": 5 + (i % 4),
            }
        )

    for i in range(same_day_rows):
        rows.append(
            {
                "game_id": f"2026029{i:03d}",
                "game_date": target_day.isoformat(),
                "_date": target_day,
                "drat_home_win_prob": 0.99,
                "sdv_home_win_prob": 0.01,
                "drat_exp_margin": 9.0,
                "sdv_exp_margin": -9.0,
                "drat_exp_total": 12.0,
                "sdv_exp_total": 2.0,
                "actual_home_win": i % 2,
                "actual_margin": 8.0,
                "actual_total": 10.0,
            }
        )

    return pd.DataFrame(rows), target_day


def test_secondary_models_use_only_dates_before_target() -> None:
    module = load_module("docs/win/hockey/nhl/scripts/03_edges/build_secondary_model_signals.py")
    history, target_day = make_history()
    bundle = module.fit_bundle_for_target(
        history,
        target_day,
        min_train_rows=100,
        disagreement_quantile=0.75,
    )
    assert bundle is not None
    assert bundle.train_rows == 110
    assert bundle.history_max_game_date < target_day.isoformat()


def test_secondary_signal_falls_back_when_sdv_missing() -> None:
    module = load_module("docs/win/hockey/nhl/scripts/03_edges/build_secondary_model_signals.py")
    base = pd.Series(
        {
            "drat_home_win_prob": 0.58,
            "drat_exp_margin": 0.8,
            "drat_exp_total": 6.1,
            "sdv_home_win_prob": np.nan,
            "sdv_exp_margin": np.nan,
            "sdv_exp_total": np.nan,
        }
    )
    row = module.signal_row(base, None, True)
    assert row["secondary_model_status"] == "sdv_unavailable"
    assert row["drat_home_win_prob"] == pytest.approx(0.58)


def secondary_config() -> dict:
    return {
        "secondary_model": {
            "enabled": True,
            "selection_mode": "high_disagreement_requires_secondary_support",
            "unavailable_behavior": "use_primary",
            "min_train_rows": 100,
            "disagreement_quantile": 0.75,
            "derived_signal_by_market": {
                "moneyline": "weighted",
                "puck_line": "meta",
                "total": "meta",
            },
        }
    }


def primary_candidate() -> dict:
    return {
        "game_date": "2026_10_10",
        "market_type": "moneyline",
        "bet_side": "home",
        "line": "",
    }


def signal_row_for_gate(*, sdv_prob: float, weighted_prob: float, status: str = "ready") -> pd.Series:
    module = load_module("docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py")
    data = {col: np.nan for col in module.SECONDARY_SIGNAL_COLUMNS}
    data.update(
        {
            "secondary_model_status": status,
            "high_prob_disagreement_flag": 1,
            "sdv_home_win_prob": sdv_prob,
            "weighted_home_win_prob": weighted_prob,
        }
    )
    return pd.Series(data)


def test_stage04_blocks_high_disagreement_without_secondary_support() -> None:
    module = load_module("docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py")
    rejections = {}
    row = signal_row_for_gate(sdv_prob=0.40, weighted_prob=0.45)
    kept = module.apply_secondary_model_gate(
        [primary_candidate()],
        row,
        secondary_config(),
        market_type="moneyline",
        rejections=rejections,
    )
    assert kept == []
    assert sum(rejections.values()) == 1


def test_stage04_keeps_high_disagreement_with_secondary_support() -> None:
    module = load_module("docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py")
    rejections = {}
    row = signal_row_for_gate(sdv_prob=0.62, weighted_prob=0.45)
    kept = module.apply_secondary_model_gate(
        [primary_candidate()],
        row,
        secondary_config(),
        market_type="moneyline",
        rejections=rejections,
    )
    assert len(kept) == 1
    assert kept[0]["secondary_decision"] == "high_disagreement_supported"
    assert rejections == {}


def test_stage04_uses_primary_when_secondary_unavailable() -> None:
    module = load_module("docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py")
    rejections = {}
    row = signal_row_for_gate(sdv_prob=np.nan, weighted_prob=np.nan, status="sdv_unavailable")
    kept = module.apply_secondary_model_gate(
        [primary_candidate()],
        row,
        secondary_config(),
        market_type="moneyline",
        rejections=rejections,
    )
    assert len(kept) == 1
    assert kept[0]["secondary_decision"] == "fallback_primary:sdv_unavailable"
    assert rejections == {}


def test_markets_yaml_secondary_model_contract() -> None:
    path = NHL_ROOT / "config" / "markets.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    secondary = payload["markets"]["nhl"]["secondary_model"]
    assert secondary["enabled"] is True
    assert secondary["min_train_rows"] == 100
    assert secondary["disagreement_quantile"] == pytest.approx(0.75)
    assert secondary["derived_signal_by_market"] == {
        "moneyline": "weighted",
        "puck_line": "meta",
        "total": "meta",
    }
