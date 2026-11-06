#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/tests/test_nhl_pipeline.py

from __future__ import annotations

import copy
import importlib.util
from collections import defaultdict
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _find_repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "docs" / "win" / "hockey" / "nhl").is_dir():
            return candidate
    raise RuntimeError(
        "Unable to locate repository root containing docs/win/hockey/nhl"
    )


REPO_ROOT = _find_repo_root()
NHL_ROOT = REPO_ROOT / "docs" / "win" / "hockey" / "nhl"

CORE_PATHS = [
    "docs/win/hockey/nhl/config/markets.yaml",
    "docs/win/hockey/nhl/config/mapping/team_map_nhl.csv",
    "docs/win/hockey/nhl/config/juice/nhl_moneyline_juice.csv",
    "docs/win/hockey/nhl/config/juice/nhl_puck_line_juice.csv",
    "docs/win/hockey/nhl/config/juice/nhl_total_juice.csv",
    "docs/win/hockey/nhl/scripts/00_intake/build_games.py",
    "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py",
    "docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py",
    "docs/win/hockey/nhl/scripts/00_intake/transform_hockey.py",
    "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py",
    "docs/win/hockey/nhl/scripts/01_merge/build_juice_files.py",
    "docs/win/hockey/nhl/scripts/02_juice/validate_juice_config.py",
    "docs/win/hockey/nhl/scripts/02_juice/apply_moneyline_juice.py",
    "docs/win/hockey/nhl/scripts/02_juice/apply_puck_line_juice.py",
    "docs/win/hockey/nhl/scripts/02_juice/apply_total_juice.py",
    "docs/win/hockey/nhl/scripts/03_edges/compute_edges.py",
    "docs/win/hockey/nhl/scripts/03_edges/compute_ev_kelly.py",
    "docs/win/hockey/nhl/scripts/04_select/validate_markets_config.py",
    "docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py",
    "docs/win/hockey/nhl/scripts/05_final_scores/01_nhl_results_grade.py",
    "docs/win/hockey/nhl/scripts/05_final_scores/03_nhl_results_reports.py",
    "docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py",
]

JUICE_CONFIG_PATHS = [
    NHL_ROOT / "config" / "juice" / "nhl_moneyline_juice.csv",
    NHL_ROOT / "config" / "juice" / "nhl_puck_line_juice.csv",
    NHL_ROOT / "config" / "juice" / "nhl_total_juice.csv",
]


FATIGUE_FEATURE_VALUES = {
    "home_days_rest": 2,
    "away_days_rest": 1,
    "home_back_to_back": 0,
    "away_back_to_back": 1,
    "home_games_in_4_days": 2,
    "away_games_in_4_days": 3,
    "home_three_in_four": 0,
    "away_three_in_four": 1,
    "home_games_in_6_days": 3,
    "away_games_in_6_days": 4,
    "home_four_in_six": 0,
    "away_four_in_six": 1,
    "home_games_in_7_days": 3,
    "away_games_in_7_days": 4,
    "rest_differential": 1,
}


def assert_fatigue_features_preserved(
    row: pd.Series,
) -> None:
    for column, expected in FATIGUE_FEATURE_VALUES.items():
        assert column in row.index
        assert float(row[column]) == pytest.approx(
            float(expected),
            abs=1e-12,
        )


TEAM_STRENGTH_FEATURE_VALUES = {
    "home_adj_xgf": 3.10,
    "away_adj_xgf": 2.80,
    "adj_xgf_differential": 0.30,
    "home_adj_xga": 2.40,
    "away_adj_xga": 2.90,
    "adj_xga_differential": -0.50,
    "home_adj_xg_net": 0.70,
    "away_adj_xg_net": -0.10,
    "adj_xg_net_differential": 0.80,
    "home_adj_gf": 3.20,
    "away_adj_gf": 2.70,
    "adj_gf_differential": 0.50,
    "home_adj_ga": 2.50,
    "away_adj_ga": 3.00,
    "adj_ga_differential": -0.50,
    "home_off_rank": 4,
    "away_off_rank": 18,
    "off_rank_differential": -14,
    "home_def_rank": 6,
    "away_def_rank": 20,
    "def_rank_differential": -14,
    "home_net_rank": 3,
    "away_net_rank": 19,
    "net_rank_differential": -16,
    "home_net_z": 1.25,
    "away_net_z": -0.20,
    "net_z_differential": 1.45,
}


def assert_team_strength_features_preserved(
    row: pd.Series,
) -> None:
    for (
        column,
        expected,
    ) in TEAM_STRENGTH_FEATURE_VALUES.items():
        assert column in row.index
        assert float(
            row[
                column
            ]
        ) == pytest.approx(
            float(expected),
            abs=1e-12,
        )


GOALIE_FEATURE_VALUES = {
    "home_expected_starter": "Jeremy Swayman",
    "away_expected_starter": "Igor Shesterkin",
    "home_starter_gsax": 4.25,
    "away_starter_gsax": 2.75,
    "home_backup_gsax": 0.50,
    "away_backup_gsax": -0.25,
    "starter_gsax_differential": 1.50,
    "home_goalie_status": "projected",
    "away_goalie_status": "projected",
    "home_goalie_status_observed_at": "2026-01-01T22:00:00+00:00",
    "away_goalie_status_observed_at": "2026-01-01T22:00:00+00:00",
    "home_goalie_status_source": "sportsdataverse_prior_goalie_usage_projection",
    "away_goalie_status_source": "sportsdataverse_prior_goalie_usage_projection",
}


def assert_goalie_features_preserved(
    row: pd.Series,
) -> None:
    for column, expected in GOALIE_FEATURE_VALUES.items():
        assert column in row.index

        if isinstance(
            expected,
            (int, float),
        ):
            assert float(
                row[column]
            ) == pytest.approx(
                float(expected),
                abs=1e-12,
            )
        else:
            assert str(
                row[column]
            ) == str(
                expected
            )




LINEUP_FEATURE_VALUES = {
    "home_skater_rapm": 0.35,
    "away_skater_rapm": 0.10,
    "skater_rapm_differential": 0.25,
    "home_skater_war": 5.50,
    "away_skater_war": 4.00,
    "skater_war_differential": 1.50,
    "home_pp_value": 1.20,
    "away_pp_value": 0.75,
    "pp_value_differential": 0.45,
    "home_pk_value": 0.80,
    "away_pk_value": 0.30,
    "pk_value_differential": 0.50,
    "home_forward_line_strength": 0.60,
    "away_forward_line_strength": 0.40,
    "forward_line_strength_differential": 0.20,
    "home_defense_pair_strength": 0.45,
    "away_defense_pair_strength": 0.25,
    "defense_pair_strength_differential": 0.20,
    "home_lineup_status": "unknown",
    "away_lineup_status": "unknown",
    "home_lineup_observed_at": "",
    "away_lineup_observed_at": "",
    "home_lineup_source": "sportsdataverse_prior_game_player_pool_no_lineup_confirmation",
    "away_lineup_source": "sportsdataverse_prior_game_player_pool_no_lineup_confirmation",
}


def assert_lineup_features_preserved(
    row: pd.Series,
) -> None:
    for column, expected in LINEUP_FEATURE_VALUES.items():
        assert column in row.index

        if isinstance(
            expected,
            (int, float),
        ):
            assert float(
                row[column]
            ) == pytest.approx(
                float(expected),
                abs=1e-12,
            )
        else:
            actual = row[column]
            if pd.isna(actual):
                actual = ""
            assert str(
                actual
            ) == str(
                expected
            )


def load_repo_module(relative_path: str):
    path = REPO_ROOT / relative_path
    if not path.is_file():
        pytest.fail(
            f"Required repository module does not exist: {relative_path}"
        )

    module_name = (
        f"nhl_pipeline_test_{path.stem}_{uuid.uuid4().hex}"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        pytest.fail(
            f"Unable to load module spec for: {relative_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def write_reconciled_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    columns = [
        "game_id",
        "sportsbook_event_id",
        "sport",
        "league",
        "game_date",
        "game_time",
        "home_team",
        "away_team",
    ]

    pd.DataFrame(
        rows,
        columns=columns,
    ).to_csv(
        path,
        index=False,
    )


def valid_reconciled_row(
    *,
    game_id: str = "2025020001",
    sportsbook_event_id: str = "book-1",
) -> dict[str, str]:
    return {
        "game_id": game_id,
        "sportsbook_event_id": sportsbook_event_id,
        "sport": "hockey",
        "league": "nhl",
        "game_date": "2026_01_01",
        "game_time": "19:00",
        "home_team": "Boston Bruins",
        "away_team": "New York Rangers",
    }


def permissive_side_rules() -> dict:
    return {
        "enabled": True,
        "prob_bands": [[0.0, 1.0]],
        "odds_bands": [[-10000.0, 10000.0]],
        "ev_bands": [[-1.0, 999.9999]],
        "kelly_bands": [[0.0, 999.9999]],
    }


def synthetic_moneyline_row() -> dict:
    return {
        "sport": "hockey",
        "league": "nhl",
        "game_date": "2026_01_01",
        "game_time": "19:00",
        "game_id": "2025020001",
        "away_team": "New York Rangers",
        "home_team": "Boston Bruins",
        "home_dk_moneyline_american": -110.0,
        "away_dk_moneyline_american": 105.0,
        "home_dk_moneyline_decimal": 1.91,
        "away_dk_moneyline_decimal": 2.05,
        "home_model_prob_moneyline": 0.55,
        "away_model_prob_moneyline": 0.50,
        "home_edge_pct_moneyline": 0.0264,
        "away_edge_pct_moneyline": 0.0122,
        "home_ev_moneyline": 0.0500,
        "away_ev_moneyline": 0.0250,
        "home_kelly_moneyline": 0.10,
        "away_kelly_moneyline": 0.05,
    }


def calibration_input_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "league": "nhl",
                "game_date": "2026_01_01",
                "game_id": "2025020001",
                "away_team": "New York Rangers",
                "home_team": "Boston Bruins",
                "market_type": "moneyline",
                "bet_side": "home",
                "line": np.nan,
                "dk_odds_american": -110,
                "dk_odds_decimal": 1.91,
                "model_prob": 0.60,
                "my_model_prob": 0.60,
                "edge": 0.05,
                "ev": 0.05,
                "kelly": 0.10,
                "away_score": 2,
                "home_score": 4,
                "total_score": 6,
                "away_puck_line_result": -2,
                "home_puck_line_result": 2,
                "bet_result": "Win",
            },
            {
                "league": "nhl",
                "game_date": "2026_01_02",
                "game_id": "2025020002",
                "away_team": "Toronto Maple Leafs",
                "home_team": "Montreal Canadiens",
                "market_type": "moneyline",
                "bet_side": "away",
                "line": np.nan,
                "dk_odds_american": 110,
                "dk_odds_decimal": 2.10,
                "model_prob": 0.40,
                "my_model_prob": 0.40,
                "edge": 0.02,
                "ev": 0.03,
                "kelly": 0.04,
                "away_score": 2,
                "home_score": 3,
                "total_score": 5,
                "away_puck_line_result": -1,
                "home_puck_line_result": 1,
                "bet_result": "Loss",
            },
            {
                "league": "nhl",
                "game_date": "2026_01_02",
                "game_id": "2025020003",
                "away_team": "Detroit Red Wings",
                "home_team": "Chicago Blackhawks",
                "market_type": "total",
                "bet_side": "over",
                "line": 6.0,
                "dk_odds_american": -105,
                "dk_odds_decimal": 1.95,
                "model_prob": 0.55,
                "my_model_prob": 0.55,
                "edge": 0.01,
                "ev": 0.01,
                "kelly": 0.02,
                "away_score": 3,
                "home_score": 3,
                "total_score": 6,
                "away_puck_line_result": 0,
                "home_puck_line_result": 0,
                "bet_result": "Push",
            },
        ]
    )


# ---------------------------------------------------------------------
# test_paths.py coverage
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "relative_path",
    CORE_PATHS,
)
def test_required_nhl_pipeline_paths_exist(
    relative_path: str,
) -> None:
    assert (
        REPO_ROOT
        / relative_path
    ).is_file(), relative_path


def test_nhl_root_is_canonical_docs_win_path() -> None:
    assert (
        NHL_ROOT
        == REPO_ROOT
        / "docs"
        / "win"
        / "hockey"
        / "nhl"
    )

    assert NHL_ROOT.is_dir()


def test_repo_root_discovery_is_not_fixed_to_parent_depth() -> None:
    assert (
        REPO_ROOT
        / ".github"
    ).is_dir()

    assert (
        REPO_ROOT
        / "docs"
        / "win"
        / "hockey"
        / "nhl"
    ).is_dir()


# ---------------------------------------------------------------------
# test_transform_hockey.py coverage
# Regression fixture: missing mappings
# ---------------------------------------------------------------------

def test_team_map_has_stable_official_identity_and_source_aliases() -> None:
    mapping_path = (
        NHL_ROOT
        / "config"
        / "mapping"
        / "team_map_nhl.csv"
    )

    df = pd.read_csv(
        mapping_path,
        dtype=str,
    ).fillna("")

    required = {
        "league",
        "source",
        "alias",
        "canonical_team",
        "nhl_team_id",
        "nhl_abbrev",
    }

    assert required.issubset(
        df.columns
    )

    official = df[
        (
            df["league"].str.lower()
            == "nhl"
        )
        & (
            df["source"].str.lower()
            == "official_nhl"
        )
    ].copy()

    assert len(official) == 32
    assert official[
        "nhl_team_id"
    ].ne("").all()

    assert official[
        "nhl_team_id"
    ].str.fullmatch(
        r"\d+"
    ).all()

    assert official[
        "nhl_team_id"
    ].nunique() == 32

    assert official[
        "nhl_abbrev"
    ].str.fullmatch(
        r"[A-Z]{3}"
    ).all()

    sources = set(
        df[
            "source"
        ].str.lower()
    )

    assert {
        "official_nhl",
        "dratings",
        "sportsbook",
    }.issubset(
        sources
    )


def test_transform_hockey_records_missing_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(
        tmp_path
    )

    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/transform_hockey.py"
    )

    no_map_records: list[
        dict
    ] = []

    result = module.normalize_team(
        "Mystery Expansion Club (10-2-1)",
        {},
        no_map_records,
        "fixture.csv",
    )

    assert (
        result
        == "Mystery Expansion Club"
    )

    assert (
        len(no_map_records)
        == 1
    )

    assert (
        no_map_records[0][
            "source_file"
        ]
        == "fixture.csv"
    )

    assert (
        no_map_records[0][
            "raw_team"
        ]
        == "Mystery Expansion Club (10-2-1)"
    )

    assert (
        no_map_records[0][
            "stripped_team"
        ]
        == "Mystery Expansion Club"
    )

    assert (
        no_map_records[0][
            "normalized_attempt"
        ]
        == "mystery expansion club"
    )


# ---------------------------------------------------------------------
# test_build_games.py coverage
# Regression fixtures: blank game IDs, duplicate games
# ---------------------------------------------------------------------

def test_build_games_accepts_valid_canonical_game_id(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/build_games.py"
    )

    module.LOG_PATH = (
        tmp_path
        / "build_games.log"
    )

    input_path = (
        tmp_path
        / "NHL_2026_01_01.csv"
    )

    write_reconciled_csv(
        input_path,
        [
            valid_reconciled_row()
        ],
    )

    game_date, rows = (
        module.read_reconciled_file(
            input_path,
            [],
        )
    )

    assert (
        game_date
        == "2026_01_01"
    )

    assert len(rows) == 1

    assert (
        rows[0][
            "game_id"
        ]
        == "2025020001"
    )


def test_build_games_rejects_blank_game_id(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/build_games.py"
    )

    module.LOG_PATH = (
        tmp_path
        / "build_games.log"
    )

    row = (
        valid_reconciled_row()
    )

    row[
        "game_id"
    ] = ""

    input_path = (
        tmp_path
        / "NHL_2026_01_01.csv"
    )

    write_reconciled_csv(
        input_path,
        [row],
    )

    with pytest.raises(
        SystemExit
    ):
        module.read_reconciled_file(
            input_path,
            [],
        )

    assert (
        "missing values for: game_id"
        in module.LOG_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_build_games_rejects_duplicate_official_game_id(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/build_games.py"
    )

    module.LOG_PATH = (
        tmp_path
        / "build_games.log"
    )

    first = (
        valid_reconciled_row(
            game_id="2025020001",
            sportsbook_event_id="book-1",
        )
    )

    second = (
        valid_reconciled_row(
            game_id="2025020001",
            sportsbook_event_id="book-2",
        )
    )

    second[
        "home_team"
    ] = "Toronto Maple Leafs"

    second[
        "away_team"
    ] = "Montreal Canadiens"

    input_path = (
        tmp_path
        / "NHL_2026_01_01.csv"
    )

    write_reconciled_csv(
        input_path,
        [
            first,
            second,
        ],
    )

    with pytest.raises(
        SystemExit
    ):
        module.read_reconciled_file(
            input_path,
            [],
        )

    assert (
        "duplicate official game_id: 2025020001"
        in module.LOG_PATH.read_text(
            encoding="utf-8"
        )
    )


# ---------------------------------------------------------------------
# test_transform_hockey_odds.py coverage
# Regression fixtures: ESPN market parsing and provider fallback
# ---------------------------------------------------------------------

def espn_provider_row(
    *,
    provider_id: str = "53",
    provider_name: str = "Titanbets",
    provider_priority: int = 0,
) -> dict:
    return {
        "espn_event_id": (
            "fixture_sb_20251008_001"
        ),
        "provider_id": (
            provider_id
        ),
        "provider_name": (
            provider_name
        ),
        "provider_priority": (
            provider_priority
        ),
        (
            "home_team_odds_current_"
            "money_line_american"
        ): 110,
        (
            "away_team_odds_current_"
            "money_line_american"
        ): -125,
        (
            "home_team_odds_current_"
            "money_line_decimal"
        ): 2.10,
        (
            "away_team_odds_current_"
            "money_line_decimal"
        ): 1.80,
        (
            "home_team_odds_current_"
            "point_spread_american"
        ): 1.5,
        (
            "away_team_odds_current_"
            "point_spread_american"
        ): -1.5,
        (
            "home_team_odds_current_"
            "point_spread_alternate_"
            "display_value"
        ): "+1.5",
        (
            "away_team_odds_current_"
            "point_spread_alternate_"
            "display_value"
        ): "-1.5",
        (
            "home_team_odds_current_"
            "spread_american"
        ): -167,
        (
            "away_team_odds_current_"
            "spread_american"
        ): 135,
        (
            "home_team_odds_current_"
            "spread_decimal"
        ): 1.60,
        (
            "away_team_odds_current_"
            "spread_decimal"
        ): 2.35,
        "current_total_american": 6.5,
        (
            "current_total_"
            "alternate_display_value"
        ): "6.5",
        "over_under": 6.5,
        "current_over_american": -110,
        "current_under_american": -110,
        "current_over_decimal": 1.91,
        "current_under_decimal": 1.91,
    }


def test_transform_hockey_odds_rejects_unsupported_total_only() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py"
    )

    low = (
        espn_provider_row()
    )

    low[
        "current_total_american"
    ] = 5.0

    low[
        "current_total_alternate_display_value"
    ] = "5.0"

    low[
        "over_under"
    ] = 5.0

    high = (
        espn_provider_row()
    )

    high[
        "current_total_american"
    ] = 8.5

    high[
        "current_total_alternate_display_value"
    ] = "8.5"

    high[
        "over_under"
    ] = 8.5

    assert (
        module.total_from_provider(
            low
        )
        is None
    )

    assert (
        module.total_from_provider(
            high
        )
        is None
    )

    assert (
        module.TOTAL_MIN
        == 5.5
    )

    assert (
        module.TOTAL_MAX
        == 7.5
    )


def test_transform_hockey_odds_parses_espn_main_lines() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py"
    )

    row = (
        espn_provider_row()
    )

    moneyline = (
        module.moneyline_from_provider(
            row
        )
    )

    puck_line = (
        module.puck_line_from_provider(
            row
        )
    )

    total = (
        module.total_from_provider(
            row
        )
    )

    assert moneyline == {
        "home_american": "+110",
        "away_american": "-125",
        "home_decimal": "2.1",
        "away_decimal": "1.8",
    }

    assert puck_line == {
        "home_line": "1.5",
        "away_line": "-1.5",
        "home_american": "-167",
        "away_american": "+135",
        "home_decimal": "1.59880239520958",
        "away_decimal": "2.35",
    }

    assert total == {
        "total": "6.5",
        "over_american": "-110",
        "under_american": "-110",
        "over_decimal": "1.90909090909091",
        "under_decimal": "1.90909090909091",
    }


def test_transform_hockey_odds_provider_priority_and_live_exclusion() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py"
    )

    rows = [
        espn_provider_row(
            provider_id="58",
            provider_name="ESPN BET",
            provider_priority=0,
        ),
        espn_provider_row(
            provider_id="41",
            provider_name="SugarHouse",
            provider_priority=0,
        ),
        espn_provider_row(
            provider_id="47",
            provider_name="MGM",
            provider_priority=0,
        ),
        espn_provider_row(
            provider_id="53",
            provider_name="Titanbets",
            provider_priority=0,
        ),
        espn_provider_row(
            provider_id="59",
            provider_name=(
                "ESPN Bet - Live Odds"
            ),
            provider_priority=-1,
        ),
    ]

    ordered = (
        module.ordered_provider_rows(
            rows
        )
    )

    assert [
        module.provider_name(
            row
        )
        for row in ordered
    ] == [
        "Titanbets",
        "MGM",
        "SugarHouse",
        "ESPN BET",
    ]


def test_transform_hockey_odds_selects_provider_independently_per_market() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py"
    )

    titanbets = (
        espn_provider_row(
            provider_id="53",
            provider_name="Titanbets",
            provider_priority=0,
        )
    )

    for key in (
        "current_total_american",
        "current_total_alternate_display_value",
        "over_under",
        "current_over_american",
        "current_under_american",
        "current_over_decimal",
        "current_under_decimal",
    ):
        titanbets.pop(
            key,
            None,
        )

    mgm = (
        espn_provider_row(
            provider_id="47",
            provider_name="MGM",
            provider_priority=0,
        )
    )

    event = {
        "id": (
            "fixture_sb_20251008_001"
        ),
        "status": "pending",
        "date": (
            "2025-10-08T23:00:00Z"
        ),
        "home": "Montreal Canadiens",
        "away": "Toronto Maple Leafs",
    }

    result = (
        module.build_row(
            event,
            [
                mgm,
                titanbets,
            ],
            (
                "2025-10-08T12:00:00"
                "-04:00"
            ),
            defaultdict(
                int
            ),
        )
    )

    assert (
        result[
            "moneyline_provider_name"
        ]
        == "Titanbets"
    )

    assert (
        result[
            "puck_line_provider_name"
        ]
        == "Titanbets"
    )

    assert (
        result[
            "total_provider_name"
        ]
        == "MGM"
    )

    assert (
        result[
            "odds_source"
        ]
        == "espn"
    )

    assert (
        result[
            "sportsbook_event_id"
        ]
        == "fixture_sb_20251008_001"
    )

    assert (
        result[
            "home_puck_line"
        ]
        == "1.5"
    )

    assert (
        result[
            "away_puck_line"
        ]
        == "-1.5"
    )

    assert (
        result[
            "total"
        ]
        == "6.5"
    )


# ---------------------------------------------------------------------
# test_pull_sdv.py coverage
# Regression fixture: opponent-adjusted ratings are strictly pregame
# ---------------------------------------------------------------------

def test_historical_team_strength_uses_only_prior_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    schedule = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020002",
                "2025020003",
            ],
            "season": [
                2025,
                2025,
                2025,
            ],
            "game_date": [
                "2025-10-01",
                "2025-10-03",
                "2025-10-05",
            ],
            "home_team_abbr": [
                "BOS",
                "NYR",
                "BOS",
            ],
            "away_team_abbr": [
                "NYR",
                "BOS",
                "NYR",
            ],
            "game_type": [
                "R",
                "R",
                "R",
            ],
        }
    )

    game_rates = module.pl.DataFrame(
        {
            "date": [
                module.date(
                    2025,
                    10,
                    1,
                ),
                module.date(
                    2025,
                    10,
                    1,
                ),
                module.date(
                    2025,
                    10,
                    3,
                ),
                module.date(
                    2025,
                    10,
                    3,
                ),
            ],
            "team": [
                "BOS",
                "NYR",
                "BOS",
                "NYR",
            ],
            "season": [
                2025,
                2025,
                2025,
                2025,
            ],
        }
    )

    monkeypatch.setattr(
        module.nhl,
        "team_game_xg_rates",
        lambda pbp, rating_schedule: game_rates,
    )

    seen_max_dates: list[
        object
    ] = []

    def fake_ratings(
        prior_rates,
    ):
        seen_max_dates.append(
            prior_rates[
                "date"
            ].max()
        )

        teams = sorted(
            set(
                prior_rates[
                    "team"
                ].to_list()
            )
        )

        return module.pl.DataFrame(
            {
                "season": [
                    2025
                    for _ in teams
                ],
                "team": teams,
                "adj_xgf": [
                    3.0
                    for _ in teams
                ],
                "adj_xga": [
                    2.5
                    for _ in teams
                ],
                "adj_xg_net": [
                    0.5
                    for _ in teams
                ],
                "adj_gf": [
                    3.1
                    for _ in teams
                ],
                "adj_ga": [
                    2.6
                    for _ in teams
                ],
                "off_rank": [
                    1
                    for _ in teams
                ],
                "def_rank": [
                    2
                    for _ in teams
                ],
                "net_rank": [
                    1
                    for _ in teams
                ],
                "net_z": [
                    1.0
                    for _ in teams
                ],
            }
        )

    monkeypatch.setattr(
        module,
        "ratings_from_game_rates",
        fake_ratings,
    )

    result = (
        module.historical_team_strength(
            schedule,
            module.pl.DataFrame(
                {
                    "fixture": [
                        1
                    ]
                }
            ),
        )
    )

    assert (
        result[
            "game_date"
        ].to_list()
        == [
            "2025-10-03",
            "2025-10-03",
            "2025-10-05",
            "2025-10-05",
        ]
    )

    assert seen_max_dates == [
        module.date(
            2025,
            10,
            1,
        ),
        module.date(
            2025,
            10,
            3,
        ),
    ]


def _team_strength_ratings_fixture(
    module,
):
    return module.pl.DataFrame(
        {
            "season": [
                2025,
                2025,
            ],
            "team": [
                "BOS",
                "NYR",
            ],
            "adj_xgf": [
                3.1,
                2.8,
            ],
            "adj_xga": [
                2.4,
                2.9,
            ],
            "adj_xg_net": [
                0.7,
                -0.1,
            ],
            "adj_gf": [
                3.2,
                2.7,
            ],
            "adj_ga": [
                2.5,
                3.0,
            ],
            "off_rank": [
                4,
                18,
            ],
            "def_rank": [
                6,
                20,
            ],
            "net_rank": [
                3,
                19,
            ],
            "net_z": [
                1.25,
                -0.20,
            ],
        }
    )


def test_current_team_strength_accepts_only_snapshot_before_pregame_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    monkeypatch.setattr(
        module,
        "official_schedule_cutoff_lookup",
        lambda: {},
    )

    ratings = (
        _team_strength_ratings_fixture(
            module
        )
    )

    schedule = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
            ],
            "game_date": [
                "2026-01-01",
            ],
            "game_time": [
                "19:00",
            ],
            "home_team_abbr": [
                "BOS",
            ],
            "away_team_abbr": [
                "NYR",
            ],
        }
    )

    safe = (
        module.normalize_team_strength_asof(
            ratings,
            schedule=schedule,
            generated_at_utc=module.datetime(
                2026,
                1,
                1,
                22,
                0,
                tzinfo=module.UTC,
            ),
        )
    )

    assert (
        safe.height
        == 2
    )

    assert set(
        safe[
            "game_id"
        ].to_list()
    ) == {
        "2025020001",
    }

    assert all(
        value
        == "2026-01-02T00:00:00Z"
        for value in safe[
            "pregame_cutoff_utc"
        ].to_list()
    )

    unsafe = (
        module.normalize_team_strength_asof(
            ratings,
            schedule=schedule,
            generated_at_utc=module.datetime(
                2026,
                1,
                2,
                0,
                0,
                tzinfo=module.UTC,
            ),
        )
    )

    assert unsafe.is_empty()


def test_historical_team_strength_stamps_strictly_before_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    schedule = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020002",
            ],
            "season": [
                2025,
                2025,
            ],
            "game_date": [
                "2025-10-01",
                "2025-10-03",
            ],
            "game_time": [
                "19:00",
                "19:00",
            ],
            "home_team_abbr": [
                "BOS",
                "NYR",
            ],
            "away_team_abbr": [
                "NYR",
                "BOS",
            ],
            "game_type": [
                "R",
                "R",
            ],
        }
    )

    game_rates = module.pl.DataFrame(
        {
            "date": [
                module.date(
                    2025,
                    10,
                    1,
                ),
                module.date(
                    2025,
                    10,
                    1,
                ),
            ],
            "team": [
                "BOS",
                "NYR",
            ],
            "season": [
                2025,
                2025,
            ],
        }
    )

    monkeypatch.setattr(
        module.nhl,
        "team_game_xg_rates",
        lambda pbp, rating_schedule: game_rates,
    )

    monkeypatch.setattr(
        module,
        "ratings_from_game_rates",
        lambda prior_rates: (
            _team_strength_ratings_fixture(
                module
            )
        ),
    )

    result = (
        module.historical_team_strength(
            schedule,
            module.pl.DataFrame(
                {
                    "fixture": [
                        1
                    ]
                }
            ),
        )
    )

    assert (
        result.height
        == 2
    )

    for row in result.to_dicts():
        cutoff = (
            module.parse_timestamp_utc(
                row[
                    "pregame_cutoff_utc"
                ]
            )
        )
        ratings_as_of = (
            module.parse_timestamp_utc(
                row[
                    "ratings_as_of_utc"
                ]
            )
        )

        assert (
            ratings_as_of
            < cutoff
        )


# ---------------------------------------------------------------------
# test_merge_intake.py coverage
# Regression fixture: SportsDataverse fatigue -> game-level features
# ---------------------------------------------------------------------

def test_merge_intake_builds_game_level_fatigue_features(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    module.FATIGUE_DIR = (
        tmp_path
        / "fatigue"
    )

    module.TEAM_MAP_PATH = (
        tmp_path
        / "team_map_nhl.csv"
    )

    module.LOG_FILE = (
        tmp_path
        / "merge_intake.log"
    )

    module.FATIGUE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            {
                "league": "nhl",
                "source": "official_nhl",
                "alias": "Boston Bruins",
                "canonical_team": (
                    "Boston Bruins"
                ),
                "nhl_team_id": "6",
                "nhl_abbrev": "BOS",
            },
            {
                "league": "nhl",
                "source": "official_nhl",
                "alias": (
                    "New York Rangers"
                ),
                "canonical_team": (
                    "New York Rangers"
                ),
                "nhl_team_id": "3",
                "nhl_abbrev": "NYR",
            },
        ]
    ).to_csv(
        module.TEAM_MAP_PATH,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "team": "BOS",
                "game_date": (
                    "2026-01-01"
                ),
                "days_rest": 2,
                "back_to_back": False,
                "games_in_4_days": 2,
                "three_in_four": False,
                "games_in_6_days": 3,
                "four_in_six": False,
                "games_in_7_days": 3,
            },
            {
                "team": "NYR",
                "game_date": (
                    "2026-01-01"
                ),
                "days_rest": 1,
                "back_to_back": True,
                "games_in_4_days": 3,
                "three_in_four": True,
                "games_in_6_days": 4,
                "four_in_six": True,
                "games_in_7_days": 4,
            },
        ]
    ).to_csv(
        (
            module.FATIGUE_DIR
            / "latest_fatigue.csv"
        ),
        index=False,
    )

    fatigue_index = (
        module.load_fatigue_index()
    )

    features = (
        module.fatigue_features_for_game(
            {
                "game_date": (
                    "2026_01_01"
                ),
                "home_team": (
                    "Boston Bruins"
                ),
                "away_team": (
                    "New York Rangers"
                ),
            },
            fatigue_index,
        )
    )

    assert features == {
        "home_days_rest": "2",
        "away_days_rest": "1",
        "home_back_to_back": "0",
        "away_back_to_back": "1",
        "home_games_in_4_days": "2",
        "away_games_in_4_days": "3",
        "home_three_in_four": "0",
        "away_three_in_four": "1",
        "home_games_in_6_days": "3",
        "away_games_in_6_days": "4",
        "home_four_in_six": "0",
        "away_four_in_six": "1",
        "home_games_in_7_days": "3",
        "away_games_in_7_days": "4",
        "rest_differential": "1",
    }


def test_merge_intake_leaves_missing_fatigue_blank() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    features = (
        module.fatigue_features_for_game(
            {
                "game_date": (
                    "2026_01_01"
                ),
                "home_team": (
                    "Boston Bruins"
                ),
                "away_team": (
                    "New York Rangers"
                ),
            },
            {},
        )
    )

    assert all(
        value == ""
        for value in (
            features.values()
        )
    )


def test_merge_intake_builds_game_level_team_strength_features(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    module.TEAM_STRENGTH_DIR = (
        tmp_path
        / "team-strength"
    )

    module.TEAM_MAP_PATH = (
        tmp_path
        / "team_map_nhl.csv"
    )

    module.LOG_FILE = (
        tmp_path
        / "merge_intake.log"
    )

    module.TEAM_STRENGTH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            {
                "league": "nhl",
                "source": "official_nhl",
                "alias": "Boston Bruins",
                "canonical_team": (
                    "Boston Bruins"
                ),
                "nhl_team_id": "6",
                "nhl_abbrev": "BOS",
            },
            {
                "league": "nhl",
                "source": "official_nhl",
                "alias": (
                    "New York Rangers"
                ),
                "canonical_team": (
                    "New York Rangers"
                ),
                "nhl_team_id": "3",
                "nhl_abbrev": "NYR",
            },
        ]
    ).to_csv(
        module.TEAM_MAP_PATH,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "game_id": "2025020001",
                "game_date": "2026-01-01",
                "team": "BOS",
                "pregame_cutoff_utc": "2026-01-02T00:00:00Z",
                "ratings_as_of_utc": "2026-01-01T22:00:00Z",
                "pregame_cutoff_source": "official_nhl_schedule",
                "adj_xgf": 3.10,
                "adj_xga": 2.40,
                "adj_xg_net": 0.70,
                "adj_gf": 3.20,
                "adj_ga": 2.50,
                "off_rank": 4,
                "def_rank": 6,
                "net_rank": 3,
                "net_z": 1.25,
            },
            {
                "game_id": "2025020001",
                "game_date": "2026-01-01",
                "team": "NYR",
                "pregame_cutoff_utc": "2026-01-02T00:00:00Z",
                "ratings_as_of_utc": "2026-01-01T22:00:00Z",
                "pregame_cutoff_source": "official_nhl_schedule",
                "adj_xgf": 2.80,
                "adj_xga": 2.90,
                "adj_xg_net": -0.10,
                "adj_gf": 2.70,
                "adj_ga": 3.00,
                "off_rank": 18,
                "def_rank": 20,
                "net_rank": 19,
                "net_z": -0.20,
            },
        ]
    ).to_csv(
        (
            module.TEAM_STRENGTH_DIR
            / "latest_team_ratings_asof.csv"
        ),
        index=False,
    )

    strength_index = (
        module.load_team_strength_index()
    )

    features = (
        module.team_strength_features_for_game(
            {
                "game_id": "2025020001",
                "game_date": "2026_01_01",
                "game_time": "19:00",
                "home_team": "Boston Bruins",
                "away_team": "New York Rangers",
            },
            strength_index,
        )
    )

    assert set(
        features
    ) == set(
        TEAM_STRENGTH_FEATURE_VALUES
    )

    for (
        column,
        expected,
    ) in TEAM_STRENGTH_FEATURE_VALUES.items():
        assert float(
            features[
                column
            ]
        ) == pytest.approx(
            float(expected),
            abs=1e-12,
        )


def test_merge_intake_leaves_missing_team_strength_blank() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    features = (
        module.team_strength_features_for_game(
            {
                "game_date": "2026_01_01",
                "home_team": "Boston Bruins",
                "away_team": "New York Rangers",
            },
            {},
        )
    )

    assert set(
        features
    ) == set(
        TEAM_STRENGTH_FEATURE_VALUES
    )

    assert all(
        value == ""
        for value in features.values()
    )



def test_merge_intake_rejects_team_strength_at_or_after_pregame_cutoff(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    module.TEAM_STRENGTH_DIR = (
        tmp_path
        / "team-strength"
    )
    module.TEAM_MAP_PATH = (
        tmp_path
        / "team_map_nhl.csv"
    )
    module.LOG_FILE = (
        tmp_path
        / "merge_intake.log"
    )

    module.TEAM_STRENGTH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            {
                "league": "nhl",
                "source": "official_nhl",
                "alias": "Boston Bruins",
                "canonical_team": "Boston Bruins",
                "nhl_team_id": "6",
                "nhl_abbrev": "BOS",
            },
        ]
    ).to_csv(
        module.TEAM_MAP_PATH,
        index=False,
    )

    row = {
        "game_id": "2025020001",
        "game_date": "2026-01-01",
        "team": "BOS",
        "pregame_cutoff_utc": "2026-01-02T00:00:00Z",
        "ratings_as_of_utc": "2026-01-02T00:00:00Z",
        "pregame_cutoff_source": "official_nhl_schedule",
        "adj_xgf": 3.10,
        "adj_xga": 2.40,
        "adj_xg_net": 0.70,
        "adj_gf": 3.20,
        "adj_ga": 2.50,
        "off_rank": 4,
        "def_rank": 6,
        "net_rank": 3,
        "net_z": 1.25,
    }

    pd.DataFrame(
        [
            row
        ]
    ).to_csv(
        (
            module.TEAM_STRENGTH_DIR
            / "latest_team_ratings_asof.csv"
        ),
        index=False,
    )

    with pytest.raises(
        SystemExit
    ):
        module.load_team_strength_index()


def test_merge_intake_rejects_cutoff_later_than_canonical_game_start() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    index = {
        (
            "2025020001",
            "Boston Bruins",
        ): {
            "game_id": "2025020001",
            "game_date": "2026_01_01",
            "team": "Boston Bruins",
            "pregame_cutoff_utc": "2026-01-02T00:30:00+00:00",
            "ratings_as_of_utc": "2026-01-01T22:00:00+00:00",
            "adj_xgf": "3.1",
            "adj_xga": "2.4",
            "adj_xg_net": "0.7",
            "adj_gf": "3.2",
            "adj_ga": "2.5",
            "off_rank": "4",
            "def_rank": "6",
            "net_rank": "3",
            "net_z": "1.25",
            "_source_file": "fixture.csv",
        },
    }

    with pytest.raises(
        SystemExit
    ):
        module.team_strength_features_for_game(
            {
                "game_id": "2025020001",
                "game_date": "2026_01_01",
                "game_time": "19:00",
                "home_team": "Boston Bruins",
                "away_team": "New York Rangers",
            },
            index,
        )


# ---------------------------------------------------------------------
# test_build_juice_files.py coverage
# Regression fixture: supplemental features survive Stage 01 market split
# ---------------------------------------------------------------------

def test_build_juice_files_preserves_fatigue_features_in_all_markets() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/build_juice_files.py"
    )

    expected = set(
        FATIGUE_FEATURE_VALUES
    )

    assert expected.issubset(
        module.MERGED_REQUIRED_COLUMNS
    )
    assert expected.issubset(
        module.MONEYLINE_COLUMNS
    )
    assert expected.issubset(
        module.PUCK_LINE_COLUMNS
    )
    assert expected.issubset(
        module.TOTAL_COLUMNS
    )


def test_build_juice_files_preserves_team_strength_features_in_all_markets() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/build_juice_files.py"
    )

    expected = set(
        TEAM_STRENGTH_FEATURE_VALUES
    )

    assert expected.issubset(
        module.MERGED_REQUIRED_COLUMNS
    )
    assert expected.issubset(
        module.MONEYLINE_COLUMNS
    )
    assert expected.issubset(
        module.PUCK_LINE_COLUMNS
    )
    assert expected.issubset(
        module.TOTAL_COLUMNS
    )


def test_stage02_juice_config_outputs_exist_and_have_calibration_adjustment() -> None:
    for path in JUICE_CONFIG_PATHS:
        assert path.is_file()

        df = pd.read_csv(
            path
        )

        assert not df.empty

        assert (
            "model_calibration_adjustment"
            in df.columns
        )

        values = pd.to_numeric(
            df[
                "model_calibration_adjustment"
            ],
            errors="coerce",
        )

        assert (
            values.notna().all()
        )

        assert (
            np.isfinite(
                values
            ).all()
        )


def test_current_juice_config_files_pass_validator(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/02_juice/validate_juice_config.py"
    )

    module.LOG_FILE = (
        tmp_path
        / "validate_juice_config.log"
    )

    module.main()

    assert (
        "STATUS: SUCCESS"
        in module.LOG_FILE.read_text(
            encoding="utf-8"
        )
    )


def test_invalid_calibration_adjustment_is_rejected(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/02_juice/validate_juice_config.py"
    )

    module.LOG_FILE = (
        tmp_path
        / "validate_juice_config.log"
    )

    invalid_path = (
        tmp_path
        / "invalid_moneyline_juice.csv"
    )

    pd.DataFrame(
        [
            {
                "band": "fixture",
                "band_min": -200,
                "band_max": -100,
                "fav_ud": "favorite",
                "venue": "home",
                (
                    "model_calibration_"
                    "adjustment"
                ): "not-a-number",
            }
        ]
    ).to_csv(
        invalid_path,
        index=False,
    )

    errors: list[
        str
    ] = []

    loaded = (
        module.load_config(
            invalid_path,
            module.MONEYLINE_REQUIRED,
            errors,
        )
    )

    assert (
        loaded
        is not None
    )

    assert any(
        (
            "INVALID NUMERIC VALUES"
            in message
        )
        for message in errors
    )


# ---------------------------------------------------------------------
# test_apply_juice.py coverage
# ---------------------------------------------------------------------

def _prepare_apply_module(
    module,
    tmp_path: Path,
) -> None:
    module.OUTPUT_DIR = (
        tmp_path
        / "out"
    )

    module.ERROR_DIR = (
        tmp_path
        / "errors"
    )

    module.LOG_FILE = (
        tmp_path
        / "apply_juice.log"
    )

    module.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def test_moneyline_juice_application_normalizes_probabilities(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/02_juice/apply_moneyline_juice.py"
    )

    _prepare_apply_module(
        module,
        tmp_path,
    )

    input_path = (
        tmp_path
        / "2026_01_01_NHL_moneyline.csv"
    )

    row = {
        col: ""
        for col
        in module.REQUIRED_INPUT_COLUMNS
    }

    row.update(
        {
            "sport": "hockey",
            "league": "nhl",
            "game_date": "2026_01_01",
            "game_time": "19:00",
            "game_id": "2025020001",
            "away_team": (
                "New York Rangers"
            ),
            "home_team": (
                "Boston Bruins"
            ),
            **FATIGUE_FEATURE_VALUES,
            **TEAM_STRENGTH_FEATURE_VALUES,
            "away_prob_moneyline": 0.55,
            "home_prob_moneyline": 0.45,
            "away_fair_decimal_moneyline": (
                1
                / 0.55
            ),
            "home_fair_decimal_moneyline": (
                1
                / 0.45
            ),
            "away_dk_moneyline_american": -120,
            "home_dk_moneyline_american": 110,
            "away_dk_moneyline_decimal": 1.83,
            "home_dk_moneyline_decimal": 2.10,
        }
    )

    pd.DataFrame(
        [row],
        columns=(
            module.REQUIRED_INPUT_COLUMNS
        ),
    ).to_csv(
        input_path,
        index=False,
    )

    juice_df = pd.DataFrame(
        [
            {
                "band": "away_favorite",
                "band_min": -200,
                "band_max": -100,
                "fav_ud": "favorite",
                "venue": "away",
                "model_calibration_adjustment": 0.05,
            },
            {
                "band": "home_underdog",
                "band_min": 100,
                "band_max": 200,
                "fav_ud": "underdog",
                "venue": "home",
                "model_calibration_adjustment": 0.02,
            },
        ]
    )

    (
        applied,
        skipped_bad,
        skipped_noband,
    ) = module.process_file(
        input_path,
        juice_df,
    )

    assert (
        applied,
        skipped_bad,
        skipped_noband,
    ) == (
        1,
        0,
        0,
    )

    output = pd.read_csv(
        module.OUTPUT_DIR
        / input_path.name
    )

    assert_fatigue_features_preserved(
        output.iloc[0]
    )

    assert_team_strength_features_preserved(
        output.iloc[0]
    )

    away = float(
        output.loc[
            0,
            "away_normalized_prob_moneyline",
        ]
    )

    home = float(
        output.loc[
            0,
            "home_normalized_prob_moneyline",
        ]
    )

    assert (
        away
        + home
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )

    assert (
        0.0
        < away
        < 1.0
    )

    assert (
        0.0
        < home
        < 1.0
    )


def test_puck_line_juice_application_normalizes_probabilities(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/02_juice/apply_puck_line_juice.py"
    )

    _prepare_apply_module(
        module,
        tmp_path,
    )

    input_path = (
        tmp_path
        / "2026_01_01_NHL_puck_line.csv"
    )

    row = {
        col: ""
        for col
        in module.REQUIRED_INPUT_COLUMNS
    }

    row.update(
        {
            "sport": "hockey",
            "league": "nhl",
            "game_date": "2026_01_01",
            "game_time": "19:00",
            "game_id": "2025020001",
            "away_team": (
                "New York Rangers"
            ),
            "home_team": (
                "Boston Bruins"
            ),
            **FATIGUE_FEATURE_VALUES,
            **TEAM_STRENGTH_FEATURE_VALUES,
            "away_puck_line": -1.5,
            "home_puck_line": 1.5,
            "away_prob_puck_line": 0.52,
            "home_prob_puck_line": 0.48,
            "away_fair_decimal_puck_line": 1.92,
            "home_fair_decimal_puck_line": 2.08,
            "away_dk_puck_line_american": 145,
            "home_dk_puck_line_american": -165,
            "away_dk_puck_line_decimal": 2.45,
            "home_dk_puck_line_decimal": 1.61,
        }
    )

    pd.DataFrame(
        [row],
        columns=(
            module.REQUIRED_INPUT_COLUMNS
        ),
    ).to_csv(
        input_path,
        index=False,
    )

    juice_df = pd.DataFrame(
        [
            {
                "band": "away_favorite",
                "band_min": -2.0,
                "band_max": -1.0,
                "venue": "away",
                "fav_ud": "favorite",
                "model_calibration_adjustment": 0.02,
            },
            {
                "band": "home_underdog",
                "band_min": 1.0,
                "band_max": 2.0,
                "venue": "home",
                "fav_ud": "underdog",
                "model_calibration_adjustment": 0.02,
            },
        ]
    )

    (
        applied,
        skipped_bad,
        skipped_noband,
    ) = module.process_file(
        input_path,
        juice_df,
    )

    assert (
        applied,
        skipped_bad,
        skipped_noband,
    ) == (
        1,
        0,
        0,
    )

    output = pd.read_csv(
        module.OUTPUT_DIR
        / input_path.name
    )

    assert_fatigue_features_preserved(
        output.iloc[0]
    )

    assert_team_strength_features_preserved(
        output.iloc[0]
    )

    away = float(
        output.loc[
            0,
            "away_normalized_prob_puck_line",
        ]
    )

    home = float(
        output.loc[
            0,
            "home_normalized_prob_puck_line",
        ]
    )

    assert (
        away
        + home
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )

    assert (
        0.0
        < away
        < 1.0
    )

    assert (
        0.0
        < home
        < 1.0
    )


def test_total_juice_application_normalizes_probabilities(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/02_juice/apply_total_juice.py"
    )

    _prepare_apply_module(
        module,
        tmp_path,
    )

    input_path = (
        tmp_path
        / "2026_01_01_NHL_total.csv"
    )

    row = {
        col: ""
        for col
        in module.REQUIRED_INPUT_COLUMNS
    }

    row.update(
        {
            "sport": "hockey",
            "league": "nhl",
            "game_date": "2026_01_01",
            "game_time": "19:00",
            "game_id": "2025020001",
            "away_team": (
                "New York Rangers"
            ),
            "home_team": (
                "Boston Bruins"
            ),
            **FATIGUE_FEATURE_VALUES,
            **TEAM_STRENGTH_FEATURE_VALUES,
            "total": 6.5,
            "total_projected_goals": 6.3,
            "over_prob_total": 0.48,
            "under_prob_total": 0.52,
            "over_fair_decimal_total": 2.08,
            "under_fair_decimal_total": 1.92,
            "dk_total_over_american": -105,
            "dk_total_under_american": -115,
            "dk_total_over_decimal": 1.95,
            "dk_total_under_decimal": 1.87,
        }
    )

    pd.DataFrame(
        [row],
        columns=(
            module.REQUIRED_INPUT_COLUMNS
        ),
    ).to_csv(
        input_path,
        index=False,
    )

    juice_df = pd.DataFrame(
        [
            {
                "band": "over_6_5",
                "band_min": 6.0,
                "band_max": 7.0,
                "side": "over",
                "model_calibration_adjustment": 0.02,
            },
            {
                "band": "under_6_5",
                "band_min": 6.0,
                "band_max": 7.0,
                "side": "under",
                "model_calibration_adjustment": 0.02,
            },
        ]
    )

    (
        applied,
        skipped_bad,
        skipped_noband,
    ) = module.process_file(
        input_path,
        juice_df,
    )

    assert (
        applied,
        skipped_bad,
        skipped_noband,
    ) == (
        1,
        0,
        0,
    )

    output = pd.read_csv(
        module.OUTPUT_DIR
        / input_path.name
    )

    assert_fatigue_features_preserved(
        output.iloc[0]
    )

    assert_team_strength_features_preserved(
        output.iloc[0]
    )

    over = float(
        output.loc[
            0,
            "over_normalized_prob_total",
        ]
    )

    under = float(
        output.loc[
            0,
            "under_normalized_prob_total",
        ]
    )

    assert (
        over
        + under
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )

    assert (
        0.0
        < over
        < 1.0
    )

    assert (
        0.0
        < under
        < 1.0
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/win/hockey/nhl/scripts/02_juice/apply_moneyline_juice.py",
        "docs/win/hockey/nhl/scripts/02_juice/apply_puck_line_juice.py",
        "docs/win/hockey/nhl/scripts/02_juice/apply_total_juice.py",
    ],
)
def test_apply_juice_scripts_preserve_fatigue_feature_contract(
    relative_path: str,
) -> None:
    module = load_repo_module(
        relative_path
    )

    expected = set(
        FATIGUE_FEATURE_VALUES
    )

    assert expected.issubset(
        module.REQUIRED_INPUT_COLUMNS
    )
    assert expected.issubset(
        module.OUTPUT_COLUMNS
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/win/hockey/nhl/scripts/02_juice/apply_moneyline_juice.py",
        "docs/win/hockey/nhl/scripts/02_juice/apply_puck_line_juice.py",
        "docs/win/hockey/nhl/scripts/02_juice/apply_total_juice.py",
    ],
)
def test_apply_juice_scripts_preserve_team_strength_feature_contract(
    relative_path: str,
) -> None:
    module = load_repo_module(
        relative_path
    )

    expected = set(
        TEAM_STRENGTH_FEATURE_VALUES
    )

    assert expected.issubset(
        module.REQUIRED_INPUT_COLUMNS
    )
    assert expected.issubset(
        module.OUTPUT_COLUMNS
    )


@pytest.mark.parametrize(
    "relative_path, config_filename",
    [
        (
            "docs/win/hockey/nhl/scripts/02_juice/apply_moneyline_juice.py",
            "nhl_moneyline_juice.csv",
        ),
        (
            "docs/win/hockey/nhl/scripts/02_juice/apply_puck_line_juice.py",
            "nhl_puck_line_juice.csv",
        ),
        (
            "docs/win/hockey/nhl/scripts/02_juice/apply_total_juice.py",
            "nhl_total_juice.csv",
        ),
    ],
)
def test_apply_juice_scripts_use_model_calibration_adjustment(
    relative_path: str,
    config_filename: str,
) -> None:
    module = load_repo_module(
        relative_path
    )

    assert (
        "model_calibration_adjustment"
        in module.REQUIRED_CONFIG_COLUMNS
    )

    assert (
        module.JUICE_FILE.name
        == config_filename
    )


# ---------------------------------------------------------------------
# test_compute_edges.py coverage
# ---------------------------------------------------------------------

def test_compute_edges_formula() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/03_edges/compute_edges.py"
    )

    result = (
        module.safe_edge_pct(
            pd.Series(
                [2.0]
            ),
            pd.Series(
                [0.60]
            ),
        )
    )

    assert (
        float(
            result.iloc[
                0
            ]
        )
        == pytest.approx(
            0.10,
            abs=1e-12,
        )
    )


@pytest.mark.parametrize(
    "decimal, probability",
    [
        (1.0, 0.60),
        (2.0, 0.0),
        (2.0, 1.0),
        (np.nan, 0.60),
        (2.0, np.nan),
    ],
)
def test_compute_edges_invalid_inputs_return_nan(
    decimal,
    probability,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/03_edges/compute_edges.py"
    )

    result = (
        module.safe_edge_pct(
            pd.Series(
                [decimal]
            ),
            pd.Series(
                [probability]
            ),
        )
    )

    assert pd.isna(
        result.iloc[
            0
        ]
    )


# ---------------------------------------------------------------------
# test_compute_ev_kelly.py coverage
# ---------------------------------------------------------------------

def test_compute_ev_and_kelly_formulas(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/03_edges/compute_ev_kelly.py"
    )

    module.LOG_FILE = (
        tmp_path
        / "compute_ev_kelly.log"
    )

    probability = pd.Series(
        [0.60]
    )

    decimal = pd.Series(
        [2.0]
    )

    ev = (
        module.compute_ev(
            probability,
            decimal,
        )
    )

    (
        kelly,
        negative_count,
    ) = module.compute_kelly(
        probability,
        decimal,
        "fixture.csv",
    )

    assert (
        float(
            ev.iloc[
                0
            ]
        )
        == pytest.approx(
            0.20,
            abs=1e-12,
        )
    )

    assert (
        float(
            kelly.iloc[
                0
            ]
        )
        == pytest.approx(
            0.20,
            abs=1e-12,
        )
    )

    assert (
        negative_count
        == 0
    )


def test_negative_kelly_is_clipped_to_zero(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/03_edges/compute_ev_kelly.py"
    )

    module.LOG_FILE = (
        tmp_path
        / "compute_ev_kelly.log"
    )

    (
        kelly,
        negative_count,
    ) = module.compute_kelly(
        pd.Series(
            [0.40]
        ),
        pd.Series(
            [2.0]
        ),
        "fixture.csv",
    )

    assert (
        float(
            kelly.iloc[
                0
            ]
        )
        == 0.0
    )

    assert (
        negative_count
        == 1
    )


# ---------------------------------------------------------------------
# test_markets_config.py coverage
# ---------------------------------------------------------------------

def test_current_markets_yaml_passes_validator(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/04_select/validate_markets_config.py"
    )

    module.LOG_FILE = (
        tmp_path
        / "validate_markets_config.log"
    )

    errors: list[
        str
    ] = []

    nhl_config = (
        module.load_config(
            errors
        )
    )

    assert (
        nhl_config
        is not None
    )

    module.validate_config(
        nhl_config,
        errors,
    )

    assert (
        errors
        == []
    )


def test_markets_validator_rejects_invalid_pick_preference(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/04_select/validate_markets_config.py"
    )

    module.LOG_FILE = (
        tmp_path
        / "validate_markets_config.log"
    )

    errors: list[
        str
    ] = []

    nhl_config = (
        module.load_config(
            errors
        )
    )

    assert (
        nhl_config
        is not None
    )

    assert (
        errors
        == []
    )

    invalid = (
        copy.deepcopy(
            nhl_config
        )
    )

    invalid[
        "moneyline"
    ][
        "pick_preference"
    ] = "not-valid"

    module.validate_config(
        invalid,
        errors,
    )

    assert any(
        (
            "pick_preference INVALID"
            in message
        )
        for message in errors
    )


# ---------------------------------------------------------------------
# test_select_bets.py coverage
# Regression fixtures: dual selections, blank game IDs
# ---------------------------------------------------------------------

def test_dual_moneyline_selections_allowed_when_pick_preference_all() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py"
    )

    rules = (
        permissive_side_rules()
    )

    config = {
        "moneyline": {
            "enabled": True,
            "pick_preference": "all",
            "home": copy.deepcopy(
                rules
            ),
            "away": copy.deepcopy(
                rules
            ),
        }
    }

    rejections: dict = {}

    selected = (
        module.process_moneyline(
            synthetic_moneyline_row(),
            config,
            "fixture_slate",
            rejections,
        )
    )

    assert (
        len(selected)
        == 2
    )

    assert {
        row[
            "bet_side"
        ]
        for row in selected
    } == {
        "home",
        "away",
    }

    assert (
        rejections
        == {}
    )


def test_best_ev_reduces_dual_moneyline_selection_to_one() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py"
    )

    rules = (
        permissive_side_rules()
    )

    config = {
        "moneyline": {
            "enabled": True,
            "pick_preference": "best_ev",
            "home": copy.deepcopy(
                rules
            ),
            "away": copy.deepcopy(
                rules
            ),
        }
    }

    row = (
        synthetic_moneyline_row()
    )

    row[
        "home_ev_moneyline"
    ] = 0.08

    row[
        "away_ev_moneyline"
    ] = 0.03

    rejections: dict = {}

    selected = (
        module.process_moneyline(
            row,
            config,
            "fixture_slate",
            rejections,
        )
    )

    assert (
        len(selected)
        == 1
    )

    assert (
        selected[0][
            "bet_side"
        ]
        == "home"
    )

    assert (
        rejections[
            (
                "2026_01_01",
                "moneyline",
                "away",
                "pick_preference",
            )
        ]
        == 1
    )


def test_selector_rejects_blank_game_id(
    tmp_path: Path,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py"
    )

    module.INPUT_DIR = (
        tmp_path
    )

    module.LOG_FILE = (
        tmp_path
        / "hockey_select_bets.log"
    )

    input_path = (
        tmp_path
        / "fixture_moneyline.csv"
    )

    pd.DataFrame(
        [
            {
                "game_id": "",
                "dummy": "value",
            }
        ]
    ).to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        SystemExit
    ):
        module.read_market_file(
            input_path,
            "moneyline",
        )

    assert (
        "blank game_id"
        in module.LOG_FILE.read_text(
            encoding="utf-8"
        )
    )


# ---------------------------------------------------------------------
# test_final_scores.py coverage
# Regression fixtures: unmatched grading rows, invalid calibration outputs
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "row, expected",
    [
        (
            {
                "market_type": "moneyline",
                "bet_side": "home",
                "away_score": 2,
                "home_score": 4,
            },
            "Win",
        ),
        (
            {
                "market_type": "puck_line",
                "bet_side": "away",
                "line": 1.5,
                "away_score": 3,
                "home_score": 4,
                "away_puck_line_result": -1,
            },
            "Win",
        ),
        (
            {
                "market_type": "total",
                "bet_side": "under",
                "line": 6.5,
                "away_score": 2,
                "home_score": 3,
                "total_score": 5,
            },
            "Win",
        ),
        (
            {
                "market_type": "total",
                "bet_side": "over",
                "line": 6.0,
                "away_score": 2,
                "home_score": 4,
                "total_score": 6,
            },
            "Push",
        ),
    ],
)
def test_final_score_outcome_logic(
    row: dict,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(
        tmp_path
    )

    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/01_nhl_results_grade.py"
    )

    module.GRADE_ERROR_LOG = (
        tmp_path
        / "grade_errors.log"
    )

    module.GRADE_SUMMARY_LOG = (
        tmp_path
        / "grade_summary.log"
    )

    assert (
        module.determine_outcome(
            row
        )
        == expected
    )


def test_unmatched_grading_row_is_preserved_as_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(
        tmp_path
    )

    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/01_nhl_results_grade.py"
    )

    module.GRADE_ERROR_LOG = (
        tmp_path
        / "grade_errors.log"
    )

    module.GRADE_SUMMARY_LOG = (
        tmp_path
        / "grade_summary.log"
    )

    bets = pd.DataFrame(
        [
            {
                "game_id": "2025020001",
                "game_date": "2026_01_01",
                "market_type": "moneyline",
                "bet_side": "home",
            }
        ]
    )

    scores = pd.DataFrame(
        columns=[
            "game_id"
        ]
    )

    statuses = pd.DataFrame(
        columns=[
            "game_id"
        ]
    )

    (
        graded,
        pending,
        unresolved,
    ) = module.grade_rows(
        bets,
        scores,
        statuses,
    )

    assert (
        graded.empty
    )

    assert (
        pending.empty
    )

    assert (
        len(unresolved)
        == 1
    )

    assert (
        unresolved.iloc[
            0
        ][
            "game_id"
        ]
        == "2025020001"
    )

    assert (
        unresolved.iloc[
            0
        ][
            "unresolved_reason"
        ]
        == "official_game_status_missing"
    )

    assert (
        len(graded)
        + len(pending)
        + len(unresolved)
        == len(bets)
    )


def _load_results_report_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(
        tmp_path
    )

    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/03_nhl_results_reports.py"
    )

    module.CALIBRATION_DIR = (
        tmp_path
        / "calibration"
    )

    module.CALIBRATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.SUMMARY_LOG = (
        tmp_path
        / "report_summary.log"
    )

    module.ERROR_LOG = (
        tmp_path
        / "report_errors.log"
    )

    return module


@pytest.mark.parametrize(
    "invalid_probability",
    [
        np.nan,
        -0.01,
        1.01,
    ],
)
def test_calibration_rejects_invalid_model_probability(
    invalid_probability,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = (
        _load_results_report_module(
            tmp_path,
            monkeypatch,
        )
    )

    df = pd.DataFrame(
        [
            {
                "game_date": "2026_01_01",
                "game_id": "2025020001",
                "market_type": "moneyline",
                "bet_side": "home",
                "model_prob": (
                    invalid_probability
                ),
                "bet_result": "Win",
            }
        ]
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "CALIBRATION BLOCKED"
        ),
    ):
        module.prepare_calibration_df(
            df
        )


def test_calibration_excludes_pushes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = (
        _load_results_report_module(
            tmp_path,
            monkeypatch,
        )
    )

    prepared = (
        module.prepare_df(
            calibration_input_rows()
        )
    )

    calibration = (
        module.prepare_calibration_df(
            prepared
        )
    )

    assert (
        len(calibration)
        == 2
    )

    assert set(
        calibration[
            "bet_result"
        ].str.lower()
    ) == {
        "win",
        "loss",
    }

    assert (
        "push"
        not in set(
            calibration[
                "bet_result"
            ].str.lower()
        )
    )


def test_calibration_metrics_brier_log_loss_expected_vs_realized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = (
        _load_results_report_module(
            tmp_path,
            monkeypatch,
        )
    )

    prepared = (
        module.prepare_df(
            calibration_input_rows()
        )
    )

    calibration = (
        module.prepare_calibration_df(
            prepared
        )
    )

    row = (
        module.calibration_metric_row(
            calibration,
            "ALL",
        )
    )

    assert (
        row[
            "bets"
        ]
        == 2
    )

    assert (
        row[
            "expected_win_rate"
        ]
        == pytest.approx(
            0.50,
            abs=1e-12,
        )
    )

    assert (
        row[
            "realized_win_rate"
        ]
        == pytest.approx(
            0.50,
            abs=1e-12,
        )
    )

    assert (
        row[
            "calibration_gap"
        ]
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        row[
            "brier_score"
        ]
        == pytest.approx(
            0.16,
            abs=1e-12,
        )
    )

    assert (
        row[
            "log_loss"
        ]
        == pytest.approx(
            -np.log(
                0.60
            ),
            abs=1e-12,
        )
    )

    assert (
        row[
            "expected_wins"
        ]
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )

    assert (
        row[
            "realized_wins"
        ]
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )


def test_calibration_writes_all_required_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = (
        _load_results_report_module(
            tmp_path,
            monkeypatch,
        )
    )

    prepared = (
        module.prepare_df(
            calibration_input_rows()
        )
    )

    module.write_calibration_reports(
        prepared
    )

    expected = {
        "nhl_calibration_metrics.csv": (
            module.CALIBRATION_METRICS_COLUMNS
        ),
        "nhl_probability_calibration.csv": (
            module.PROBABILITY_CALIBRATION_COLUMNS
        ),
        "nhl_expected_vs_realized.csv": (
            module.EXPECTED_VS_REALIZED_COLUMNS
        ),
        "nhl_walk_forward_performance.csv": (
            module.WALK_FORWARD_COLUMNS
        ),
    }

    for (
        filename,
        columns,
    ) in expected.items():
        path = (
            module.CALIBRATION_DIR
            / filename
        )

        assert (
            path.is_file()
        ), filename

        report = (
            pd.read_csv(
                path
            )
        )

        assert (
            list(
                report.columns
            )
            == list(
                columns
            )
        )


def test_walk_forward_performance_is_cumulative_expanding_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = (
        _load_results_report_module(
            tmp_path,
            monkeypatch,
        )
    )

    prepared = (
        module.prepare_df(
            calibration_input_rows()
        )
    )

    calibration = (
        module.prepare_calibration_df(
            prepared
        )
    )

    module.write_walk_forward_performance(
        prepared,
        calibration,
    )

    report = (
        pd.read_csv(
            module.CALIBRATION_DIR
            / "nhl_walk_forward_performance.csv"
        )
    )

    all_rows = report[
        report[
            "market_type"
        ]
        == "ALL"
    ].reset_index(
        drop=True
    )

    assert list(
        all_rows[
            "through_game_date"
        ]
    ) == [
        "2026_01_01",
        "2026_01_02",
    ]

    first = (
        all_rows.iloc[
            0
        ]
    )

    second = (
        all_rows.iloc[
            1
        ]
    )

    assert (
        int(
            first[
                "Win"
            ]
        )
        == 1
    )

    assert (
        int(
            first[
                "Loss"
            ]
        )
        == 0
    )

    assert (
        int(
            first[
                "Push"
            ]
        )
        == 0
    )

    assert (
        int(
            first[
                "bets_excluding_pushes"
            ]
        )
        == 1
    )

    assert (
        int(
            first[
                "bets_including_pushes"
            ]
        )
        == 1
    )

    assert (
        int(
            second[
                "Win"
            ]
        )
        == 1
    )

    assert (
        int(
            second[
                "Loss"
            ]
        )
        == 1
    )

    assert (
        int(
            second[
                "Push"
            ]
        )
        == 1
    )

    assert (
        int(
            second[
                "bets_excluding_pushes"
            ]
        )
        == 2
    )

    assert (
        int(
            second[
                "bets_including_pushes"
            ]
        )
        == 3
    )

    assert (
        float(
            second[
                "expected_win_rate"
            ]
        )
        == pytest.approx(
            0.50,
            abs=1e-12,
        )
    )

    assert (
        float(
            second[
                "realized_win_rate"
            ]
        )
        == pytest.approx(
            0.50,
            abs=1e-12,
        )
    )


def test_reporting_blocks_unresolved_grading_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = (
        _load_results_report_module(
            tmp_path,
            monkeypatch,
        )
    )

    module.UNRESOLVED_FILE = (
        tmp_path
        / "unresolved.csv"
    )

    pd.DataFrame(
        [
            {
                "game_id": "2025020001",
                "unresolved_reason": (
                    "official_game_status_missing"
                ),
            }
        ]
    ).to_csv(
        module.UNRESOLVED_FILE,
        index=False,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "FINAL REPORTING BLOCKED"
        ),
    ):
        module.fail_if_unresolved_rows_exist()

# ---------------------------------------------------------------------
# SDV-P3 goalie GSAx / strict-as-of coverage
# ---------------------------------------------------------------------

def test_pull_sdv_goalie_features_use_only_prior_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    monkeypatch.setattr(
        module,
        "official_schedule_cutoff_lookup",
        lambda: {},
    )

    schedule = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020002",
            ],
            "game_date": [
                "2026-01-01",
                "2026-01-03",
            ],
            "game_time": [
                "19:00",
                "19:00",
            ],
            "home_team_abbr": [
                "BOS",
                "BOS",
            ],
            "away_team_abbr": [
                "NYR",
                "NYR",
            ],
        }
    )

    pbp = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020001",
                "2025020002",
                "2025020002",
            ],
            "home_abbr": [
                "BOS",
                "BOS",
                "BOS",
                "BOS",
            ],
            "away_abbr": [
                "NYR",
                "NYR",
                "NYR",
                "NYR",
            ],
            "home_goalie_id": [
                10,
                10,
                11,
                11,
            ],
            "away_goalie_id": [
                20,
                20,
                21,
                21,
            ],
            "home_goalie": [
                "Prior Boston",
                "Prior Boston",
                "Target Boston",
                "Target Boston",
            ],
            "away_goalie": [
                "Prior New York",
                "Prior New York",
                "Target New York",
                "Target New York",
            ],
        }
    )

    seen_game_ids: list[
        list[str]
    ] = []

    def fake_gsax(
        prior_pbp,
        _prior_shifts,
    ):
        ids = sorted(
            set(
                prior_pbp[
                    "game_id"
                ].cast(
                    module.pl.Utf8
                ).to_list()
            )
        )

        seen_game_ids.append(
            ids
        )

        return module.pl.DataFrame(
            {
                "player_id": [
                    10,
                    20,
                ],
                "goalie": [
                    "Prior Boston",
                    "Prior New York",
                ],
                "shots": [
                    30,
                    30,
                ],
                "xga": [
                    2.5,
                    3.0,
                ],
                "ga": [
                    2,
                    3,
                ],
                "gsax": [
                    0.5,
                    0.0,
                ],
                "gsax_per_60": [
                    0.5,
                    0.0,
                ],
            }
        )

    monkeypatch.setattr(
        module.nhl,
        "nhl_goalie_gsax",
        fake_gsax,
    )

    result = (
        module.build_game_goalie_features_asof(
            schedule,
            pbp,
            module.pl.DataFrame(),
            current=False,
            generated_at_utc=module.datetime(
                2026,
                1,
                3,
                20,
                0,
                tzinfo=module.UTC,
            ),
        )
    )

    target = (
        result.filter(
            module.pl.col(
                "game_id"
            )
            == "2025020002"
        )
        .to_dicts()[
            0
        ]
    )

    assert seen_game_ids == [
        [
            "2025020001",
        ]
    ]

    assert (
        target[
            "home_expected_starter"
        ]
        == "Prior Boston"
    )

    assert (
        target[
            "away_expected_starter"
        ]
        == "Prior New York"
    )

    assert (
        target[
            "home_goalie_status"
        ]
        == "projected"
    )

    assert (
        target[
            "away_goalie_status"
        ]
        == "projected"
    )

    assert float(
        target[
            "starter_gsax_differential"
        ]
    ) == pytest.approx(
        0.5,
        abs=1e-12,
    )

    assert (
        target[
            "goalie_decision_cutoff_utc"
        ]
        == "2026-01-03T23:00:00Z"
    )

    assert (
        target[
            "goalie_snapshot_as_of_utc"
        ]
        == "2026-01-03T23:00:00Z"
    )

    assert (
        target[
            "home_goalie_status_observed_at"
        ]
        == "2026-01-03T23:00:00Z"
    )


def test_pull_sdv_rejects_current_goalie_snapshot_after_t60_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    monkeypatch.setattr(
        module,
        "official_schedule_cutoff_lookup",
        lambda: {},
    )

    schedule = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
            ],
            "game_date": [
                "2026-01-01",
            ],
            "game_time": [
                "19:00",
            ],
            "home_team_abbr": [
                "BOS",
            ],
            "away_team_abbr": [
                "NYR",
            ],
        }
    )

    pbp = module.pl.DataFrame(
        {
            "game_id": [
                "2024010001",
                "2024010001",
            ],
            "game_date": [
                "2025-12-30",
                "2025-12-30",
            ],
            "home_abbr": [
                "BOS",
                "BOS",
            ],
            "away_abbr": [
                "NYR",
                "NYR",
            ],
            "home_goalie_id": [
                10,
                10,
            ],
            "away_goalie_id": [
                20,
                20,
            ],
            "home_goalie": [
                "Prior Boston",
                "Prior Boston",
            ],
            "away_goalie": [
                "Prior New York",
                "Prior New York",
            ],
        }
    )

    monkeypatch.setattr(
        module.nhl,
        "nhl_goalie_gsax",
        lambda prior_pbp, prior_shifts: (
            module.pl.DataFrame(
                {
                    "player_id": [
                        10,
                        20,
                    ],
                    "goalie": [
                        "Prior Boston",
                        "Prior New York",
                    ],
                    "shots": [
                        30,
                        30,
                    ],
                    "xga": [
                        2.5,
                        2.5,
                    ],
                    "ga": [
                        2,
                        2,
                    ],
                    "gsax": [
                        0.5,
                        0.5,
                    ],
                    "gsax_per_60": [
                        0.5,
                        0.5,
                    ],
                }
            )
        ),
    )

    accepted = (
        module.build_game_goalie_features_asof(
            schedule,
            pbp,
            module.pl.DataFrame(),
            current=True,
            generated_at_utc=module.datetime(
                2026,
                1,
                1,
                23,
                0,
                tzinfo=module.UTC,
            ),
        )
    )

    assert (
        accepted.height
        == 1
    )

    rejected = (
        module.build_game_goalie_features_asof(
            schedule,
            pbp,
            module.pl.DataFrame(),
            current=True,
            generated_at_utc=module.datetime(
                2026,
                1,
                1,
                23,
                0,
                1,
                tzinfo=module.UTC,
            ),
        )
    )

    assert rejected.is_empty()


def test_merge_intake_builds_strict_asof_goalie_features() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    index = {
        "2025020001": {
            "game_id": "2025020001",
            "game_date": "2026_01_01",
            "home_team": "Boston Bruins",
            "away_team": "New York Rangers",
            "pregame_cutoff_utc": "2026-01-02T00:00:00+00:00",
            "goalie_decision_cutoff_utc": "2026-01-01T23:00:00+00:00",
            "goalie_decision_cutoff_source": "fixed_60_minutes_before_puck_drop",
            "goalie_snapshot_as_of_utc": "2026-01-01T22:00:00+00:00",
            "home_expected_starter": "Jeremy Swayman",
            "away_expected_starter": "Igor Shesterkin",
            "home_starter_gsax": "4.25",
            "away_starter_gsax": "2.75",
            "home_backup_gsax": "0.5",
            "away_backup_gsax": "-0.25",
            "starter_gsax_differential": "1.5",
            "home_goalie_status": "projected",
            "away_goalie_status": "projected",
            "home_goalie_status_observed_at": "2026-01-01T22:00:00+00:00",
            "away_goalie_status_observed_at": "2026-01-01T22:00:00+00:00",
            "home_goalie_status_source": "sportsdataverse_prior_goalie_usage_projection",
            "away_goalie_status_source": "sportsdataverse_prior_goalie_usage_projection",
            "_source_file": "fixture.csv",
        }
    }

    features = (
        module.goalie_features_for_game(
            {
                "game_id": "2025020001",
                "game_date": "2026_01_01",
                "game_time": "19:00",
                "home_team": "Boston Bruins",
                "away_team": "New York Rangers",
            },
            index,
        )
    )

    assert (
        features[
            "home_expected_starter"
        ]
        == "Jeremy Swayman"
    )

    assert float(
        features[
            "starter_gsax_differential"
        ]
    ) == pytest.approx(
        1.5,
        abs=1e-12,
    )


def test_merge_intake_accepts_goalie_observation_at_t60_cutoff() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    index = {
        "2025020001": {
            "game_id": "2025020001",
            "game_date": "2026_01_01",
            "home_team": "Boston Bruins",
            "away_team": "New York Rangers",
            "pregame_cutoff_utc": "2026-01-02T00:00:00+00:00",
            "goalie_decision_cutoff_utc": "2026-01-01T23:00:00+00:00",
            "goalie_decision_cutoff_source": "fixed_60_minutes_before_puck_drop",
            "goalie_snapshot_as_of_utc": "2026-01-01T23:00:00+00:00",
            "home_expected_starter": "Jeremy Swayman",
            "away_expected_starter": "Igor Shesterkin",
            "home_starter_gsax": "4.25",
            "away_starter_gsax": "2.75",
            "home_backup_gsax": "0.5",
            "away_backup_gsax": "-0.25",
            "starter_gsax_differential": "1.5",
            "home_goalie_status": "expected",
            "away_goalie_status": "expected",
            "home_goalie_status_observed_at": "2026-01-01T23:00:00+00:00",
            "away_goalie_status_observed_at": "2026-01-01T23:00:00+00:00",
            "home_goalie_status_source": "fixture_expected",
            "away_goalie_status_source": "fixture_expected",
            "_source_file": "fixture.csv",
        }
    }

    features = module.goalie_features_for_game(
        {
            "game_id": "2025020001",
            "game_date": "2026_01_01",
            "game_time": "19:00",
            "home_team": "Boston Bruins",
            "away_team": "New York Rangers",
        },
        index,
    )

    assert (
        features[
            "home_goalie_status"
        ]
        == "expected"
    )


def test_merge_intake_rejects_goalie_observation_after_t60_cutoff() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    index = {
        "2025020001": {
            "game_id": "2025020001",
            "game_date": "2026_01_01",
            "home_team": "Boston Bruins",
            "away_team": "New York Rangers",
            "pregame_cutoff_utc": "2026-01-02T00:00:00+00:00",
            "goalie_decision_cutoff_utc": "2026-01-01T23:00:00+00:00",
            "goalie_decision_cutoff_source": "fixed_60_minutes_before_puck_drop",
            "goalie_snapshot_as_of_utc": "2026-01-01T22:00:00+00:00",
            "home_expected_starter": "Jeremy Swayman",
            "away_expected_starter": "Igor Shesterkin",
            "home_starter_gsax": "4.25",
            "away_starter_gsax": "2.75",
            "home_backup_gsax": "0.5",
            "away_backup_gsax": "-0.25",
            "starter_gsax_differential": "1.5",
            "home_goalie_status": "confirmed",
            "away_goalie_status": "projected",
            "home_goalie_status_observed_at": "2026-01-01T23:00:01+00:00",
            "away_goalie_status_observed_at": "2026-01-01T22:00:00+00:00",
            "home_goalie_status_source": "fixture_confirmation",
            "away_goalie_status_source": "fixture_projection",
            "_source_file": "fixture.csv",
        }
    }

    with pytest.raises(
        SystemExit
    ):
        module.goalie_features_for_game(
            {
                "game_id": "2025020001",
                "game_date": "2026_01_01",
                "game_time": "19:00",
                "home_team": "Boston Bruins",
                "away_team": "New York Rangers",
            },
            index,
        )


def test_build_juice_files_preserves_goalie_features_in_all_markets() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/build_juice_files.py"
    )

    expected = set(
        GOALIE_FEATURE_VALUES
    )

    assert expected.issubset(
        module.MERGED_REQUIRED_COLUMNS
    )

    assert expected.issubset(
        module.MONEYLINE_COLUMNS
    )

    assert expected.issubset(
        module.PUCK_LINE_COLUMNS
    )

    assert expected.issubset(
        module.TOTAL_COLUMNS
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/win/hockey/nhl/scripts/02_juice/apply_moneyline_juice.py",
        "docs/win/hockey/nhl/scripts/02_juice/apply_puck_line_juice.py",
        "docs/win/hockey/nhl/scripts/02_juice/apply_total_juice.py",
    ],
)
def test_apply_juice_scripts_preserve_goalie_feature_contract(
    relative_path: str,
) -> None:
    module = load_repo_module(
        relative_path
    )

    expected = set(
        GOALIE_FEATURE_VALUES
    )

    assert expected.issubset(
        module.REQUIRED_INPUT_COLUMNS
    )

    assert expected.issubset(
        module.OUTPUT_COLUMNS
    )



# ---------------------------------------------------------------------
# SDV-P4 lineup/player/unit/special-teams strict-as-of coverage
# ---------------------------------------------------------------------

def test_sdv_pregame_feature_evaluation_covers_required_families() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    evaluation = module.pregame_feature_evaluation()

    assert (
        evaluation[
            "evaluation_status"
        ]
        == "VERIFIED"
    )

    by_function = {
        row["function"]: row
        for row in evaluation[
            "families"
        ]
    }

    assert {
        "nhl_xg",
        "nhl_goalie_gsax",
        "nhl_skater_rapm",
        "nhl_skater_war",
        "nhl_special_teams_value",
        "nhl_unit_ratings",
        "nhl_penalty_value",
        "nhl_faceoff_value",
        "nhl_edge_skating_value",
        "nhl_expected_assists",
        "nhl_zone_transitions",
    }.issubset(
        by_function
    )

    assert (
        by_function[
            "nhl_skater_rapm"
        ][
            "decision"
        ]
        == "production"
    )

    assert (
        by_function[
            "nhl_penalty_value"
        ][
            "decision"
        ]
        == "evaluated_not_selected"
    )

    assert (
        by_function[
            "nhl_edge_skating_value"
        ][
            "decision"
        ]
        == "research_only_not_production_safe"
    )

    assert (
        by_function[
            "nhl_edge_skating_value"
        ][
            "as_of_capability"
        ]
        == "not_reconstructable_at_t60"
    )

    assert (
        by_function[
            "nhl_expected_assists"
        ][
            "decision"
        ]
        == "research_only"
    )

    assert (
        by_function[
            "nhl_expected_assists"
        ][
            "as_of_capability"
        ]
        == "prior_pbp_safe"
    )

    assert (
        by_function[
            "nhl_zone_transitions"
        ][
            "decision"
        ]
        == "research_only"
    )

    assert (
        by_function[
            "nhl_zone_transitions"
        ][
            "as_of_capability"
        ]
        == "prior_pbp_safe"
    )

    assert all(
        row[
            "evaluation_status"
        ]
        == "VERIFIED"
        for row in by_function.values()
    )


def test_sdv_research_microstat_input_excludes_target_and_later_games() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    schedule = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020002",
                "2025020003",
            ],
            "game_date": [
                "2026-01-01",
                "2026-01-03",
                "2026-01-05",
            ],
            "home_team_abbr": [
                "BOS",
                "BOS",
                "BOS",
            ],
            "away_team_abbr": [
                "NYR",
                "NYR",
                "NYR",
            ],
        }
    )

    pbp = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020002",
                "2025020003",
            ],
            "game_date": [
                "2026-01-01",
                "2026-01-03",
                "2026-01-05",
            ],
            "type_desc_key": [
                "goal",
                "goal",
                "goal",
            ],
        }
    )

    prior = (
        module.research_pbp_strictly_before_target(
            schedule,
            pbp,
            target_day=module.date(
                2026,
                1,
                3,
            ),
        )
    )

    assert (
        prior[
            "game_id"
        ]
        .cast(
            module.pl.Utf8
        )
        .to_list()
        == [
            "2025020001",
        ]
    )


def test_pull_sdv_lineup_features_use_only_prior_games_and_t60(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    monkeypatch.setattr(
        module,
        "official_schedule_cutoff_lookup",
        lambda: {},
    )

    schedule = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020002",
            ],
            "game_date": [
                "2026-01-01",
                "2026-01-03",
            ],
            "game_time": [
                "19:00",
                "19:00",
            ],
            "home_team_abbr": [
                "BOS",
                "BOS",
            ],
            "away_team_abbr": [
                "NYR",
                "NYR",
            ],
        }
    )

    pbp = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020002",
            ],
            "game_date": [
                "2026-01-01",
                "2026-01-03",
            ],
        }
    )

    shifts = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
                "2025020002",
            ],
            "game_date": [
                "2026-01-01",
                "2026-01-03",
            ],
        }
    )

    seen: list[
        list[str]
    ] = []

    def fake_metrics(
        prior_pbp,
        _prior_shifts,
    ):
        ids = sorted(
            prior_pbp[
                "game_id"
            ].cast(
                module.pl.Utf8
            ).to_list()
        ) if not prior_pbp.is_empty() else []

        seen.append(
            ids
        )

        return {
            "BOS": {
                "skater_rapm": 0.35,
                "skater_war": 5.5,
                "pp_value": 1.2,
                "pk_value": 0.8,
                "forward_line_strength": 0.6,
                "defense_pair_strength": 0.45,
            },
            "NYR": {
                "skater_rapm": 0.10,
                "skater_war": 4.0,
                "pp_value": 0.75,
                "pk_value": 0.30,
                "forward_line_strength": 0.40,
                "defense_pair_strength": 0.25,
            },
        }

    monkeypatch.setattr(
        module,
        "compute_lineup_team_metrics",
        fake_metrics,
    )

    result = (
        module.build_game_lineup_features_asof(
            schedule,
            pbp,
            shifts,
            current=False,
            generated_at_utc=module.datetime(
                2026,
                1,
                3,
                20,
                0,
                tzinfo=module.UTC,
            ),
        )
    )

    target = (
        result.filter(
            module.pl.col(
                "game_id"
            )
            == "2025020002"
        )
        .to_dicts()[
            0
        ]
    )

    assert seen == [
        [],
        [
            "2025020001",
        ],
    ]

    assert (
        target[
            "lineup_decision_cutoff_utc"
        ]
        == "2026-01-03T23:00:00Z"
    )

    assert (
        target[
            "lineup_snapshot_as_of_utc"
        ]
        == "2026-01-03T23:00:00Z"
    )

    assert (
        target[
            "home_lineup_status"
        ]
        == "unknown"
    )

    assert (
        target[
            "home_lineup_observed_at"
        ]
        == ""
    )

    assert float(
        target[
            "skater_war_differential"
        ]
    ) == pytest.approx(
        1.5,
        abs=1e-12,
    )


def test_pull_sdv_rejects_current_lineup_snapshot_after_t60(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )

    monkeypatch.setattr(
        module,
        "official_schedule_cutoff_lookup",
        lambda: {},
    )

    schedule = module.pl.DataFrame(
        {
            "game_id": [
                "2025020001",
            ],
            "game_date": [
                "2026-01-01",
            ],
            "game_time": [
                "19:00",
            ],
            "home_team_abbr": [
                "BOS",
            ],
            "away_team_abbr": [
                "NYR",
            ],
        }
    )

    accepted = module.build_game_lineup_features_asof(
        schedule,
        module.pl.DataFrame(),
        module.pl.DataFrame(),
        current=True,
        generated_at_utc=module.datetime(
            2026,
            1,
            1,
            23,
            0,
            tzinfo=module.UTC,
        ),
    )

    assert (
        accepted.height
        == 1
    )

    rejected = module.build_game_lineup_features_asof(
        schedule,
        module.pl.DataFrame(),
        module.pl.DataFrame(),
        current=True,
        generated_at_utc=module.datetime(
            2026,
            1,
            1,
            23,
            0,
            1,
            tzinfo=module.UTC,
        ),
    )

    assert rejected.is_empty()


def test_merge_intake_builds_strict_asof_lineup_features() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    index = {
        "2025020001": {
            "game_id": "2025020001",
            "game_date": "2026_01_01",
            "home_team": "Boston Bruins",
            "away_team": "New York Rangers",
            "pregame_cutoff_utc": "2026-01-02T00:00:00+00:00",
            "lineup_decision_cutoff_utc": "2026-01-01T23:00:00+00:00",
            "lineup_decision_cutoff_source": "fixed_60_minutes_before_puck_drop",
            "lineup_snapshot_as_of_utc": "2026-01-01T23:00:00+00:00",
            **{
                key: str(value)
                for key, value in LINEUP_FEATURE_VALUES.items()
            },
            "_source_file": "fixture.csv",
        }
    }

    features = module.lineup_features_for_game(
        {
            "game_id": "2025020001",
            "game_date": "2026_01_01",
            "game_time": "19:00",
            "home_team": "Boston Bruins",
            "away_team": "New York Rangers",
        },
        index,
    )

    assert float(
        features[
            "skater_rapm_differential"
        ]
    ) == pytest.approx(
        0.25,
        abs=1e-12,
    )

    assert (
        features[
            "home_lineup_status"
        ]
        == "unknown"
    )


def test_merge_intake_rejects_lineup_snapshot_after_t60() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/merge_intake.py"
    )

    index = {
        "2025020001": {
            "game_id": "2025020001",
            "game_date": "2026_01_01",
            "home_team": "Boston Bruins",
            "away_team": "New York Rangers",
            "pregame_cutoff_utc": "2026-01-02T00:00:00+00:00",
            "lineup_decision_cutoff_utc": "2026-01-01T23:00:00+00:00",
            "lineup_decision_cutoff_source": "fixed_60_minutes_before_puck_drop",
            "lineup_snapshot_as_of_utc": "2026-01-01T23:00:01+00:00",
            **{
                key: str(value)
                for key, value in LINEUP_FEATURE_VALUES.items()
            },
            "_source_file": "fixture.csv",
        }
    }

    with pytest.raises(
        SystemExit
    ):
        module.lineup_features_for_game(
            {
                "game_id": "2025020001",
                "game_date": "2026_01_01",
                "game_time": "19:00",
                "home_team": "Boston Bruins",
                "away_team": "New York Rangers",
            },
            index,
        )


def test_build_juice_files_preserves_lineup_features_in_all_markets() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/build_juice_files.py"
    )

    expected = set(
        LINEUP_FEATURE_VALUES
    )

    assert expected.issubset(
        module.MERGED_REQUIRED_COLUMNS
    )
    assert expected.issubset(
        module.MONEYLINE_COLUMNS
    )
    assert expected.issubset(
        module.PUCK_LINE_COLUMNS
    )
    assert expected.issubset(
        module.TOTAL_COLUMNS
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/win/hockey/nhl/scripts/02_juice/apply_moneyline_juice.py",
        "docs/win/hockey/nhl/scripts/02_juice/apply_puck_line_juice.py",
        "docs/win/hockey/nhl/scripts/02_juice/apply_total_juice.py",
    ],
)
def test_apply_juice_scripts_preserve_lineup_feature_contract(
    relative_path: str,
) -> None:
    module = load_repo_module(
        relative_path
    )

    expected = set(
        LINEUP_FEATURE_VALUES
    )

    assert expected.issubset(
        module.REQUIRED_INPUT_COLUMNS
    )

    assert expected.issubset(
        module.OUTPUT_COLUMNS
    )

# ---------------------------------------------------------------------
# P7/P8 provider provenance and CLV regression coverage
# ---------------------------------------------------------------------

def _single_side_market_config(
    market_type: str,
    enabled_side: str,
) -> dict:
    rules_enabled = permissive_side_rules()
    rules_disabled = copy.deepcopy(
        rules_enabled
    )
    rules_disabled["enabled"] = False

    if market_type == "moneyline":
        return {
            "moneyline": {
                "enabled": True,
                "pick_preference": "all",
                "home": (
                    copy.deepcopy(
                        rules_enabled
                    )
                    if enabled_side == "home"
                    else copy.deepcopy(
                        rules_disabled
                    )
                ),
                "away": (
                    copy.deepcopy(
                        rules_enabled
                    )
                    if enabled_side == "away"
                    else copy.deepcopy(
                        rules_disabled
                    )
                ),
            }
        }

    if market_type == "puck_line":
        home = (
            copy.deepcopy(
                rules_enabled
            )
            if enabled_side == "home"
            else copy.deepcopy(
                rules_disabled
            )
        )
        away = (
            copy.deepcopy(
                rules_enabled
            )
            if enabled_side == "away"
            else copy.deepcopy(
                rules_disabled
            )
        )
        home["line_bands"] = [
            [-100.0, 100.0]
        ]
        away["line_bands"] = [
            [-100.0, 100.0]
        ]

        return {
            "puck_line": {
                "enabled": True,
                "pick_preference": "all",
                "home": home,
                "away": away,
            }
        }

    if market_type == "total":
        over = (
            copy.deepcopy(
                rules_enabled
            )
            if enabled_side == "over"
            else copy.deepcopy(
                rules_disabled
            )
        )
        under = (
            copy.deepcopy(
                rules_enabled
            )
            if enabled_side == "under"
            else copy.deepcopy(
                rules_disabled
            )
        )
        over["line_bands"] = [
            [0.0, 100.0]
        ]
        under["line_bands"] = [
            [0.0, 100.0]
        ]

        return {
            "total": {
                "enabled": True,
                "pick_preference": "all",
                "over": over,
                "under": under,
            }
        }

    raise AssertionError(
        f"Unsupported market_type fixture: {market_type}"
    )


def test_stage04_moneyline_preserves_selected_provider() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py"
    )

    row = synthetic_moneyline_row()
    row.update(
        {
            "moneyline_provider_id": "53",
            "moneyline_provider_name": "Titanbets",
            "odds_source": "espn",
            "pulled_at": "2026-01-01T18:00:00-05:00",
        }
    )

    selected = module.process_moneyline(
        row,
        _single_side_market_config(
            "moneyline",
            "home",
        ),
        "fixture_slate",
        {},
    )

    assert len(selected) == 1
    assert (
        selected[0][
            "selected_provider_id"
        ]
        == "53"
    )
    assert (
        selected[0][
            "selected_provider_name"
        ]
        == "Titanbets"
    )
    assert (
        selected[0][
            "odds_source"
        ]
        == "espn"
    )


def test_stage04_puck_line_preserves_selected_provider() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py"
    )

    row = {
        "sport": "hockey",
        "league": "nhl",
        "game_date": "2026_01_01",
        "game_time": "19:00",
        "game_id": "2025020001",
        "away_team": "New York Rangers",
        "home_team": "Boston Bruins",
        "home_puck_line": 1.5,
        "away_puck_line": -1.5,
        "home_dk_puck_line_american": -165.0,
        "away_dk_puck_line_american": 145.0,
        "home_dk_puck_line_decimal": 1.61,
        "away_dk_puck_line_decimal": 2.45,
        "home_model_prob_puck_line": 0.62,
        "away_model_prob_puck_line": 0.38,
        "home_edge_pct_puck_line": 0.03,
        "away_edge_pct_puck_line": 0.01,
        "home_ev_puck_line": 0.04,
        "away_ev_puck_line": 0.01,
        "home_kelly_puck_line": 0.08,
        "away_kelly_puck_line": 0.02,
        "puck_line_provider_id": "47",
        "puck_line_provider_name": "MGM",
        "odds_source": "espn",
        "pulled_at": "2026-01-01T18:00:00-05:00",
    }

    selected = module.process_puck_line(
        row,
        _single_side_market_config(
            "puck_line",
            "home",
        ),
        "fixture_slate",
        {},
    )

    assert len(selected) == 1
    assert (
        selected[0][
            "selected_provider_id"
        ]
        == "47"
    )
    assert (
        selected[0][
            "selected_provider_name"
        ]
        == "MGM"
    )


def test_stage04_total_preserves_selected_provider() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py"
    )

    row = {
        "sport": "hockey",
        "league": "nhl",
        "game_date": "2026_01_01",
        "game_time": "19:00",
        "game_id": "2025020001",
        "away_team": "New York Rangers",
        "home_team": "Boston Bruins",
        "total": 6.5,
        "dk_total_over_american": -110.0,
        "dk_total_under_american": -110.0,
        "dk_total_over_decimal": 1.91,
        "dk_total_under_decimal": 1.91,
        "over_model_prob_total": 0.56,
        "under_model_prob_total": 0.44,
        "over_edge_pct_total": 0.02,
        "under_edge_pct_total": 0.01,
        "over_ev_total": 0.03,
        "under_ev_total": 0.01,
        "over_kelly_total": 0.05,
        "under_kelly_total": 0.02,
        "total_provider_id": "38",
        "total_provider_name": "Caesars Sportsbook",
        "odds_source": "espn",
        "pulled_at": "2026-01-01T18:00:00-05:00",
    }

    selected = module.process_total(
        row,
        _single_side_market_config(
            "total",
            "over",
        ),
        "fixture_slate",
        {},
    )

    assert len(selected) == 1
    assert (
        selected[0][
            "selected_provider_id"
        ]
        == "38"
    )
    assert (
        selected[0][
            "selected_provider_name"
        ]
        == "Caesars Sportsbook"
    )


def test_stage05_grading_preserves_selected_provider_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(
        tmp_path
    )

    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/01_nhl_results_grade.py"
    )

    module.GRADE_ERROR_LOG = (
        tmp_path
        / "grade_errors.log"
    )
    module.GRADE_SUMMARY_LOG = (
        tmp_path
        / "grade_summary.log"
    )

    bets = pd.DataFrame(
        [
            {
                "game_id": "2025020001",
                "game_date": "2026_01_01",
                "market_type": "moneyline",
                "bet_side": "home",
                "line": "",
                "selected_provider_id": "53",
                "selected_provider_name": "Titanbets",
            }
        ]
    )

    scores = pd.DataFrame(
        [
            {
                "game_id": "2025020001",
                "away_score": 2,
                "home_score": 4,
                "total_score": 6,
                "away_puck_line_result": -2,
                "home_puck_line_result": 2,
                "source_score_file": (
                    "2026_01_01_NHL_final_scores.csv"
                ),
            }
        ]
    )

    statuses = pd.DataFrame(
        [
            {
                "game_id": "2025020001",
                "game_state": "FINAL",
                "game_schedule_state": "OK",
                "is_final": "1",
                "status_observed_at": (
                    "2026-01-02T03:00:00Z"
                ),
                "source_status_file": (
                    "nhl_game_status.csv"
                ),
            }
        ]
    )

    (
        graded,
        pending,
        unresolved,
    ) = module.grade_rows(
        bets,
        scores,
        statuses,
    )

    assert len(graded) == 1
    assert pending.empty
    assert unresolved.empty
    assert (
        graded.iloc[0][
            "selected_provider_id"
        ]
        == "53"
    )
    assert (
        graded.iloc[0][
            "selected_provider_name"
        ]
        == "Titanbets"
    )
    assert (
        graded.iloc[0][
            "bet_result"
        ]
        == "Win"
    )

    assert {
        "selected_provider_id",
        "selected_provider_name",
    }.issubset(
        module.GRADED_OUTPUT_COLUMNS
    )


def test_clv_reference_market_uses_provider_priority_and_excludes_live() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py"
    )

    live = espn_provider_row(
        provider_id="59",
        provider_name="ESPN Bet - Live Odds",
        provider_priority=-1,
    )
    mgm = espn_provider_row(
        provider_id="47",
        provider_name="MGM",
        provider_priority=0,
    )
    titanbets = espn_provider_row(
        provider_id="53",
        provider_name="Titanbets",
        provider_priority=0,
    )

    reference = module.reference_market(
        [
            live,
            mgm,
            titanbets,
        ],
        "moneyline",
        "home",
    )

    assert reference is not None
    assert (
        reference[
            "provider_id"
        ]
        == "53"
    )
    assert (
        reference[
            "provider_name"
        ]
        == "Titanbets"
    )
    assert float(
        reference[
            "decimal"
        ]
    ) == pytest.approx(
        2.10,
        abs=1e-12,
    )


def _clv_moneyline_bet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sport": "hockey",
                "league": "nhl",
                "game_date": "2026_01_01",
                "game_time": "19:00",
                "game_id": "2025020001",
                "away_team": "New York Rangers",
                "home_team": "Boston Bruins",
                "market_type": "moneyline",
                "bet_side": "home",
                "line": "",
                "take_bet": "home_moneyline",
                "dk_odds_american": 100,
                "dk_odds_decimal": 2.00,
                "model_prob": 0.55,
                "edge": 0.05,
                "ev": 0.10,
                "kelly": 0.10,
                "selected_provider_id": "53",
                "selected_provider_name": "Titanbets",
                "odds_source": "espn",
                "pulled_at": "2026-01-01T21:00:00Z",
                "source_select_file": (
                    "2026_01_01_NHL.csv"
                ),
            }
        ]
    )


def _clv_snapshot(
    _module,
    *,
    snapshot_at,
    home_decimal: float,
    status: str = "pending",
    provider_id: str = "53",
    provider_name: str = "Titanbets",
) -> dict:
    row = espn_provider_row(
        provider_id=provider_id,
        provider_name=provider_name,
        provider_priority=0,
    )
    row[
        "home_team_odds_current_money_line_decimal"
    ] = home_decimal

    if home_decimal >= 2.0:
        american = (
            (home_decimal - 1.0)
            * 100.0
        )
    else:
        american = (
            -100.0
            / (home_decimal - 1.0)
        )

    row[
        "home_team_odds_current_money_line_american"
    ] = american

    return {
        "snapshot_at": snapshot_at,
        "source": "espn",
        "event_status": {
            "fixture_sb_20251008_001": status,
        },
        "rows_by_event": {
            "fixture_sb_20251008_001": [
                row
            ],
        },
    }


def test_clv_uses_latest_valid_pregame_snapshot_as_closing_reference() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py"
    )

    snapshots = [
        _clv_snapshot(
            module,
            snapshot_at=module.datetime(
                2026,
                1,
                1,
                22,
                0,
                tzinfo=module.UTC,
            ),
            home_decimal=1.90,
        ),
        _clv_snapshot(
            module,
            snapshot_at=module.datetime(
                2026,
                1,
                1,
                23,
                30,
                tzinfo=module.UTC,
            ),
            home_decimal=1.80,
        ),
        _clv_snapshot(
            module,
            snapshot_at=module.datetime(
                2026,
                1,
                2,
                0,
                1,
                tzinfo=module.UTC,
            ),
            home_decimal=1.50,
            status="in",
        ),
    ]

    (
        summary_rows,
        history_rows,
    ) = module.process_bets(
        _clv_moneyline_bet(),
        {
            "2025020001": (
                "fixture_sb_20251008_001"
            )
        },
        snapshots,
    )

    assert len(summary_rows) == 1
    summary = summary_rows[0]

    assert (
        summary[
            "closing_snapshot_at_utc"
        ]
        == "2026-01-01T23:30:00+00:00"
    )
    assert float(
        summary[
            "closing_odds_decimal"
        ]
    ) == pytest.approx(
        1.80,
        abs=1e-12,
    )
    assert (
        summary[
            "closing_provider_id"
        ]
        == "53"
    )
    assert (
        summary[
            "closing_provider_name"
        ]
        == "Titanbets"
    )
    assert (
        summary[
            "later_pregame_snapshot_count"
        ]
        == 2
    )
    assert (
        summary[
            "clv_status"
        ]
        == "evaluated"
    )

    assert len(history_rows) == 2
    closing_history = [
        row
        for row in history_rows
        if int(
            row[
                "is_closing_reference"
            ]
        )
        == 1
    ]
    assert len(
        closing_history
    ) == 1
    assert (
        closing_history[0][
            "snapshot_at_utc"
        ]
        == "2026-01-01T23:30:00+00:00"
    )


def test_clv_price_math_matches_expected_values() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py"
    )

    (
        implied_probability_clv,
        decimal_ratio_clv,
    ) = module.price_clv(
        2.00,
        1.80,
        True,
    )

    assert (
        implied_probability_clv
        == pytest.approx(
            (
                1.0
                / 1.80
                - 1.0
                / 2.00
            ),
            abs=1e-12,
        )
    )
    assert (
        decimal_ratio_clv
        == pytest.approx(
            (
                2.00
                / 1.80
                - 1.0
            ),
            abs=1e-12,
        )
    )

    assert (
        module.favorable_line_clv(
            "puck_line",
            "home",
            1.5,
            1.0,
        )
        == pytest.approx(
            0.5,
            abs=1e-12,
        )
    )
    assert (
        module.favorable_line_clv(
            "total",
            "over",
            6.0,
            6.5,
        )
        == pytest.approx(
            0.5,
            abs=1e-12,
        )
    )
    assert (
        module.favorable_line_clv(
            "total",
            "under",
            6.5,
            6.0,
        )
        == pytest.approx(
            0.5,
            abs=1e-12,
        )
    )


def test_clv_reports_no_reference_snapshot_without_crashing() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py"
    )

    (
        summary_rows,
        history_rows,
    ) = module.process_bets(
        _clv_moneyline_bet(),
        {
            "2025020001": (
                "fixture_sb_20251008_001"
            )
        },
        [],
    )

    assert len(summary_rows) == 1
    assert history_rows == []
    assert (
        summary_rows[0][
            "clv_status"
        ]
        == "no_pregame_reference_snapshot"
    )


def test_clv_reports_no_later_closing_snapshot_when_only_earlier_price_exists() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py"
    )

    snapshot = _clv_snapshot(
        module,
        snapshot_at=module.datetime(
            2026,
            1,
            1,
            20,
            0,
            tzinfo=module.UTC,
        ),
        home_decimal=1.95,
    )

    (
        summary_rows,
        history_rows,
    ) = module.process_bets(
        _clv_moneyline_bet(),
        {
            "2025020001": (
                "fixture_sb_20251008_001"
            )
        },
        [
            snapshot
        ],
    )

    assert len(summary_rows) == 1
    assert history_rows == []
    assert (
        summary_rows[0][
            "later_pregame_snapshot_count"
        ]
        == 0
    )
    assert (
        summary_rows[0][
            "clv_status"
        ]
        == "no_later_closing_snapshot"
    )


def test_clv_line_change_blocks_price_comparison_but_keeps_line_clv() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py"
    )

    assert (
        module.same_line(
            "total",
            6.5,
            6.0,
        )
        is False
    )

    (
        implied_probability_clv,
        decimal_ratio_clv,
    ) = module.price_clv(
        1.91,
        1.80,
        False,
    )

    assert (
        implied_probability_clv
        is None
    )
    assert (
        decimal_ratio_clv
        is None
    )
    assert (
        module.favorable_line_clv(
            "total",
            "over",
            6.5,
            6.0,
        )
        == pytest.approx(
            -0.5,
            abs=1e-12,
        )
    )



# Regression tests: NHL SDV empty historical snapshots.
@pytest.mark.parametrize(
    "category,label",
    [
        ("fatigue", "fatigue"),
        ("team-strength", "team_ratings_asof"),
        ("goalie", "game_goalie_features_asof"),
        ("lineup-strength", "game_lineup_features_asof"),
    ],
)
def test_nhl_sdv_producer_does_not_write_headerless_feature_files(
    tmp_path: Path, monkeypatch, category: str, label: str
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py"
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / category).mkdir()
    prefix = "2026_08_20T183742_ET"
    written = module.save_object(
        category, label, pd.DataFrame(), prefix=prefix, current=True
    )
    assert written == []
    assert not (tmp_path / category / f"{prefix}_{label}.csv").exists()
    assert not (tmp_path / category / f"latest_{label}.csv").exists()


def test_nhl_sdv_restore_skips_headerless_archives(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/restore_nhl_sdv_history.py"
    )
    sdv = tmp_path / "sdv"
    folder = sdv / "fatigue"
    folder.mkdir(parents=True)
    monkeypatch.setattr(module, "SDV", sdv)
    monkeypatch.setattr(
        module, "FAMILIES", {"fatigue": module.FAMILIES["fatigue"]}
    )
    (folder / "2026_08_20T183742_ET_fatigue.csv").write_text(
        "\n", encoding="utf-8"
    )
    valid = folder / "2026_10_10T040000_ET_fatigue.csv"
    valid.write_text(
        "team,game_date,previous_game_date,days_rest,back_to_back,"
        "games_in_4_days,three_in_four,games_in_6_days,four_in_six,"
        "games_in_7_days\n"
        "BOS,2026-10-10,2026-10-08,2,False,2,False,3,False,3\n",
        encoding="utf-8",
    )
    module.prepare_history()
    season = folder / "season_2026_fatigue.csv"
    assert season.is_file()
    rows = pd.read_csv(season)
    assert len(rows) == 1
    assert rows.loc[0, "team"] == "BOS"
    assert "SDV empty historical snapshot skipped" in capsys.readouterr().out


def test_nhl_sdv_restore_rejects_duplicate_headers(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/01_merge/restore_nhl_sdv_history.py"
    )
    sdv = tmp_path / "sdv"
    folder = sdv / "fatigue"
    folder.mkdir(parents=True)
    monkeypatch.setattr(module, "SDV", sdv)
    monkeypatch.setattr(
        module, "FAMILIES", {"fatigue": module.FAMILIES["fatigue"]}
    )
    (folder / "2026_10_10T040000_ET_fatigue.csv").write_text(
        "team,team\nBOS,BOS\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Missing or duplicate CSV headers"):
        module.prepare_history()

# ---------------------------------------------------------------------
# NHL canonical sportsbook payout regression tests (2026-11-04)
# ---------------------------------------------------------------------

def test_american_odds_are_canonical_for_all_six_intake_prices() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py"
    )
    row = espn_provider_row()
    market_rows = [
        module.moneyline_from_provider(row),
        module.puck_line_from_provider(row),
        module.total_from_provider(row),
    ]
    assert all(market_rows)
    for parsed in market_rows:
        for side in ("home", "away", "over", "under"):
            american = parsed.get(f"{side}_american")
            decimal = parsed.get(f"{side}_decimal")
            if american is None:
                continue
            assert float(decimal) == pytest.approx(
                float(module.american_to_decimal(american)), abs=1e-12
            )


def test_intake_espn_rounding_and_fraction_validation() -> None:
    module = load_repo_module(
        "docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py"
    )
    row = espn_provider_row()
    row["away_team_odds_current_spread_american"] = "-218"
    row["away_team_odds_current_spread_decimal"] = 1.45
    row["away_team_odds_current_spread_fraction"] = "50/109"
    assert float(module.puck_line_from_provider(row)["away_decimal"]) == pytest.approx(
        1.0 + 100.0 / 218.0, abs=1e-12
    )
    row["away_team_odds_current_spread_fraction"] = "1/3"
    assert module.puck_line_from_provider(row) is None


def test_price_contract_reconciles_all_markets_and_metrics() -> None:
    prices = load_repo_module(
        "docs/win/hockey/nhl/scripts/odds_price_common.py"
    )
    df = pd.DataFrame({
        "american": [-218, -110, 125, np.nan, 0],
        "decimal": [1.45, 1.91, 2.25, 1.80, 1.8],
    })
    prices.reconcile_dataframe_prices(df, [("american", "decimal")])
    assert df.loc[0, "decimal"] == pytest.approx(1 + 100/218)
    assert df.loc[1, "decimal"] == pytest.approx(1 + 100/110)
    assert df.loc[2, "decimal"] == pytest.approx(2.25)
    assert np.isnan(df.loc[3, "decimal"])
    assert np.isnan(df.loc[4, "decimal"])

    edge, ev, kelly = prices.price_metrics(
        pd.Series([0.7973035387]),
        pd.Series([df.loc[0, "decimal"]]),
    )
    decimal = 1 + 100 / 218
    assert edge.iloc[0] == pytest.approx(0.7973035387 - 1/decimal)
    assert ev.iloc[0] == pytest.approx(0.7973035387 * decimal - 1)
    assert kelly.iloc[0] == pytest.approx(
        max(0, (0.7973035387 * decimal - 1)/(decimal - 1))
    )


def test_clv_and_settlement_use_same_canonical_payout() -> None:
    clv = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py"
    )
    report = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/03_nhl_results_reports.py"
    )
    quoted = clv.market_price(
        {
            "price_american": -218,
            "price_decimal": 1.45,
        },
        ("price_american",),
    )
    assert quoted is not None
    assert quoted[1] == pytest.approx(1 + 100/218)
    assert report.grade_to_units({
        "bet_result": "win", "dk_odds_american": -218,
    }) == pytest.approx(quoted[1] - 1)


def test_reporting_revalues_legacy_ev_without_mutating_input(tmp_path, monkeypatch):
    report = load_repo_module(
        "docs/win/hockey/nhl/scripts/05_final_scores/03_nhl_results_reports.py"
    )
    monkeypatch.setattr(report, "FINAL_ROOT", tmp_path)
    (tmp_path / "reports").mkdir(parents=True)
    df = pd.DataFrame([{
        "league": "nhl", "market_type": "moneyline",
        "bet_side": "home", "game_date": "2026_10_01",
        "game_id": "2026020001", "selected_provider_name": "MGM",
        "pulled_at": "2026-10-01T08:00:00-04:00",
        "dk_odds_american": -218, "dk_odds_decimal": 1.45,
        "model_prob": 0.7973035387, "my_model_prob": 0.80,
        "edge": 0.01, "ev": 0.15609, "kelly": 0.03,
        "line": np.nan, "away_score": 1, "home_score": 2,
        "total_score": 3, "bet_result": "win",
    }])
    old = df.copy(deep=True)
    out = report.prepare_df(df)
    assert df.equals(old)
    decimal = 1 + 100/218
    assert out.loc[0, "dk_odds_decimal"] == pytest.approx(decimal)
    assert out.loc[0, "ev"] == pytest.approx(0.7973035387*decimal-1)
    assert out.loc[0, "units"] == pytest.approx(decimal-1)
    trail = pd.read_csv(tmp_path / "reports" / "nhl_price_revaluations.csv")
    assert len(trail) == 1
    assert trail.loc[0, "original_decimal"] == pytest.approx(1.45)
