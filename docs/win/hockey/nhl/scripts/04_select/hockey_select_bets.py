#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/04_select/hockey_select_bets.py

import traceback
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Never

import pandas as pd
import yaml


SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# noinspection PyPep8
from selection_schema_common import (
    SELECTION_COLUMNS as OUTPUT_COLUMNS,
)


INPUT_DIR = Path("docs/win/hockey/nhl/03_edges/secondary_signals")
OUTPUT_DIR = Path("docs/win/hockey/nhl/04_select")
CONFIG_PATH = Path("docs/win/hockey/nhl/config/markets.yaml")

ERROR_DIR = Path("docs/win/hockey/nhl/errors/04_select")
LOG_FILE = ERROR_DIR / "hockey_select_bets.txt"
REJECTION_FILE = ERROR_DIR / "selection_rejections.csv"

LEAGUE_CODE = "NHL"

BLOCKED_PATH_PARTS = {
    "05_final_scores",
    "graded",
    "results",
    "reports",
}

REJECTION_COLUMNS = [
    "game_date",
    "market_type",
    "bet_side",
    "failing_condition",
    "rejection_count",
]

REJECTION_ORDER = {
    "missing_data": 0,
    "probability": 1,
    "my_model_probability": 2,
    "odds": 3,
    "line": 4,
    "edge": 5,
    "ev": 6,
    "kelly": 7,
    "secondary_model": 8,
    "pick_preference": 9,
}

SECONDARY_SIGNAL_COLUMNS = [
    "drat_home_win_prob",
    "drat_exp_margin",
    "drat_exp_total",
    "sdv_home_win_prob",
    "sdv_exp_margin",
    "sdv_exp_total",
    "prob_disagreement",
    "margin_disagreement",
    "total_disagreement",
    "prob_disagreement_threshold_p75_prior",
    "margin_disagreement_threshold_p75_prior",
    "total_disagreement_threshold_p75_prior",
    "high_prob_disagreement_flag",
    "high_margin_disagreement_flag",
    "high_total_disagreement_flag",
    "ensemble_train_rows",
    "weighted_prob_drat_weight",
    "weighted_margin_drat_weight",
    "weighted_total_drat_weight",
    "weighted_home_win_prob",
    "weighted_exp_margin",
    "weighted_exp_total",
    "meta_home_win_prob",
    "meta_exp_margin",
    "meta_exp_total",
    "my_model_home_prob_moneyline",
    "my_model_away_prob_moneyline",
    "my_model_home_prob_puck_line",
    "my_model_away_prob_puck_line",
    "my_model_over_prob_total",
    "my_model_under_prob_total",
    "secondary_history_max_game_date",
    "secondary_model_status",
    "secondary_signal_version",
]


def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def fail(msg: str) -> Never:
    _log(msg, "ERROR")
    raise SystemExit(msg)


def path_parts(path: Path) -> set:
    return set(path.as_posix().split("/"))


def assert_read_path(path: Path):
    parts = path_parts(path)
    blocked = sorted(parts & BLOCKED_PATH_PARTS)
    if blocked:
        fail(f"Blocked read path contains forbidden folder(s): {path} | blocked={blocked}")

    p = path.as_posix()
    if not (p.startswith(INPUT_DIR.as_posix() + "/") or p == CONFIG_PATH.as_posix()):
        fail(f"Blocked read path outside allowed Stage 04 inputs/config: {path}")


def assert_write_path(path: Path):
    parts = path_parts(path)
    blocked = sorted(parts & BLOCKED_PATH_PARTS)
    if blocked:
        fail(f"Blocked write path contains forbidden folder(s): {path} | blocked={blocked}")

    p = path.as_posix()
    allowed_output = OUTPUT_DIR.as_posix()
    allowed_log = ERROR_DIR.as_posix()

    if not (p.startswith(allowed_output + "/") or p.startswith(allowed_log + "/")):
        fail(f"Blocked write path outside allowed Stage 04 output/log folders: {path}")


def ensure_dirs():
    assert_write_path(OUTPUT_DIR / "dummy.csv")
    assert_write_path(LOG_FILE)
    assert_write_path(REJECTION_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def reset_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== NHL hockey_select_bets RUN {_now()} ===\n")


def load_config():
    assert_read_path(CONFIG_PATH)

    if not CONFIG_PATH.exists():
        fail(f"Config file not found: {CONFIG_PATH}")

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        fail(f"Malformed YAML in {CONFIG_PATH}: {e}")

    try:
        return raw["markets"]["nhl"]
    except Exception as e:
        fail(f"Missing expected config path markets -> nhl in {CONFIG_PATH}: {e}")


def fv(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return None


def sv(x):
    if pd.isna(x):
        return ""
    return str(x)


def in_range(val, ranges):
    if val is None:
        return False
    if ranges is None:
        return True
    return any(float(lo) <= val <= float(hi) for lo, hi in ranges)


def add_rejection(
    rejections: dict,
    *,
    game_date: str,
    market_type: str,
    bet_side: str,
    failing_condition: str,
):
    key = (
        game_date,
        market_type,
        bet_side,
        failing_condition,
    )
    rejections[key] = rejections.get(key, 0) + 1


def side_rule_failures(
    *,
    rules: dict,
    odds,
    line,
    prob,
    my_model_prob,
    edge,
    ev,
    kelly,
    check_line: bool,
):
    if not rules.get("enabled", False):
        return None

    values = [
        odds,
        prob,
        edge,
        ev,
        kelly,
    ]

    if check_line:
        values.append(line)

    if any(value is None for value in values):
        return ["missing_data"]

    failures = []

    if not in_range(odds, rules.get("odds_bands", [])):
        failures.append("odds")

    if check_line and not in_range(line, rules.get("line_bands", [])):
        failures.append("line")

    if not in_range(prob, rules.get("prob_bands", [])):
        failures.append("probability")

    # Secondary-model probability is an additional configurable filter.
    # Missing secondary probability follows secondary_model.unavailable_behavior
    # and therefore does not become primary missing_data here.
    if (
        my_model_prob is not None
        and not in_range(
            my_model_prob,
            rules.get("my_model_prob_bands", None),
        )
    ):
        failures.append("my_model_probability")

    if not in_range(edge, rules.get("edge_bands", None)):
        failures.append("edge")

    if not in_range(ev, rules.get("ev_bands", [])):
        failures.append("ev")

    if not in_range(kelly, rules.get("kelly_bands", [])):
        failures.append("kelly")

    return failures


def market_side_values(
    row,
    side: str,
    market_type: str,
) -> dict[str, float | None]:
    if market_type == "moneyline":
        suffix = "moneyline"
        return {
            "odds": fv(row.get(f"{side}_dk_{suffix}_american")),
            "decimal": fv(row.get(f"{side}_dk_{suffix}_decimal")),
            "line": None,
            "prob": fv(row.get(f"{side}_model_prob_{suffix}")),
            "my_model_prob": fv(
                row.get(f"my_model_{side}_prob_moneyline")
            ),
            "edge": fv(row.get(f"{side}_edge_pct_{suffix}")),
            "ev": fv(row.get(f"{side}_ev_{suffix}")),
            "kelly": fv(row.get(f"{side}_kelly_{suffix}")),
        }

    if market_type == "puck_line":
        suffix = "puck_line"
        return {
            "odds": fv(row.get(f"{side}_dk_{suffix}_american")),
            "decimal": fv(row.get(f"{side}_dk_{suffix}_decimal")),
            "line": fv(row.get(f"{side}_{suffix}")),
            "prob": fv(row.get(f"{side}_model_prob_{suffix}")),
            "my_model_prob": fv(
                row.get(f"my_model_{side}_prob_puck_line")
            ),
            "edge": fv(row.get(f"{side}_edge_pct_{suffix}")),
            "ev": fv(row.get(f"{side}_ev_{suffix}")),
            "kelly": fv(row.get(f"{side}_kelly_{suffix}")),
        }

    if market_type == "total":
        return {
            "odds": fv(row.get(f"dk_total_{side}_american")),
            "decimal": fv(row.get(f"dk_total_{side}_decimal")),
            "line": fv(row.get("total")),
            "prob": fv(row.get(f"{side}_model_prob_total")),
            "my_model_prob": fv(
                row.get(f"my_model_{side}_prob_total")
            ),
            "edge": fv(row.get(f"{side}_edge_pct_total")),
            "ev": fv(row.get(f"{side}_ev_total")),
            "kelly": fv(row.get(f"{side}_kelly_total")),
        }

    raise ValueError(
        f"Unsupported market_type: {market_type}"
    )


def check_side_rules(
    *,
    rules: dict,
    odds,
    line,
    prob,
    my_model_prob,
    edge,
    ev,
    kelly,
    check_line: bool,
):
    failures = side_rule_failures(
        rules=rules,
        odds=odds,
        line=line,
        prob=prob,
        my_model_prob=my_model_prob,
        edge=edge,
        ev=ev,
        kelly=kelly,
        check_line=check_line,
    )

    return failures == []


def require_columns(df: pd.DataFrame, cols: list[str], market_type: str, path: Path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fail(f"{market_type} missing required column(s): {missing} | file={path}")


def read_market_file(path: Path, market_type: str):
    if not path.exists():
        return None

    assert_read_path(path)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return fail(f"Failed reading {market_type} file: {path} | {e}")

    if "game_id" not in df.columns:
        fail(f"{market_type} file missing game_id: {path}")

    if df["game_id"].isna().any():
        fail(f"{market_type} file has blank game_id: {path}")

    dupes = df[df.duplicated(subset=["game_id"], keep=False)]
    if not dupes.empty:
        ids = sorted(dupes["game_id"].astype(str).unique().tolist())
        fail(f"multiple {market_type} rows for one game_id | file={path} | game_id={ids}")

    return df


def get_base_meta(row):
    return {
        "sport": sv(row.get("sport")),
        "league": sv(row.get("league")),
        "game_date": sv(row.get("game_date")),
        "game_time": sv(row.get("game_time")),
        "game_id": row.get("game_id"),
        "away_team": sv(row.get("away_team")),
        "home_team": sv(row.get("home_team")),
    }



def support_label(
    *,
    market_type: str,
    bet_side: str,
    prediction,
    line,
) -> str:
    value = fv(prediction)
    if value is None:
        return "unavailable"

    if market_type == "moneyline":
        if abs(value - 0.5) < 1e-12:
            return "neutral"
        supports_home = value > 0.5
        supports = supports_home if bet_side == "home" else not supports_home
        return "supports" if supports else "opposes"

    if market_type == "puck_line":
        line_value = fv(line)
        if line_value is None:
            return "unavailable"
        cover_margin = (
            value + line_value
            if bet_side == "home"
            else -value + line_value
        )
        if abs(cover_margin) < 1e-12:
            return "neutral"
        return "supports" if cover_margin > 0 else "opposes"

    if market_type == "total":
        line_value = fv(line)
        if line_value is None:
            return "unavailable"
        if abs(value - line_value) < 1e-12:
            return "neutral"
        supports_over = value > line_value
        supports = supports_over if bet_side == "over" else not supports_over
        return "supports" if supports else "opposes"

    return fail(f"Unknown market_type for secondary support: {market_type}")


def secondary_market_fields(
    market_type: str,
    derived_model: str,
) -> tuple[str, str, str]:
    if market_type == "moneyline":
        derived_field = (
            "weighted_home_win_prob"
            if derived_model == "weighted"
            else "meta_home_win_prob"
        )
        return "high_prob_disagreement_flag", "sdv_home_win_prob", derived_field

    if market_type == "puck_line":
        derived_field = (
            "weighted_exp_margin"
            if derived_model == "weighted"
            else "meta_exp_margin"
        )
        return "high_margin_disagreement_flag", "sdv_exp_margin", derived_field

    if market_type == "total":
        derived_field = (
            "weighted_exp_total"
            if derived_model == "weighted"
            else "meta_exp_total"
        )
        return "high_total_disagreement_flag", "sdv_exp_total", derived_field

    return fail(f"Unknown market_type for secondary model: {market_type}")


def apply_secondary_model_gate(
    candidates: list[dict],
    row,
    config: dict,
    *,
    market_type: str,
    rejections: dict,
) -> list[dict]:
    if not candidates:
        return []

    secondary = config.get("secondary_model", {})
    enabled = bool(secondary.get("enabled", False))

    if not enabled:
        for candidate in candidates:
            for col in SECONDARY_SIGNAL_COLUMNS:
                candidate[col] = row.get(col)
            candidate["secondary_challenger_support"] = "unavailable"
            candidate["secondary_derived_model"] = ""
            candidate["secondary_derived_support"] = "unavailable"
            candidate["secondary_decision"] = "disabled_primary_only"
        return candidates

    selection_mode = secondary.get("selection_mode")
    if selection_mode != "high_disagreement_requires_secondary_support":
        fail(
            "Unsupported secondary_model.selection_mode: "
            f"{selection_mode!r}"
        )

    derived_by_market = secondary.get("derived_signal_by_market", {})
    derived_model = str(derived_by_market.get(market_type, "")).strip().lower()

    if derived_model not in {"weighted", "meta"}:
        fail(
            f"secondary_model derived signal is invalid for {market_type}: "
            f"{derived_model!r}"
        )

    high_field, challenger_field, derived_field = secondary_market_fields(
        market_type,
        derived_model,
    )

    kept: list[dict] = []

    for candidate in candidates:
        for col in SECONDARY_SIGNAL_COLUMNS:
            candidate[col] = row.get(col)

        candidate["secondary_derived_model"] = derived_model
        status = sv(row.get("secondary_model_status"))

        line = candidate.get("line")
        side = candidate["bet_side"]
        challenger_support = support_label(
            market_type=market_type,
            bet_side=side,
            prediction=row.get(challenger_field),
            line=line,
        )
        derived_support = support_label(
            market_type=market_type,
            bet_side=side,
            prediction=row.get(derived_field),
            line=line,
        )

        candidate["secondary_challenger_support"] = challenger_support
        candidate["secondary_derived_support"] = derived_support

        if status != "ready":
            if secondary.get("unavailable_behavior") != "use_primary":
                fail(
                    "Unsupported secondary_model.unavailable_behavior: "
                    f"{secondary.get('unavailable_behavior')!r}"
                )
            candidate["secondary_decision"] = f"fallback_primary:{status or 'unavailable'}"
            kept.append(candidate)
            continue

        high_value = fv(row.get(high_field))
        high_disagreement = high_value is not None and high_value >= 0.5

        if not high_disagreement:
            candidate["secondary_decision"] = "normal_disagreement_primary"
            kept.append(candidate)
            continue

        support_exists = (
            challenger_support == "supports"
            or derived_support == "supports"
        )

        if support_exists:
            candidate["secondary_decision"] = "high_disagreement_supported"
            kept.append(candidate)
            continue

        candidate["secondary_decision"] = "high_disagreement_no_secondary_support"
        add_rejection(
            rejections,
            game_date=candidate["game_date"],
            market_type=market_type,
            bet_side=side,
            failing_condition="secondary_model",
        )

    return kept

def apply_pick_preference(
    candidates: list[dict],
    pick_preference: str,
    slate_key: str,
    game_id,
    market_type: str,
    rejections: dict,
):
    if not candidates:
        return []

    if pick_preference == "all":
        return candidates

    if pick_preference == "best_ev":
        max_ev = max(c["ev"] for c in candidates)
        winners = [c for c in candidates if c["ev"] == max_ev]
    elif pick_preference == "best_prob":
        max_prob = max(c["model_prob"] for c in candidates)
        winners = [c for c in candidates if c["model_prob"] == max_prob]
    else:
        return fail(
            f"Invalid pick_preference for {market_type}: {pick_preference} | "
            f"slate={slate_key} | game_id={game_id}"
        )

    if len(winners) > 1:
        fail(
            f"pick_preference tie for {market_type} | preference={pick_preference} | "
            f"slate={slate_key} | game_id={game_id}"
        )

    winner_ids = {id(winner) for winner in winners}

    for candidate in candidates:
        if id(candidate) in winner_ids:
            continue

        add_rejection(
            rejections,
            game_date=candidate["game_date"],
            market_type=market_type,
            bet_side=candidate["bet_side"],
            failing_condition="pick_preference",
        )

    return winners


MARKET_SIDES = {
    "moneyline": ("home", "away"),
    "puck_line": ("home", "away"),
    "total": ("over", "under"),
}


def process_market(
    row,
    config,
    slate_key,
    rejections,
    market_type: str,
):
    market_config = config.get(
        market_type,
        {},
    )
    if not market_config.get(
        "enabled",
        False,
    ):
        return []

    sides = MARKET_SIDES.get(
        market_type
    )
    if sides is None:
        return fail(
            f"Unsupported market_type during selection: {market_type}"
        )

    for side in sides:
        if side not in market_config:
            fail(
                f"{market_type} config missing side: {side}"
            )

    candidates = []
    meta = get_base_meta(row)
    check_line = (
        market_type != "moneyline"
    )

    for side in sides:
        side_rules = market_config[side]
        values = market_side_values(
            row,
            side,
            market_type,
        )

        failures = side_rule_failures(
            rules=side_rules,
            odds=values["odds"],
            line=values["line"],
            prob=values["prob"],
            my_model_prob=values["my_model_prob"],
            edge=values["edge"],
            ev=values["ev"],
            kelly=values["kelly"],
            check_line=check_line,
        )

        if failures is None:
            continue

        if failures:
            for condition in failures:
                add_rejection(
                    rejections,
                    game_date=meta["game_date"],
                    market_type=market_type,
                    bet_side=side,
                    failing_condition=condition,
                )
            continue

        candidates.append(
            {
                **meta,
                "market_type": market_type,
                "bet_side": side,
                "line": (
                    ""
                    if market_type == "moneyline"
                    else values["line"]
                ),
                "take_bet": (
                    f"{side}_{market_type}"
                ),
                "dk_odds_american": values[
                    "odds"
                ],
                "dk_odds_decimal": values[
                    "decimal"
                ],
                "model_prob": values[
                    "prob"
                ],
                "my_model_prob": values[
                    "my_model_prob"
                ],
                "edge": values[
                    "edge"
                ],
                "ev": values[
                    "ev"
                ],
                "kelly": values[
                    "kelly"
                ],
                "selected_provider_id": sv(
                    row.get(
                        f"{market_type}_provider_id"
                    )
                ),
                "selected_provider_name": sv(
                    row.get(
                        f"{market_type}_provider_name"
                    )
                ),
                "odds_source": sv(
                    row.get("odds_source")
                ),
                "pulled_at": sv(
                    row.get("pulled_at")
                ),
            }
        )

    candidates = apply_secondary_model_gate(
        candidates,
        row,
        config,
        market_type=market_type,
        rejections=rejections,
    )

    return apply_pick_preference(
        candidates,
        market_config.get(
            "pick_preference",
            "all",
        ),
        slate_key,
        row.get("game_id"),
        market_type,
        rejections,
    )


def process_moneyline(
    row,
    config,
    slate_key,
    rejections,
):
    return process_market(
        row,
        config,
        slate_key,
        rejections,
        "moneyline",
    )


def process_puck_line(
    row,
    config,
    slate_key,
    rejections,
):
    return process_market(
        row,
        config,
        slate_key,
        rejections,
        "puck_line",
    )


def process_total(
    row,
    config,
    slate_key,
    rejections,
):
    return process_market(
        row,
        config,
        slate_key,
        rejections,
        "total",
    )


def reset_rejection_file():
    assert_write_path(REJECTION_FILE)
    pd.DataFrame(columns=REJECTION_COLUMNS).to_csv(
        REJECTION_FILE,
        index=False,
    )


def write_rejections(rejections: dict):
    rows = []

    for (
        game_date,
        market_type,
        bet_side,
        failing_condition,
    ), rejection_count in rejections.items():
        rows.append({
            "game_date": game_date,
            "market_type": market_type,
            "bet_side": bet_side,
            "failing_condition": failing_condition,
            "rejection_count": rejection_count,
        })

    rows.sort(
        key=lambda row: (
            row["game_date"],
            row["market_type"],
            row["bet_side"],
            REJECTION_ORDER.get(
                row["failing_condition"],
                999,
            ),
            row["failing_condition"],
        )
    )

    assert_write_path(REJECTION_FILE)

    pd.DataFrame(
        rows,
        columns=REJECTION_COLUMNS,
    ).to_csv(
        REJECTION_FILE,
        index=False,
    )

    _log(
        f"WROTE: {REJECTION_FILE} | "
        f"rows={len(rows)} | "
        f"rejections={sum(rejections.values())}"
    )


def wipe_outputs():
    for old in OUTPUT_DIR.glob("*.csv"):
        assert_write_path(old)
        old.unlink()


def find_slates():
    assert_read_path(INPUT_DIR / "dummy.csv")

    files = sorted(INPUT_DIR.glob("*_NHL_*.csv"))
    slates = {}

    for fpath in files:
        assert_read_path(fpath)

        name = fpath.name

        if name.endswith("_NHL_moneyline.csv"):
            slate_key = name.replace("_NHL_moneyline.csv", "")
            slates.setdefault(slate_key, {})["moneyline"] = fpath
        elif name.endswith("_NHL_puck_line.csv"):
            slate_key = name.replace("_NHL_puck_line.csv", "")
            slates.setdefault(slate_key, {})["puck_line"] = fpath
        elif name.endswith("_NHL_total.csv"):
            slate_key = name.replace("_NHL_total.csv", "")
            slates.setdefault(slate_key, {})["total"] = fpath

    return slates


def validate_market_columns(df, market_type, path):
    base_cols = [
        "sport",
        "league",
        "game_date",
        "game_time",
        "game_id",
        "away_team",
        "home_team",
        *SECONDARY_SIGNAL_COLUMNS,
    ]
    cols = base_cols

    if market_type == "moneyline":
        cols = base_cols + [
            "away_dk_moneyline_american",
            "home_dk_moneyline_american",
            "away_dk_moneyline_decimal",
            "home_dk_moneyline_decimal",
            "odds_source",
            "moneyline_provider_id",
            "moneyline_provider_name",
            "pulled_at",
            "away_model_prob_moneyline",
            "home_model_prob_moneyline",
            "away_edge_pct_moneyline",
            "home_edge_pct_moneyline",
            "away_ev_moneyline",
            "home_ev_moneyline",
            "away_kelly_moneyline",
            "home_kelly_moneyline",
        ]
    elif market_type == "puck_line":
        cols = base_cols + [
            "away_puck_line",
            "home_puck_line",
            "away_dk_puck_line_american",
            "home_dk_puck_line_american",
            "away_dk_puck_line_decimal",
            "home_dk_puck_line_decimal",
            "odds_source",
            "puck_line_provider_id",
            "puck_line_provider_name",
            "pulled_at",
            "away_model_prob_puck_line",
            "home_model_prob_puck_line",
            "away_edge_pct_puck_line",
            "home_edge_pct_puck_line",
            "away_ev_puck_line",
            "home_ev_puck_line",
            "away_kelly_puck_line",
            "home_kelly_puck_line",
        ]
    elif market_type == "total":
        cols = base_cols + [
            "total",
            "dk_total_over_american",
            "dk_total_under_american",
            "dk_total_over_decimal",
            "dk_total_under_decimal",
            "odds_source",
            "total_provider_id",
            "total_provider_name",
            "pulled_at",
            "over_model_prob_total",
            "under_model_prob_total",
            "over_edge_pct_total",
            "under_edge_pct_total",
            "over_ev_total",
            "under_ev_total",
            "over_kelly_total",
            "under_kelly_total",
        ]
    else:
        fail(f"Unknown market_type during validation: {market_type}")

    require_columns(df, cols, market_type, path)


def row_for_game(df, game_id, market_type):
    if df is None:
        return None

    match = df[df["game_id"].astype(str) == str(game_id)]

    if len(match) > 1:
        fail(f"multiple {market_type} rows for one game_id | game_id={game_id}")

    if len(match) == 0:
        return None

    return match.iloc[0]


def process_slate(slate_key, paths, config, rejections):
    _log(f"--- SLATE: {slate_key}")

    ml_path = paths.get("moneyline")
    pl_path = paths.get("puck_line")
    td_path = paths.get("total")

    ml_df = read_market_file(ml_path, "moneyline") if ml_path else None
    pl_df = read_market_file(pl_path, "puck_line") if pl_path else None
    td_df = read_market_file(td_path, "total") if td_path else None

    if ml_df is not None:
        validate_market_columns(ml_df, "moneyline", ml_path)
    else:
        _log(f"{slate_key} missing moneyline file — skipping moneyline only", "WARN")

    if pl_df is not None:
        validate_market_columns(pl_df, "puck_line", pl_path)
    else:
        _log(f"{slate_key} missing puck_line file — skipping puck_line only", "WARN")

    if td_df is not None:
        validate_market_columns(td_df, "total", td_path)
    else:
        _log(f"{slate_key} missing total file — skipping total only", "WARN")

    game_ids = set()

    for df in [ml_df, pl_df, td_df]:
        if df is not None:
            game_ids.update(df["game_id"].astype(str).tolist())

    game_ids = sorted(game_ids)

    final_rows = []

    for game_id in game_ids:
        ml_row = row_for_game(ml_df, game_id, "moneyline")
        pl_row = row_for_game(pl_df, game_id, "puck_line")
        td_row = row_for_game(td_df, game_id, "total")

        if ml_row is not None:
            final_rows.extend(
                process_moneyline(
                    ml_row,
                    config,
                    slate_key,
                    rejections,
                )
            )

        if pl_row is not None:
            final_rows.extend(
                process_puck_line(
                    pl_row,
                    config,
                    slate_key,
                    rejections,
                )
            )

        if td_row is not None:
            final_rows.extend(
                process_total(
                    td_row,
                    config,
                    slate_key,
                    rejections,
                )
            )

    out_path = OUTPUT_DIR / f"{slate_key}_NHL.csv"
    assert_write_path(out_path)

    df_out = pd.DataFrame(final_rows, columns=OUTPUT_COLUMNS)
    df_out.to_csv(out_path, index=False)

    ml_count = int((df_out["market_type"] == "moneyline").sum()) if not df_out.empty else 0
    pl_count = int((df_out["market_type"] == "puck_line").sum()) if not df_out.empty else 0
    td_count = int((df_out["market_type"] == "total").sum()) if not df_out.empty else 0

    _log(
        f"WROTE: {out_path} | bets={len(df_out)} | "
        f"moneyline={ml_count} | puck_line={pl_count} | total={td_count}"
    )

    return {
        "slate": slate_key,
        "bets": len(df_out),
        "moneyline": ml_count,
        "puck_line": pl_count,
        "total": td_count,
    }


def write_summary(summary_rows):
    total_slates = len(summary_rows)
    total_bets = sum(r["bets"] for r in summary_rows)
    total_ml = sum(r["moneyline"] for r in summary_rows)
    total_pl = sum(r["puck_line"] for r in summary_rows)
    total_td = sum(r["total"] for r in summary_rows)

    lines = [
        "",
        "=" * 60,
        f"SUMMARY {_now()}",
        "=" * 60,
        f"slates_written : {total_slates}",
        f"total_bets     : {total_bets}",
        f"moneyline_bets : {total_ml}",
        f"puck_line_bets : {total_pl}",
        f"total_bets_mkt : {total_td}",
        "",
        f"{'slate':<20} {'bets':>6} {'ml':>6} {'pl':>6} {'total':>6}",
    ]

    for r in summary_rows:
        lines.append(
            f"{r['slate']:<20} {r['bets']:>6} "
            f"{r['moneyline']:>6} {r['puck_line']:>6} {r['total']:>6}"
        )

    lines.extend([
        "",
        "STATUS: SUCCESS",
        "=" * 60,
    ])

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ensure_dirs()
    reset_log()

    try:
        config = load_config()

        _log(f"INPUT_DIR: {INPUT_DIR}")
        _log(f"OUTPUT_DIR: {OUTPUT_DIR}")
        _log(f"CONFIG_PATH: {CONFIG_PATH}")
        _log(f"LOG_FILE: {LOG_FILE}")
        _log(f"REJECTION_FILE: {REJECTION_FILE}")

        wipe_outputs()
        reset_rejection_file()

        slates = find_slates()
        _log(f"Slates found: {len(slates)}")

        summary_rows = []
        rejections = {}

        for slate_key in sorted(slates):
            summary_rows.append(
                process_slate(
                    slate_key,
                    slates[slate_key],
                    config,
                    rejections,
                )
            )

        write_rejections(rejections)
        write_summary(summary_rows)

        print("hockey_select_bets complete.")

    except SystemExit:
        raise
    except Exception as e:
        # noinspection PyBroadException
        try:
            _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        except Exception:
            pass
        raise SystemExit(1)


if __name__ == "__main__":
    main()