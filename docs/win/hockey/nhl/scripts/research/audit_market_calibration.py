#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/research/audit_market_calibration.py

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


NHL_ROOT = Path(__file__).resolve().parents[2]
STAGE02_DIR = NHL_ROOT / "02_juice"
SCORE_DIR = NHL_ROOT / "05_final_scores" / "final_scores"
CONFIG_PATH = NHL_ROOT / "config" / "markets.yaml"
OUTPUT_DIR = NHL_ROOT / "research" / "calibration"

SUMMARY_FILE = OUTPUT_DIR / "market_side_summary.csv"
RELIABILITY_FILE = OUTPUT_DIR / "reliability_buckets.csv"
TIME_SPLIT_FILE = OUTPUT_DIR / "season_time_split.csv"
SIGNAL_BUCKET_FILE = OUTPUT_DIR / "ev_kelly_buckets.csv"
SIGNAL_TIME_SPLIT_FILE = OUTPUT_DIR / "ev_kelly_time_split.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-15

MARKETS = {
    "moneyline": {
        "sides": ("home", "away"),
        "pattern": "*_NHL_moneyline.csv",
        "prob": lambda s: f"{s}_normalized_prob_moneyline",
        "american": lambda s: f"{s}_dk_moneyline_american",
        "decimal": lambda s: f"{s}_dk_moneyline_decimal",
        "line": lambda s: None,
    },
    "puck_line": {
        "sides": ("home", "away"),
        "pattern": "*_NHL_puck_line.csv",
        "prob": lambda s: f"{s}_normalized_prob_puck_line",
        "american": lambda s: f"{s}_dk_puck_line_american",
        "decimal": lambda s: f"{s}_dk_puck_line_decimal",
        "line": lambda s: f"{s}_puck_line",
    },
    "total": {
        "sides": ("over", "under"),
        "pattern": "*_NHL_total.csv",
        "prob": lambda s: f"{s}_normalized_prob_total",
        "american": lambda s: f"dk_total_{s}_american",
        "decimal": lambda s: f"dk_total_{s}_decimal",
        "line": lambda s: "total",
    },
}


def to_float(value):
    try:
        text = str(value).strip()

        if not text:
            return None

        value = float(text)

        return value if math.isfinite(value) else None

    except (TypeError, ValueError, OverflowError):
        return None


def parse_date(value):
    value = pd.to_datetime(
        str(value).strip().replace("_", "-"),
        errors="coerce",
    )

    return (
        None
        if pd.isna(value)
        else value.normalize()
    )


def valid_game_id(value):
    value = str(value).strip()

    return (
        len(value) == 10
        and value.isdigit()
    )


def valid_american(value):
    return (
        value is not None
        and (
            value >= 100
            or value <= -100
        )
    )


def american_to_decimal(value):
    if not valid_american(value):
        return None

    if value > 0:
        return (
            1.0
            + value / 100.0
        )

    return (
        1.0
        + 100.0 / abs(value)
    )


def resolve_decimal(
    decimal_value,
    american_value,
):
    decimal_value = to_float(
        decimal_value
    )

    if (
        decimal_value is not None
        and decimal_value > 1
    ):
        return decimal_value

    return american_to_decimal(
        american_value
    )


def edge_pct(
    probability,
    decimal_odds,
):
    if (
        probability is None
        or decimal_odds is None
        or not 0 < probability < 1
        or decimal_odds <= 1
    ):
        return None

    return (
        probability
        - (
            1.0
            / decimal_odds
        )
    )


def expected_value(
    probability,
    decimal_odds,
):
    if (
        probability is None
        or decimal_odds is None
        or not 0 < probability < 1
        or decimal_odds <= 1
    ):
        return None

    return (
        probability
        * decimal_odds
        - 1.0
    )


def kelly_fraction(
    probability,
    decimal_odds,
):
    if (
        probability is None
        or decimal_odds is None
        or not 0 < probability < 1
        or decimal_odds <= 1
    ):
        return None

    b = (
        decimal_odds
        - 1.0
    )

    kelly = (
        (
            b
            * probability
        )
        - (
            1.0
            - probability
        )
    ) / b

    return max(
        kelly,
        0.0,
    )


def in_range(
    value,
    bands,
):
    if value is None:
        return False

    if bands is None:
        return True

    if (
        not isinstance(
            bands,
            list,
        )
        or not bands
    ):
        return False

    for band in bands:
        if (
            not isinstance(
                band,
                list,
            )
            or len(band) != 2
        ):
            raise RuntimeError(
                f"INVALID BAND IN "
                f"{CONFIG_PATH}: "
                f"{band!r}"
            )

        if (
            float(band[0])
            <= value
            <= float(band[1])
        ):
            return True

    return False


def atomic_write(
    df,
    path,
):
    tmp = path.with_suffix(
        ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(
        path
    )


def load_config():
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"MISSING CONFIG | "
            f"{CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        raw = yaml.safe_load(f)

    try:
        return raw[
            "markets"
        ][
            "nhl"
        ]

    except Exception as exc:
        raise RuntimeError(
            "MISSING CONFIG PATH "
            "markets -> nhl | "
            f"{CONFIG_PATH}"
        ) from exc


def required_columns(
    market,
):
    spec = MARKETS[
        market
    ]

    columns = {
        "game_date",
        "game_id",
        "away_team",
        "home_team",
    }

    for side in spec[
        "sides"
    ]:
        columns.update(
            {
                spec["prob"](
                    side
                ),
                spec["american"](
                    side
                ),
                spec["decimal"](
                    side
                ),
            }
        )

        line_col = spec[
            "line"
        ](
            side
        )

        if line_col:
            columns.add(
                line_col
            )

    return sorted(
        columns
    )


def make_candidate(
    row,
    market,
    side,
    source_file,
):
    spec = MARKETS[
        market
    ]

    game_id = str(
        row.get(
            "game_id",
            "",
        )
    ).strip()

    game_date = parse_date(
        row.get(
            "game_date"
        )
    )

    if (
        not valid_game_id(
            game_id
        )
        or game_date is None
    ):
        return None

    probability = to_float(
        row.get(
            spec["prob"](
                side
            )
        )
    )

    american = to_float(
        row.get(
            spec["american"](
                side
            )
        )
    )

    decimal_odds = resolve_decimal(
        row.get(
            spec["decimal"](
                side
            )
        ),
        american,
    )

    line_col = spec[
        "line"
    ](
        side
    )

    line = (
        None
        if line_col is None
        else to_float(
            row.get(
                line_col
            )
        )
    )

    if not valid_american(
        american
    ):
        return None

    edge = edge_pct(
        probability,
        decimal_odds,
    )

    ev = expected_value(
        probability,
        decimal_odds,
    )

    kelly = kelly_fraction(
        probability,
        decimal_odds,
    )

    if any(
        value is None
        for value in (
            probability,
            decimal_odds,
            edge,
            ev,
            kelly,
        )
    ):
        return None

    if (
        market
        in {
            "puck_line",
            "total",
        }
        and line is None
    ):
        return None

    return {
        "game_date": game_date,
        "game_id": game_id,
        "away_team": str(
            row.get(
                "away_team",
                "",
            )
        ).strip(),
        "home_team": str(
            row.get(
                "home_team",
                "",
            )
        ).strip(),
        "market_type": market,
        "bet_side": side,
        "line": line,
        "odds_american": american,
        "odds_decimal": decimal_odds,
        "model_prob": probability,
        "edge": edge,
        "ev": ev,
        "kelly": kelly,
        "source_file": source_file,
    }


def load_candidates():
    rows = []

    stats = {
        "files_seen": 0,
        "files_loaded": 0,
        "files_skipped_schema": 0,
        "source_rows_seen": 0,
        "invalid_candidate_sides": 0,
        "invalid_puck_line_rows": 0,
    }

    for (
        market,
        spec,
    ) in MARKETS.items():

        required = (
            required_columns(
                market
            )
        )

        files = sorted(
            STAGE02_DIR.glob(
                spec[
                    "pattern"
                ]
            )
        )

        for path in files:
            stats[
                "files_seen"
            ] += 1

            df = pd.read_csv(
                path,
                dtype=str,
            )

            if df.empty:
                stats[
                    "files_loaded"
                ] += 1

                continue

            missing = [
                column
                for column in required
                if column
                not in df.columns
            ]

            if missing:
                stats[
                    "files_skipped_schema"
                ] += 1

                print(
                    "WARNING: skipped "
                    f"{path}; "
                    "missing columns: "
                    f"{missing}"
                )

                continue

            stats[
                "files_loaded"
            ] += 1

            stats[
                "source_rows_seen"
            ] += len(df)

            for _, row in (
                df.iterrows()
            ):

                if (
                    market
                    == "puck_line"
                ):
                    away_line = (
                        to_float(
                            row.get(
                                "away_puck_line"
                            )
                        )
                    )

                    home_line = (
                        to_float(
                            row.get(
                                "home_puck_line"
                            )
                        )
                    )

                    if (
                        away_line
                        is None
                        or home_line
                        is None
                        or abs(
                            away_line
                            + home_line
                        )
                        > 1e-9
                    ):
                        stats[
                            "invalid_puck_line_rows"
                        ] += 1

                        stats[
                            "invalid_candidate_sides"
                        ] += 2

                        continue

                for side in spec[
                    "sides"
                ]:
                    candidate = (
                        make_candidate(
                            row,
                            market,
                            side,
                            path.name,
                        )
                    )

                    if (
                        candidate
                        is None
                    ):
                        stats[
                            "invalid_candidate_sides"
                        ] += 1

                    else:
                        rows.append(
                            candidate
                        )

    if not rows:
        raise RuntimeError(
            "NO VALID STAGE 02 "
            "CANDIDATES | "
            f"{STAGE02_DIR}"
        )

    candidates = (
        pd.DataFrame(
            rows
        )
    )

    key = [
        "game_id",
        "market_type",
        "bet_side",
    ]

    duplicate_rows = (
        candidates[
            candidates.duplicated(
                key,
                keep=False,
            )
        ]
    )

    for (
        values,
        group,
    ) in duplicate_rows.groupby(
        key
    ):
        compare = [
            "game_date",
            "line",
            "odds_american",
            "odds_decimal",
            "model_prob",
            "edge",
            "ev",
            "kelly",
        ]

        if (
            len(
                group[
                    compare
                ].drop_duplicates()
            )
            > 1
        ):
            raise RuntimeError(
                "CONFLICTING DUPLICATE "
                "CANDIDATE | "
                f"game_id={values[0]} "
                f"market={values[1]} "
                f"side={values[2]}"
            )

    candidates = (
        candidates.drop_duplicates(
            key,
            keep="first",
        ).copy()
    )

    return (
        candidates,
        stats,
    )


def passes_side_rules(
    row,
    rules,
    market,
):
    if not rules.get(
        "enabled",
        False,
    ):
        return False

    if not in_range(
        row[
            "odds_american"
        ],
        rules.get(
            "odds_bands",
            [],
        ),
    ):
        return False

    if (
        market
        in {
            "puck_line",
            "total",
        }
        and not in_range(
            row[
                "line"
            ],
            rules.get(
                "line_bands",
                [],
            ),
        )
    ):
        return False

    if not in_range(
        row[
            "model_prob"
        ],
        rules.get(
            "prob_bands",
            [],
        ),
    ):
        return False

    if not in_range(
        row[
            "edge"
        ],
        rules.get(
            "edge_bands",
            None,
        ),
    ):
        return False

    if not in_range(
        row[
            "ev"
        ],
        rules.get(
            "ev_bands",
            [],
        ),
    ):
        return False

    if not in_range(
        row[
            "kelly"
        ],
        rules.get(
            "kelly_bands",
            [],
        ),
    ):
        return False

    return True


def apply_current_config(
    candidates,
    config,
):
    qualified = []

    for row in (
        candidates.to_dict(
            "records"
        )
    ):
        market = row[
            "market_type"
        ]

        side = row[
            "bet_side"
        ]

        market_config = (
            config.get(
                market,
                {},
            )
        )

        if not market_config.get(
            "enabled",
            False,
        ):
            continue

        rules = market_config.get(
            side
        )

        if not isinstance(
            rules,
            dict,
        ):
            raise RuntimeError(
                "MISSING SIDE CONFIG | "
                f"market={market} "
                f"side={side}"
            )

        if passes_side_rules(
            row,
            rules,
            market,
        ):
            qualified.append(
                row
            )

    if not qualified:
        return pd.DataFrame(
            columns=(
                candidates.columns
            )
        )

    qualified = pd.DataFrame(
        qualified
    )

    selected = []

    for (
        market,
        market_df,
    ) in qualified.groupby(
        "market_type",
        sort=False,
    ):
        if not isinstance(market, str):
            raise RuntimeError(
                "market_type must be a string"
            )

        preference = config[
            market
        ].get(
            "pick_preference",
            "all",
        )

        for (
            _,
            game_df,
        ) in market_df.groupby(
            [
                "game_date",
                "game_id",
            ],
            sort=False,
        ):
            if (
                preference
                == "all"
            ):
                selected.append(
                    game_df
                )

                continue

            if (
                preference
                == "best_ev"
            ):
                metric = "ev"

            elif (
                preference
                == "best_prob"
            ):
                metric = (
                    "model_prob"
                )

            else:
                raise RuntimeError(
                    "INVALID "
                    "pick_preference | "
                    f"market={market} "
                    f"value={preference!r}"
                )

            winners = (
                game_df[
                    game_df[
                        metric
                    ]
                    == game_df[
                        metric
                    ].max()
                ]
            )

            if (
                len(
                    winners
                )
                != 1
            ):
                raise RuntimeError(
                    "pick_preference tie | "
                    f"market={market} "
                    f"preference={preference} "
                    "game_id="
                    f"{game_df.iloc[0]['game_id']}"
                )

            selected.append(
                winners
            )

    if not selected:
        return pd.DataFrame(
            columns=(
                candidates.columns
            )
        )

    return pd.concat(
        selected,
        ignore_index=True,
    )


def load_scores():
    files = sorted(
        SCORE_DIR.glob(
            "*_NHL_final_scores.csv"
        )
    )

    if not files:
        raise RuntimeError(
            "NO FINAL SCORE FILES | "
            f"{SCORE_DIR}"
        )

    scores = {}

    for path in files:
        df = pd.read_csv(
            path,
            dtype=str,
        )

        if df.empty:
            continue

        required = [
            "game_id",
            "away_score",
            "home_score",
        ]

        missing = [
            column
            for column in required
            if column
            not in df.columns
        ]

        if missing:
            raise RuntimeError(
                "MISSING FINAL SCORE "
                "COLUMNS | "
                f"{path} | "
                f"{missing}"
            )

        for row in (
            df.to_dict(
                "records"
            )
        ):
            game_id = str(
                row.get(
                    "game_id",
                    "",
                )
            ).strip()

            away_score = (
                to_float(
                    row.get(
                        "away_score"
                    )
                )
            )

            home_score = (
                to_float(
                    row.get(
                        "home_score"
                    )
                )
            )

            if (
                not valid_game_id(
                    game_id
                )
                or away_score
                is None
                or home_score
                is None
            ):
                continue

            value = (
                away_score,
                home_score,
                path.name,
            )

            if (
                game_id
                in scores
                and scores[
                    game_id
                ][:2]
                != value[:2]
            ):
                raise RuntimeError(
                    "CONFLICTING FINAL "
                    "SCORES | "
                    f"game_id={game_id}"
                )

            scores[
                game_id
            ] = value

    return scores


def outcome(
    row,
    away_score,
    home_score,
):
    market = row[
        "market_type"
    ]

    side = row[
        "bet_side"
    ]

    line = row[
        "line"
    ]

    if market == "moneyline":
        if (
            away_score
            == home_score
        ):
            return "Push"

        if side == "home":
            return (
                "Win"
                if home_score
                > away_score
                else "Loss"
            )

        if side == "away":
            return (
                "Win"
                if away_score
                > home_score
                else "Loss"
            )

    if market == "puck_line":
        if side == "home":
            diff = (
                home_score
                - away_score
                + line
            )

        else:
            diff = (
                away_score
                - home_score
                + line
            )

        if (
            abs(diff)
            < 1e-9
        ):
            return "Push"

        return (
            "Win"
            if diff > 0
            else "Loss"
        )

    if market == "total":
        diff = (
            away_score
            + home_score
            - line
        )

        if (
            abs(diff)
            < 1e-9
        ):
            return "Push"

        if side == "over":
            return (
                "Win"
                if diff > 0
                else "Loss"
            )

        if side == "under":
            return (
                "Win"
                if diff < 0
                else "Loss"
            )

    return "Unknown"


def grade(
    selected,
    scores,
):
    rows = []

    for row in (
        selected.to_dict(
            "records"
        )
    ):
        score = scores.get(
            row[
                "game_id"
            ]
        )

        out = dict(
            row
        )

        if score is None:
            out.update(
                {
                    "bet_result": "Unknown",
                    "profit_units": np.nan,
                    "source_score_file": "",
                }
            )

        else:
            result = outcome(
                row,
                score[0],
                score[1],
            )

            if (
                result
                == "Win"
            ):
                profit = (
                    row[
                        "odds_decimal"
                    ]
                    - 1.0
                )

            elif (
                result
                == "Loss"
            ):
                profit = -1.0

            elif (
                result
                == "Push"
            ):
                profit = 0.0

            else:
                profit = np.nan

            out.update(
                {
                    "bet_result": result,
                    "profit_units": profit,
                    "source_score_file": score[2],
                }
            )

        rows.append(
            out
        )

    columns = (
        list(
            selected.columns
        )
        + [
            "bet_result",
            "profit_units",
            "source_score_file",
        ]
    )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


SETTLED_RESULTS = (
    "Win",
    "Loss",
    "Push",
)


def settled_rows(
    graded,
):
    return graded[
        graded[
            "bet_result"
        ].isin(
            SETTLED_RESULTS
        )
    ].copy()


def add_time_split_columns(
    settled,
):
    out = settled.copy()
    out[
        "season"
    ] = out[
        "game_date"
    ].map(
        season_label
    )
    out[
        "season_half"
    ] = out[
        "game_date"
    ].map(
        season_half_label
    )
    return out


def metrics(
    df,
):
    settled = settled_rows(df)

    decisions = settled[
        settled[
            "bet_result"
        ].isin(
            [
                "Win",
                "Loss",
            ]
        )
    ].copy()

    wins = int(
        (
            settled[
                "bet_result"
            ]
            == "Win"
        ).sum()
    )

    losses = int(
        (
            settled[
                "bet_result"
            ]
            == "Loss"
        ).sum()
    )

    pushes = int(
        (
            settled[
                "bet_result"
            ]
            == "Push"
        ).sum()
    )

    sample_size = len(
        settled
    )

    decision_count = len(
        decisions
    )

    realized = np.nan
    expected = np.nan
    brier = np.nan
    log_loss = np.nan

    if decision_count:
        y = (
            (
                decisions[
                    "bet_result"
                ]
                == "Win"
            )
            .astype(float)
            .to_numpy()
        )

        p = (
            decisions[
                "model_prob"
            ]
            .astype(float)
            .to_numpy()
        )

        safe_p = np.clip(
            p,
            EPS,
            1.0 - EPS,
        )

        realized = float(
            y.mean()
        )

        expected = float(
            p.mean()
        )

        brier = float(
            np.mean(
                (
                    p
                    - y
                )
                ** 2
            )
        )

        log_loss = float(
            -np.mean(
                (
                    y
                    * np.log(
                        safe_p
                    )
                )
                + (
                    (
                        1.0
                        - y
                    )
                    * np.log(
                        1.0
                        - safe_p
                    )
                )
            )
        )

    net_units = (
        float(
            settled[
                "profit_units"
            ].sum()
        )
        if sample_size
        else 0.0
    )

    return {
        "sample_size": sample_size,
        "decision_count": decision_count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "realized_win_rate": realized,
        "expected_probability": expected,
        "calibration_gap": (
            realized
            - expected
            if decision_count
            else np.nan
        ),
        "brier_score": brier,
        "log_loss": log_loss,
        "net_units": net_units,
        "roi": (
            net_units
            / sample_size
            if sample_size
            else np.nan
        ),
        "avg_decimal_odds": (
            float(
                settled[
                    "odds_decimal"
                ].mean()
            )
            if sample_size
            else np.nan
        ),
        "avg_ev": (
            float(
                settled[
                    "ev"
                ].mean()
            )
            if sample_size
            else np.nan
        ),
        "avg_kelly": (
            float(
                settled[
                    "kelly"
                ].mean()
            )
            if sample_size
            else np.nan
        ),
    }


def build_summary(
    selected,
    graded,
    config,
):
    rows = []

    for (
        market,
        spec,
    ) in MARKETS.items():

        for side in spec[
            "sides"
        ]:
            selected_group = (
                selected[
                    (
                        selected[
                            "market_type"
                        ]
                        == market
                    )
                    & (
                        selected[
                            "bet_side"
                        ]
                        == side
                    )
                ]
            )

            graded_group = (
                graded[
                    (
                        graded[
                            "market_type"
                        ]
                        == market
                    )
                    & (
                        graded[
                            "bet_side"
                        ]
                        == side
                    )
                ]
            )

            market_config = (
                config.get(
                    market,
                    {},
                )
            )

            side_config = (
                market_config.get(
                    side,
                    {},
                )
            )

            rows.append(
                {
                    "market_type": market,
                    "bet_side": side,
                    "configured_enabled": bool(
                        market_config.get(
                            "enabled",
                            False,
                        )
                        and side_config.get(
                            "enabled",
                            False,
                        )
                    ),
                    "pick_preference": market_config.get(
                        "pick_preference",
                        "",
                    ),
                    "selected_candidates": len(
                        selected_group
                    ),
                    "missing_final_score": int(
                        (
                            graded_group[
                                "bet_result"
                            ]
                            == "Unknown"
                        ).sum()
                    ),
                    **metrics(
                        graded_group
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def bucket_label(
    probability,
):
    index = min(
        int(
            probability
            * 10
        ),
        9,
    )

    return (
        f"{index / 10:.1f}-"
        f"{(index + 1) / 10:.1f}"
    )


def build_reliability(
    graded,
):
    settled = settled_rows(graded)

    if settled.empty:
        return pd.DataFrame()

    settled[
        "probability_bucket"
    ] = settled[
        "model_prob"
    ].map(
        bucket_label
    )

    rows = []

    grouped = settled.groupby(
        [
            "market_type",
            "bet_side",
            "probability_bucket",
        ],
        sort=True,
    )

    for (
        market,
        side,
        bucket,
    ), group in grouped:

        rows.append(
            {
                "market_type": market,
                "bet_side": side,
                "probability_bucket": bucket,
                **metrics(
                    group
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _positive_signal_bucket(
    value: float,
):
    buckets = (
        (
            0.02,
            1,
            "0.00-0.02",
        ),
        (
            0.05,
            2,
            "0.02-0.05",
        ),
        (
            0.10,
            3,
            "0.05-0.10",
        ),
        (
            0.20,
            4,
            "0.10-0.20",
        ),
    )

    for (
        threshold,
        order,
        label,
    ) in buckets:
        if value < threshold:
            return (
                order,
                label,
            )

    return (
        5,
        "0.20+",
    )


def ev_bucket(
    value,
):
    value = float(
        value
    )

    if value < 0.0:
        return (
            0,
            "<0.00",
        )

    return _positive_signal_bucket(
        value
    )


def kelly_bucket(
    value,
):
    value = float(
        value
    )

    if value <= 0.0:
        return (
            0,
            "0.00",
        )

    return _positive_signal_bucket(
        value
    )


def add_signal_bucket_columns(
    df,
    signal_type,
):
    out = df.copy()

    if signal_type == "ev":
        bucket_func = ev_bucket

    elif signal_type == "kelly":
        bucket_func = kelly_bucket

    else:
        raise RuntimeError(
            "INVALID SIGNAL TYPE | "
            f"{signal_type}"
        )

    bucket_values = out[
        signal_type
    ].map(
        bucket_func
    )

    out[
        "signal_type"
    ] = signal_type

    out[
        "signal_bucket_order"
    ] = bucket_values.map(
        lambda value: value[0]
    )

    out[
        "signal_bucket"
    ] = bucket_values.map(
        lambda value: value[1]
    )

    return out


def build_signal_buckets(
    graded,
):
    settled = settled_rows(graded)

    if settled.empty:
        return pd.DataFrame()

    rows = []

    for signal_type in (
        "ev",
        "kelly",
    ):
        signal_df = (
            add_signal_bucket_columns(
                settled,
                signal_type,
            )
        )

        grouped = signal_df.groupby(
            [
                "market_type",
                "bet_side",
                "signal_bucket_order",
                "signal_bucket",
            ],
            sort=True,
        )

        for (
            market,
            side,
            bucket_order,
            bucket,
        ), group in grouped:

            signal_values = (
                group[
                    signal_type
                ].astype(float)
            )

            rows.append(
                {
                    "signal_type": signal_type,
                    "market_type": market,
                    "bet_side": side,
                    "signal_bucket_order": bucket_order,
                    "signal_bucket": bucket,
                    "avg_signal_value": float(
                        signal_values.mean()
                    ),
                    "min_signal_value": float(
                        signal_values.min()
                    ),
                    "max_signal_value": float(
                        signal_values.max()
                    ),
                    **metrics(
                        group
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def season_label(
    game_date,
):
    date = pd.Timestamp(
        game_date
    )

    start_year = (
        date.year
        if date.month >= 7
        else date.year - 1
    )

    return (
        f"{start_year}-"
        f"{str(start_year + 1)[-2:]}"
    )


def season_half_label(
    game_date,
):
    date = pd.Timestamp(
        game_date
    )

    half = (
        "first_half"
        if date.month >= 7
        else "second_half"
    )

    return (
        f"{season_label(date)}_"
        f"{half}"
    )


def build_time_splits(
    graded,
):
    settled = settled_rows(graded)

    if settled.empty:
        return pd.DataFrame()

    settled = add_time_split_columns(settled)

    rows = []

    for split_type in (
        "season",
        "season_half",
    ):
        grouped = settled.groupby(
            [
                split_type,
                "market_type",
                "bet_side",
            ],
            sort=True,
        )

        for (
            value,
            market,
            side,
        ), group in grouped:

            rows.append(
                {
                    "split_type": split_type,
                    "split_value": value,
                    "market_type": market,
                    "bet_side": side,
                    **metrics(
                        group
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_signal_time_splits(
    graded,
):
    settled = settled_rows(graded)

    if settled.empty:
        return pd.DataFrame()

    settled = add_time_split_columns(settled)

    rows = []

    for signal_type in (
        "ev",
        "kelly",
    ):
        signal_df = (
            add_signal_bucket_columns(
                settled,
                signal_type,
            )
        )

        for split_type in (
            "season",
            "season_half",
        ):
            grouped = signal_df.groupby(
                [
                    split_type,
                    "market_type",
                    "bet_side",
                    "signal_bucket_order",
                    "signal_bucket",
                ],
                sort=True,
            )

            for (
                split_value,
                market,
                side,
                bucket_order,
                bucket,
            ), group in grouped:

                signal_values = (
                    group[
                        signal_type
                    ].astype(float)
                )

                rows.append(
                    {
                        "split_type": split_type,
                        "split_value": split_value,
                        "signal_type": signal_type,
                        "market_type": market,
                        "bet_side": side,
                        "signal_bucket_order": bucket_order,
                        "signal_bucket": bucket,
                        "avg_signal_value": float(
                            signal_values.mean()
                        ),
                        "min_signal_value": float(
                            signal_values.min()
                        ),
                        "max_signal_value": float(
                            signal_values.max()
                        ),
                        **metrics(
                            group
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def main():
    config = load_config()

    (
        candidates,
        stats,
    ) = load_candidates()

    selected = (
        apply_current_config(
            candidates,
            config,
        )
    )

    scores = load_scores()

    graded = grade(
        selected,
        scores,
    )

    summary = build_summary(
        selected,
        graded,
        config,
    )

    reliability = (
        build_reliability(
            graded
        )
    )

    time_splits = (
        build_time_splits(
            graded
        )
    )

    signal_buckets = (
        build_signal_buckets(
            graded
        )
    )

    signal_time_splits = (
        build_signal_time_splits(
            graded
        )
    )

    atomic_write(
        summary,
        SUMMARY_FILE,
    )

    atomic_write(
        reliability,
        RELIABILITY_FILE,
    )

    atomic_write(
        time_splits,
        TIME_SPLIT_FILE,
    )

    atomic_write(
        signal_buckets,
        SIGNAL_BUCKET_FILE,
    )

    atomic_write(
        signal_time_splits,
        SIGNAL_TIME_SPLIT_FILE,
    )

    print(
        "NHL market calibration "
        "audit complete."
    )

    print(
        "Stage 02 files seen: "
        f"{stats['files_seen']}"
    )

    print(
        "Stage 02 files loaded: "
        f"{stats['files_loaded']}"
    )

    print(
        "Stage 02 files skipped "
        "for schema: "
        f"{stats['files_skipped_schema']}"
    )

    print(
        "Stage 02 source rows seen: "
        f"{stats['source_rows_seen']}"
    )

    print(
        "Invalid candidate sides "
        "skipped: "
        f"{stats['invalid_candidate_sides']}"
    )

    print(
        "Invalid puck-line source "
        "rows skipped: "
        f"{stats['invalid_puck_line_rows']}"
    )

    print(
        "Valid candidate sides: "
        f"{len(candidates)}"
    )

    print(
        "Selected by current "
        "markets.yaml: "
        f"{len(selected)}"
    )

    missing_scores = (
        int(
            (
                graded[
                    "bet_result"
                ]
                == "Unknown"
            ).sum()
        )
        if not graded.empty
        else 0
    )

    print(
        "Selected rows missing "
        "final score: "
        f"{missing_scores}"
    )

    print(
        f"WROTE: "
        f"{SUMMARY_FILE}"
    )

    print(
        f"WROTE: "
        f"{RELIABILITY_FILE}"
    )

    print(
        f"WROTE: "
        f"{TIME_SPLIT_FILE}"
    )

    print(
        f"WROTE: "
        f"{SIGNAL_BUCKET_FILE}"
    )

    print(
        f"WROTE: "
        f"{SIGNAL_TIME_SPLIT_FILE}"
    )

    print(
        "Treat these results as "
        "out-of-sample only for dates "
        "that were not used to choose "
        "or tune the thresholds in "
        "markets.yaml."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as main_exc:
        print(
            "audit_market_calibration "
            f"failed: {main_exc}",
            file=sys.stderr,
        )

        raise