#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/03_edges/build_secondary_model_signals.py

from __future__ import annotations

import math
import sys

import traceback
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Never

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.stats import poisson, skellam


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# noinspection PyPep8
from model_blend_common import (
    design_matrix,
    fit_blend_weights,
    ridge_linear_coefficients,
)


BASE_DIR = Path(__file__).resolve().parents[2]
EV_DIR = BASE_DIR / "03_edges" / "ev_kelly"
MERGE_DIR = BASE_DIR / "01_merge"
OUTPUT_DIR = BASE_DIR / "03_edges" / "secondary_signals"
HISTORY_ROOT = BASE_DIR / "research" / "sdv_challenger"
CONFIG_PATH = BASE_DIR / "config" / "markets.yaml"
ERROR_DIR = BASE_DIR / "errors" / "03_edges"
LOG_FILE = ERROR_DIR / "build_secondary_model_signals.txt"

SIGNAL_VERSION = "P6-production-v1"
EPS = 1e-6

DIRECT_PROBABILITY_COLUMNS = [
    "my_model_home_prob_moneyline",
    "my_model_away_prob_moneyline",
    "my_model_home_prob_puck_line",
    "my_model_away_prob_puck_line",
    "my_model_over_prob_total",
    "my_model_under_prob_total",
]

SIGNAL_COLUMNS = [
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
    "secondary_history_max_game_date",
    "secondary_model_status",
    "secondary_signal_version",
]

HISTORY_REQUIRED_COLUMNS = [
    "game_id",
    "game_date",
    "sdv_home_win_prob",
    "sdv_exp_margin",
    "sdv_exp_total",
    "drat_home_win_prob",
    "drat_exp_margin",
    "drat_exp_total",
    "actual_home_win",
    "actual_margin",
    "actual_total",
]

MERGED_REQUIRED_COLUMNS = [
    "game_id",
    "game_date",
    "home_prob_moneyline",
    "away_projected_goals",
    "home_projected_goals",
    "total_projected_goals",
    "sdv_home_win_prob",
    "sdv_exp_margin",
    "sdv_exp_total",
]


@dataclass(frozen=True)
class LinearModel:
    coefficients: np.ndarray


@dataclass(frozen=True)
class LogisticModel:
    coefficients: np.ndarray


@dataclass(frozen=True)
class ModelBundle:
    train_rows: int
    history_max_game_date: str
    prob_threshold: float
    margin_threshold: float
    total_threshold: float
    prob_weight: float
    margin_weight: float
    total_weight: float
    drat_cal: LogisticModel
    sdv_cal: LogisticModel
    prob_meta: LogisticModel
    margin_meta: LinearModel
    total_meta: LinearModel


def now() -> str:
    return datetime.now(UTC).isoformat()


def reset_log() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== build_secondary_model_signals RUN {now()} ===\n",
        encoding="utf-8",
    )


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{now()} | {message}\n")


def fail(message: str) -> Never:
    log(f"FATAL | {message}")
    raise SystemExit(message)


def canonical_game_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalized_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype("string").str.replace("_", "-", regex=False),
        errors="coerce",
    ).dt.date


def load_secondary_config() -> dict:
    if not CONFIG_PATH.exists():
        fail(f"Missing markets config: {CONFIG_PATH}")

    try:
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"Unable to read {CONFIG_PATH}: {exc}")

    try:
        config = payload["markets"]["nhl"]["secondary_model"]
    except Exception as exc:
        return fail(f"Missing markets.nhl.secondary_model in {CONFIG_PATH}: {exc}")

    return config


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        fail(f"{label} missing required columns: {missing}")


def load_merged_predictions() -> pd.DataFrame:
    files = sorted(MERGE_DIR.glob("*_NHL_merged.csv"))
    if not files:
        fail(f"No Stage 01 merged files found in {MERGE_DIR}")

    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, dtype={"game_id": "string"})
        require_columns(df, MERGED_REQUIRED_COLUMNS, str(path))
        if df.empty:
            continue
        df = df[MERGED_REQUIRED_COLUMNS].copy()
        df["_source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=MERGED_REQUIRED_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out["game_id"] = out["game_id"].map(canonical_game_id)
    if out["game_id"].eq("").any():
        fail("Stage 01 merged predictions contain blank game_id")

    dupes = out[out.duplicated("game_id", keep=False)]
    if not dupes.empty:
        ids = sorted(dupes["game_id"].unique().tolist())
        fail(f"Stage 01 merged predictions contain duplicate game_id: {ids[:10]}")

    numeric = [
        "home_prob_moneyline",
        "away_projected_goals",
        "home_projected_goals",
        "total_projected_goals",
        "sdv_home_win_prob",
        "sdv_exp_margin",
        "sdv_exp_total",
    ]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["drat_home_win_prob"] = out["home_prob_moneyline"]
    out["drat_exp_margin"] = (
        out["home_projected_goals"] - out["away_projected_goals"]
    )
    out["drat_exp_total"] = out["total_projected_goals"]
    out["_target_date"] = normalized_date_series(out["game_date"])

    if out["_target_date"].isna().any():
        bad = out.loc[out["_target_date"].isna(), ["game_id", "game_date"]]
        fail(f"Stage 01 merged predictions contain invalid game_date: {bad.to_dict(orient='records')[:5]}")

    return out[
        [
            "game_id",
            "game_date",
            "_target_date",
            "drat_home_win_prob",
            "drat_exp_margin",
            "drat_exp_total",
            "sdv_home_win_prob",
            "sdv_exp_margin",
            "sdv_exp_total",
        ]
    ].copy()


def load_history() -> pd.DataFrame:
    files = sorted(HISTORY_ROOT.glob("season_*/standalone_comparison.csv"))
    frames: list[pd.DataFrame] = []

    for path in files:
        try:
            df = pd.read_csv(path, dtype={"game_id": "string"})
        except Exception as exc:
            log(f"WARN | unable to read historical comparison {path}: {exc}")
            continue

        missing = [c for c in HISTORY_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            log(f"WARN | skipping {path}; missing columns={missing}")
            continue

        df = df[HISTORY_REQUIRED_COLUMNS].copy()
        df["_source_file"] = str(path)
        frames.append(df)

    if not frames:
        log("No usable P6 standalone comparison history found.")
        return pd.DataFrame(columns=HISTORY_REQUIRED_COLUMNS + ["_date"])

    out = pd.concat(frames, ignore_index=True)
    out["game_id"] = out["game_id"].map(canonical_game_id)
    out["_date"] = normalized_date_series(out["game_date"])

    numeric = [c for c in HISTORY_REQUIRED_COLUMNS if c not in {"game_id", "game_date"}]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["_date", *numeric]).copy()
    out = out.sort_values(["_date", "game_id", "_source_file"])
    out = out.drop_duplicates("game_id", keep="last")

    log(
        f"Loaded P6 completed comparison history: rows={len(out)} files={len(files)}"
    )
    return out


def clipped_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = clipped_prob(p)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(y, float)) ** 2)))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> LogisticModel:
    y = np.asarray(y, dtype=float)
    design = design_matrix(x)

    def objective(beta: np.ndarray) -> float:
        z = np.clip(design @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        nll = -np.sum(
            y * np.log(np.clip(p, EPS, 1.0 - EPS))
            + (1.0 - y) * np.log(np.clip(1.0 - p, EPS, 1.0 - EPS))
        )
        return float(nll + 1e-4 * np.sum(beta[1:] ** 2))

    result = minimize(objective, np.zeros(design.shape[1]), method="BFGS")
    if not np.isfinite(result.fun):
        raise RuntimeError("logistic calibration fit produced a non-finite objective")
    return LogisticModel(result.x)


def apply_logistic(model: LogisticModel, x: np.ndarray) -> np.ndarray:
    design = design_matrix(x)
    z = np.clip(design @ model.coefficients, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_linear(
    x: np.ndarray,
    y: np.ndarray,
) -> LinearModel:
    return LinearModel(
        ridge_linear_coefficients(x, y)
    )


def apply_linear(model: LinearModel, x: np.ndarray) -> np.ndarray:
    design = design_matrix(x)
    return design @ model.coefficients


def fit_bundle_for_target(
    history: pd.DataFrame,
    target_day: date,
    *,
    min_train_rows: int,
    disagreement_quantile: float,
) -> ModelBundle | None:
    train = history[history["_date"] < target_day].copy()
    if len(train) < min_train_rows:
        return None
    if train["actual_home_win"].nunique() < 2:
        return None

    y = train["actual_home_win"].to_numpy(float)
    (
        drat_cal,
        sdv_cal,
        prob_weight,
        margin_weight,
        total_weight,
    ) = fit_blend_weights(
        train,
        y,
        fit_logistic,
        apply_logistic,
    )

    prob_disagreement = (train["sdv_home_win_prob"] - train["drat_home_win_prob"]).abs()
    margin_disagreement = (train["sdv_exp_margin"] - train["drat_exp_margin"]).abs()
    total_disagreement = (train["sdv_exp_total"] - train["drat_exp_total"]).abs()

    prob_meta = fit_logistic(
        np.column_stack(
            [
                train["drat_home_win_prob"].to_numpy(float),
                train["sdv_home_win_prob"].to_numpy(float),
                prob_disagreement.to_numpy(float),
            ]
        ),
        y,
    )
    margin_meta = fit_linear(
        np.column_stack(
            [
                train["drat_exp_margin"].to_numpy(float),
                train["sdv_exp_margin"].to_numpy(float),
                margin_disagreement.to_numpy(float),
            ]
        ),
        train["actual_margin"].to_numpy(float),
    )
    total_meta = fit_linear(
        np.column_stack(
            [
                train["drat_exp_total"].to_numpy(float),
                train["sdv_exp_total"].to_numpy(float),
                total_disagreement.to_numpy(float),
            ]
        ),
        train["actual_total"].to_numpy(float),
    )

    history_max = max(train["_date"]).isoformat()
    return ModelBundle(
        train_rows=len(train),
        history_max_game_date=history_max,
        prob_threshold=float(prob_disagreement.quantile(disagreement_quantile)),
        margin_threshold=float(margin_disagreement.quantile(disagreement_quantile)),
        total_threshold=float(total_disagreement.quantile(disagreement_quantile)),
        prob_weight=prob_weight,
        margin_weight=margin_weight,
        total_weight=total_weight,
        drat_cal=drat_cal,
        sdv_cal=sdv_cal,
        prob_meta=prob_meta,
        margin_meta=margin_meta,
        total_meta=total_meta,
    )


def blank_signal_row(base: pd.Series, status: str) -> dict:
    result = {column: np.nan for column in SIGNAL_COLUMNS}
    result.update(
        {
            "drat_home_win_prob": base.get("drat_home_win_prob"),
            "drat_exp_margin": base.get("drat_exp_margin"),
            "drat_exp_total": base.get("drat_exp_total"),
            "sdv_home_win_prob": base.get("sdv_home_win_prob"),
            "sdv_exp_margin": base.get("sdv_exp_margin"),
            "sdv_exp_total": base.get("sdv_exp_total"),
            "secondary_model_status": status,
            "secondary_signal_version": SIGNAL_VERSION,
        }
    )
    return result


def signal_row(base: pd.Series, bundle: ModelBundle | None, history_available: bool) -> dict:
    sdv_values = [
        base.get("sdv_home_win_prob"),
        base.get("sdv_exp_margin"),
        base.get("sdv_exp_total"),
    ]
    if any(pd.isna(v) for v in sdv_values):
        return blank_signal_row(base, "sdv_unavailable")

    prob_disagreement = abs(float(base["sdv_home_win_prob"]) - float(base["drat_home_win_prob"]))
    margin_disagreement = abs(float(base["sdv_exp_margin"]) - float(base["drat_exp_margin"]))
    total_disagreement = abs(float(base["sdv_exp_total"]) - float(base["drat_exp_total"]))

    if bundle is None:
        row = blank_signal_row(
            base,
            "insufficient_history" if history_available else "history_unavailable",
        )
        row.update(
            {
                "prob_disagreement": prob_disagreement,
                "margin_disagreement": margin_disagreement,
                "total_disagreement": total_disagreement,
            }
        )
        return row

    drat_prob = np.array([[float(base["drat_home_win_prob"])]])
    sdv_prob = np.array([[float(base["sdv_home_win_prob"])]])
    drat_cal = float(apply_logistic(bundle.drat_cal, drat_prob)[0])
    sdv_cal = float(apply_logistic(bundle.sdv_cal, sdv_prob)[0])
    weighted_prob = bundle.prob_weight * drat_cal + (1.0 - bundle.prob_weight) * sdv_cal

    weighted_margin = (
        bundle.margin_weight * float(base["drat_exp_margin"])
        + (1.0 - bundle.margin_weight) * float(base["sdv_exp_margin"])
    )
    weighted_total = (
        bundle.total_weight * float(base["drat_exp_total"])
        + (1.0 - bundle.total_weight) * float(base["sdv_exp_total"])
    )

    meta_prob = float(
        apply_logistic(
            bundle.prob_meta,
            np.array(
                [[
                    float(base["drat_home_win_prob"]),
                    float(base["sdv_home_win_prob"]),
                    prob_disagreement,
                ]]
            ),
        )[0]
    )
    meta_margin = float(
        apply_linear(
            bundle.margin_meta,
            np.array(
                [[
                    float(base["drat_exp_margin"]),
                    float(base["sdv_exp_margin"]),
                    margin_disagreement,
                ]]
            ),
        )[0]
    )
    meta_total = float(
        apply_linear(
            bundle.total_meta,
            np.array(
                [[
                    float(base["drat_exp_total"]),
                    float(base["sdv_exp_total"]),
                    total_disagreement,
                ]]
            ),
        )[0]
    )

    return {
        "drat_home_win_prob": float(base["drat_home_win_prob"]),
        "drat_exp_margin": float(base["drat_exp_margin"]),
        "drat_exp_total": float(base["drat_exp_total"]),
        "sdv_home_win_prob": float(base["sdv_home_win_prob"]),
        "sdv_exp_margin": float(base["sdv_exp_margin"]),
        "sdv_exp_total": float(base["sdv_exp_total"]),
        "prob_disagreement": prob_disagreement,
        "margin_disagreement": margin_disagreement,
        "total_disagreement": total_disagreement,
        "prob_disagreement_threshold_p75_prior": bundle.prob_threshold,
        "margin_disagreement_threshold_p75_prior": bundle.margin_threshold,
        "total_disagreement_threshold_p75_prior": bundle.total_threshold,
        "high_prob_disagreement_flag": int(prob_disagreement >= bundle.prob_threshold),
        "high_margin_disagreement_flag": int(margin_disagreement >= bundle.margin_threshold),
        "high_total_disagreement_flag": int(total_disagreement >= bundle.total_threshold),
        "ensemble_train_rows": bundle.train_rows,
        "weighted_prob_drat_weight": bundle.prob_weight,
        "weighted_margin_drat_weight": bundle.margin_weight,
        "weighted_total_drat_weight": bundle.total_weight,
        "weighted_home_win_prob": float(np.clip(weighted_prob, 0.0, 1.0)),
        "weighted_exp_margin": weighted_margin,
        "weighted_exp_total": weighted_total,
        "meta_home_win_prob": float(np.clip(meta_prob, 0.0, 1.0)),
        "meta_exp_margin": meta_margin,
        "meta_exp_total": meta_total,
        "secondary_history_max_game_date": bundle.history_max_game_date,
        "secondary_model_status": "ready",
        "secondary_signal_version": SIGNAL_VERSION,
    }


def build_signal_frame(current: pd.DataFrame, history: pd.DataFrame, config: dict) -> pd.DataFrame:
    min_train_rows = int(config["min_train_rows"])
    disagreement_quantile = float(config["disagreement_quantile"])
    bundles: dict[date, ModelBundle | None] = {}

    for target_day in sorted(current["_target_date"].dropna().unique()):
        bundles[target_day] = fit_bundle_for_target(
            history,
            target_day,
            min_train_rows=min_train_rows,
            disagreement_quantile=disagreement_quantile,
        )

    rows: list[dict] = []
    history_available = not history.empty
    for _, base in current.iterrows():
        target_day = base["_target_date"]
        row = {"game_id": base["game_id"]}
        row.update(signal_row(base, bundles.get(target_day), history_available))
        rows.append(row)

    return pd.DataFrame(rows, columns=["game_id", *SIGNAL_COLUMNS])


def _derived_field(derived_model: str, weighted_field: str, meta_field: str) -> str:
    if derived_model == "weighted":
        return weighted_field
    if derived_model == "meta":
        return meta_field
    return fail(f"Unsupported secondary derived model: {derived_model!r}")


def _puck_probability(
    line,
    projected_goals,
    opponent_projected_goals,
):
    values = (
        line,
        projected_goals,
        opponent_projected_goals,
    )
    if any(pd.isna(value) for value in values):
        return np.nan

    line = float(line)
    projected_goals = float(projected_goals)
    opponent_projected_goals = float(opponent_projected_goals)

    if (
        not math.isfinite(line)
        or not math.isfinite(projected_goals)
        or not math.isfinite(opponent_projected_goals)
        or projected_goals <= 0
        or opponent_projected_goals <= 0
    ):
        return np.nan

    threshold = math.floor(-line)
    probability = 1.0 - skellam.cdf(
        threshold,
        projected_goals,
        opponent_projected_goals,
    )
    if pd.isna(probability):
        return np.nan

    return float(np.clip(probability, 0.01, 0.99))


def _total_probabilities(total_line, projected_total):
    if pd.isna(total_line) or pd.isna(projected_total):
        return np.nan, np.nan

    total_line = float(total_line)
    projected_total = float(projected_total)

    if (
        not math.isfinite(total_line)
        or not math.isfinite(projected_total)
        or projected_total <= 0
    ):
        return np.nan, np.nan

    if total_line.is_integer():
        push_total = int(total_line)
        under_prob = poisson.cdf(
            push_total - 1,
            projected_total,
        )
        over_prob = 1.0 - poisson.cdf(
            push_total,
            projected_total,
        )
        no_push_prob = under_prob + over_prob
        if pd.isna(no_push_prob) or no_push_prob <= 0:
            return np.nan, np.nan
        under_prob /= no_push_prob
        over_prob /= no_push_prob
    else:
        cutoff = math.floor(total_line)
        under_prob = poisson.cdf(
            cutoff,
            projected_total,
        )
        over_prob = 1.0 - under_prob

    if pd.isna(over_prob) or pd.isna(under_prob):
        return np.nan, np.nan

    return (
        float(np.clip(over_prob, 0.01, 0.99)),
        float(np.clip(under_prob, 0.01, 0.99)),
    )


def _validate_ready_probabilities(
    df: pd.DataFrame,
    ready: pd.Series,
    columns: list[str],
    market_type: str,
) -> None:
    missing_mask = ready & df[columns].isna().any(axis=1)
    if missing_mask.any():
        ids = (
            df.loc[missing_mask, "game_id"]
            .astype(str)
            .tolist()
        )
        fail(
            f"{market_type} secondary model is ready but direct "
            f"model probability is unavailable for game_id values: "
            f"{ids[:10]}"
        )


def add_direct_model_probabilities(
    df: pd.DataFrame,
    market_type: str,
    derived_model: str,
) -> pd.DataFrame:
    out = df.copy()

    for column in DIRECT_PROBABILITY_COLUMNS:
        out[column] = np.nan

    ready = (
        out["secondary_model_status"]
        .astype("string")
        .fillna("")
        .eq("ready")
    )

    if market_type == "moneyline":
        probability_field = _derived_field(
            derived_model,
            "weighted_home_win_prob",
            "meta_home_win_prob",
        )
        home_prob = pd.to_numeric(
            out[probability_field],
            errors="coerce",
        ).clip(0.0, 1.0)

        out.loc[
            ready,
            "my_model_home_prob_moneyline",
        ] = home_prob.loc[ready]
        out.loc[
            ready,
            "my_model_away_prob_moneyline",
        ] = 1.0 - home_prob.loc[ready]

        _validate_ready_probabilities(
            out,
            ready,
            [
                "my_model_home_prob_moneyline",
                "my_model_away_prob_moneyline",
            ],
            market_type,
        )
        return out

    if market_type == "puck_line":
        required = [
            "home_puck_line",
            "away_puck_line",
        ]
        missing = [column for column in required if column not in out.columns]
        if missing:
            fail(
                f"Puck-line EV file missing columns required for "
                f"secondary model probabilities: {missing}"
            )

        margin_field = _derived_field(
            derived_model,
            "weighted_exp_margin",
            "meta_exp_margin",
        )
        total_field = _derived_field(
            derived_model,
            "weighted_exp_total",
            "meta_exp_total",
        )

        for idx in out.index[ready]:
            margin = pd.to_numeric(
                pd.Series([out.at[idx, margin_field]]),
                errors="coerce",
            ).iloc[0]
            total = pd.to_numeric(
                pd.Series([out.at[idx, total_field]]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(margin) or pd.isna(total):
                continue

            home_goals = (float(total) + float(margin)) / 2.0
            away_goals = (float(total) - float(margin)) / 2.0

            out.at[
                idx,
                "my_model_home_prob_puck_line",
            ] = _puck_probability(
                out.at[idx, "home_puck_line"],
                home_goals,
                away_goals,
            )
            out.at[
                idx,
                "my_model_away_prob_puck_line",
            ] = _puck_probability(
                out.at[idx, "away_puck_line"],
                away_goals,
                home_goals,
            )

        _validate_ready_probabilities(
            out,
            ready,
            [
                "my_model_home_prob_puck_line",
                "my_model_away_prob_puck_line",
            ],
            market_type,
        )
        return out

    if market_type == "total":
        if "total" not in out.columns:
            fail(
                "Total EV file missing total column required for "
                "secondary model probabilities"
            )

        total_field = _derived_field(
            derived_model,
            "weighted_exp_total",
            "meta_exp_total",
        )

        for idx in out.index[ready]:
            over_prob, under_prob = _total_probabilities(
                out.at[idx, "total"],
                out.at[idx, total_field],
            )
            out.at[
                idx,
                "my_model_over_prob_total",
            ] = over_prob
            out.at[
                idx,
                "my_model_under_prob_total",
            ] = under_prob

        _validate_ready_probabilities(
            out,
            ready,
            [
                "my_model_over_prob_total",
                "my_model_under_prob_total",
            ],
            market_type,
        )
        return out

    return fail(f"Unsupported market type for secondary model probabilities: {market_type!r}")


def _market_type_from_path(path: Path) -> str:
    name = path.name
    if name.endswith("_NHL_moneyline.csv"):
        return "moneyline"
    if name.endswith("_NHL_puck_line.csv"):
        return "puck_line"
    if name.endswith("_NHL_total.csv"):
        return "total"
    return fail(f"Unable to determine market type from EV/Kelly file: {path}")


def wipe_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*.csv"):
        path.unlink()


def enrich_ev_files(signals: pd.DataFrame, config: dict) -> int:
    files = sorted(EV_DIR.glob("*_NHL_*.csv"))
    if not files:
        fail(f"No EV/Kelly market files found in {EV_DIR}")

    written = 0
    signal_ids = set(signals["game_id"].astype(str))

    for path in files:
        df = pd.read_csv(path, dtype={"game_id": "string"})
        if "game_id" not in df.columns:
            fail(f"EV/Kelly file missing game_id: {path}")
        df["game_id"] = df["game_id"].map(canonical_game_id)

        dupes = df[df.duplicated("game_id", keep=False)]
        if not dupes.empty:
            fail(f"EV/Kelly file has duplicate game_id: {path}")

        unknown = sorted(set(df["game_id"].astype(str)) - signal_ids)
        if unknown:
            fail(
                f"EV/Kelly file contains game_id absent from Stage 01 merged predictions: "
                f"{path} ids={unknown[:10]}"
            )

        existing_signal_columns = [
            col
            for col in [*SIGNAL_COLUMNS, *DIRECT_PROBABILITY_COLUMNS]
            if col in df.columns
        ]
        if existing_signal_columns:
            df = df.drop(columns=existing_signal_columns)

        enriched = df.merge(signals, on="game_id", how="left", validate="one_to_one")

        market_type = _market_type_from_path(path)
        derived_by_market = config.get("derived_signal_by_market", {})
        derived_model = str(
            derived_by_market.get(market_type, "")
        ).strip().lower()

        if derived_model not in {"weighted", "meta"}:
            fail(
                f"secondary_model derived signal is invalid for {market_type}: "
                f"{derived_model!r}"
            )

        enriched = add_direct_model_probabilities(
            enriched,
            market_type,
            derived_model,
        )

        out_path = OUTPUT_DIR / path.name
        enriched.to_csv(out_path, index=False)
        log(f"WROTE {out_path} rows={len(enriched)}")
        written += 1

    return written


def main() -> None:
    reset_log()
    try:
        config = load_secondary_config()
        current = load_merged_predictions()
        history = load_history()
        signals = build_signal_frame(current, history, config)

        wipe_outputs()
        written = enrich_ev_files(signals, config)

        statuses = signals["secondary_model_status"].value_counts(dropna=False).to_dict()
        log(
            f"STATUS: SUCCESS | signal_rows={len(signals)} files_written={written} "
            f"status_counts={statuses}"
        )
        print(
            "NHL secondary model signals complete: "
            f"signal_rows={len(signals)} files_written={written}"
        )
    except SystemExit:
        raise
    except Exception as exc:
        log(f"FATAL | {exc}\n{traceback.format_exc()}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
