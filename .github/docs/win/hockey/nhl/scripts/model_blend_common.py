#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6


def select_probability_weight(
    y: np.ndarray,
    p_drat: np.ndarray,
    p_sdv: np.ndarray,
) -> float:
    grid = np.linspace(0.0, 1.0, 101)
    losses = []
    for weight in grid:
        probability = weight * p_drat + (1.0 - weight) * p_sdv
        probability = np.clip(probability, EPS, 1.0 - EPS)
        losses.append(
            float(
                -np.mean(
                    y * np.log(probability)
                    + (1.0 - y) * np.log(1.0 - probability)
                )
            )
        )
    return float(grid[int(np.argmin(losses))])


def select_numeric_weight(
    y: np.ndarray,
    drat: np.ndarray,
    sdv: np.ndarray,
) -> float:
    grid = np.linspace(0.0, 1.0, 101)
    losses = [
        float(
            np.sqrt(
                np.mean(
                    (weight * drat + (1.0 - weight) * sdv - y) ** 2
                )
            )
        )
        for weight in grid
    ]
    return float(grid[int(np.argmin(losses))])


def design_matrix(
    x: np.ndarray,
) -> np.ndarray:
    features = np.asarray(x, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    return np.column_stack(
        [np.ones(len(features)), features]
    )


def ridge_linear_coefficients(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    target = np.asarray(y, dtype=float)
    design = design_matrix(x)
    ridge = 1e-6 * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + ridge,
        design.T @ target,
    )


def fit_blend_weights(
    train: pd.DataFrame,
    y: np.ndarray,
    fit_logistic: Callable[[np.ndarray, np.ndarray], Any],
    apply_logistic: Callable[[Any, np.ndarray], np.ndarray],
) -> tuple[Any, Any, float, float, float]:
    drat_probability = train[["drat_home_win_prob"]].to_numpy(float)
    sdv_probability = train[["sdv_home_win_prob"]].to_numpy(float)

    drat_cal = fit_logistic(drat_probability, y)
    sdv_cal = fit_logistic(sdv_probability, y)

    prob_weight = select_probability_weight(
        y,
        apply_logistic(drat_cal, drat_probability),
        apply_logistic(sdv_cal, sdv_probability),
    )
    margin_weight = select_numeric_weight(
        train["actual_margin"].to_numpy(float),
        train["drat_exp_margin"].to_numpy(float),
        train["sdv_exp_margin"].to_numpy(float),
    )
    total_weight = select_numeric_weight(
        train["actual_total"].to_numpy(float),
        train["drat_exp_total"].to_numpy(float),
        train["sdv_exp_total"].to_numpy(float),
    )

    return (
        drat_cal,
        sdv_cal,
        prob_weight,
        margin_weight,
        total_weight,
    )
