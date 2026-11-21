#!/usr/bin/env python3
"""Single NHL price contract: the American quote determines executable decimal payout.

ESPN's supplied decimal is typically displayed to two places; its fractional
quote agrees exactly with its American quote in verified snapshots. Retain raw
snapshots for provenance. Never infer an actual ticket execution from this feed.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def american_to_decimal(value) -> float | None:
    try:
        american = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(american) or american == 0:
        return None
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def reconcile_dataframe_prices(
    df: pd.DataFrame,
    price_pairs: Iterable[tuple[str, str]],
    *,
    log: Callable[[str, str], None] | None = None,
) -> pd.DataFrame:
    """Replace inherited rounded decimals with decimal payouts from American.

    If an American column is available but a row's quote is invalid, its decimal
    becomes NaN rather than falling back to the potentially contradictory feed.
    Legacy tables without American columns retain existing decimals.
    """
    for american_col, decimal_col in price_pairs:
        if american_col not in df.columns:
            if log is not None:
                log(f"PRICE WARNING missing American column: {american_col}", "WARN")
            continue
        american = pd.to_numeric(df[american_col], errors="coerce")
        existing = pd.to_numeric(df.get(decimal_col), errors="coerce")
        canonical = pd.Series(np.nan, index=df.index, dtype="float64")
        pos = (american > 0) & np.isfinite(american)
        neg = (american < 0) & np.isfinite(american)
        canonical.loc[pos] = 1.0 + american.loc[pos] / 100.0
        canonical.loc[neg] = 1.0 + 100.0 / american.loc[neg].abs()
        mismatch = (
            canonical.notna()
            & existing.notna()
            & np.isfinite(existing)
            & (canonical.sub(existing).abs() > 1e-6)
        )
        if log is not None and mismatch.any():
            log(
                f"PRICE RECONCILED {decimal_col}: {int(mismatch.sum())} "
                "ESPN rounded decimal quotes replaced using American odds",
                "WARN",
            )
        df[decimal_col] = canonical
    return df


def price_metrics(probability, decimal):
    """Return edge, EV and nonnegative full Kelly for a consistent payout."""
    prob = pd.to_numeric(probability, errors="coerce")
    dec = pd.to_numeric(decimal, errors="coerce")
    valid = (
        prob.notna() & dec.notna()
        & np.isfinite(prob) & np.isfinite(dec)
        & (prob > 0) & (prob < 1) & (dec > 1)
    )
    edge = pd.Series(np.nan, index=prob.index, dtype="float64")
    ev = edge.copy()
    kelly = edge.copy()
    edge.loc[valid] = prob.loc[valid] - 1.0 / dec.loc[valid]
    ev.loc[valid] = prob.loc[valid] * dec.loc[valid] - 1.0
    kelly.loc[valid] = (
        (prob.loc[valid] * (dec.loc[valid] - 1.0) - (1.0 - prob.loc[valid]))
        / (dec.loc[valid] - 1.0)
    ).clip(lower=0)
    return edge, ev, kelly
