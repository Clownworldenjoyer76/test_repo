#!/usr/bin/env python3
from __future__ import annotations

from datetime import date

import polars as pl


def prior_adjusted_ratings(
    game_rates: pl.DataFrame,
    target_date: date,
) -> pl.DataFrame:
    from sportsdataverse.nhl.nhl_prediction_constants import get_constants
    from sportsdataverse.nhl.nhl_team_ratings import adjust_rate_opponent

    prior = game_rates.filter(
        pl.col("date") < pl.lit(target_date)
    )
    if prior.is_empty():
        return pl.DataFrame()

    const = get_constants("nhl")

    xg_adj = adjust_rate_opponent(
        prior,
        for_col="xgf",
        against_col="xga",
        hfa=const.hfa,
        avg=const.avg_xgf,
        shrink_k=const.shrink_k,
    )
    goal_adj = adjust_rate_opponent(
        prior,
        for_col="gf",
        against_col="ga",
        hfa=const.hfa,
        avg=const.avg_total_goals / 2.0,
        shrink_k=const.shrink_k,
    )

    if xg_adj.is_empty():
        return pl.DataFrame()

    return xg_adj.join(
        goal_adj.select(
            "team",
            pl.col("adj_for").alias("adj_gf"),
            pl.col("adj_against").alias("adj_ga"),
        ),
        on="team",
        how="left",
    ).rename(
        {
            "adj_for": "adj_xgf",
            "adj_against": "adj_xga",
            "adj_net": "adj_xg_net",
        }
    )
