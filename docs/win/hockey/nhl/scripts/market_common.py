#!/usr/bin/env python3
from __future__ import annotations


def normalize_market(value) -> str:
    value = str(value).strip().lower()

    if value in {"moneyline", "ml"}:
        return "moneyline"

    if value in {"puck_line", "puckline", "spread"}:
        return "puck_line"

    if value in {"total", "totals"}:
        return "total"

    return value


def normalize_side(value) -> str:
    return str(value).strip().lower()
