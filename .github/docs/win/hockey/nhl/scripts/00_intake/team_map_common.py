#!/usr/bin/env python3
from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Mapping, MutableMapping
from pathlib import Path
from typing import Any


TeamIdentity = dict[str, str]


def strip_record(value: str) -> str:
    return re.sub(
        r"\s*\(\d+[-–]\d+[-–]?\d*\)\s*$",
        "",
        str(value),
    ).strip()


def normalize_alias_key(value: str) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(value).strip(),
    )
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )
    text = text.lower().replace("&", " and ")
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_team_alias_key(value: str) -> str:
    return normalize_alias_key(
        strip_record(value)
    )


def parse_nhl_team_map_row(
    row: Mapping[str, Any],
    row_number: int,
    map_path: Path,
    *,
    supported_sources: Collection[str] | None = None,
) -> tuple[str, str, str, str, TeamIdentity] | None:
    if str(row.get("league", "")).strip().lower() != "nhl":
        return None

    source = str(row.get("source", "")).strip().lower()
    alias = str(row.get("alias", "")).strip()
    canonical = str(row.get("canonical_team", "")).strip()
    team_id = str(row.get("nhl_team_id", "")).strip()
    abbrev = str(row.get("nhl_abbrev", "")).strip().upper()

    if not source or not alias or not canonical:
        return None

    if (
        supported_sources is not None
        and source not in supported_sources
    ):
        raise ValueError(
            f"{map_path} row {row_number} has unsupported "
            f"source={source!r}"
        )

    if canonical != "TBD":
        if not team_id or not team_id.isdigit():
            raise ValueError(
                f"{map_path} row {row_number} has invalid "
                f"nhl_team_id={team_id!r}"
            )

        if not re.fullmatch(r"[A-Z]{3}", abbrev):
            raise ValueError(
                f"{map_path} row {row_number} has invalid "
                f"nhl_abbrev={abbrev!r}"
            )

    identity = {
        "canonical_team": canonical,
        "nhl_team_id": team_id,
        "nhl_abbrev": abbrev,
    }

    return (
        source,
        alias,
        team_id,
        abbrev,
        identity,
    )


def register_team_identity(
    by_id: MutableMapping[str, TeamIdentity],
    team_id: str,
    identity: TeamIdentity,
    map_path: Path,
) -> None:
    if not team_id:
        return

    prior = by_id.get(team_id)
    if prior is not None and prior != identity:
        raise ValueError(
            f"{map_path} has conflicting identity for "
            f"nhl_team_id={team_id}: {prior} != {identity}"
        )

    by_id[team_id] = identity
