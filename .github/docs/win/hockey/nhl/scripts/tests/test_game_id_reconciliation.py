#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/tests/test_game_id_reconciliation.py

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any


REPO_ROOT = Path.cwd()
BASE_REL = Path("docs/win/hockey/nhl")
BASE_DIR = REPO_ROOT / BASE_REL

FIXTURE_ROOT = (
    BASE_DIR
    / "test_fixture"
    / "game_id_reconciliation"
)

FIXTURE_ODDS_DIR = FIXTURE_ROOT / "odds"
FIXTURE_PREDICTIONS_DIR = FIXTURE_ROOT / "predictions"
FIXTURE_SPORTSBOOK_DIR = FIXTURE_ROOT / "sportsbook"

TEST_OUTPUT_ROOT = (
    BASE_DIR
    / "test_output"
    / "game_id_reconciliation"
)

MARKETS = (
    "moneyline",
    "puck_line",
    "total",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        rows = list(csv.DictReader(handle))
    return rows


def normalize_team_name(value: str) -> str:
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
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fixture_dates() -> list[str]:
    odds_dates = {
        path.stem
        for path in FIXTURE_ODDS_DIR.glob("*.json")
    }

    prediction_dates = {
        path.name[
            : -len("_nhl_predictions.csv")
        ]
        for path in FIXTURE_PREDICTIONS_DIR.glob(
            "*_nhl_predictions.csv"
        )
    }

    sportsbook_dates = {
        path.stem[len("NHL_") :]
        for path in FIXTURE_SPORTSBOOK_DIR.glob(
            "NHL_*.csv"
        )
    }

    if not odds_dates:
        raise RuntimeError(
            "Fixture odds directory contains no fixture dates."
        )

    if (
        odds_dates != prediction_dates
        or odds_dates != sportsbook_dates
    ):
        raise RuntimeError(
            "Fixture date sets must match exactly across odds, "
            "predictions, and sportsbook. "
            f"odds={sorted(odds_dates)} "
            f"predictions={sorted(prediction_dates)} "
            f"sportsbook={sorted(sportsbook_dates)}"
        )

    return sorted(odds_dates)


def fixture_sportsbook_expectations(
    test_date: str,
) -> dict[str, Any]:
    path = (
        FIXTURE_SPORTSBOOK_DIR
        / f"NHL_{test_date}.csv"
    )

    rows = read_csv(path)

    if len(rows) != 1:
        raise RuntimeError(
            "Fixture sportsbook must contain exactly one row "
            f"for {test_date}. Found: {len(rows)}"
        )

    row = rows[0]

    sportsbook_event_id = row.get(
        "sportsbook_event_id",
        "",
    ).strip()
    odds_source = row.get(
        "odds_source",
        "",
    ).strip()
    home_team = row.get(
        "home_team",
        "",
    ).strip()
    away_team = row.get(
        "away_team",
        "",
    ).strip()

    required_values = {
        "sportsbook_event_id": sportsbook_event_id,
        "odds_source": odds_source,
        "home_team": home_team,
        "away_team": away_team,
    }

    for name, value in required_values.items():
        if not value:
            raise RuntimeError(
                f"Fixture sportsbook {path} has blank {name}."
            )

    providers: dict[str, dict[str, str]] = {}

    for market in MARKETS:
        provider_id = row.get(
            f"{market}_provider_id",
            "",
        ).strip()
        provider_name = row.get(
            f"{market}_provider_name",
            "",
        ).strip()

        if not provider_id or not provider_name:
            raise RuntimeError(
                f"Fixture sportsbook {path} has blank "
                f"{market} provider metadata."
            )

        providers[market] = {
            "id": provider_id,
            "name": provider_name,
        }

    return {
        "sportsbook_event_id": sportsbook_event_id,
        "odds_source": odds_source,
        "home_team": home_team,
        "away_team": away_team,
        "providers": providers,
    }


def prepare_workspace(
    workspace_root: Path,
    test_date: str,
) -> Path:
    work_base = (
        workspace_root
        / BASE_REL
    )

    shutil.copytree(
        BASE_DIR / "scripts" / "00_intake",
        work_base / "scripts" / "00_intake",
    )

    shutil.copy2(
        BASE_DIR / "scripts" / "odds_price_common.py",
        work_base / "scripts" / "odds_price_common.py",
    )

    shutil.copytree(
        BASE_DIR / "config" / "mapping",
        work_base / "config" / "mapping",
    )

    odds_dir = work_base / "odds"
    scraper_dir = (
        work_base
        / "00_intake"
        / "predictions"
        / "scraper"
    )
    sportsbook_dir = (
        work_base
        / "00_intake"
        / "sportsbook"
    )

    odds_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    scraper_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    sportsbook_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        FIXTURE_ODDS_DIR
        / f"{test_date}.json",
        odds_dir
        / f"{test_date}.json",
    )

    shutil.copy2(
        FIXTURE_PREDICTIONS_DIR
        / f"{test_date}_nhl_predictions.csv",
        scraper_dir
        / f"{test_date}_nhl_predictions.csv",
    )

    return work_base


def run_script(
    workspace_root: Path,
    name: str,
) -> None:
    script_path = (
        BASE_REL
        / "scripts"
        / "00_intake"
        / name
    )

    subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=workspace_root,
        check=True,
    )


def validate_transformed_sportsbook(
    work_base: Path,
    test_date: str,
    expected: dict[str, Any],
) -> None:
    generated_path = (
        work_base
        / "00_intake"
        / "sportsbook"
        / f"NHL_{test_date}.csv"
    )

    expected_path = (
        FIXTURE_SPORTSBOOK_DIR
        / f"NHL_{test_date}.csv"
    )

    if not generated_path.exists():
        raise RuntimeError(
            f"Sportsbook output missing: {generated_path}"
        )

    generated = read_csv(
        generated_path
    )

    expected_rows = read_csv(
        expected_path
    )

    if generated != expected_rows:
        raise RuntimeError(
            "Generated sportsbook output does not match "
            f"fixture sportsbook file: {expected_path}"
        )

    if len(generated) != 1:
        raise RuntimeError(
            "Fixture sportsbook must contain exactly "
            f"one row. Found: {len(generated)}"
        )

    row = generated[0]

    if "game_id" in row:
        raise RuntimeError(
            "Pre-reconciliation sportsbook output "
            "contains game_id."
        )

    if (
        row.get(
            "sportsbook_event_id",
            "",
        ).strip()
        != expected["sportsbook_event_id"]
    ):
        raise RuntimeError(
            "Unexpected sportsbook_event_id."
        )

    if (
        row.get(
            "odds_source",
            "",
        ).strip()
        != expected["odds_source"]
    ):
        raise RuntimeError(
            "Unexpected odds_source."
        )

    providers = expected["providers"]

    if not isinstance(providers, dict):
        raise RuntimeError(
            "Invalid fixture provider expectations."
        )

    for market in MARKETS:
        market_provider = providers[market]

        if (
            row.get(
                f"{market}_provider_id",
                "",
            ).strip()
            != market_provider["id"]
        ):
            raise RuntimeError(
                f"Unexpected {market}_provider_id."
            )

        if (
            row.get(
                f"{market}_provider_name",
                "",
            ).strip()
            != market_provider["name"]
        ):
            raise RuntimeError(
                f"Unexpected {market}_provider_name."
            )


def stage_fixture_sportsbook(
    work_base: Path,
    test_date: str,
) -> None:
    target = (
        work_base
        / "00_intake"
        / "sportsbook"
        / f"NHL_{test_date}.csv"
    )

    shutil.copy2(
        FIXTURE_SPORTSBOOK_DIR
        / f"NHL_{test_date}.csv",
        target,
    )


def validate_transformed_predictions(
    work_base: Path,
    test_date: str,
) -> None:
    path = (
        work_base
        / "00_intake"
        / "predictions"
        / f"hockey_{test_date}.csv"
    )

    if not path.exists():
        raise RuntimeError(
            f"Prediction output missing: {path}"
        )

    rows = read_csv(path)

    if len(rows) != 1:
        raise RuntimeError(
            "Fixture prediction must produce exactly "
            f"one row. Found: {len(rows)}"
        )

    if rows[0].get(
        "game_id",
        "",
    ).strip():
        raise RuntimeError(
            "Prediction received game_id before "
            "reconciliation."
        )


def validate_schedule(
    work_base: Path,
    test_date: str,
    sportsbook_expected: dict[str, Any],
) -> dict[str, str]:
    path = (
        work_base
        / "00_intake"
        / "nhl_schedule"
        / f"NHL_{test_date}.csv"
    )

    if not path.exists():
        raise RuntimeError(
            f"Official NHL schedule missing: {path}"
        )

    rows = read_csv(path)

    if not rows:
        raise RuntimeError(
            "Official NHL schedule contains zero rows."
        )

    official_ids = {
        row.get(
            "game_id",
            "",
        ).strip()
        for row in rows
    }

    bad_ids = sorted(
        game_id
        for game_id in official_ids
        if (
            len(game_id) != 10
            or not game_id.isdigit()
        )
    )

    if bad_ids:
        raise RuntimeError(
            f"Invalid official NHL game IDs: {bad_ids}"
        )

    fixture_teams = {
        normalize_team_name(
            str(sportsbook_expected["home_team"])
        ),
        normalize_team_name(
            str(sportsbook_expected["away_team"])
        ),
    }

    matches = []

    for row in rows:
        schedule_teams = {
            normalize_team_name(
                row.get("home_team", "")
            ),
            normalize_team_name(
                row.get("away_team", "")
            ),
        }

        if schedule_teams == fixture_teams:
            matches.append(row)

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one official NHL schedule match "
            f"for fixture {test_date}. Found: {len(matches)}"
        )

    match = matches[0]
    game_id = match.get(
        "game_id",
        "",
    ).strip()
    home_team = match.get(
        "home_team",
        "",
    ).strip()
    away_team = match.get(
        "away_team",
        "",
    ).strip()
    game_type = match.get(
        "game_type",
        "",
    ).strip()

    if game_type not in {"2", "3"}:
        raise RuntimeError(
            "Fixture matched a non-regular-season/playoff game: "
            f"game_type={game_type!r}"
        )

    return {
        "official_game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
    }


def validate_reconciliation(
    work_base: Path,
    test_date: str,
    sportsbook_expected: dict[str, Any],
    schedule_expected: dict[str, str],
) -> None:
    sportsbook_path = (
        work_base
        / "00_intake"
        / "sportsbook"
        / f"NHL_{test_date}.csv"
    )

    prediction_path = (
        work_base
        / "00_intake"
        / "predictions"
        / f"hockey_{test_date}.csv"
    )

    reconciled_path = (
        work_base
        / "00_intake"
        / "reconciled"
        / f"NHL_{test_date}.csv"
    )

    audit_path = (
        work_base
        / "00_intake"
        / "reconciled"
        / "audit"
        / f"NHL_{test_date}_reconciliation.csv"
    )

    for path in (
        sportsbook_path,
        prediction_path,
        reconciled_path,
        audit_path,
    ):
        if not path.exists():
            raise RuntimeError(
                f"Expected reconciliation output missing: {path}"
            )

    sportsbook_rows = read_csv(
        sportsbook_path
    )

    prediction_rows = read_csv(
        prediction_path
    )

    reconciled_rows = read_csv(
        reconciled_path
    )

    audit_rows = read_csv(
        audit_path
    )

    if len(reconciled_rows) != 1:
        raise RuntimeError(
            "Expected exactly one reconciled game. "
            f"Found: {len(reconciled_rows)}"
        )

    row = reconciled_rows[0]
    expected_game_id = schedule_expected[
        "official_game_id"
    ]
    expected_sportsbook_event_id = sportsbook_expected[
        "sportsbook_event_id"
    ]

    if (
        row.get(
            "game_id",
            "",
        ).strip()
        != expected_game_id
    ):
        raise RuntimeError(
            "Reconciled official game_id does not "
            "match expected NHL ID."
        )

    if (
        row.get(
            "sportsbook_event_id",
            "",
        ).strip()
        != expected_sportsbook_event_id
    ):
        raise RuntimeError(
            "Reconciled sportsbook_event_id does not "
            "match expected provider ID."
        )

    if (
        row["game_id"]
        == row["sportsbook_event_id"]
    ):
        raise RuntimeError(
            "Official NHL game_id equals "
            "sportsbook_event_id."
        )

    if (
        normalize_team_name(
            row.get(
                "home_team",
                "",
            )
        )
        != normalize_team_name(
            schedule_expected["home_team"]
        )
    ):
        raise RuntimeError(
            "Reconciled home team does not match "
            "the official NHL schedule."
        )

    if (
        normalize_team_name(
            row.get(
                "away_team",
                "",
            )
        )
        != normalize_team_name(
            schedule_expected["away_team"]
        )
    ):
        raise RuntimeError(
            "Reconciled away team does not match "
            "the official NHL schedule."
        )

    providers = sportsbook_expected["providers"]

    if not isinstance(providers, dict):
        raise RuntimeError(
            "Invalid fixture provider expectations."
        )

    for source_name, rows in (
        (
            "sportsbook",
            sportsbook_rows,
        ),
        (
            "prediction",
            prediction_rows,
        ),
    ):
        if len(rows) != 1:
            raise RuntimeError(
                f"{source_name} reconciled output "
                "must contain one row."
            )

        source_row = rows[0]

        if (
            source_row.get(
                "game_id",
                "",
            ).strip()
            != expected_game_id
        ):
            raise RuntimeError(
                f"{source_name} does not contain "
                "the official NHL game_id."
            )

        if source_name == "sportsbook":
            if (
                source_row.get(
                    "odds_source",
                    "",
                ).strip()
                != sportsbook_expected["odds_source"]
            ):
                raise RuntimeError(
                    "Reconciled sportsbook odds_source mismatch."
                )

            for market in MARKETS:
                market_provider = providers[market]

                if (
                    source_row.get(
                        f"{market}_provider_id",
                        "",
                    ).strip()
                    != market_provider["id"]
                ):
                    raise RuntimeError(
                        "Reconciled sportsbook "
                        f"{market}_provider_id mismatch."
                    )

                if (
                    source_row.get(
                        f"{market}_provider_name",
                        "",
                    ).strip()
                    != market_provider["name"]
                ):
                    raise RuntimeError(
                        "Reconciled sportsbook "
                        f"{market}_provider_name mismatch."
                    )

    bad_audit = [
        audit
        for audit in audit_rows
        if audit.get(
            "status",
            "",
        ).strip()
        != "reconciled"
    ]

    if bad_audit:
        raise RuntimeError(
            "Reconciliation audit contains failures: "
            f"{bad_audit}"
        )

    corrections = [
        audit
        for audit in audit_rows
        if audit.get(
            "orientation_corrected",
            "",
        ).strip()
        == "yes"
    ]

    if not corrections:
        raise RuntimeError(
            "Fixture did not exercise reversed "
            "home/away correction."
        )


def validate_games(
    work_base: Path,
    test_date: str,
    sportsbook_expected: dict[str, Any],
    schedule_expected: dict[str, str],
) -> None:
    path = (
        work_base
        / "00_intake"
        / "games"
        / f"{test_date}_nhl_games.csv"
    )

    if not path.exists():
        raise RuntimeError(
            f"Games output missing: {path}"
        )

    rows = read_csv(path)

    if len(rows) != 1:
        raise RuntimeError(
            "Games output must contain exactly "
            f"one fixture row. Found: {len(rows)}"
        )

    row = rows[0]

    if (
        row.get(
            "game_id",
            "",
        ).strip()
        != schedule_expected["official_game_id"]
    ):
        raise RuntimeError(
            "Games output official game_id mismatch."
        )

    if (
        row.get(
            "sportsbook_event_id",
            "",
        ).strip()
        != sportsbook_expected["sportsbook_event_id"]
    ):
        raise RuntimeError(
            "Games output sportsbook_event_id mismatch."
        )


def save_test_output(
    work_base: Path,
    test_date: str,
    sportsbook_expected: dict[str, Any],
    schedule_expected: dict[str, str],
) -> Path:
    output_dir = (
        TEST_OUTPUT_ROOT
        / test_date
    )

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir = (
        output_dir
        / "logs"
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copies = {
        (
            work_base
            / "00_intake"
            / "nhl_schedule"
            / f"NHL_{test_date}.csv"
        ): (
            output_dir
            / "nhl_schedule.csv"
        ),
        (
            work_base
            / "00_intake"
            / "sportsbook"
            / f"NHL_{test_date}.csv"
        ): (
            output_dir
            / "sportsbook_reconciled.csv"
        ),
        (
            work_base
            / "00_intake"
            / "predictions"
            / f"hockey_{test_date}.csv"
        ): (
            output_dir
            / "predictions_reconciled.csv"
        ),
        (
            work_base
            / "00_intake"
            / "reconciled"
            / f"NHL_{test_date}.csv"
        ): (
            output_dir
            / "reconciled_games.csv"
        ),
        (
            work_base
            / "00_intake"
            / "reconciled"
            / "audit"
            / f"NHL_{test_date}_reconciliation.csv"
        ): (
            output_dir
            / "reconciliation_audit.csv"
        ),
        (
            work_base
            / "00_intake"
            / "games"
            / f"{test_date}_nhl_games.csv"
        ): (
            output_dir
            / "games.csv"
        ),
    }

    for source, target in copies.items():
        shutil.copy2(
            source,
            target,
        )

    error_dir = (
        work_base
        / "errors"
        / "00_intake"
    )

    for name in (
        "transform_hockey_odds.txt",
        "transform_hockey.txt",
        "pull_nhl_schedule.txt",
        "reconcile_game_ids.txt",
        "build_games.txt",
    ):
        source = error_dir / name

        if source.exists():
            shutil.copy2(
                source,
                logs_dir / name,
            )

    providers = sportsbook_expected["providers"]

    if not isinstance(providers, dict):
        raise RuntimeError(
            "Invalid fixture provider expectations."
        )

    summary = [
        "NHL GAME ID RECONCILIATION TEST",
        f"TEST_DATE={test_date}",
        "STATUS=PASSED",
        "",
        (
            "official_nhl_game_id="
            f"{schedule_expected['official_game_id']}"
        ),
        (
            "sportsbook_event_id="
            f"{sportsbook_expected['sportsbook_event_id']}"
        ),
        (
            "odds_source="
            f"{sportsbook_expected['odds_source']}"
        ),
        (
            "official_home_team="
            f"{schedule_expected['home_team']}"
        ),
        (
            "official_away_team="
            f"{schedule_expected['away_team']}"
        ),
    ]

    for market in MARKETS:
        market_provider = providers[market]
        summary.append(
            f"{market}_provider="
            f"{market_provider['name']} "
            f"({market_provider['id']})"
        )

    summary.extend(
        [
            "fixture_source=game_id_reconciliation",
            "workspace=isolated_temp_directory",
        ]
    )

    (
        output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    return output_dir


def run_fixture_date(
    test_date: str,
) -> Path:
    sportsbook_expected = (
        fixture_sportsbook_expectations(
            test_date
        )
    )

    print(
        f"Fixture test date: {test_date}"
    )

    with tempfile.TemporaryDirectory(
        prefix=(
            "nhl_game_id_reconciliation_"
            f"{test_date}_"
        )
    ) as temp_dir:
        workspace_root = Path(temp_dir)

        work_base = prepare_workspace(
            workspace_root,
            test_date,
        )

        run_script(
            workspace_root,
            "transform_hockey_odds.py",
        )

        validate_transformed_sportsbook(
            work_base,
            test_date,
            sportsbook_expected,
        )

        stage_fixture_sportsbook(
            work_base,
            test_date,
        )

        run_script(
            workspace_root,
            "transform_hockey.py",
        )

        validate_transformed_predictions(
            work_base,
            test_date,
        )

        run_script(
            workspace_root,
            "pull_nhl_schedule.py",
        )

        schedule_expected = validate_schedule(
            work_base,
            test_date,
            sportsbook_expected,
        )

        run_script(
            workspace_root,
            "reconcile_game_ids.py",
        )

        validate_reconciliation(
            work_base,
            test_date,
            sportsbook_expected,
            schedule_expected,
        )

        run_script(
            workspace_root,
            "build_games.py",
        )

        validate_games(
            work_base,
            test_date,
            sportsbook_expected,
            schedule_expected,
        )

        output_dir = save_test_output(
            work_base,
            test_date,
            sportsbook_expected,
            schedule_expected,
        )

    print(
        f"Fixture date passed: {test_date}"
    )

    print(
        "official_game_id="
        f"{schedule_expected['official_game_id']}"
    )

    print(
        "sportsbook_event_id="
        f"{sportsbook_expected['sportsbook_event_id']}"
    )

    print(
        f"output_dir={output_dir}"
    )

    return output_dir


def main() -> None:
    dates = fixture_dates()

    print(
        "Fixture test dates: "
        + ", ".join(dates)
    )

    for test_date in dates:
        run_fixture_date(
            test_date
        )

    print(
        "NHL GAME ID RECONCILIATION TEST PASSED"
    )

    print(
        f"fixture_dates_tested={len(dates)}"
    )


if __name__ == "__main__":
    main()