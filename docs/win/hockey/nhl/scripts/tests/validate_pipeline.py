#!/usr/bin/env python3

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


NHL = Path("docs/win/hockey/nhl")
WORKFLOW = Path(".github/workflows/nhl_01_pipeline.yml")
REQUIREMENTS = NHL / "requirements.txt"

REQUIRED_DIRS = [
    NHL,
    NHL / "00_intake",
    NHL / "01_merge",
    NHL / "02_juice",
    NHL / "03_edges",
    NHL / "04_select",
    NHL / "05_final_scores",
    NHL / "config",
    NHL / "config" / "juice",
    NHL / "config" / "mapping",
    NHL / "scripts",
    NHL / "scripts" / "00_intake",
    NHL / "scripts" / "01_merge",
    NHL / "scripts" / "02_juice",
    NHL / "scripts" / "03_edges",
    NHL / "scripts" / "04_select",
    NHL / "scripts" / "05_final_scores",
    NHL / "scripts" / "tests",
]

REQUIRED_FILES = [
    WORKFLOW,
    REQUIREMENTS,
    NHL / "config" / "markets.yaml",
    NHL / "config" / "mapping" / "team_map_nhl.csv",
    NHL / "config" / "juice" / "nhl_moneyline_juice.csv",
    NHL / "config" / "juice" / "nhl_puck_line_juice.csv",
    NHL / "config" / "juice" / "nhl_total_juice.csv",
    NHL / "scripts" / "02_juice" / "validate_juice_config.py",
    NHL / "scripts" / "04_select" / "validate_markets_config.py",
]

TEAM_MAP_COLUMNS = [
    "league",
    "source",
    "alias",
    "canonical_team",
    "nhl_team_id",
    "nhl_abbrev",
]

CONFIG_VALIDATORS = [
    NHL / "scripts" / "02_juice" / "validate_juice_config.py",
    NHL / "scripts" / "04_select" / "validate_markets_config.py",
]

RUNTIME_REF_RE = re.compile(
    r"(docs/win/hockey/nhl/[A-Za-z0-9_./-]+\.py)"
)

SDV_PIN_RE = re.compile(
    r"^\s*sportsdataverse\s*==\s*([^\s;#]+)",
    re.IGNORECASE,
)


def repo_root() -> Path:
    here = Path(__file__).resolve()

    for path in here.parents:
        if (
            (path / ".github").is_dir()
            and (path / REQUIREMENTS).is_file()
            and (path / NHL).is_dir()
        ):
            return path

    raise RuntimeError(
        f"Repository root not found from {here}"
    )


def fail(
    errors: list[str],
    message: str,
) -> None:
    errors.append(message)
    print(f"FAIL | {message}")


def passed(message: str) -> None:
    print(f"PASS | {message}")


def check_directories(
    root: Path,
    errors: list[str],
) -> None:
    missing = [
        path
        for path in REQUIRED_DIRS
        if not (root / path).is_dir()
    ]

    if missing:
        for path in missing:
            fail(
                errors,
                f"Missing required directory: {path.as_posix()}",
            )
    else:
        passed(
            f"Required directories present: {len(REQUIRED_DIRS)}"
        )


def check_required_files(
    root: Path,
    errors: list[str],
) -> None:
    missing = [
        path
        for path in REQUIRED_FILES
        if not (root / path).is_file()
    ]

    if missing:
        for path in missing:
            fail(
                errors,
                f"Missing required file: {path.as_posix()}",
            )
    else:
        passed(
            f"Required files present: {len(REQUIRED_FILES)}"
        )


def check_workflow_runtime_references(
    root: Path,
    errors: list[str],
) -> None:
    path = root / WORKFLOW

    if not path.is_file():
        return

    refs = sorted(
        set(
            RUNTIME_REF_RE.findall(
                path.read_text(encoding="utf-8")
            )
        )
    )

    if not refs:
        fail(
            errors,
            "No NHL Python references found in "
            f"{WORKFLOW.as_posix()}",
        )
        return

    missing = [
        ref
        for ref in refs
        if not (root / ref).is_file()
    ]

    if missing:
        for ref in missing:
            fail(
                errors,
                f"Stale workflow runtime reference: {ref}",
            )
    else:
        passed(
            f"Workflow Python references resolve: {len(refs)}"
        )


def check_team_map(
    root: Path,
    errors: list[str],
) -> None:
    path = (
        root
        / NHL
        / "config"
        / "mapping"
        / "team_map_nhl.csv"
    )

    if not path.is_file():
        return

    before = len(errors)

    try:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        fail(
            errors,
            f"Could not read "
            f"{path.relative_to(root).as_posix()}: {exc}",
        )
        return

    if list(df.columns) != TEAM_MAP_COLUMNS:
        fail(
            errors,
            "team_map_nhl.csv schema mismatch: "
            f"expected={TEAM_MAP_COLUMNS} "
            f"actual={list(df.columns)}",
        )
        return

    if df.empty:
        fail(
            errors,
            "team_map_nhl.csv is empty",
        )
        return

    for column in TEAM_MAP_COLUMNS:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    blank = (
        df["league"].eq("")
        | df["source"].eq("")
        | df["alias"].eq("")
        | df["canonical_team"].eq("")
    )

    if blank.any():
        rows = [
            int(index) + 2
            for index in df.index[blank]
        ]

        fail(
            errors,
            "Blank required mapping values "
            f"at rows {rows[:20]}",
        )

    duplicate = df.duplicated(
        [
            "league",
            "source",
            "alias",
        ],
        keep=False,
    )

    if duplicate.any():
        rows = [
            int(index) + 2
            for index in df.index[duplicate]
        ]

        fail(
            errors,
            "Duplicate league/source/alias mappings "
            f"at rows {rows[:20]}",
        )

    non_nhl = (
        ~df["league"]
        .str.lower()
        .eq("nhl")
    )

    if non_nhl.any():
        rows = [
            int(index) + 2
            for index in df.index[non_nhl]
        ]

        fail(
            errors,
            "Non-NHL league values in team map "
            f"at rows {rows[:20]}",
        )

    official = df[
        df["source"]
        .str.lower()
        .eq("official_nhl")
        & df["league"]
        .str.lower()
        .eq("nhl")
    ].copy()

    if len(official) != 32:
        fail(
            errors,
            "Official NHL mapping row count "
            f"must be 32; found {len(official)}",
        )

    if official.empty:
        return

    official_blank = (
        official["alias"].eq("")
        | official["canonical_team"].eq("")
        | official["nhl_team_id"].eq("")
        | official["nhl_abbrev"].eq("")
    )

    if official_blank.any():
        rows = [
            int(index) + 2
            for index in official.index[official_blank]
        ]

        fail(
            errors,
            "Blank official NHL mapping values "
            f"at rows {rows[:20]}",
        )

    bad_id = (
        ~official["nhl_team_id"]
        .str.fullmatch(r"\d+")
    )

    if bad_id.any():
        rows = [
            int(index) + 2
            for index in official.index[bad_id]
        ]

        fail(
            errors,
            "Invalid official NHL team IDs "
            f"at rows {rows[:20]}",
        )

    bad_abbrev = (
        ~official["nhl_abbrev"]
        .str.fullmatch(r"[A-Z]{3}")
    )

    if bad_abbrev.any():
        rows = [
            int(index) + 2
            for index in official.index[bad_abbrev]
        ]

        fail(
            errors,
            "Invalid official NHL abbreviations "
            f"at rows {rows[:20]}",
        )

    for column in (
        "canonical_team",
        "nhl_team_id",
        "nhl_abbrev",
    ):
        duplicate_official = (
            official.duplicated(
                column,
                keep=False,
            )
        )

        if duplicate_official.any():
            rows = [
                int(index) + 2
                for index
                in official.index[
                    duplicate_official
                ]
            ]

            fail(
                errors,
                f"Duplicate official {column} values "
                f"at rows {rows[:20]}",
            )

    official_lookup = {
        row["canonical_team"]: (
            row["nhl_team_id"],
            row["nhl_abbrev"],
        )
        for _, row in official.iterrows()
    }

    bad_alias_rows = []

    for index, row in df.iterrows():
        if row["source"].lower() == "shared":
            continue

        expected = official_lookup.get(
            row["canonical_team"]
        )

        actual = (
            row["nhl_team_id"],
            row["nhl_abbrev"],
        )

        if expected is None or expected != actual:
            bad_alias_rows.append(
                int(index) + 2
            )

    if bad_alias_rows:
        fail(
            errors,
            "Alias mappings disagree with official "
            "NHL ID/abbreviation at rows "
            f"{bad_alias_rows[:20]}",
        )

    if len(errors) == before:
        passed(
            "team_map_nhl.csv schema and "
            "official NHL mappings"
        )


def required_sdv_version(
    path: Path,
) -> str | None:
    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        match = SDV_PIN_RE.match(line)

        if match:
            return match.group(1)

    return None


def check_sdv_version(
    root: Path,
    errors: list[str],
) -> None:
    path = root / REQUIREMENTS

    if not path.is_file():
        return

    expected = required_sdv_version(
        path
    )

    if expected is None:
        fail(
            errors,
            "requirements.txt does not contain "
            "sportsdataverse==<version>",
        )
        return

    try:
        installed = (
            importlib.metadata.version(
                "sportsdataverse"
            )
        )
    except importlib.metadata.PackageNotFoundError:
        fail(
            errors,
            "sportsdataverse is listed in "
            "requirements.txt but is not installed",
        )
        return

    if installed != expected:
        fail(
            errors,
            "sportsdataverse version mismatch: "
            f"requirements.txt={expected}, "
            f"installed={installed}",
        )
    else:
        passed(
            "sportsdataverse version matches "
            f"requirements.txt: {installed}"
        )


def run_config_validator(
    root: Path,
    script: Path,
    errors: list[str],
) -> None:
    path = root / script

    if not path.is_file():
        return

    result = subprocess.run(
        [
            sys.executable,
            str(path),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout.strip():
        print(
            result.stdout.strip()
        )

    if result.stderr.strip():
        print(
            result.stderr.strip(),
            file=sys.stderr,
        )

    if result.returncode != 0:
        fail(
            errors,
            "Configuration/schema validator failed: "
            f"{script.as_posix()}",
        )
    else:
        passed(
            "Configuration/schema validator passed: "
            f"{script.as_posix()}"
        )


def main() -> int:
    try:
        root = repo_root()
    except Exception as exc:
        print(
            f"FAIL | {exc}"
        )
        return 1

    errors: list[str] = []

    check_directories(
        root,
        errors,
    )

    check_required_files(
        root,
        errors,
    )

    check_workflow_runtime_references(
        root,
        errors,
    )

    check_team_map(
        root,
        errors,
    )

    check_sdv_version(
        root,
        errors,
    )

    for script in CONFIG_VALIDATORS:
        run_config_validator(
            root,
            script,
            errors,
        )

    if errors:
        print(
            "NHL pre-run pipeline validation FAILED: "
            f"{len(errors)} error(s)."
        )
        return 1

    print(
        "NHL pre-run pipeline validation PASSED."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
