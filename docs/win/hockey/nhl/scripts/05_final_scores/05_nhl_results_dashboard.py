#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/05_final_scores/05_nhl_results_dashboard.py
#
# Builds the NHL dashboard data read live by the React NHL page from the
# Stage 05 grading and report CSVs.
#
# Output:
#   frontend/data/dashboards/nhl.json
#   {"generated_at": <UTC ISO>, "data": {<dataset>: [<row>, ...]}}
#   Each row is one CSV row as {header: string value}.

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import json
import math
import os
import sys


BASE = Path("docs/win/hockey/nhl/05_final_scores")
OUTPUT = Path("frontend/data/dashboards/nhl.json")

DATASETS = {
    "bets": BASE / "graded/NHL_final.csv",
    "market_tally": BASE / "nhl_market_tally.csv",
    "moneyline_ev": BASE / "reports/moneyline/nhl_moneyline_by_ev.csv",
    "moneyline_kelly": BASE / "reports/moneyline/nhl_moneyline_by_kelly.csv",
    "moneyline_odds": BASE / "reports/moneyline/nhl_moneyline_by_odds.csv",
    "moneyline_win_prob": BASE / "reports/moneyline/nhl_moneyline_by_win_prob.csv",
    "puckline_ev": BASE / "reports/puckline/nhl_puck_line_by_ev.csv",
    "puckline_kelly": BASE / "reports/puckline/nhl_puck_line_by_kelly.csv",
    "puckline_odds": BASE / "reports/puckline/nhl_puck_line_by_odds.csv",
    "puckline_side": BASE / "reports/puckline/nhl_puck_line_by_side.csv",
    "puckline_win_prob": BASE / "reports/puckline/nhl_puck_line_by_win_prob.csv",
    "total_ev": BASE / "reports/total/nhl_total_by_ev.csv",
    "total_kelly": BASE / "reports/total/nhl_total_by_kelly.csv",
    "total_odds": BASE / "reports/total/nhl_total_by_odds.csv",
    "total_side": BASE / "reports/total/nhl_total_by_side.csv",
    "total_range": BASE / "reports/total/nhl_total_by_total_range.csv",
    "total_win_prob": BASE / "reports/total/nhl_total_by_win_prob.csv",
}


# Same rules as the React page's CSV reader: every value stays a string, rows
# whose cells are all empty are skipped, header cells are trimmed, missing
# cells become "" and extra cells are ignored.
JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def read_csv_text(path: Path) -> str:
    text = path.read_bytes().decode("utf-8", errors="replace")
    return text[1:] if text.startswith("\ufeff") else text


def parse_csv_rows(text: str) -> list[dict[str, str]]:
    rows: list[list[str]] = []
    row: list[str] = []
    value: list[str] = []
    quoted = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        following = text[index + 1] if index + 1 < length else ""

        if char == '"' and quoted and following == '"':
            value.append('"')
            index += 2
            continue

        if char == '"':
            quoted = not quoted
            index += 1
            continue

        if char == "," and not quoted:
            row.append("".join(value))
            value = []
            index += 1
            continue

        if char in "\n\r" and not quoted:
            if char == "\r" and following == "\n":
                index += 1
            row.append("".join(value))
            value = []
            if any(cell != "" for cell in row):
                rows.append(row)
            row = []
            index += 1
            continue

        value.append(char)
        index += 1

    if value or row:
        row.append("".join(value))
        if any(cell != "" for cell in row):
            rows.append(row)

    if not rows:
        return []

    headers = [cell.strip(JS_WHITESPACE) for cell in rows[0]]
    records: list[dict[str, str]] = []
    for values in rows[1:]:
        record: dict[str, str] = {}
        for position, header in enumerate(headers):
            record[header] = values[position] if position < len(values) else ""
        records.append(record)
    return records


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data) -> int:
    document = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": json_safe(data),
    }
    content = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(content, encoding="utf-8")
    os.replace(staging, path)
    return len(content.encode("utf-8"))


def main() -> int:
    data: dict[str, list[dict[str, str]]] = {}

    for name, path in DATASETS.items():
        if not path.is_file():
            print(f"WARNING: missing NHL dashboard input {path}; dataset {name} is empty", file=sys.stderr)
            data[name] = []
            continue
        data[name] = parse_csv_rows(read_csv_text(path))
        print(f"{name}: {len(data[name])} rows")

    bytes_written = write_json(OUTPUT, data)
    print(f"NHL dashboard data generated: {OUTPUT} ({bytes_written} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
