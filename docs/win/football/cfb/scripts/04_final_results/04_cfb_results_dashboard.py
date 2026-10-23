#!/usr/bin/env python3
# docs/win/football/cfb/scripts/04_final_results/04_cfb_results_dashboard.py
#
# Builds the CFB dashboard data read live by the React CFB page from the
# grading reports written by 03_cfb_results_reports.py.
#
# Output:
#   frontend/data/dashboards/cfb.json
#   {"generated_at": <UTC ISO>, "data": {<dataset>: [<row>, ...]}}
#   Each row is one CSV row as {header: string value}.

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import json
import math
import os
import sys


REPORTS = Path("docs/win/football/cfb/04_final_results/reports")
OUTPUT = Path("frontend/data/dashboards/cfb.json")

DATASETS = {
    "bet_log": REPORTS / "overview/cfb_bet_log.csv",
    "calibration": REPORTS / "overview/cfb_calibration_by_probability.csv",
    "probability_metrics": REPORTS / "overview/cfb_probability_metrics.csv",
    "cumulative_units_by_week": REPORTS / "overview/cfb_cumulative_units_by_week.csv",
    "summary_overall": REPORTS / "overview/cfb_summary_overall.csv",
    "summary_by_market": REPORTS / "overview/cfb_summary_by_market.csv",
    "summary_by_side_group": REPORTS / "overview/cfb_summary_by_side_group.csv",
    "summary_by_week": REPORTS / "overview/cfb_summary_by_week.csv",
    "summary_by_day_night": REPORTS / "overview/cfb_summary_by_day_night.csv",
    "moneyline_by_ev": REPORTS / "moneyline/cfb_moneyline_by_ev.csv",
    "moneyline_by_ev_split": REPORTS / "moneyline/cfb_moneyline_by_ev_home_away_summary.csv",
    "moneyline_by_home_away": REPORTS / "moneyline/cfb_moneyline_by_home_away.csv",
    "moneyline_by_kelly": REPORTS / "moneyline/cfb_moneyline_by_kelly.csv",
    "moneyline_by_kelly_split": REPORTS / "moneyline/cfb_moneyline_by_kelly_home_away_summary.csv",
    "moneyline_by_odds": REPORTS / "moneyline/cfb_moneyline_by_odds.csv",
    "moneyline_by_odds_split": REPORTS / "moneyline/cfb_moneyline_by_odds_home_away_summary.csv",
    "moneyline_by_win_prob": REPORTS / "moneyline/cfb_moneyline_by_win_prob.csv",
    "moneyline_by_win_prob_split": REPORTS / "moneyline/cfb_moneyline_by_win_prob_home_away_summary.csv",
    "spread_by_ev": REPORTS / "spread/cfb_spread_by_ev.csv",
    "spread_by_ev_split": REPORTS / "spread/cfb_spread_by_ev_home_away_summary.csv",
    "spread_by_favorite_underdog": REPORTS / "spread/cfb_spread_by_favorite_underdog.csv",
    "spread_by_home_away": REPORTS / "spread/cfb_spread_by_home_away.csv",
    "spread_by_kelly": REPORTS / "spread/cfb_spread_by_kelly.csv",
    "spread_by_kelly_split": REPORTS / "spread/cfb_spread_by_kelly_home_away_summary.csv",
    "spread_by_line": REPORTS / "spread/cfb_spread_by_line.csv",
    "spread_by_line_split": REPORTS / "spread/cfb_spread_by_line_home_away_summary.csv",
    "spread_by_odds": REPORTS / "spread/cfb_spread_by_odds.csv",
    "spread_by_odds_split": REPORTS / "spread/cfb_spread_by_odds_home_away_summary.csv",
    "spread_by_win_prob": REPORTS / "spread/cfb_spread_by_win_prob.csv",
    "spread_by_win_prob_split": REPORTS / "spread/cfb_spread_by_win_prob_home_away_summary.csv",
    "total_by_ev": REPORTS / "totals/cfb_total_by_ev.csv",
    "total_by_ev_split": REPORTS / "totals/cfb_total_by_ev_over_under_summary.csv",
    "total_by_kelly": REPORTS / "totals/cfb_total_by_kelly.csv",
    "total_by_kelly_split": REPORTS / "totals/cfb_total_by_kelly_over_under_summary.csv",
    "total_by_odds": REPORTS / "totals/cfb_total_by_odds.csv",
    "total_by_odds_split": REPORTS / "totals/cfb_total_by_odds_over_under_summary.csv",
    "total_by_over_under": REPORTS / "totals/cfb_total_by_over_under.csv",
    "total_by_total_range": REPORTS / "totals/cfb_total_by_total_range.csv",
    "total_by_total_range_split": REPORTS / "totals/cfb_total_by_total_range_over_under_summary.csv",
    "total_by_win_prob": REPORTS / "totals/cfb_total_by_win_prob.csv",
    "total_by_win_prob_split": REPORTS / "totals/cfb_total_by_win_prob_over_under_summary.csv",
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
            print(f"WARNING: missing CFB dashboard input {path}; dataset {name} is empty", file=sys.stderr)
            data[name] = []
            continue
        data[name] = parse_csv_rows(read_csv_text(path))
        print(f"{name}: {len(data[name])} rows")

    bytes_written = write_json(OUTPUT, data)
    print(f"CFB dashboard data generated: {OUTPUT} ({bytes_written} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
