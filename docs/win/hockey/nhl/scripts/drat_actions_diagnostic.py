#!/usr/bin/env python3
"""Diagnose DRatings NHL page loading from a GitHub Actions runner.

Does not import or run the production scraper and does not read NHL schedules.
"""

import json
import sys
import traceback
from datetime import datetime, timezone
from contextlib import ExitStack
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://www.dratings.com/predictor/nhl-hockey-predictions/"
OUTPUT = Path("drat_diagnostic_output")


def save_text(filename, text):
    (OUTPUT / filename).write_text(str(text), encoding="utf-8")


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = []
    report = {
        "url_requested": URL,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "result": "FAILED",
    }
    browser = None
    page = None
    resources = ExitStack()
    try:
        playwright = resources.enter_context(sync_playwright())
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        def on_response(response):
            try:
                kind = response.request.resource_type
                if kind in ("document", "xhr", "fetch") or response.status >= 400:
                    events.append(f"HTTP {response.status} ({kind}) {response.url}")
            except Exception as exc:
                events.append(f"Response callback error: {exc!r}")

        def on_request_failed(request):
            events.append(f"REQUEST FAILED {request.url}: {request.failure}")

        def on_console(message):
            if message.type == "error":
                events.append(f"CONSOLE ERROR: {message.text}")

        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)
        page.on("console", on_console)
        page.on("pageerror", lambda error: events.append(f"JAVASCRIPT ERROR: {error}"))

        # Keep this user-agent setting identical to the production scraper.
        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        })

        response = page.goto(URL, wait_until="load", timeout=45000)
        report["navigation_http_status"] = response.status if response is not None else None
        report["final_url"] = page.url
        report["page_title"] = page.title()
        page.wait_for_selector("table", state="visible", timeout=30000)
        report["table_count"] = page.locator("table").count()
        report["visible_table_count"] = page.locator("table:visible").count()
        report["tbody_row_count"] = page.locator("table tbody tr").count()
        report["result"] = "TABLE_VISIBLE"
        return_code = 0
    
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        return_code = 1
    finally:
        if page is not None:
            try:
                report["final_url"] = page.url
                report["page_title"] = page.title(timeout=10000)
            except Exception as exc:
                events.append(f"PAGE METADATA ERROR: {exc!r}")
            for key, method in (
                ("table_count", lambda: page.locator("table").count()),
                ("visible_table_count", lambda: page.locator("table:visible").count()),
                ("tbody_row_count", lambda: page.locator("table tbody tr").count()),
                ("frame_urls", lambda: [frame.url for frame in page.frames]),
            ):
                try:
                    report[key] = method()
                except Exception as exc:
                    events.append(f"{key} ERROR: {exc!r}")
            try:
                save_text("page.html", page.content())
            except Exception as exc:
                events.append(f"HTML SAVE ERROR: {exc!r}")
            try:
                save_text("page_text.txt", page.locator("body").inner_text(timeout=10000))
            except Exception as exc:
                events.append(f"PAGE TEXT ERROR: {exc!r}")
            try:
                page.screenshot(path=str(OUTPUT / "screenshot.png"), full_page=True, timeout=15000)
            except Exception as exc:
                events.append(f"SCREENSHOT ERROR: {exc!r}")
        if browser is not None:
            try:
                browser.close()
            except Exception as exc:
                events.append(f"BROWSER CLOSE ERROR: {exc!r}")
        try:
            resources.close()
        except Exception as exc:
            events.append(f"PLAYWRIGHT CLOSE ERROR: {exc!r}")

        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save_text("report.json", json.dumps(report, indent=2, ensure_ascii=False))
        save_text("browser_events.txt", "\n".join(events) + "\n")
        print(f"RESULT: {report['result']}")
        print(f"HTTP STATUS: {report.get('navigation_http_status', 'not recorded')}")
        print(f"TABLE COUNT: {report.get('table_count', 'not recorded')}")
        print(f"VISIBLE TABLE COUNT: {report.get('visible_table_count', 'not recorded')}")
        print(f"OUTPUT: {OUTPUT.resolve()}")
        if report.get("error"):
            print(f"ERROR: {report['error']}")
        return return_code


if __name__ == "__main__":
    sys.exit(run())
