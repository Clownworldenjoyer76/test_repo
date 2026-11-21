#!/usr/bin/env python3
"""One-time, atomic, idempotent setup for daily official NHL targets refresh.

Patches ONLY the local build_historical_targets.py entry point and its default
output path. Its original historical rebuild functionality remains intact.
Run after placing refresh_targets_official.py beside build_historical_targets.py.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

ROOT = Path(r'G:\hockey\docs\win\hockey\nhl\prop_engine')
BUILDER = ROOT / 'scripts' / '00_intake' / 'build_historical_targets.py'
REFRESH = ROOT / 'scripts' / '00_intake' / 'refresh_targets_official.py'
MARKER = 'from refresh_targets_official import main as _refresh_main'
OLD_END = 'if __name__ == "__main__":\n    raise SystemExit(main())'
NEW_END = '''if __name__ == "__main__":
    # Standard pipeline step: acquire and validate NEW daily official NHL results.
    # Explicit original history-builder CLI arguments still rebuild old Parquets.
    if len(sys.argv) == 1 or "--refresh-daily" in sys.argv[1:]:
        from refresh_targets_official import main as _refresh_main
        params = [part for part in sys.argv[1:] if part != "--refresh-daily"]
        raise SystemExit(_refresh_main(params))
    raise SystemExit(main())'''


def main():
    if not BUILDER.is_file() or not REFRESH.is_file():
        raise FileNotFoundError(f'Required file absent: {BUILDER if not BUILDER.is_file() else REFRESH}')
    raw = BUILDER.read_text(encoding='utf-8')
    normalized = raw.replace('\r\n', '\n')
    if MARKER in normalized:
        print(f'ALREADY CONFIGURED | {BUILDER}')
        return 0
    if normalized.count(OLD_END) != 1:
        raise ValueError(f'Unexpected historical target-builder entry point; NOT modified: {BUILDER}')
    older = 'DEFAULT_OUTPUT = Path.home() / "Downloads"'
    newer = 'DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data"'
    if normalized.count(older) != 1:
        raise ValueError(f'Unexpected historical target-builder output configuration; NOT modified')
    patched = normalized.replace(older, newer).replace(OLD_END, NEW_END)
    compile(patched, str(BUILDER), 'exec')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=BUILDER.parent, suffix='.tmp',
                                         delete=False) as f:
            tmp = Path(f.name)
            f.write(patched)
        os.replace(tmp, BUILDER)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    print(f'CONFIGURED DAILY NHL RESULTS REFRESH | {BUILDER}')
    print(f'CURRENT TARGET CSV | {ROOT / "data" / "nhl_historical_player_game_targets_2009_2026.csv"}')
    print('REBUILD MODE PRESERVED | passing --start-season/--end-season runs the original builder')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        raise SystemExit(f'FAILED | {e}')
