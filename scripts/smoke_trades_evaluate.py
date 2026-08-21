#!/usr/bin/env python3
"""Smoke-test POST /trades/evaluate against a running API.

Requires a live RosterContext (import Excel first via /roster/import + claim).

Example
-------
    export API_URL=http://localhost:8000
    export API_KEY=...
    python scripts/smoke_trades_evaluate.py \\
        --context-id <uuid> \\
        --sheet "Serie A" \\
        --team "LaMiaRosa" \\
        --give 1234 --receive 5678 \\
        --mode mantra
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    p.add_argument("--context-id", required=True)
    p.add_argument("--sheet", required=True, dest="sheet_name")
    p.add_argument("--team", required=True, dest="team_name")
    p.add_argument("--mode", choices=("classic", "mantra"), default="mantra")
    p.add_argument("--give", action="append", default=[], help="player_id (repeatable)")
    p.add_argument("--receive", action="append", default=[], help="player_id (repeatable)")
    p.add_argument("--tolerance", type=float, default=10.0)
    args = p.parse_args(argv)

    if not args.give and not args.receive:
        print("Provide at least one --give or --receive", file=sys.stderr)
        return 2

    body = {
        "contextId": args.context_id,
        "sheetName": args.sheet_name,
        "teamName": args.team_name,
        "mode": args.mode,
        "give": args.give,
        "receive": args.receive,
        "formationPrefs": ["4-3-3", "3-5-2", "3-4-3"],
        "tolerancePercent": args.tolerance,
    }
    url = args.api_url.rstrip("/") + "/trades/evaluate"
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            status = resp.status
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {err_body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return 1

    print(f"HTTP {status}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    # Minimal assertions for CI-style smoke
    ok = True
    for key in ("mode", "valid", "give", "receive", "rationale"):
        if key not in payload:
            print(f"MISSING key: {key}", file=sys.stderr)
            ok = False
    if payload.get("valid") and payload.get("verdict") not in (
        "vantaggioso",
        "equilibrato",
        "sfavorevole",
    ):
        print(f"Unexpected verdict: {payload.get('verdict')}", file=sys.stderr)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
