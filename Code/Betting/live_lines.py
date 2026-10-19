"""Live lines for the published board: the latest nflverse lines, prices and scores of the newest
season's games, written as the document the board reads from its shared database (lines/current).

    python live_lines.py OUT.json                      # write the document
    python live_lines.py OUT.json --against PREV.json  # ... and exit 3 if nothing differs from PREV

The board lays this document over the lines it was built with, so odds that nflverse posts or moves
after a build reach the page without republishing it. Scheduled claude.ai routines run this hourly
on game days and copy OUT.json into lines/current when it changed. Standard library only, so it runs in a
fresh checkout without the project's data or packages.
"""
import csv
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
LINE_COLUMNS = ["spread_line", "total_line", "home_moneyline", "away_moneyline", "home_spread_odds",
                "away_spread_odds", "over_odds", "under_odds", "result", "total", "home_score", "away_score"]


def live_lines():
    """{game_id: {column: float or None}} for the newest season's games that have any line or score."""
    with urllib.request.urlopen(GAMES_URL, timeout=60) as response:
        rows = list(csv.DictReader(io.TextIOWrapper(response, encoding="utf-8")))
    season = max(int(r["season"]) for r in rows)
    games = {}
    for r in rows:
        if int(r["season"]) != season:
            continue
        values = {c: float(r[c]) if r.get(c, "").strip() not in ("", "NA") else None for c in LINE_COLUMNS}
        if any(v is not None for v in values.values()):
            games[r["game_id"]] = values
    return games


if __name__ == "__main__":
    if len(sys.argv) not in (2, 4) or (len(sys.argv) == 4 and sys.argv[2] != "--against"):
        sys.exit(__doc__)
    games = live_lines()
    previous = {}
    if len(sys.argv) == 4:
        try:
            with open(sys.argv[3]) as f:
                previous = json.load(f).get("games") or {}
        except (OSError, ValueError):
            pass  # no previous document: everything is new
    changed = sorted(g for g in games if games[g] != previous.get(g))
    with open(sys.argv[1], "w") as f:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "games": games},
                  f, separators=(",", ":"))
    print(f"{len(games)} games with lines, {len(changed)} changed"
          + (f": {', '.join(changed[:8])}{' ...' if len(changed) > 8 else ''}" if changed else ""))
    if not changed:
        sys.exit(3)
