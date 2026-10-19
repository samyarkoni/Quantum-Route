"""Betting board: one deterministic pick (or pass) per game from the team model's simulations.

    python betting_board.py predict 2026 6     # simulate week 6 -> Data/Team_Model/predictions_2026_06.csv
    python betting_board.py predict 2026       # every week of 2026
    python betting_board.py build              # -> Data/Betting/betting_board.html
    python betting_board.py build --refresh    # re-download the nflverse schedule (new lines/results) first

The board reads every backtest_*.csv and predictions_*.csv in Data/Team_Model, joins the lines and
prices from the nflverse schedule, and writes one self-contained HTML page. The pick rule runs in
the page, so lines can be edited there to match a sportsbook and the rule's thresholds tuned
against the graded games.

build also writes betting_board_page.html, the same page without its document wrapper, which is
what gets published to the shared claude.ai board. There, the "Bet it" boxes and the "Bets we
placed" tab save to the board's shared database (one bets/<game_id> document per bet); the local
file shows the board without them. Lines and scores posted after a build reach the published page
through its lines/current document (see live_lines.py), so the board needs rebuilding only for new
predictions.
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR / "Team_Model"))
from data import GAMES_PATH, GAMES_URL, OUTPUT_DIR, load_epa_plays  # noqa: E402
from live_lines import LINE_COLUMNS  # noqa: E402

BOARD_DIR = OUTPUT_DIR.parent / "Betting"
TEMPLATE = Path(__file__).resolve().parent / "board_template.html"


def load_schedule(refresh=False):
    """The full nflverse schedule, every season (load_games stops at the last season with plays)."""
    if refresh or not GAMES_PATH.exists():
        GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(GAMES_URL).to_csv(GAMES_PATH, index=False)
    return pd.read_csv(GAMES_PATH)


def team_sides(schedule):
    """One row per (game, team) in schedule order, with that team's listed starter (qb_id, qb_name),
    the most recent listed starter at or before this game (qb_now, carried forward through games not
    yet listed) and the previous game's starter (qb_prev, qb_prev_name)."""
    cols = ["game_id", "season", "week", "gameday", "gametime"]
    sides = pd.concat([schedule[cols + [f"{s}_team", f"{s}_qb_id", f"{s}_qb_name"]].set_axis(
        cols + ["team", "qb_id", "qb_name"], axis=1).assign(side=s) for s in ("home", "away")])
    sides = sides[sides["team"].notna()].sort_values(["season", "week", "gameday", "gametime", "game_id"])
    by_team = sides.groupby("team")
    sides["qb_now"] = by_team["qb_id"].ffill()
    sides["qb_prev"] = by_team["qb_now"].shift()
    sides["qb_prev_name"] = by_team["qb_name"].ffill().groupby(sides["team"]).shift()
    return sides


def qb_changes(schedule):
    """Per game, whether each team starts a different QB than in its previous game (across seasons
    too: a new week-1 starter counts). Null until nflverse lists the starter."""
    sides = team_sides(schedule)
    listed = sides["qb_id"].notna() & sides["qb_prev"].notna()
    sides["qb_change"] = np.where(listed, sides["qb_id"] != sides["qb_prev"], None)
    out = sides.pivot(index="game_id", columns="side", values=["qb_change", "qb_name", "qb_prev_name"])
    out.columns = [f"{side}_{col}" for col, side in out.columns]
    return out


def predict(season, weeks=None):
    """Simulate every game in the given weeks, fitting only on plays before each week. Seeds match
    run_backtest.py, so a week that has been backtested reproduces the same numbers."""
    from game_sim import GamePredictor

    plays, schedule = load_epa_plays(), load_schedule()
    # A starter nflverse hasn't listed yet is the team's most recent listed one
    current = team_sides(schedule).set_index(["game_id", "side"])["qb_now"]
    schedule = schedule.assign(home_qb_id=schedule["game_id"].map(current.xs("home", level="side")),
                               away_qb_id=schedule["game_id"].map(current.xs("away", level="side")))
    season_games = schedule[(schedule["season"] == season) & schedule["home_team"].notna()]
    weeks = weeks or sorted(season_games["week"].unique())
    last = plays.loc[plays["season"] == plays["season"].max(), ["season", "week"]].max()
    if season > last["season"]:
        print(f"Warning: plays end at {last['season']} week {last['week']}; {season} ratings are "
              f"last season's form, carried over.")

    predictor, fitted_on = None, None
    for week in weeks:
        # Refit only when new plays have arrived since the last fit
        n_history = int(((plays["season"] < season) | ((plays["season"] == season) & (plays["week"] < week))).sum())
        if n_history != fitted_on:
            predictor, fitted_on = GamePredictor(plays, season, week, verbose=True), n_history
        rows = []
        for g in season_games[season_games["week"] == week].itertuples():
            pred = predictor.predict(g.home_team, g.away_team, neutral=g.location == "Neutral",
                                     seed=zlib.crc32(g.game_id.encode()), home_qb=g.home_qb_id, away_qb=g.away_qb_id)
            pred["total_pred"] = pred.pop("total")
            rows.append({"game_id": g.game_id, "season": season, "week": week, "home_team": g.home_team,
                         "away_team": g.away_team, **pred})
        path = OUTPUT_DIR / f"predictions_{season}_{week:02d}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{season} week {week}: {len(rows)} games -> {path.name}", flush=True)


def load_predictions():
    """Backtests win over standalone predictions for the same game (both use the same seeds)."""
    backtests = [pd.read_csv(f) for f in sorted(OUTPUT_DIR.glob("backtest_*.csv"))]
    predictions = [pd.read_csv(f) for f in sorted(OUTPUT_DIR.glob("predictions_*.csv"))]
    frames = [df.assign(source="backtest") for df in backtests] + [df.assign(source="prediction") for df in predictions]
    if not frames:
        raise FileNotFoundError(f"No backtest_*.csv or predictions_*.csv in {OUTPUT_DIR}")
    keep = ["game_id", "season", "week", "home_team", "away_team", "margin", "margin_sd", "total_pred", "source"]
    return pd.concat([df[keep] for df in frames]).drop_duplicates("game_id", keep="first")


def calibration(games):
    """Per-season margin slope and residual SDs, each estimated from the other seasons' graded
    games so a season's picks never see its own results. Seasons with no graded history of their
    own (the upcoming one) use every graded season."""
    graded = games[games["source"].eq("backtest") & games["result"].notna()]
    stats = {}
    for season, df in graded.groupby("season"):
        stats[season] = {"slope": float((df["margin"] * df["result"]).sum() / (df["margin"] ** 2).sum()),
                         "games": df}
    out = {}
    for season in sorted(games["season"].unique()):
        others = [s for s in stats if s != season] or list(stats)
        slope = float(np.mean([stats[s]["slope"] for s in others]))
        pool = pd.concat([stats[s]["games"] for s in others])
        out[int(season)] = {
            "slope": slope,
            "sd_margin_raw": float((pool["result"] - pool["margin"]).std()),
            "sd_margin_cal": float((pool["result"] - slope * pool["margin"]).std()),
            "sd_total": float((pool["total"] - pool["total_pred"]).std()),
            "from_seasons": [int(s) for s in others],
        }
    return out


def build(refresh=False):
    schedule = load_schedule(refresh).set_index("game_id")
    games = load_predictions()
    extra = schedule.reindex(games["game_id"])[LINE_COLUMNS + ["gameday", "gametime", "location"]]
    qb = qb_changes(schedule.reset_index()).reindex(games["game_id"])
    games = pd.concat([games.reset_index(drop=True), extra.reset_index(drop=True), qb.reset_index(drop=True)], axis=1)
    games = games.sort_values(["season", "week", "gameday", "gametime", "game_id"]).reset_index(drop=True)

    plays = load_epa_plays()
    last = plays.loc[plays["season"] == plays["season"].max(), ["season", "week"]].max()
    payload = {
        "games": json.loads(games.to_json(orient="records")),
        "calibration": calibration(games),
        "plays_through": {"season": int(last["season"]), "week": int(last["week"])},
        "built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    page = TEMPLATE.read_text().replace("/*__BOARD_DATA__*/null", json.dumps(payload, separators=(",", ":")))
    # The published page gets its document wrapper from claude.ai; the local copy needs its own
    (BOARD_DIR / "betting_board_page.html").write_text(page)
    path = BOARD_DIR / "betting_board.html"
    path.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
                    + page + "</body>\n</html>\n")
    print(f"{len(games)} games ({games['result'].notna().sum()} graded) -> {path}")


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    if command == "predict":
        predict(int(args[0]), [int(w) for w in args[1:]] or None)
    elif command == "build":
        build(refresh="--refresh" in args)
    else:
        sys.exit(__doc__)
