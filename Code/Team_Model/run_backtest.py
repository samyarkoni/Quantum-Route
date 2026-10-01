"""Walk-forward backtest: for each week, fit on games played before it, then simulate that week.

    python run_backtest.py 2025 [note]   # one season -> Data/Team_Model/backtest_2025.csv,
                                         # and a new run in the backtest history (Code/db.py)
    python run_backtest.py report        # score every saved season against the closing lines
"""
import math
import sys
import time
import zlib

import numpy as np
import pandas as pd

from data import CODE_DIR, OUTPUT_DIR, load_epa_plays, load_games
from game_sim import GamePredictor

sys.path.insert(0, str(CODE_DIR))
import db  # noqa: E402


def backtest_season(season, first_week=1, note=None):
    plays, games = load_epa_plays(), load_games()
    season_games = games[games["season"] == season]
    run_id = db.start_run(season, first_week, note)
    print(f"backtest run {run_id}", flush=True)
    rows = []
    for week in sorted(season_games["week"].unique()):
        if week < first_week:
            continue
        start = time.time()
        predictor = GamePredictor(plays, season, week, verbose=True)
        for g in season_games[season_games["week"] == week].itertuples():
            pred = predictor.predict(g.home_team, g.away_team, g.spread_line, neutral=g.location == "Neutral",
                                     seed=zlib.crc32(g.game_id.encode()))
            pred["total_pred"] = pred.pop("total")
            rows.append({"game_id": g.game_id, "season": season, "week": week, "home_team": g.home_team,
                         "away_team": g.away_team, "spread_line": g.spread_line, "total_line": g.total_line,
                         "home_moneyline": g.home_moneyline, "away_moneyline": g.away_moneyline,
                         "result": g.result, "total": g.total, **pred})
        print(f"{season} week {week}: {time.time() - start:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"backtest_{season}.csv", index=False)
        db.save_predictions(run_id, [r for r in rows if r["week"] == week])
    db.finish_run(run_id)


def implied_probability(moneyline):
    moneyline = np.asarray(moneyline, dtype=float)
    return np.where(moneyline < 0, -moneyline / (-moneyline + 100), 100 / (moneyline + 100))


def score(df, slope=1.0):
    """Accuracy against the closing market. slope rescales the model's margin around zero."""
    margin = slope * df["margin"]
    home_p = implied_probability(df["home_moneyline"])
    away_p = implied_probability(df["away_moneyline"])
    market_win = home_p / (home_p + away_p)  # de-vigged, proportional method
    home_won = np.where(df["result"] == 0, 0.5, (df["result"] > 0).astype(float))
    # Win probability from the rescaled margin, using the simulated spread of outcomes
    model_win = 1 - 0.5 * (1 + np.vectorize(math.erf)((0 - margin) / (df["margin_sd"] * np.sqrt(2))))

    edge = margin - df["spread_line"]
    graded = df["result"] != df["spread_line"]
    cover = (df["result"] > df["spread_line"]) == (edge > 0)
    out = {
        "games": len(df),
        "mae_model": np.abs(df["result"] - margin).mean(),
        "mae_market": np.abs(df["result"] - df["spread_line"]).mean(),
        "corr_model_market": np.corrcoef(margin, df["spread_line"])[0, 1],
        "brier_model": np.mean((model_win - home_won) ** 2),
        "brier_market": np.nanmean((market_win - home_won) ** 2),
        "total_mae_model": np.abs(df["total"] - df["total_pred"]).mean(),
        "total_mae_market": np.abs(df["total"] - df["total_line"]).mean(),
    }
    for threshold in (0, 1.5, 3):
        pick = graded & (np.abs(edge) > threshold)
        out[f"ats_edge>{threshold}"] = f"{cover[pick].sum()}-{(~cover[pick]).sum()} ({cover[pick].mean():.1%})"
    return out


def report():
    files = sorted(OUTPUT_DIR.glob("backtest_*.csv"))
    frames = {int(f.stem.split("_")[1]): pd.read_csv(f) for f in files}

    rows, slopes = {}, {}
    for season, df in frames.items():
        # Least-squares slope of the actual margin on the model margin (through the origin)
        slopes[season] = float((df["margin"] * df["result"]).sum() / (df["margin"] ** 2).sum())
        rows[f"{season} raw"] = score(df)
    for season, df in frames.items():
        others = [slopes[s] for s in frames if s != season]
        if others:
            rows[f"{season} slope from other seasons ({np.mean(others):.2f})"] = score(df, np.mean(others))

    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nMargin calibration slope (actual ~ slope * model) by season:",
          {s: round(v, 2) for s, v in slopes.items()})


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        backtest_season(int(sys.argv[1]), note=" ".join(sys.argv[2:]) or None)
