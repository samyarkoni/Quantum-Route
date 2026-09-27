import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR / "Drive_Model"))
sys.path.insert(0, str(CODE_DIR / "Scoring_Model"))
from expected_points import ExpectedPointsModel, add_epa, label_next_score, prepare_plays  # noqa: E402
from plays import PROJECT_DIR  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "Data" / "Team_Model"
CACHE_PATH = OUTPUT_DIR / "plays_epa_full.parquet"
GAMES_PATH = PROJECT_DIR / "Data" / "games.csv"
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
SEASONS = range(2018, 2026)


def load_games():
    """nflverse schedule: scores, closing lines, coaches, starting QBs. result and spread_line are
    both home minus away (spread_line > 0 means the home team is favored)."""
    if not GAMES_PATH.exists():
        GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(GAMES_URL).to_csv(GAMES_PATH, index=False)
    games = pd.read_csv(GAMES_PATH)
    return games[games["season"].isin(SEASONS)].reset_index(drop=True)


def load_epa_plays(refresh=False):
    """Every play with possession, game state, ep and epa, plus week and neutral-site flags.
    The EP model is fit on all seasons: it is a league-wide valuation of field position, so the
    leak into backtests is a few coefficients' worth."""
    if CACHE_PATH.exists() and not refresh:
        return pd.read_parquet(CACHE_PATH)

    plays = label_next_score(prepare_plays(SEASONS))
    plays = add_epa(plays, ExpectedPointsModel().fit(plays))
    plays = plays.drop(columns=["offense_player_ids", "defense_player_ids"])

    games = load_games().set_index("game_id")
    plays["week"] = plays["game_id"].str.split("_").str[1].astype(int)
    plays["neutral"] = plays["game_id"].map(games["location"]).eq("Neutral")
    plays["posteam_is_home"] = plays["posteam_is_home"].astype(float)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plays.to_parquet(CACHE_PATH, index=False)
    return plays


def before(plays, season, week):
    """Plays from games played before (season, week)."""
    return plays[(plays["season"] < season) | ((plays["season"] == season) & (plays["week"] < week))]


def recency_weights(plays, season, week, decay=0.95, carryover=0.5):
    """Weights for estimating team form going into (season, week): this season's games decay by
    `decay` per week back; last season's start at `carryover` (rosters and coaches change in the
    offseason) and keep decaying from its final week; anything older gets 0."""
    this_season = plays["season"] == season
    last_season = plays["season"] == season - 1
    last_week = plays.loc[last_season, "week"].max() if last_season.any() else 0
    weeks_back = np.where(this_season, week - plays["week"], last_week + 1 - plays["week"])
    weights = decay ** (weeks_back - 1) * np.where(this_season, 1.0, carryover)
    return np.where(this_season | last_season, weights, 0.0)


def neutral_filter(plays):
    """Drop likely garbage time: 4th-quarter plays with the offense up or down by more than 21."""
    return ~((plays["quarter"] == 4) & (plays["score_diff"].abs() > 21))
