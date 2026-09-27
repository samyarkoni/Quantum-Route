import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
GAMES_DIR = PROJECT_DIR / "GAMES"
CACHE_PATH = PROJECT_DIR / "Data" / "plays.parquet"

COLUMNS = [
    "game_id", "play_id", "play_type", "yards_gained", "starting_yard", "ending_yard", "time_on_clock_start",
    "down", "yds_to_go", "home_team_score", "away_team_score", "defense_coverage_type",
    "offense_player_ids", "defense_player_ids",
]

# Plays that happen inside a drive (have a down and a line of scrimmage)
SCRIMMAGE_TYPES = ["run", "pass", "turnover", "no_play", "punt", "field_goal", "qb_kneel", "qb_spike"]
# Plays after which the next scrimmage play belongs to a new drive
DRIVE_ENDING_TYPES = ["kickoff", "punt", "turnover", "field_goal", "extra_point",
                      "start_quarter_3", "start_quarter_5", "end_game"]

# Same distance cutoffs as First_Down_Distance/Distance_Stats.py (inclusive upper bounds)
DISTANCE_UPPER = [3, 7, 12, 20]
DISTANCE_LABELS = ["1-3", "4-7", "8-12", "13-20", "21+"]
# Yards from the opponent's end zone
FIELD_UPPER = [10, 20, 30, 40, 50]
FIELD_LABELS = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-99"]


def distance_bucket(yds_to_go):
    return np.asarray(DISTANCE_LABELS)[np.searchsorted(DISTANCE_UPPER, yds_to_go)]


def field_zone(yardline):
    return np.asarray(FIELD_LABELS)[np.searchsorted(FIELD_UPPER, yardline)]


def _parquet_available():
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return True
        except ImportError:
            pass
    return False


def load_plays(refresh=False):
    """All plays from GAMES/, cached to Data/plays.parquet after the first load
    (the cache is skipped if neither pyarrow nor fastparquet is installed)."""
    use_cache = _parquet_available()
    if use_cache and CACHE_PATH.exists() and not refresh:
        cached = pd.read_parquet(CACHE_PATH)
        if set(COLUMNS) <= set(cached.columns):
            return cached

    files = sorted(GAMES_DIR.rglob("*.csv"))
    plays = pd.concat((pd.read_csv(f, usecols=COLUMNS, low_memory=False) for f in files), ignore_index=True)
    plays = plays.sort_values(["game_id", "play_id"], ignore_index=True)
    plays["season"] = plays["game_id"].str[:4].astype(int)

    if use_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plays.to_parquet(CACHE_PATH, index=False)
    return plays


def add_features(plays):
    """Label play calls, scoring, and drive membership. There is no possession column,
    so drives are split on possession-changing plays and points are signed by who scored."""
    plays = plays.copy()
    by_game = plays.groupby("game_id", sort=False)
    ptype = plays["play_type"]

    total = (plays["home_team_score"] + plays["away_team_score"]).groupby(plays["game_id"]).ffill().fillna(0)
    plays["score_delta"] = total.groupby(plays["game_id"]).diff().fillna(total).clip(lower=0)
    delta = plays["score_delta"]

    plays["is_scrimmage"] = ptype.isin(SCRIMMAGE_TYPES) & plays["down"].notna()
    plays["is_turnover"] = ptype.eq("turnover")

    # Turnovers aren't labeled run/pass; coverage type is only charted on dropbacks
    plays["call"] = ptype.where(ptype.isin(["run", "pass", "punt", "field_goal"]))
    # 4th-and-long turnovers with no coverage are punt plays (blocks, muffs, bad snaps), not runs
    punt_turnover = (plays["down"] == 4) & (plays["yds_to_go"] >= 4)
    turnover_call = np.select([plays["defense_coverage_type"].notna(), punt_turnover], ["pass", "punt"], default="run")
    plays.loc[plays["is_turnover"], "call"] = turnover_call[plays["is_turnover"].to_numpy()]

    # Score columns occasionally update a play late, so TDs come from field position and the
    # other scores keep only the part of the change that the play could have produced (mod 6)
    offense_play = plays["is_scrimmage"] & ptype.isin(["run", "pass"])
    plays["offense_td"] = offense_play & plays["ending_yard"].eq(0)
    plays["fg_made"] = plays["is_scrimmage"] & ptype.eq("field_goal") & (delta % 6 == 3)
    plays["safety"] = (plays["is_scrimmage"] & ptype.isin(["run", "pass", "punt"])
                       & ~plays["offense_td"] & (delta == 2))
    plays["defense_score"] = plays["is_scrimmage"] & ptype.isin(["turnover", "punt"]) & (delta >= 6)
    plays["failed_4th"] = (offense_play & (plays["down"] == 4) & ~plays["offense_td"]
                           & (plays["yards_gained"] < plays["yds_to_go"]))

    end_event = (ptype.isin(DRIVE_ENDING_TYPES) | plays["offense_td"] | plays["safety"] | plays["failed_4th"])
    plays["end_label"] = np.select(
        [plays["offense_td"], plays["safety"], plays["defense_score"], plays["failed_4th"],
         ptype.isin(["start_quarter_3", "start_quarter_5", "end_game"]), end_event],
        ["touchdown", "safety", "defensive_td", "downs", "end_of_half", ptype],
        default="",
    )

    # Rows after an end event start a new drive; non-scrimmage rows (XP, timeouts) join the last drive
    seq = end_event.groupby(plays["game_id"]).shift(fill_value=False).groupby(plays["game_id"]).cumsum()
    drive_id = plays["game_id"] + "_" + seq.astype(str)
    plays["drive_id"] = drive_id.where(plays["is_scrimmage"]).groupby(plays["game_id"]).ffill()

    # Points from the drive team's perspective; conversions follow the sign of the score before them
    scoring_sign = pd.Series(np.nan, index=plays.index)
    scoring_sign[plays["offense_td"]] = 1
    scoring_sign[plays["defense_score"]] = -1
    scoring_sign[plays["fg_made"] | plays["safety"] | (ptype.eq("kickoff") & (delta > 0))] = 0
    last_sign = scoring_sign.groupby(plays["game_id"]).ffill().fillna(0)
    is_conversion = ptype.eq("extra_point") | (ptype.isin(["run", "pass"]) & plays["down"].isna())
    plays["drive_points"] = np.select(
        [plays["offense_td"], plays["defense_score"], plays["fg_made"], plays["safety"], is_conversion],
        [6, -6, 3, -2, last_sign * (delta % 6)],
        default=0,
    )
    return plays


def build_drives(plays):
    """One row per drive: starting field position, points, net yards, and how it ended."""
    in_drive = plays[plays["drive_id"].notna()]
    scrimmage = in_drive[in_drive["is_scrimmage"]]
    gains = scrimmage[scrimmage["play_type"].isin(["run", "pass"])]

    drives = pd.DataFrame({
        "game_id": scrimmage.groupby("drive_id")["game_id"].first(),
        "season": scrimmage.groupby("drive_id")["season"].first(),
        "start_yardline": scrimmage.groupby("drive_id")["starting_yard"].first(),
        "plays": gains.groupby("drive_id").size(),
        "yards": gains.groupby("drive_id")["yards_gained"].sum(),
        "points": in_drive.groupby("drive_id")["drive_points"].sum(),
        "outcome": in_drive[in_drive["end_label"] != ""].groupby("drive_id")["end_label"].first(),
    })
    drives[["plays", "yards"]] = drives[["plays", "yards"]].fillna(0)
    drives["outcome"] = drives["outcome"].fillna("end_of_half")
    return drives.dropna(subset=["start_yardline"])


def add_game_state(plays):
    """Quarter, half, seconds left in the half, and cleaned running scores. The raw score columns
    are the score after each play but sometimes go stale (mostly on timeout rows) and dip, so each
    team's score is carried forward as a running max."""
    plays = plays.copy()
    game = plays["game_id"]
    quarter_start = plays["play_type"].str.startswith("start_quarter", na=False)
    plays["quarter"] = quarter_start.groupby(game).cumsum().clip(lower=1)
    plays["half"] = np.select([plays["quarter"] <= 2, plays["quarter"] <= 4], [1, 2], default=3)  # 3 = overtime

    clock = plays["time_on_clock_start"].str.split(":", expand=True).astype(float)
    quarter_seconds = (clock[0] * 60 + clock[1]).groupby(game).ffill()
    plays["half_seconds_left"] = quarter_seconds + np.where(plays["quarter"].isin([1, 3]), 900, 0)

    for side in ("home", "away"):
        score = plays[f"{side}_team_score"].groupby(game).ffill().fillna(0)
        plays[f"{side}_score"] = score.groupby(game).cummax()
    return plays


def _split_ids(ids):
    return [i for i in ids.split(";") if i] if isinstance(ids, str) else []


def _label_sides(offense, defense, iterations=5):
    """Split one game's plays by which team is on offense: +1 / -1, 0 if the play lists no players.
    Players listed on the same side of a play are teammates, but a handful of plays misattribute a
    player, so teams are found by majority vote rather than strict grouping."""
    seed = next(i for i, ids in enumerate(offense) if len(ids) >= 5)
    team = {pid: 1 for pid in offense[seed]} | {pid: -1 for pid in defense[seed]}
    for _ in range(iterations):
        sides = np.sign([sum(team.get(p, 0) for p in off) - sum(team.get(p, 0) for p in de)
                         for off, de in zip(offense, defense)])
        votes = {}
        for side, off, de in zip(sides, offense, defense):
            for p in off:
                votes[p] = votes.get(p, 0) + side
            for p in de:
                votes[p] = votes.get(p, 0) - side
        team = {p: np.sign(v) for p, v in votes.items()}
    return sides


def add_possession(plays):
    """posteam / defteam for every play that lists players (kickoffs and punts belong to the kicking
    team). game_id is season_week_AWAY_HOME; which side is home comes from which score column moves
    when that side scores. Expects add_game_state() to have run."""
    plays = plays.copy()
    offense = plays["offense_player_ids"].map(_split_ids)
    defense = plays["defense_player_ids"].map(_split_ids)
    side = pd.Series(0, index=plays.index)
    for _, idx in plays.groupby("game_id", sort=False).indices.items():
        side.iloc[idx] = _label_sides(offense.iloc[idx].tolist(), defense.iloc[idx].tolist())

    # The last play with players before a score is almost always the scoring team's
    game = plays["game_id"]
    home_scored = plays["home_score"].groupby(game).diff().fillna(0) > 0
    away_scored = plays["away_score"].groupby(game).diff().fillna(0) > 0
    last_side = side.where(side != 0).groupby(game).ffill().fillna(0)
    vote = (last_side * (home_scored.astype(int) - away_scored.astype(int))).groupby(game).sum()
    home_side = plays["game_id"].map(np.sign(vote))

    teams = plays["game_id"].str.split("_", expand=True)
    is_home = side == home_side
    has_team = (side != 0) & (home_side != 0)
    plays["posteam"] = np.where(is_home, teams[3], teams[2])
    plays["defteam"] = np.where(is_home, teams[2], teams[3])
    plays.loc[~has_team, ["posteam", "defteam"]] = None
    plays["posteam_is_home"] = is_home.where(has_team)

    # Offense's lead before the snap
    home_lead = (plays["home_score"] - plays["away_score"]).groupby(game).shift(fill_value=0)
    plays["score_diff"] = np.where(is_home, home_lead, -home_lead)
    plays.loc[~has_team, "score_diff"] = np.nan
    return plays
