"""Aggregate play-by-play into per-team-game offense metrics, penalties, special teams, and QB-game lines.

Outputs (parquet) in data_cache/:
  team_game_off.parquet  one row per (game_id, team): offensive metrics for that team in that game
  qb_game.parquet        one row per (game_id, team, qb_id): dropback count and EPA sum
"""
import sys
import time
import nflreadpy as nfl
import polars as pl
from common import D

OUT = D
OUT.mkdir(parents=True, exist_ok=True)
SEASONS = list(range(1999, 2027))

NEED = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
        "play_type", "qb_dropback", "epa", "success", "wp", "down", "ydstogo", "yardline_100",
        "yards_gained", "first_down", "third_down_converted", "third_down_failed", "sack",
        "interception", "fumble_lost", "qb_scramble", "penalty", "penalty_team", "penalty_yards",
        "complete_pass", "pass_attempt", "cpoe", "passer_player_id", "rusher_player_id",
        "play_deleted"]

off_parts, qb_parts = [], []
missing_report = {}
t0 = time.time()

for yr in SEASONS:
    p = nfl.load_pbp(yr)
    present = [c for c in NEED if c in p.columns]
    missing = [c for c in NEED if c not in p.columns]
    if missing:
        missing_report[yr] = missing
    p = p.select(present)
    if "play_deleted" in p.columns:
        p = p.filter(pl.col("play_deleted").fill_null(0) != 1)

    # numeric flags -> 0/1 with nulls as 0
    flag_cols = ["qb_dropback", "sack", "interception", "fumble_lost", "complete_pass", "pass_attempt",
                 "first_down", "third_down_converted", "third_down_failed", "penalty"]
    p = p.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in flag_cols if c in p.columns])

    sc = p.filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null()
                  & pl.col("posteam").is_not_null())
    sc = sc.with_columns(
        (pl.col("qb_dropback") == 1).alias("db"),
        ((pl.col("play_type") == "run") & (pl.col("qb_dropback") != 1)).alias("rush"),
        ((pl.col("wp") >= 0.05) & (pl.col("wp") <= 0.95)).fill_null(False).alias("gt"),
        (pl.col("down") <= 2).fill_null(False).alias("ed"),
        (pl.col("down") >= 3).fill_null(False).alias("ld"),
        ((pl.col("wp") >= 0.2) & (pl.col("wp") <= 0.8) & (pl.col("down") <= 2)).fill_null(False).alias("neutral"),
        (pl.col("yardline_100") <= 20).fill_null(False).alias("rz"),
    )

    g = sc.group_by(["game_id", "posteam"]).agg(
        pl.first("season").alias("season"),
        pl.first("week").alias("week"),
        pl.first("season_type").alias("season_type"),
        pl.first("home_team").alias("home_team"),
        pl.first("away_team").alias("away_team"),
        pl.len().alias("n_plays"),
        pl.col("db").sum().alias("n_db"),
        pl.col("rush").sum().alias("n_rush"),
        pl.col("epa").sum().alias("epa_sum"),
        pl.col("epa").filter(pl.col("gt")).sum().alias("epa_gt_sum"),
        pl.col("gt").sum().alias("n_gt"),
        pl.col("epa").filter(pl.col("db")).sum().alias("pass_epa_sum"),
        pl.col("epa").filter(pl.col("rush")).sum().alias("rush_epa_sum"),
        pl.col("success").sum().alias("succ_sum"),
        pl.col("success").filter(pl.col("db")).sum().alias("pass_succ_sum"),
        pl.col("success").filter(pl.col("rush")).sum().alias("rush_succ_sum"),
        pl.col("epa").filter(pl.col("ed")).sum().alias("ed_epa_sum"),
        pl.col("ed").sum().alias("n_ed"),
        pl.col("epa").filter(pl.col("ld")).sum().alias("ld_epa_sum"),
        pl.col("ld").sum().alias("n_ld"),
        ((pl.col("yards_gained") >= 20) & (pl.col("complete_pass") == 1)).sum().alias("expl_pass_n"),
        ((pl.col("yards_gained") >= 10) & pl.col("rush")).sum().alias("expl_rush_n"),
        pl.col("sack").sum().alias("sacks"),
        pl.col("interception").sum().alias("ints"),
        pl.col("fumble_lost").sum().alias("fum_lost"),
        pl.col("yards_gained").sum().alias("yds"),
        (pl.col("down") == 3).sum().alias("third_att"),
        (pl.col("third_down_converted") == 1).sum().alias("third_conv"),
        pl.col("rz").sum().alias("rz_plays"),
        pl.col("epa").filter(pl.col("rz")).sum().alias("rz_epa_sum"),
        pl.col("cpoe").mean().alias("cpoe"),
        pl.col("complete_pass").sum().alias("comp"),
        pl.col("pass_attempt").sum().alias("att"),
        pl.col("neutral").sum().alias("n_neutral"),
        (pl.col("neutral") & pl.col("db")).sum().alias("neutral_db"),
    ).rename({"posteam": "team"})

    # Penalties by the penalized team (offense, defense and special teams combined)
    pen = (p.filter((pl.col("penalty") == 1) & pl.col("penalty_team").is_not_null())
             .group_by(["game_id", "penalty_team"])
             .agg(pl.len().alias("pen_n"), pl.col("penalty_yards").fill_null(0).sum().alias("pen_yds"))
             .rename({"penalty_team": "team"}))

    # Special teams: kicking-unit EPA (own kicks) + return-unit EPA (opponent punts/kickoffs, sign-flipped)
    kk = p.filter(pl.col("play_type").is_in(["punt", "kickoff", "field_goal", "extra_point"])
                  & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
    st_kick = kk.group_by(["game_id", "posteam"]).agg(pl.col("epa").sum().alias("st_kick_epa")).rename({"posteam": "team"})
    st_fg = (kk.filter(pl.col("play_type").is_in(["field_goal", "extra_point"]))
               .group_by(["game_id", "posteam"]).agg(pl.col("epa").sum().alias("fg_epa")).rename({"posteam": "team"}))
    st_ret = (kk.filter(pl.col("play_type").is_in(["punt", "kickoff"]) & pl.col("defteam").is_not_null())
                .group_by(["game_id", "defteam"]).agg((-pl.col("epa")).sum().alias("st_ret_epa")).rename({"defteam": "team"}))

    for extra in (pen, st_kick, st_fg, st_ret):
        g = g.join(extra, on=["game_id", "team"], how="left")
    off_parts.append(g)

    # QB lines: dropback plays attributed to passer, else rusher (scrambles)
    qb = (sc.filter(pl.col("db"))
            .with_columns(pl.coalesce(["passer_player_id", "rusher_player_id"]).alias("qb_id"))
            .filter(pl.col("qb_id").is_not_null())
            .group_by(["game_id", "posteam", "qb_id"])
            .agg(pl.first("season").alias("season"), pl.first("week").alias("week"),
                 pl.len().alias("qb_db"), pl.col("epa").sum().alias("qb_epa_sum"))
            .rename({"posteam": "team"}))
    qb_parts.append(qb)
    print(yr, "done", f"{time.time()-t0:.0f}s", flush=True)

off = pl.concat(off_parts, how="diagonal_relaxed")
qb = pl.concat(qb_parts, how="diagonal_relaxed")
off.write_parquet(OUT / "team_game_off.parquet")
qb.write_parquet(OUT / "qb_game.parquet")
print("off:", off.shape, "qb:", qb.shape)
print("missing columns by season:", {k: v for k, v in list(missing_report.items())[:5]}, "... total seasons with missing:", len(missing_report))
