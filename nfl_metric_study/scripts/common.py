"""Shared loaders: schedule/games table, per-team-game long table, and opponent-adjusted ridge ratings.

No betting lines or odds are loaded: those columns are dropped immediately after reading the schedule.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data_cache"

CODE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}  # franchise continuity (pbp already uses current codes)
ODDS_COLS = ["spread_line", "total_line", "away_moneyline", "home_moneyline",
             "away_spread_odds", "home_spread_odds", "under_odds", "over_odds"]

# metric -> direction in which a HIGHER number is good for the offense
OFF_METRICS = {
    "pts": +1, "epa_pp": +1, "epa_gt": +1, "pass_epa": +1, "rush_epa": +1,
    "succ": +1, "pass_succ": +1, "rush_succ": +1, "ed_epa": +1, "ld_epa": +1,
    "expl_pass": +1, "expl_rush": +1, "sack_rate": -1, "int_rate": -1, "fum_rate": -1,
    "ypp": +1, "third_rate": +1, "rz_epa": +1,
}
TEAMLEVEL_METRICS = {"pen_yds": -1, "st_epa": +1, "fg_epa": +1}
ALL_RATED = {**OFF_METRICS, **TEAMLEVEL_METRICS}


def load_games():
    sch = nfl.load_schedules(list(range(1999, 2027))).to_pandas()
    sch = sch.drop(columns=[c for c in ODDS_COLS if c in sch.columns] + ["total"])
    for c in ("home_team", "away_team"):
        sch[c + "_orig"] = sch[c]
        sch[c] = sch[c].replace(CODE)
    sch["gameday"] = pd.to_datetime(sch["gameday"])
    sch = sch.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    sch["played"] = sch["result"].notna()
    sch["neutral"] = (sch["location"] == "Neutral").astype(int)
    return sch


def load_offense():
    off = pd.read_parquet(f"{D}/team_game_off.parquet")
    o = pd.DataFrame({"game_id": off.game_id, "team": off.team})
    o["epa_pp"] = off.epa_sum / off.n_plays
    o["epa_gt"] = off.epa_gt_sum / off.n_gt.replace(0, np.nan)
    o["pass_epa"] = off.pass_epa_sum / off.n_db.replace(0, np.nan)
    o["rush_epa"] = off.rush_epa_sum / off.n_rush.replace(0, np.nan)
    o["succ"] = off.succ_sum / off.n_plays
    o["pass_succ"] = off.pass_succ_sum / off.n_db.replace(0, np.nan)
    o["rush_succ"] = off.rush_succ_sum / off.n_rush.replace(0, np.nan)
    o["ed_epa"] = off.ed_epa_sum / off.n_ed.replace(0, np.nan)
    o["ld_epa"] = off.ld_epa_sum / off.n_ld.replace(0, np.nan)
    o["expl_pass"] = off.expl_pass_n / off.n_db.replace(0, np.nan)
    o["expl_rush"] = off.expl_rush_n / off.n_rush.replace(0, np.nan)
    o["sack_rate"] = off.sacks / off.n_db.replace(0, np.nan)
    o["int_rate"] = off.ints / off.n_db.replace(0, np.nan)
    o["fum_rate"] = off.fum_lost / off.n_plays
    o["ypp"] = off.yds / off.n_plays
    o["third_rate"] = off.third_conv / off.third_att.replace(0, np.nan)
    o["rz_epa"] = off.rz_epa_sum / off.rz_plays.replace(0, np.nan)
    o["n_plays"] = off.n_plays
    o["n_db"] = off.n_db
    o["pen_yds"] = off.pen_yds.fillna(0.0)
    o["st_epa"] = off.st_kick_epa.fillna(0.0) + off.st_ret_epa.fillna(0.0)
    o["fg_epa"] = off.fg_epa.fillna(0.0)
    o["neutral_pass_rate"] = off.neutral_db / off.n_neutral.replace(0, np.nan)
    return o


def build_long(sch, o):
    """Two rows per game (one per team). Own offense metrics + opponent offense metrics (= defense allowed)."""
    common = ["game_id", "season", "week", "game_type", "gameday", "neutral", "played"]
    h = sch[common].copy()
    h["team"], h["opp"], h["is_home"] = sch.home_team, sch.away_team, 1
    h["pts"], h["pts_allowed"] = sch.home_score, sch.away_score
    h["rest"], h["coach"], h["qb_id"] = sch.home_rest, sch.home_coach, sch.home_qb_id
    a = sch[common].copy()
    a["team"], a["opp"], a["is_home"] = sch.away_team, sch.home_team, 0
    a["pts"], a["pts_allowed"] = sch.away_score, sch.home_score
    a["rest"], a["coach"], a["qb_id"] = sch.away_rest, sch.away_coach, sch.away_qb_id
    L = pd.concat([h, a], ignore_index=True)
    L = L.merge(o, on=["game_id", "team"], how="left")
    oo = o.add_prefix("opp_").rename(columns={"opp_game_id": "game_id", "opp_team": "opp"})
    L = L.merge(oo, on=["game_id", "opp"], how="left")
    L["win"] = np.where(L.played, (L.pts > L.pts_allowed).astype(float) + 0.5 * (L.pts == L.pts_allowed), np.nan)
    L["margin"] = L.pts - L.pts_allowed
    L = L.sort_values(["gameday", "game_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)
    return L


class RidgeRatings:
    """Opponent-adjusted team ratings, refit every (season, week) using ONLY games played before that week.

    Model per metric m (offense team i vs defense team j in game g):
        y = mu + h*home + off_i + def_j       (weights decay exponentially with age in calendar weeks)
    off_i, def_j are ridge-shrunk toward 0. 'def_j' is what defense j allows relative to average
    (positive = allows more of the metric).
    """

    def __init__(self, L, sch, metrics, half_life_weeks, lam, window_years=4, offseason_credit=0.0):
        # offseason_credit: calendar weeks of age removed per season boundary crossed (0 = real calendar time)
        self.credit = offseason_credit
        self.metrics = list(metrics)
        self.teams = sorted(set(L.team) | set(L.opp))
        self.T = len(self.teams)
        tix = {t: i for i, t in enumerate(self.teams)}
        R = L[L.played & L.epa_pp.notna()].copy()           # rows with pbp data (3 old games lack it)
        R = R.sort_values(["gameday", "game_id", "is_home"], kind="mergesort").reset_index(drop=True)  # deterministic ties
        self.date = R.gameday.values.astype("datetime64[D]")
        self.season = R.season.values.astype(int)
        self.o = R.team.map(tix).values
        self.d = R.opp.map(tix).values
        self.home = (R.is_home * (1 - R.neutral)).values.astype(float)
        self.pl = (R.n_plays.fillna(R.n_plays.mean()) / 62.0).values
        Y = R[self.metrics].copy()
        for c in Y.columns:                                   # rare NaNs -> mean of EARLIER rows only (no look-ahead)
            past_mean = Y[c].expanding().mean().shift(1)
            Y[c] = Y[c].fillna(past_mean).fillna(0.0)
        self.Y = Y.values.astype(float)
        self.hl, self.lam, self.win = half_life_weeks, lam, window_years * 365
        ws = sch.groupby(["season", "week"]).gameday.min()
        self.wk_start = {k: v.to_datetime64().astype("datetime64[D]") for k, v in ws.items()}

    def fit_snapshot(self, season, week):
        S = self.wk_start[(season, week)]
        lo = np.searchsorted(self.date, S - np.timedelta64(self.win, "D"), side="left")
        hi = np.searchsorted(self.date, S, side="left")        # strictly before the week's first kickoff
        n = hi - lo
        T, M = self.T, len(self.metrics)
        if n < 60:
            return None
        age_w = (S - self.date[lo:hi]).astype(float) / 7.0
        if self.credit:
            age_w = np.maximum(age_w - self.credit * (season - self.season[lo:hi]), 0.0)
        w =np.exp(-np.log(2) * age_w / self.hl) * self.pl[lo:hi]
        X = np.zeros((n, 2 + 2 * T))
        X[:, 0] = 1.0
        X[:, 1] = self.home[lo:hi]
        r = np.arange(n)
        X[r, 2 + self.o[lo:hi]] = 1.0
        X[r, 2 + T + self.d[lo:hi]] = 1.0
        Xw = X * w[:, None]
        A = X.T @ Xw
        A[np.arange(2, 2 + 2 * T), np.arange(2, 2 + 2 * T)] += self.lam
        B = np.linalg.solve(A, Xw.T @ self.Y[lo:hi])
        return {"mu": B[0], "home": B[1], "off": B[2:2 + T], "def": B[2 + T:], "n": n}

    def fit_all(self, keys):
        out = {}
        for k in keys:
            res = self.fit_snapshot(*k)
            if res is not None:
                out[k] = res
        return out
