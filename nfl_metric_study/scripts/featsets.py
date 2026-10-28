"""Feature groups (home-minus-away differences) shared by evaluation scripts. Same definitions as models.py."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import numpy as np
import pandas as pd
from common import ROOT, D

G = pd.read_parquet(f"{D}/model_table.parquet")
G = G[(G.game_type == "REG") & (G.season >= 2002)].reset_index(drop=True)


def dd(cfg, m, part):
    return G[f"{cfg}__{m}__{part}_h"] - G[f"{cfg}__{m}__{part}_a"]


def rating_block(cfg, metrics, parts=("off", "def")):
    return {f"{cfg}.{m}.{p}": dd(cfg, m, p) for m in metrics for p in parts}


_g = {
    "ratings_points": rating_block("L20", ["pts"]),
    "ratings_epa": rating_block("L20", ["epa_pp", "pass_epa", "rush_epa", "succ", "expl_pass", "expl_rush"]),
    "ratings_other_eff": rating_block("L20", ["epa_gt", "ed_epa", "ld_epa", "ypp", "pass_succ", "rush_succ"]),
    "turnovers": rating_block("L20", ["int_rate", "fum_rate"]),
    "sacks": rating_block("L20", ["sack_rate"]),
    "situational": rating_block("L20", ["third_rate", "rz_epa"]),
    "special_penalty": rating_block("L20", ["st_epa", "fg_epa", "pen_yds"], parts=("off",)),
    "form_short": rating_block("S6", ["pts", "epa_pp"]),
    "history_long": rating_block("H80", ["pts", "epa_pp"]),
    "record_unadj": {c: G[f"h_{c}"] - G[f"a_{c}"] for c in
                     ["ytd_win", "ytd_pd", "ytd_pyth", "prev1_win", "prev1_pd", "prev3_win", "prev3_pd"]},
    "results_form": {c: G[f"h_{c}"] - G[f"a_{c}"] for c in ["last3_pd", "last3_win", "last1_margin"]},
    "qb": {"qb_rating": G.h_qb_rating - G.a_qb_rating, "qb_exp": G.h_qb_exp - G.a_qb_exp,
           "qb_delta": G.h_qb_delta - G.a_qb_delta, "h_qb_changed": G.h_qb_changed, "a_qb_changed": G.a_qb_changed,
           "qb_rookie": G.h_qb_rookie - G.a_qb_rookie},
    "coach": {"coach_tenure": np.log1p(G.h_coach_tenure) - np.log1p(G.a_coach_tenure),
              "h_coach_new": (G.h_coach_tenure < 17).astype(float), "a_coach_new": (G.a_coach_tenure < 17).astype(float)},
    "h2h": {"h2h_win": G.h2h_win, "h2h_margin": G.h2h_margin},
    "rest": {c: G[c] for c in ["rest_diff", "home_bye", "away_bye", "home_short", "away_short", "home_long", "away_long"]},
    "travel": {c: G[c] for c in ["away_miles", "away_tz_east", "away_tz_west", "away_body_early"]},
    "weather_venue": {c: G[c] for c in ["dome", "temp_cold", "temp_hot", "wind", "turf"]},
    "slot": {c: G[c] for c in ["thu", "mon", "sat", "prime", "late_window"]},
    "division": {"div_game": G.div_game},
    "hfa_dyn": {"hfa_pts": G["L20__hfa_pts"]},
}
GROUPS = {k: pd.DataFrame(v) for k, v in _g.items()}
BASE = pd.DataFrame({"neutral": G.neutral})
ALL = list(GROUPS)
STRENGTH = ["ratings_points", "ratings_epa", "ratings_other_eff"]
SEQ = ["ratings_points", "ratings_epa", "ratings_other_eff", "qb", "history_long", "form_short", "record_unadj",
       "results_form", "turnovers", "sacks", "situational", "special_penalty", "rest", "travel", "weather_venue",
       "slot", "division", "coach", "h2h", "hfa_dyn"]


def X_for(groups, extra=None):
    parts = [BASE] + [GROUPS[g] for g in groups] + ([extra] if extra is not None else [])
    return pd.concat(parts, axis=1).astype(float)
