"""Effect sizes in POINTS (home margin) for context factors, controlling for team strength; plus home-field
advantage by era and offense-vs-defense persistence.  OLS with HC1 robust standard errors."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import *

pd.set_option("display.width", 200)
G = pd.read_parquet(f"{D}/model_table.parquet")
G = G[(G.game_type == "REG") & G.played & (G.season.between(2002, 2025))].reset_index(drop=True)

def ols(y, X, names):
    X1 = np.column_stack([np.ones(len(y)), X]); nm = ["const"] + names
    XtXi = np.linalg.inv(X1.T @ X1)
    b = XtXi @ X1.T @ y
    e = y - X1 @ b
    meat = (X1 * e[:, None]).T @ (X1 * e[:, None])
    cov = XtXi @ meat @ XtXi * len(y) / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=nm), 1 - e.var() / y.var()

def dd(cfg, m, part):
    return G[f"{cfg}__{m}__{part}_h"] - G[f"{cfg}__{m}__{part}_a"]

# strength controls: points ratings + EPA ratings (off/def) + QB rating, home-minus-away
ctrl = pd.DataFrame({
    "pts_off": dd("L20", "pts", "off"), "pts_def": dd("L20", "pts", "def"),
    "epa_off": dd("L20", "epa_pp", "off"), "epa_def": dd("L20", "epa_pp", "def"),
    "qb": G.h_qb_rating - G.a_qb_rating,
}).fillna(0.0)
y = G.result.values.astype(float)

# ---------- home-field advantage by era
era = pd.cut(G.season, [2001, 2007, 2013, 2019, 2025], labels=["2002-07", "2008-13", "2014-19", "2020-25"])
E = pd.get_dummies(era).astype(float)
Xe = pd.concat([E, ctrl, G[["neutral"]].astype(float)], axis=1)
res, r2 = ols(y, Xe.drop(columns=[E.columns[0]]).values, list(Xe.drop(columns=[E.columns[0]]).columns))
# re-fit with all era dummies and no constant for readable intercepts
X1 = Xe.values; b = np.linalg.lstsq(X1, y, rcond=None)[0]; e = y - X1 @ b
cov = np.linalg.inv(X1.T @ X1) @ ((X1 * e[:, None]).T @ (X1 * e[:, None])) @ np.linalg.inv(X1.T @ X1) * len(y) / (len(y) - X1.shape[1])
print("=== Home-field advantage by era (points; strength-adjusted, non-neutral) ===")
for i, c in enumerate(E.columns):
    print(f"{c}: {b[i]:+.2f} pts  (SE {np.sqrt(cov[i, i]):.2f})")
raw = G[G.neutral == 0].groupby(era[G.neutral == 0], observed=True).agg(games=("result", "size"), home_win=("result", lambda s: (s > 0).mean() + 0.5 * (s == 0).mean()), raw_margin=("result", "mean"))
print(raw.round(3).to_string())

# ---------- context effects (points of home margin, controlling for strength)
def run(spec_name, cols):
    X = pd.concat([ctrl, G[["neutral"]].astype(float), cols], axis=1)
    out, r2 = ols(y, X.values, list(X.columns))
    print(f"\n=== {spec_name}  (n={len(y)}, R2={r2:.3f}) — coefficients are POINTS added to the HOME team's margin ===")
    print(out.loc[cols.columns].round(2).to_string())

run("Rest as difference", pd.DataFrame({"rest_diff_per_day": G.rest_diff}))
run("Rest as flags", pd.DataFrame({c: G[c] for c in ["home_bye", "away_bye", "home_short", "away_short", "home_long", "away_long"]}))
run("Travel", pd.DataFrame({"away_miles_per_1000": G.away_miles, "away_tz_east_hrs": G.away_tz_east,
                            "away_tz_west_hrs": G.away_tz_west, "away_body_clock_early": G.away_body_early}))
run("Weather / venue", pd.DataFrame({"dome": G.dome, "cold_per_10F_below_40": G.temp_cold / 10, "hot_per_10F_above_80": G.temp_hot / 10,
                                     "wind_per_10mph": G.wind / 10, "turf": G.turf}))
run("Slot / schedule", pd.DataFrame({c: G[c] for c in ["thu", "mon", "sat", "prime", "late_window", "div_game"]}))
run("QB / coach situation", pd.DataFrame({"home_qb_changed": G.h_qb_changed, "away_qb_changed": G.a_qb_changed,
                                          "home_qb_rookie": G.h_qb_rookie, "away_qb_rookie": G.a_qb_rookie,
                                          "home_new_coach": (G.h_coach_tenure < 17).astype(float),
                                          "away_new_coach": (G.a_coach_tenure < 17).astype(float)}))

# interaction worth checking: Thursday short-rest on BOTH sides vs one side
Gc = G.copy()
Gc["both_short"] = ((Gc.home_rest <= 5) & (Gc.away_rest <= 5)).astype(float)
Gc["away_short_only"] = ((Gc.home_rest > 5) & (Gc.away_rest <= 5)).astype(float)
Gc["home_short_only"] = ((Gc.home_rest <= 5) & (Gc.away_rest > 5)).astype(float)
print("\ngames: both short", int(Gc.both_short.sum()), "| away short only", int(Gc.away_short_only.sum()), "| home short only", int(Gc.home_short_only.sum()))
run("Short rest by side", Gc[["home_short_only", "away_short_only", "both_short"]])

# ---------- offense vs defense persistence (team-season level)
L = build_long(load_games(), load_offense())
reg = L[(L.game_type == "REG") & L.played & L.season.between(2002, 2025) & L.epa_pp.notna()]
rows = []
for (team, season), d in reg.groupby(["team", "season"]):
    if len(d) < 14:
        continue
    d = d.sort_values("gameday"); f, l = d.iloc[:8], d.iloc[8:]
    rows.append(dict(team=team, season=season, pf=f.pts.mean(), pa=f.pts_allowed.mean(),
                     epa_o=f.epa_pp.mean(), epa_d=f.opp_epa_pp.mean(), pd_last=l.margin.mean(),
                     pf_all=d.pts.mean(), pa_all=d.pts_allowed.mean()))
T = pd.DataFrame(rows)
nx = T[["team", "season", "pf_all", "pa_all"]].copy(); nx["season"] -= 1; nx["pd_next"] = nx.pf_all - nx.pa_all
T = T.merge(nx[["team", "season", "pd_next"]], on=["team", "season"], how="left")
print("\n=== Offense vs defense persistence ===")
out, r2 = ols(T.pd_last.values, np.column_stack([T.pf.values, -T.pa.values]), ["points_scored_first8", "neg_points_allowed_first8"])
print(f"Future (games 9+) point diff on first-8 points scored vs points allowed (n={len(T)}):\n", out.round(3).to_string())
out, r2 = ols(T.pd_last.values, np.column_stack([T.epa_o.values, -T.epa_d.values]), ["off_epa_per_play_first8", "neg_def_epa_allowed_first8"])
print("Same with EPA/play (per +0.1 EPA/play):\n", (out * [1, 0.1, 0.1] if False else out).round(2).to_string())
m = T.pd_next.notna()
out, r2 = ols(T.pd_next[m].values, np.column_stack([T.pf_all[m].values, -T.pa_all[m].values]), ["points_scored_season", "neg_points_allowed_season"])
print("Next-season point diff on this season's points scored vs allowed:\n", out.round(3).to_string())
