"""Which stats are signal vs noise?  (REG seasons 2002-2025)

A. Per metric: odd/even split-half reliability, first-8 -> last-9 persistence, and how well first-8 predicts
   FUTURE point differential, with incremental value beyond first-8 point differential.
B. Year-over-year: season s metric vs season s+1 point differential / win%.
C. How many games of last season is worth one game of this season (shrinkage constant k).
D. Trailing-window length: which window of recent results predicts the next game best.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import numpy as np
import pandas as pd
from common import *
from features import team_history_features

pd.set_option("display.width", 220)
sch = load_games()
L = build_long(sch, load_offense())

# ---------------------------------------------------------------- A/B
specs = []  # (label, column, direction: +1 if higher is good for the team)
for m, s in OFF_METRICS.items():
    if m == "pts":
        specs.append(("OFF points", "pts", +1))
    else:
        specs.append((f"OFF {m}", m, s))
DEF = {"epa_pp": -1, "pass_epa": -1, "rush_epa": -1, "succ": -1, "pass_succ": -1, "rush_succ": -1, "ed_epa": -1,
       "ld_epa": -1, "expl_pass": -1, "expl_rush": -1, "sack_rate": +1, "int_rate": +1, "fum_rate": +1,
       "ypp": -1, "third_rate": -1, "rz_epa": -1, "epa_gt": -1}
for m, s in DEF.items():
    specs.append((f"DEF {m}", f"opp_{m}", s))
specs.append(("DEF points allowed", "pts_allowed", -1))
for m, s in TEAMLEVEL_METRICS.items():
    specs.append((f"TEAM {m}", m, s))
L["net_epa"] = L.epa_pp - L.opp_epa_pp
specs.append(("NET epa_pp (off-def)", "net_epa", +1))
L["turnover_margin"] = (L.opp_int_rate * L.opp_n_db + L.opp_fum_rate * L.opp_n_plays) - (L.int_rate * L.n_db + L.fum_rate * L.n_plays)
specs.append(("NET turnover margin/game", "turnover_margin", +1))

reg = L[(L.game_type == "REG") & L.played & L.season.between(2002, 2025) & L.epa_pp.notna()].copy()
rows = []
for (team, season), d in reg.groupby(["team", "season"]):
    if len(d) < 14:
        continue
    d = d.sort_values("gameday")
    odd, even, first, last = d.iloc[::2], d.iloc[1::2], d.iloc[:8], d.iloc[8:]
    rec = dict(team=team, season=season, pd_first=first.margin.mean(), pd_last=last.margin.mean(),
               pd_all=d.margin.mean(), win_first=first.win.mean(), win_all=d.win.mean(), win_last=last.win.mean())
    for lab, col, dr in specs:
        rec[f"{lab}|odd"] = dr * odd[col].mean(); rec[f"{lab}|even"] = dr * even[col].mean()
        rec[f"{lab}|first"] = dr * first[col].mean(); rec[f"{lab}|all"] = dr * d[col].mean()
    rows.append(rec)
T = pd.DataFrame(rows)
T["win_first_"] = T.win_first
nxt = T[["team", "season", "pd_all", "win_all"]].copy()
nxt["season"] -= 1
T = T.merge(nxt.rename(columns={"pd_all": "pd_next", "win_all": "win_next"}), on=["team", "season"], how="left")

def corr(a, b):
    m = a.notna() & b.notna()
    return np.corrcoef(a[m], b[m])[0, 1]

def r2_ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ b
    return 1 - res.var() / y.var(), b

base_r2, _ = r2_ols(T.pd_last.values, T[["pd_first"]].values)
out = []
for lab, col, dr in specs:
    x_odd, x_even, x_first, x_all = (T[f"{lab}|{k}"] for k in ("odd", "even", "first", "all"))
    r = corr(x_odd, x_even)
    sb = 2 * r / (1 + r)
    m = x_first.notna()
    z = (x_first[m] - x_first[m].mean()) / x_first[m].std()
    r2_add, b = r2_ols(T.pd_last[m].values, np.column_stack([T.pd_first[m].values, z.values]))
    r2_base_m, _ = r2_ols(T.pd_last[m].values, T.pd_first[m].values.reshape(-1, 1))
    out.append(dict(metric=lab, reliability_SB=round(sb, 2),
                    corr_with_season_pd=round(corr(x_all, T.pd_all), 2),
                    corr_first8_to_future_pd=round(corr(x_first, T.pd_last), 2),
                    corr_to_next_year_pd=round(corr(x_all, T.pd_next), 2),
                    add_R2_beyond_pd=round(r2_add - r2_base_m, 3),
                    beta_pts_per_sd=round(b[2], 2)))
A = pd.DataFrame(out).sort_values("corr_first8_to_future_pd", ascending=False)
print(f"Team-seasons: {len(T)} (SE of a correlation ~ {1/np.sqrt(len(T)):.3f})")
print("Reference: first-8 point diff -> future pd corr =", round(corr(T.pd_first, T.pd_last), 3),
      "| first-8 win% -> future pd =", round(corr(T.win_first, T.pd_last), 3),
      "| odd/even pd reliability(SB) =", round(2 * corr(T.pd_first, T.pd_last) / (1 + corr(T.pd_first, T.pd_last)), 2))
print("Reference: season pd -> next-season pd corr =", round(corr(T.pd_all, T.pd_next), 3),
      "| season win% -> next-season win% =", round(corr(T.win_all, T.win_next), 3))
print("\n=== A/B. Metric signal table (sorted by predictive corr of first 8 games -> future pd) ===")
print(A.to_string(index=False))
A.to_csv(f"{ROOT}/reliability_table.csv", index=False)

# ---------------------------------------------------------------- C: prior-season shrinkage constant k
Lh = team_history_features(L)
Lh["net_epa_f"] = Lh.net_epa.where(Lh.played).fillna(0.0)
Lh["net_epa_n"] = (Lh.played & Lh.net_epa.notna()).astype(float)
g = Lh.groupby(["team", "season"], sort=False)
Lh["net_epa_ytd"] = g["net_epa_f"].cumsum() - Lh["net_epa_f"]
Lh["net_epa_n_ytd"] = g["net_epa_n"].cumsum() - Lh["net_epa_n"]
sea = Lh[(Lh.game_type == "REG") & Lh.played].groupby(["team", "season"]).agg(
    epa_s=("net_epa_f", "sum"), epa_n=("net_epa_n", "sum")).reset_index()
sea["season"] += 1
Lh = Lh.merge(sea.rename(columns={"epa_s": "pe_s", "epa_n": "pe_n"}), on=["team", "season"], how="left")
Lh["prev_pd"] = np.where(Lh.g1 > 0, Lh.pd1 / Lh.g1.replace(0, np.nan), 0.0)
Lh["prev_epa"] = (Lh.pe_s / Lh.pe_n).fillna(0.0)

def team_view(df, cols):
    return df[["game_id", "team"] + cols]

Hh = Lh[Lh.is_home == 1]
Aw = team_view(Lh[Lh.is_home == 0], ["margin_f_ytd", "pl_ytd", "prev_pd", "net_epa_ytd", "net_epa_n_ytd", "prev_epa"])
D = Hh.merge(Aw.add_prefix("a_").rename(columns={"a_game_id": "game_id"}), on="game_id")
D = D[(D.game_type == "REG") & D.played & D.season.between(2003, 2025)].copy()
D["result"] = D.margin
print(f"\n=== C. Weight on last season: predictor = (g*ytd + k*prev)/(g + k), {len(D)} games ===")
rowsC = []
for k in [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 1e6]:
    for name, ytd_s, n_ytd, prev, a_ytd_s, a_n, a_prev in [
        ("point diff", "margin_f_ytd", "pl_ytd", "prev_pd", "a_margin_f_ytd", "a_pl_ytd", "a_prev_pd"),
        ("net EPA/play", "net_epa_ytd", "net_epa_n_ytd", "prev_epa", "a_net_epa_ytd", "a_net_epa_n_ytd", "a_prev_epa")]:
        ph = (D[ytd_s] + k * D[prev]) / (D[n_ytd] + k) if k < 1e5 else D[prev]
        pa = (D[a_ytd_s] + k * D[a_prev]) / (D[a_n] + k) if k < 1e5 else D[a_prev]
        x = (ph - pa).values
        ok = np.isfinite(x)
        for bucket, mask in [("all", np.ones(len(D), bool)), ("wk1-4", D.week.values <= 4),
                             ("wk5-9", (D.week.values >= 5) & (D.week.values <= 9)), ("wk10+", D.week.values >= 10)]:
            m = ok & mask
            r2, _ = r2_ols(D.result.values[m], x[m].reshape(-1, 1))
            rowsC.append(dict(k=k, metric=name, bucket=bucket, r2=r2))
C = pd.DataFrame(rowsC)
pivot = C.pivot_table(index=["metric", "k"], columns="bucket", values="r2").round(4)
print(pivot.to_string())
best = C.loc[C.groupby(["metric", "bucket"]).r2.idxmax()][["metric", "bucket", "k", "r2"]]
print("\nBest k by bucket:\n", best.to_string(index=False))

# ---------------------------------------------------------------- D: trailing window length
print("\n=== D. Trailing-window length (within-season), games where BOTH teams have >= 8 prior games ===")
Lh = Lh.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
g = Lh.groupby(["team", "season"], sort=False)
for N in (1, 2, 3, 4, 6, 8):
    Lh[f"pd_w{N}"] = g["margin"].shift(1).groupby([Lh.team, Lh.season]).transform(lambda s: s.rolling(N, min_periods=N).mean())
    Lh[f"epa_w{N}"] = g["net_epa"].shift(1).groupby([Lh.team, Lh.season]).transform(lambda s: s.rolling(N, min_periods=N).mean())
Lh["pd_ytd"] = Lh.margin_f_ytd / Lh.pl_ytd.replace(0, np.nan)
Lh["epa_ytd"] = Lh.net_epa_ytd / Lh.net_epa_n_ytd.replace(0, np.nan)
cols = [f"pd_w{N}" for N in (1, 2, 3, 4, 6, 8)] + ["pd_ytd"] + [f"epa_w{N}" for N in (1, 2, 3, 4, 6, 8)] + ["epa_ytd", "pl_ytd"]
Hh = Lh[Lh.is_home == 1]; Aw = Lh[Lh.is_home == 0][["game_id"] + cols].add_prefix("a_").rename(columns={"a_game_id": "game_id"})
D2 = Hh.merge(Aw, on="game_id")
D2 = D2[(D2.game_type == "REG") & D2.played & D2.season.between(2002, 2025) & (D2.pl_ytd >= 8) & (D2.a_pl_ytd >= 8)]
rowsD = []
for c in cols[:-1]:
    x = (D2[c] - D2["a_" + c]).values
    ok = np.isfinite(x)
    r2, b = r2_ols(D2.margin.values[ok], x[ok].reshape(-1, 1))
    rowsD.append(dict(window=c, games=int(ok.sum()), R2_margin=round(r2, 4), pts_per_unit=round(b[1], 2)))
print(pd.DataFrame(rowsD).to_string(index=False))
pd.DataFrame(rowsD).to_csv(f"{ROOT}/windows_table.csv", index=False)
C.to_csv(f"{ROOT}/prior_weight_table.csv", index=False)
