"""Independent verification (separate code path from the feature pipeline):
 1. headline numbers recomputed straight from the schedule
 2. hand-recompute record features for random games
 3. rest days vs calendar dates
 4. ratings face validity (do stat-only ratings line up with actual records?)
 5. QB effect in points
 6. column audit: nothing from betting markets anywhere in the feature table
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nflreadpy as nfl
from common import ROOT, D, CODE

pd.set_option("display.width", 200)
raw = nfl.load_schedules(list(range(1999, 2027))).to_pandas()
raw["gameday"] = pd.to_datetime(raw.gameday)
for c in ("home_team", "away_team"):
    raw[c] = raw[c].replace(CODE)
reg = raw[(raw.game_type == "REG") & raw.result.notna()].copy()

# ------------------------------------------------------------------ 1. independent headline numbers
long = pd.concat([
    pd.DataFrame({"game_id": reg.game_id, "season": reg.season, "week": reg.week, "date": reg.gameday, "team": reg.home_team,
                  "pf": reg.home_score, "pa": reg.away_score}),
    pd.DataFrame({"game_id": reg.game_id, "season": reg.season, "week": reg.week, "date": reg.gameday, "team": reg.away_team,
                  "pf": reg.away_score, "pa": reg.home_score})]).sort_values(["team", "date", "game_id"])
long["w"] = np.where(long.pf > long.pa, 1.0, np.where(long.pf == long.pa, 0.5, 0.0))
long["g_before"] = long.groupby(["team", "season"]).cumcount()
long["w_before"] = long.groupby(["team", "season"]).w.cumsum() - long.w
long["pd_before"] = long.groupby(["team", "season"]).apply(lambda d: (d.pf - d.pa).cumsum() - (d.pf - d.pa), include_groups=False).reset_index(level=[0, 1], drop=True)
seas = long.groupby(["team", "season"]).agg(g=("w", "size"), w=("w", "sum"), pdiff=("pf", "sum"), pa=("pa", "sum")).reset_index()
seas["pd"] = seas.pdiff - seas.pa
seas["season"] += 1
seas = seas.rename(columns={"g": "g_prev", "w": "w_prev", "pd": "pd_prev"})[["team", "season", "g_prev", "w_prev", "pd_prev"]]
long = long.merge(seas, on=["team", "season"], how="left")
long["prev_win"] = (long.w_prev / long.g_prev).fillna(0.5)
long["prev_pd"] = (long.pd_prev / long.g_prev).fillna(0.0)
long["ytd_win"] = long.w_before / long.g_before.replace(0, np.nan)
long["ytd_pd"] = long.pd_before / long.g_before.replace(0, np.nan)

H = long.merge(reg[["game_id", "home_team", "result"]], left_on=["game_id", "team"], right_on=["game_id", "home_team"])
A = long.merge(reg[["game_id", "away_team"]], left_on=["game_id", "team"], right_on=["game_id", "away_team"])
M = H.merge(A, on="game_id", suffixes=("_h", "_a"))
T = M[(M.season_h.between(2015, 2025)) & (M.result != 0)].copy()
T["home_win"] = (T.result > 0).astype(int)
print(f"[1] test games 2015-2025 (non-tie): {len(T)} | home win rate {T.home_win.mean():.4f}")
def rule(d, kind):
    late = d.week_h >= 5
    if kind == "win":
        s = np.where(late, d.ytd_win_h - d.ytd_win_a, d.prev_win_h - d.prev_win_a)
    else:
        s = np.where(late, d.ytd_pd_h - d.ytd_pd_a, d.prev_pd_h - d.prev_pd_a)
    s = np.nan_to_num(s)
    pick = np.where(s == 0, True, s > 0)
    return (pick == d.home_win.values).mean()
print(f"[1] 'better season-to-date win%' accuracy: {rule(T, 'win'):.4f}   (pipeline said 0.631)")
print(f"[1] 'better season-to-date point diff' accuracy: {rule(T, 'pd'):.4f} (pipeline said 0.631)")
pv = T.prev_win_h - T.prev_win_a
print(f"[1] 'better prior-season win%' accuracy: {(np.where(pv == 0, True, pv > 0) == T.home_win.values).mean():.4f} (pipeline said 0.588)")

# ------------------------------------------------------------------ 2. hand-check features
G = pd.read_parquet(f"{D}/model_table.parquet")
rng = np.random.default_rng(11)
samp = G[(G.game_type == "REG") & G.played & (G.season >= 2010) & (G.week >= 4)].sample(4, random_state=5)
print("\n[2] hand-check of record features (pipeline vs recomputed from schedule):")
for r in samp.itertuples():
    lh = long[(long.game_id == r.game_id) & (long.team == r.home_team)].iloc[0]
    g, w = lh.g_before, lh.w_before
    manual_ytd = (w + 1.5) / (g + 3)
    manual_prev = (lh.w_prev + 1.5) / (lh.g_prev + 3) if pd.notna(lh.g_prev) else 0.5
    print(f"   {r.game_id}: home {r.home_team} games_before={int(g)} wins_before={w}  ytd_win pipeline={r.h_ytd_win:.4f} manual={manual_ytd:.4f} | prev1_win pipeline={r.h_prev1_win:.4f} manual={manual_prev:.4f}")
    assert abs(manual_ytd - r.h_ytd_win) < 1e-9 and abs(manual_prev - r.h_prev1_win) < 1e-9

# ------------------------------------------------------------------ 3. rest days vs dates
allg = raw[raw.result.notna() | (raw.gameday < pd.Timestamp("2026-10-19"))].copy()
tl = pd.concat([allg[["game_id", "gameday", "home_team", "home_rest", "week", "season"]].rename(columns={"home_team": "team", "home_rest": "rest"}),
                allg[["game_id", "gameday", "away_team", "away_rest", "week", "season"]].rename(columns={"away_team": "team", "away_rest": "rest"})]).sort_values(["team", "gameday"])
tl["days_since"] = tl.groupby("team").gameday.diff().dt.days
chk = tl[(tl.week >= 2) & (tl.season >= 2005) & tl.days_since.notna()]
print(f"\n[3] rest column vs calendar days since previous game (week>=2): exact match {(chk.rest == chk.days_since).mean():.4f}, within 1 day {((chk.rest - chk.days_since).abs() <= 1).mean():.4f}")
print("    mismatches (>1 day):", int(((chk.rest - chk.days_since).abs() > 1).sum()), "of", len(chk))

# ------------------------------------------------------------------ 4. ratings face validity
rows = []
for yr, wk in [(2025, 18), (2024, 18), (2023, 18)]:
    g = G[(G.season == yr) & (G.week == wk) & (G.game_type == "REG")]
    t = pd.concat([
        pd.DataFrame({"team": g.home_team, "off": g["L20__pts__off_h"], "dfn": g["L20__pts__def_h"], "epa_off": g["L20__epa_pp__off_h"], "epa_def": g["L20__epa_pp__def_h"]}),
        pd.DataFrame({"team": g.away_team, "off": g["L20__pts__off_a"], "dfn": g["L20__pts__def_a"], "epa_off": g["L20__epa_pp__off_a"], "epa_def": g["L20__epa_pp__def_a"]})])
    t["pts_rating"] = t.off + t.dfn
    t["epa_rating"] = t.epa_off + t.epa_def
    rec = long[(long.season == yr) & (long.week < wk)].groupby("team").agg(win=("w", "mean"), g=("w", "size"), pdpg=("pf", "mean"), pa=("pa", "mean"))
    rec["pdpg"] = rec.pdpg - rec.pa
    t = t.merge(rec, on="team")
    print(f"\n[4] {yr} snapshot before week {wk}: corr(points rating, actual win%) = {np.corrcoef(t.pts_rating, t.win)[0,1]:.2f}; corr(EPA rating, win%) = {np.corrcoef(t.epa_rating, t.win)[0,1]:.2f}; corr(points rating, actual pt diff/game) = {np.corrcoef(t.pts_rating, t.pdpg)[0,1]:.2f}  (n={len(t)})")
    if yr == 2025:
        print(t.sort_values("pts_rating", ascending=False).head(4)[["team", "pts_rating", "epa_rating", "win", "pdpg"]].round(2).to_string(index=False))
        print(t.sort_values("pts_rating").head(4)[["team", "pts_rating", "epa_rating", "win", "pdpg"]].round(2).to_string(index=False))

# ------------------------------------------------------------------ 5. QB effect in points
Gp = G[(G.game_type == "REG") & G.played & G.season.between(2002, 2025)].reset_index(drop=True)
def d(cfg, m, part):
    return Gp[f"{cfg}__{m}__{part}_h"] - Gp[f"{cfg}__{m}__{part}_a"]
X = pd.DataFrame({"pts_off": d("L20", "pts", "off"), "pts_def": d("L20", "pts", "def"),
                  "epa_off": d("L20", "epa_pp", "off"), "epa_def": d("L20", "epa_pp", "def"),
                  "neutral": Gp.neutral, "qb_rating": Gp.h_qb_rating - Gp.a_qb_rating,
                  "qb_delta": Gp.h_qb_delta - Gp.a_qb_delta}).fillna(0.0)
X1 = np.column_stack([np.ones(len(X)), X.values]); y = Gp.result.values.astype(float)
XtXi = np.linalg.inv(X1.T @ X1); b = XtXi @ X1.T @ y; e = y - X1 @ b
cov = XtXi @ ((X1 * e[:, None]).T @ (X1 * e[:, None])) @ XtXi * len(y) / (len(y) - X1.shape[1]); se = np.sqrt(np.diag(cov))
print("\n[5] margin ~ team ratings + QB (HC1 SE):")
for n_, bb, ss in zip(["const"] + list(X.columns), b, se):
    print(f"    {n_:10s} {bb:8.2f}  (se {ss:.2f})")
print(f"    => +0.10 EPA/dropback QB-rating edge = {0.1*b[list(X.columns).index('qb_rating')+1]:+.2f} points of margin beyond team ratings")
print(f"    residual sd = {e.std():.2f} points -> 1 point of margin ~ {100*0.3989/e.std():.1f} percentage points of win probability near a toss-up")

# ------------------------------------------------------------------ 6. column audit
bad = [c for c in G.columns if any(k in c.lower() for k in ("spread", "moneyline", "odds", "total_line", "vegas", "line"))]
print("\n[6] feature table columns:", G.shape[1], "| columns mentioning spread/odds/lines:", bad)
print("    'total' column present:", "total" in G.columns)
