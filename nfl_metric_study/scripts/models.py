"""Out-of-sample model comparison: which feature groups improve win prediction?

Protocol
  * target: home win (ties dropped); regular season only
  * rolling-origin: for each test season Y in 2015..2025 train on seasons 2002..Y-1, predict Y
  * C (L2 strength) per feature set tuned ONCE on 2002-2014 with season-blocked CV, then frozen
  * 2026 weeks 1-6 (games already played) scored as an extra untouched holdout (train 2002-2025)
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from common import ROOT, D

pd.set_option("display.width", 220)
G = pd.read_parquet(f"{D}/model_table.parquet")
G = G[(G.game_type == "REG") & (G.season >= 2002)].reset_index(drop=True)

def dd(cfg, m, part):
    return G[f"{cfg}__{m}__{part}_h"] - G[f"{cfg}__{m}__{part}_a"]

def rating_block(cfg, metrics, parts=("off", "def")):
    return {f"{cfg}.{m}.{p}": dd(cfg, m, p) for m in metrics for p in parts}

GROUPS = {
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
BASE = {"neutral": G.neutral}
GROUPS = {k: pd.DataFrame(v) for k, v in GROUPS.items()}
BASE = pd.DataFrame(BASE)

def X_for(groups):
    return pd.concat([BASE] + [GROUPS[g] for g in groups], axis=1).astype(float)

SEQ = ["ratings_points", "ratings_epa", "ratings_other_eff", "qb", "history_long", "form_short", "record_unadj",
       "results_form", "turnovers", "sacks", "situational", "special_penalty", "rest", "travel", "weather_venue",
       "slot", "division", "coach", "h2h", "hfa_dyn"]
ALL = list(GROUPS)
STRENGTH = ["ratings_points", "ratings_epa", "ratings_other_eff"]

CONFIGS = {"baseline_home": []}
for g in ALL:
    CONFIGS[f"only:{g}"] = [g]
CONFIGS["strength_core"] = STRENGTH
CONFIGS["strength+qb"] = STRENGTH + ["qb"]
CONFIGS["strength+qb+context"] = STRENGTH + ["qb", "rest", "travel", "weather_venue", "slot", "division"]
CONFIGS["full"] = ALL
for g in ALL:
    CONFIGS[f"full-{g}"] = [x for x in ALL if x != g]
for i in range(2, len(SEQ) + 1):
    CONFIGS[f"seq{i:02d}+{SEQ[i-1]}"] = SEQ[:i]

y_win = (G.result > 0).astype(int).values
keep = (G.result != 0).values & G.played.values                 # played, no ties
season = G.season.values

def make_model(C):
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))

CS = [0.0005, 0.002, 0.008, 0.03, 0.1, 0.5]
dev_idx = np.where(keep & (season <= 2014))[0]

def tune_C(X):
    best, best_ll = None, 9
    gkf = GroupKFold(n_splits=6)
    Xd, yd, gd = X.values[dev_idx], y_win[dev_idx], season[dev_idx]
    for C in CS:
        ll = []
        for tr, va in gkf.split(Xd, yd, gd):
            m = make_model(C).fit(Xd[tr], yd[tr])
            ll.append(log_loss(yd[va], m.predict_proba(Xd[va])[:, 1], labels=[0, 1]) * len(va))
        tot = sum(ll) / len(dev_idx)
        if tot < best_ll:
            best, best_ll = C, tot
    return best

TEST = list(range(2015, 2026))
test_idx = np.where(keep & np.isin(season, TEST))[0]
preds, Cs = {}, {}
t0 = time.time()
for name, groups in CONFIGS.items():
    X = X_for(groups)
    C = tune_C(X) if groups else 1.0
    Cs[name] = C
    p = np.zeros(len(G))
    for Y in TEST:
        tr = np.where(keep & (season < Y))[0]
        te = np.where(keep & (season == Y))[0]
        m = make_model(C).fit(X.values[tr], y_win[tr])
        p[te] = m.predict_proba(X.values[te])[:, 1]
    preds[name] = p
    print(f"{name:34s} C={C:<7} {time.time()-t0:5.0f}s", flush=True)

yt = y_win[test_idx]
def metrics(p):
    p = np.clip(p[test_idx], 1e-6, 1 - 1e-6)
    return dict(logloss=log_loss(yt, p), brier=brier_score_loss(yt, p), acc=((p > 0.5) == yt).mean(), auc=roc_auc_score(yt, p))

res = pd.DataFrame({k: metrics(v) for k, v in preds.items()}).T
res["C"] = pd.Series(Cs)
base_ll = res.loc["baseline_home", "logloss"]
res["dLL_vs_base"] = res.logloss - base_ll
res.to_csv(f"{ROOT}/model_results.csv")
pd.DataFrame({k: v[test_idx] for k, v in preds.items()}).assign(y=yt, season=season[test_idx], week=G.week.values[test_idx],
                                                                 game_id=G.game_id.values[test_idx]).to_parquet(f"{D}/test_preds.parquet")

# paired bootstrap on log-loss differences
rng = np.random.default_rng(7)
def ll_vec(p):
    p = np.clip(p[test_idx], 1e-6, 1 - 1e-6)
    return -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
LL = {k: ll_vec(v) for k, v in preds.items()}
B = rng.integers(0, len(yt), size=(2000, len(yt)))
def boot_diff(a, b):
    d = LL[a] - LL[b]
    means = d[B].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

print(f"\nTest games 2015-2025: {len(yt)} | home win rate {yt.mean():.3f}")
print("\n=== Each group alone (+ home field) vs baseline; negative dLL = better ===")
only = res.loc[[c for c in res.index if c.startswith("only:")]].copy()
only["ci_lo"], only["ci_hi"] = zip(*[boot_diff(c, "baseline_home")[1:] for c in only.index])
print(only.sort_values("logloss")[["logloss", "acc", "auc", "dLL_vs_base", "ci_lo", "ci_hi"]].round(4).to_string())

print("\n=== Key composite models ===")
key = ["baseline_home", "only:ratings_points", "strength_core", "strength+qb", "strength+qb+context", "full"]
print(res.loc[key][["logloss", "brier", "acc", "auc", "dLL_vs_base"]].round(4).to_string())

print("\n=== Leave-one-group-out from FULL (positive = group helped; CI on full-minus vs full) ===")
rows = []
for g in ALL:
    d, lo, hi = boot_diff(f"full-{g}", "full")
    rows.append(dict(group=g, loss_when_removed=round(d, 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5)))
print(pd.DataFrame(rows).sort_values("loss_when_removed", ascending=False).to_string(index=False))

print("\n=== Forward additions in order (change vs previous step) ===")
rows, prev = [], "baseline_home"
for i in range(1, len(SEQ) + 1):
    nm = f"seq{i:02d}+{SEQ[i-1]}" if i >= 2 else "only:ratings_points"
    d, lo, hi = boot_diff(nm, prev)
    rows.append(dict(step=SEQ[i - 1], logloss=round(res.loc[nm, "logloss"], 4), acc=round(res.loc[nm, "acc"], 4),
                     step_change=round(d, 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5)))
    prev = nm
print(pd.DataFrame(rows).to_string(index=False))

print("\nFull vs strength_core:", [round(x, 5) for x in boot_diff("full", "strength_core")])
print("Full vs strength+qb+context:", [round(x, 5) for x in boot_diff("full", "strength+qb+context")])

# by week bucket and by season for the full model + strength core
wk = G.week.values[test_idx]
print("\n=== Full model by week bucket / season (accuracy | logloss) ===")
for nm in ["baseline_home", "strength_core", "full"]:
    row = {}
    for lab, m in [("wk1-4", wk <= 4), ("wk5-9", (wk >= 5) & (wk <= 9)), ("wk10+", wk >= 10)]:
        pp = np.clip(preds[nm][test_idx][m], 1e-6, 1 - 1e-6)
        row[lab] = f"{((pp > .5) == yt[m]).mean():.3f} | {log_loss(yt[m], pp):.3f}"
    print(nm, row)
sea_acc = pd.DataFrame({nm: [((preds[nm][test_idx][season[test_idx] == Y] > .5) == yt[season[test_idx] == Y]).mean() for Y in TEST]
                        for nm in ["baseline_home", "strength_core", "full"]}, index=TEST).round(3)
print(sea_acc.T.to_string())

# calibration of full model
p = preds["full"][test_idx]
cal = pd.DataFrame({"p": p, "y": yt})
cal["bin"] = pd.cut(cal.p, [0, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
print("\n=== Calibration, full model ===")
print(cal.groupby("bin", observed=True).agg(games=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3).to_string())
