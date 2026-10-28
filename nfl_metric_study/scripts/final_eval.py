"""Final checks: simple-rule accuracies, nonlinear model, margin model, coefficients, and the untouched 2026 holdout."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from featsets import *

pd.set_option("display.width", 200)
Cres = pd.read_csv(f"{ROOT}/model_results.csv", index_col=0)["C"]
y_win = (G.result > 0).astype(int).values
keep = ((G.result != 0) & G.played).values
season = G.season.values
TEST = list(range(2015, 2026))
test_idx = np.where(keep & np.isin(season, TEST))[0]
yt = y_win[test_idx]

def logit(C):
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(), LogisticRegression(C=C, max_iter=3000))

def rolling_logit(X, C):
    p = np.zeros(len(G))
    for Y in TEST:
        tr = np.where(keep & (season < Y))[0]; te = np.where(keep & (season == Y))[0]
        p[te] = logit(C).fit(X.values[tr], y_win[tr]).predict_proba(X.values[te])[:, 1]
    return p

def score(p, idx=None, y=None):
    idx = test_idx if idx is None else idx
    y = yt if y is None else y
    q = np.clip(p[idx], 1e-6, 1 - 1e-6)
    return dict(n=len(idx), acc=round(((q > .5) == y).mean(), 4), logloss=round(log_loss(y, q), 4),
                brier=round(brier_score_loss(y, q), 4), auc=round(roc_auc_score(y, q), 4))

# ---- simple rules: accuracy of "pick the team with the better X" (ties -> home), 2015-2025
print("=== Simple rules, 2015-2025 (n=%d) ===" % len(yt))
rules = {
    "always home": np.ones(len(G)),
    "better prior-season win%": (G.h_prev1_win - G.a_prev1_win).values,
    "better 3-season win%": (G.h_prev3_win - G.a_prev3_win).values,
    "better season-to-date win% (wk>=5)": np.where(G.week.values >= 5, (G.h_ytd_win - G.a_ytd_win).values, (G.h_prev1_win - G.a_prev1_win).values),
    "better season-to-date point diff (wk>=5)": np.where(G.week.values >= 5, (G.h_ytd_pd - G.a_ytd_pd).values, (G.h_prev1_pd - G.a_prev1_pd).values),
    "better last-3-games point diff": (G.h_last3_pd - G.a_last3_pd).values,
    "better rested team": G.rest_diff.values,
    "better QB rating": (G.h_qb_rating - G.a_qb_rating).values,
}
for name, s in rules.items():
    if name == "always home":
        pick = np.ones(len(G), bool)
    else:
        s = np.nan_to_num(s)
        pick = s > 0
        pick = np.where(s == 0, True, pick)               # ties -> home
    print(f"{name:42s} {(pick[test_idx] == yt).mean():.3f}")

# ---- logistic models on the same rolling-origin split
Xs = {"strength_core": X_for(STRENGTH), "strength+qb": X_for(STRENGTH + ["qb"]),
      "full": X_for(ALL), "strength+qb+context": X_for(STRENGTH + ["qb", "rest", "travel", "weather_venue", "slot", "division"])}
simple = pd.DataFrame({"pts_off": GROUPS["ratings_points"]["L20.pts.off"], "pts_def": GROUPS["ratings_points"]["L20.pts.def"],
                       "qb_rating": GROUPS["qb"]["qb_rating"], "h_qb_changed": GROUPS["qb"]["h_qb_changed"],
                       "a_qb_changed": GROUPS["qb"]["a_qb_changed"]})
Xs["simple5 (pts off/def + QB)"] = pd.concat([BASE, simple], axis=1).astype(float)
P = {}
for k, X in Xs.items():
    C = Cres.get({"simple5 (pts off/def + QB)": "only:ratings_points"}.get(k, k), 0.01)
    P[k] = rolling_logit(X, C)
    print(f"{k:28s}", score(P[k]))

# ---- nonlinear check
def rolling_hgb(X):
    p = np.zeros(len(G))
    for Y in TEST:
        tr = np.where(keep & (season < Y))[0]; te = np.where(keep & (season == Y))[0]
        m = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=250, max_depth=3, min_samples_leaf=60,
                                           l2_regularization=10.0, random_state=0)
        p[te] = m.fit(X.values[tr], y_win[tr]).predict_proba(X.values[te])[:, 1]
    return p
print("\n=== Nonlinear (gradient boosting) vs logistic ===")
for k in ["strength+qb", "full"]:
    ph = rolling_hgb(Xs[k]); P["hgb:" + k] = ph
    print(f"hgb {k:24s}", score(ph))
    print(f"avg(logit,hgb) {k:18s}", score((ph + P[k]) / 2))

# ---- margin (points) model: RMSE and probabilities via normal CDF
print("\n=== Margin regression (points) ===")
y_m = G.result.values.astype(float)
def rolling_ridge(X, alpha):
    pm = np.zeros(len(G)); sig = np.zeros(len(G))
    for Y in TEST:
        tr = np.where(keep & (season < Y))[0]; te = np.where(keep & (season == Y))[0]
        m = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(), Ridge(alpha=alpha)).fit(X.values[tr], y_m[tr])
        pm[te] = m.predict(X.values[te]); sig[te] = np.std(y_m[tr] - m.predict(X.values[tr]))
    return pm, sig
base_pm = np.zeros(len(G))
for Y in TEST:
    tr = np.where(keep & (season < Y))[0]; te = np.where(keep & (season == Y))[0]
    base_pm[te] = y_m[tr].mean()
rm = lambda p: np.sqrt(np.mean((y_m[test_idx] - p[test_idx]) ** 2))
print("RMSE constant (home-field only):", round(rm(base_pm), 3), "| sd of margin:", round(y_m[test_idx].std(), 3))
for k in ["strength_core", "strength+qb", "full"]:
    pm, sg = rolling_ridge(Xs[k], 300.0)
    pp = norm.cdf(pm / sg)
    print(f"ridge {k:22s} RMSE {rm(pm):.3f} | MAE {np.mean(np.abs(y_m[test_idx]-pm[test_idx])):.3f} | win-prob from margin:", score(pp))

# ---- coefficients of strength+qb trained on all 2002-2025 (standardized)
tr = np.where(keep & (season <= 2025))[0]
m = logit(Cres["strength+qb"]).fit(Xs["strength+qb"].values[tr], y_win[tr])
coef = pd.Series(m[-1].coef_[0], index=Xs["strength+qb"].columns).sort_values(key=np.abs, ascending=False)
print("\n=== Standardized logistic coefficients, strength+qb, trained 2002-2025 (top 15; collinear, read in groups) ===")
print(coef.head(15).round(3).to_string())
grp = {}
for g_ in STRENGTH + ["qb"]:
    grp[g_] = coef[[c for c in GROUPS[g_].columns]].abs().sum()
print("sum |coef| by group:", {k: round(v, 3) for k, v in grp.items()})

# ---- untouched 2026 holdout (games already played), trained on 2002-2025
idx26 = np.where(keep & (season == 2026))[0]
y26 = y_win[idx26]
print(f"\n=== 2026 holdout: {len(idx26)} completed games through week {int(G.week.values[idx26].max())}; home win rate {y26.mean():.3f} ===")
trn = np.where(keep & (season <= 2025))[0]
base_rate = y_win[trn].mean()
q = np.full(len(idx26), base_rate)
print("baseline home-rate", {"acc": round(((q > .5) == y26).mean(), 4), "logloss": round(log_loss(y26, q), 4)})
for k in ["simple5 (pts off/def + QB)", "strength_core", "strength+qb", "strength+qb+context", "full"]:
    C = Cres.get({"simple5 (pts off/def + QB)": "only:ratings_points"}.get(k, k), 0.01)
    mm = logit(C).fit(Xs[k].values[trn], y_win[trn])
    p26 = mm.predict_proba(Xs[k].values[idx26])[:, 1]
    q = np.clip(p26, 1e-6, 1 - 1e-6)
    se_acc = np.sqrt(0.64 * 0.36 / len(idx26))
    print(f"{k:28s} acc {((q > .5) == y26).mean():.3f} (+/- {1.96*se_acc:.3f}) | logloss {log_loss(y26, q):.4f} | brier {brier_score_loss(y26, q):.4f}")
pd.DataFrame({k: v[test_idx] for k, v in P.items()}).assign(y=yt, season=season[test_idx]).to_parquet(f"{D}/final_preds.parquet")
