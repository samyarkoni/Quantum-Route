"""Do injuries (weighted by how much each missing player normally plays) add anything beyond strength + QB?

Pre-game info only: final weekly injury report status (Out/Doubtful/Questionable) x the player's average snap share
over his previous (up to) 4 games. QBs excluded here (the QB block already covers them)."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import warnings
warnings.filterwarnings("ignore")
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from featsets import *

pd.set_option("display.width", 200)
snaps = nfl.load_snap_counts(list(range(2012, 2027))).to_pandas()
inj = nfl.load_injuries(list(range(2012, 2027))).to_pandas()
pl_ = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")

snaps = snaps[snaps.pfr_player_id.notna()].copy()
snaps["share"] = np.maximum(snaps.offense_pct.fillna(0), snaps.defense_pct.fillna(0))
snaps["t"] = (snaps.season * 100 + snaps.week).astype("int64")
snaps = snaps.sort_values(["pfr_player_id", "t"])
snaps["roll4"] = snaps.groupby("pfr_player_id").share.transform(lambda s: s.rolling(4, min_periods=1).mean())
snaps = snaps.sort_values("t")[["pfr_player_id", "t", "roll4"]]

inj = inj.sort_values("date_modified").drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
inj = inj.merge(pl_, on="gsis_id", how="left")
inj = inj[inj.report_status.isin(["Out", "Doubtful", "Questionable"]) & inj.pfr_id.notna() & (inj.position != "QB")].copy()
inj["t"] = (inj.season * 100 + inj.week).astype("int64")
inj = inj.sort_values("t")
m = pd.merge_asof(inj, snaps.rename(columns={"pfr_player_id": "pfr_id", "t": "t_prev"}), left_on="t", right_on="t_prev",
                  by="pfr_id", direction="backward", allow_exact_matches=False, tolerance=120)
m["prior_share"] = m.roll4.fillna(0.0)
print("injury rows used:", len(m), "| with a prior-snap match:", int(m.roll4.notna().sum()))

UNIT = {"T": "ol", "G": "ol", "C": "ol", "OL": "ol", "WR": "skill", "TE": "skill", "RB": "skill", "FB": "skill",
        "DE": "dl", "DT": "dl", "NT": "dl", "DL": "dl", "LB": "lb", "ILB": "lb", "OLB": "lb", "MLB": "lb",
        "CB": "db", "S": "db", "SS": "db", "FS": "db", "DB": "db"}
m["unit"] = m.position.map(UNIT)
m = m[m.unit.notna()]
out = m[m.report_status.isin(["Out", "Doubtful"])].pivot_table(index=["season", "week", "team"], columns="unit", values="prior_share", aggfunc="sum").add_prefix("out_")
q = m[m.report_status == "Questionable"].assign(side=lambda d: np.where(d.unit.isin(["ol", "skill"]), "off", "def")) \
    .pivot_table(index=["season", "week", "team"], columns="side", values="prior_share", aggfunc="sum").add_prefix("q_")
F = out.join(q, how="outer").fillna(0.0).reset_index()
for c in ["out_ol", "out_skill", "out_dl", "out_lb", "out_db", "q_off", "q_def"]:
    if c not in F:
        F[c] = 0.0
print(F.describe().T[["mean", "std", "max"]].round(2).to_string())

g = G[["season", "week", "home_team", "away_team"]].copy()
g = g.merge(F.add_prefix("h_").rename(columns={"h_season": "season", "h_week": "week", "h_team": "home_team"}),
            on=["season", "week", "home_team"], how="left")
g = g.merge(F.add_prefix("a_").rename(columns={"a_season": "season", "a_week": "week", "a_team": "away_team"}),
            on=["season", "week", "away_team"], how="left")
cols = ["out_ol", "out_skill", "out_dl", "out_lb", "out_db", "q_off", "q_def"]
INJ = pd.DataFrame({f"inj_{c}": g[f"h_{c}"].fillna(0) - g[f"a_{c}"].fillna(0) for c in cols}, index=G.index)
INJ["inj_out_total"] = INJ[[f"inj_{c}" for c in cols[:5]]].sum(axis=1)
INJ.to_parquet(f"{D}/inj_features.parquet")

# ---------- (A) effect sizes in points, controlling for strength + QB
def ols(y, X, names):
    X1 = np.column_stack([np.ones(len(y)), X]); XtXi = np.linalg.inv(X1.T @ X1)
    b = XtXi @ X1.T @ y; e = y - X1 @ b
    cov = XtXi @ ((X1 * e[:, None]).T @ (X1 * e[:, None])) @ XtXi * len(y) / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=["const"] + names)

reg = (G.played & (G.season.between(2013, 2025))).values
ctrl = pd.concat([GROUPS["ratings_points"], GROUPS["ratings_epa"][["L20.epa_pp.off", "L20.epa_pp.def"]], GROUPS["qb"][["qb_rating"]], BASE], axis=1).fillna(0.0)
Xa = pd.concat([ctrl, INJ[["inj_out_ol", "inj_out_skill", "inj_out_dl", "inj_out_lb", "inj_out_db", "inj_q_off", "inj_q_def"]]], axis=1)
res = ols(G.result.values[reg].astype(float), Xa.values[reg], list(Xa.columns))
print(f"\n=== Effect on HOME margin (points) per 1.0 starter-equivalent missing (home minus away), n={reg.sum()} ===")
print(res.loc[[c for c in res.index if c.startswith("inj_")]].round(2).to_string())
mean_burden = INJ[reg][["inj_out_total"]].describe().T
print("typical |home-away| Out burden (starter-equivalents):", round(INJ[reg].inj_out_total.abs().mean(), 2))

# ---------- (B) predictive value beyond strength+qb, equal training windows (2013..Y-1), test 2018-2025
Cres = pd.read_csv(f"{ROOT}/model_results.csv", index_col=0)["C"]
y_win = (G.result > 0).astype(int).values
keep = ((G.result != 0) & G.played).values
season = G.season.values
TEST = list(range(2018, 2026))
def run(X, C):
    p = np.zeros(len(G))
    for Y in TEST:
        tr = np.where(keep & (season < Y) & (season >= 2013))[0]; te = np.where(keep & (season == Y))[0]
        mdl = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        p[te] = mdl.fit(X.values[tr], y_win[tr]).predict_proba(X.values[te])[:, 1]
    return p
tidx = np.where(keep & np.isin(season, TEST))[0]; yt = y_win[tidx]
base = X_for(STRENGTH + ["qb"])
C = Cres["strength+qb"]
variants = {"strength+qb": base,
            "+ injuries (Out/Doubt by unit)": pd.concat([base, INJ[[f"inj_{c}" for c in cols[:5]]]], axis=1),
            "+ injuries (total Out only)": pd.concat([base, INJ[["inj_out_total"]]], axis=1),
            "+ injuries (all 7)": pd.concat([base, INJ[[f"inj_{c}" for c in cols]]], axis=1)}
P = {k: np.clip(run(X, C)[tidx], 1e-6, 1 - 1e-6) for k, X in variants.items()}
rng = np.random.default_rng(3); B = rng.integers(0, len(yt), size=(2000, len(yt)))
ll = {k: -(yt * np.log(p) + (1 - yt) * np.log(1 - p)) for k, p in P.items()}
print(f"\n=== Out-of-sample 2018-2025 (n={len(yt)}), training 2013..Y-1 ===")
for k, p in P.items():
    d = ll[k] - ll["strength+qb"]; mb = d[B].mean(axis=1)
    print(f"{k:34s} logloss {log_loss(yt, p):.4f} acc {((p > .5) == yt).mean():.4f} auc {roc_auc_score(yt, p):.4f} | d vs base {d.mean():+.5f} [{np.percentile(mb, 2.5):+.5f}, {np.percentile(mb, 97.5):+.5f}]")
