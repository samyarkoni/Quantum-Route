"""Tune half-life and ridge strength for opponent-adjusted ratings on DEV seasons only (2004-2014)."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import itertools
import time
import numpy as np
import pandas as pd
from common import *

sch = load_games()
o = load_offense()
L = build_long(sch, o)
print("long rows:", len(L), "| rows with pbp:", L.epa_pp.notna().sum())

dev = sch[(sch.game_type == "REG") & sch.played & sch.season.between(2004, 2014)].copy()
keys = sorted(set(zip(dev.season, dev.week)))
tix_all = None

HLS = [12, 15, 20, 25, 30]
LAMS = [1, 2, 4]
rows = []
t0 = time.time()
for hl, lam in itertools.product(HLS, LAMS):
    rr = RidgeRatings(L, sch, ["pts", "epa_pp"], hl, lam)
    fits = rr.fit_all(keys)
    tix = {t: i for i, t in enumerate(rr.teams)}
    edge_pts, edge_epa, y = [], [], []
    for _, g in dev.iterrows():
        f = fits.get((g.season, g.week))
        if f is None:
            continue
        i, j = tix[g.home_team], tix[g.away_team]
        hn = 0.0 if g.neutral else 1.0
        # expected points: home offense vs away defense minus away offense vs home defense (drop mu, cancels except home)
        ep = (f["off"][i, 0] + f["def"][j, 0]) - (f["off"][j, 0] + f["def"][i, 0])
        ee = (f["off"][i, 1] + f["def"][j, 1]) - (f["off"][j, 1] + f["def"][i, 1])
        edge_pts.append(ep); edge_epa.append(ee); y.append(g.result)
    y = np.array(y); ep = np.array(edge_pts); ee = np.array(edge_epa)
    def r2(x):
        b = np.polyfit(x, y, 1)
        res = y - np.polyval(b, x)
        return 1 - res.var() / y.var(), b[0]
    r2p, bp = r2(ep); r2e, be = r2(ee)
    rows.append(dict(hl=hl, lam=lam, r2_pts=round(r2p, 4), slope_pts=round(bp, 3), r2_epa=round(r2e, 4), slope_epa=round(be, 1)))
    print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame(rows)
print("\nBest by points-based rating:\n", res.sort_values("r2_pts", ascending=False).head(5))
print("\nBest by EPA-based rating:\n", res.sort_values("r2_epa", ascending=False).head(5))
res.to_csv(f"{ROOT}/tune_ratings.csv", index=False)
