"""Does treating the offseason as shorter than real calendar time help? Dev seasons only (2004-2014)."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import itertools
import numpy as np
import pandas as pd
from common import *

sch = load_games()
L = build_long(sch, load_offense())
dev = sch[(sch.game_type == "REG") & sch.played & sch.season.between(2004, 2014)].copy()
keys = sorted(set(zip(dev.season, dev.week)))

rows = []
for hl, credit in itertools.product([8, 12, 16, 20], [0, 10, 20, 26]):
    rr = RidgeRatings(L, sch, ["pts", "epa_pp"], hl, 3, offseason_credit=credit)
    fits = rr.fit_all(keys)
    tix = {t: i for i, t in enumerate(rr.teams)}
    ep, ee, y, wk = [], [], [], []
    for g in dev.itertuples():
        f = fits.get((g.season, g.week))
        if f is None:
            continue
        i, j = tix[g.home_team], tix[g.away_team]
        ep.append((f["off"][i, 0] + f["def"][j, 0]) - (f["off"][j, 0] + f["def"][i, 0]))
        ee.append((f["off"][i, 1] + f["def"][j, 1]) - (f["off"][j, 1] + f["def"][i, 1]))
        y.append(g.result); wk.append(g.week)
    y, ep, ee, wk = map(np.array, (y, ep, ee, wk))
    def r2(x, m=slice(None)):
        b = np.polyfit(x[m], y[m], 1)
        return 1 - (y[m] - np.polyval(b, x[m])).var() / y[m].var()
    early = wk <= 4
    rows.append(dict(hl=hl, credit=credit, r2_pts=round(r2(ep), 4), r2_epa=round(r2(ee), 4),
                     r2_pts_wk1_4=round(r2(ep, early), 4), r2_pts_wk5p=round(r2(ep, ~early), 4)))
    print(rows[-1], flush=True)

res = pd.DataFrame(rows).sort_values("r2_pts", ascending=False)
print(res.head(8).to_string(index=False))
res.to_csv(f"{ROOT}/tune_offseason.csv", index=False)
