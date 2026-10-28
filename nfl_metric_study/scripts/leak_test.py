"""Truncation test: features for a week's games must be IDENTICAL when everything from that week onward is
deleted (scores removed, games marked unplayed, pbp and QB lines dropped). Any difference = look-ahead leak."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import numpy as np
import pandas as pd
from common import *
from features import build_model_table

sch = load_games()
o = load_offense()
qb = pd.read_parquet(f"{D}/qb_game.parquet")
full = build_model_table(sch, o, qb)
full.to_parquet(f"{D}/model_table.parquet")
print("full table:", full.shape)

feat_cols = [c for c in full.columns if c not in
             ("game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "home_score",
              "away_score", "result", "overtime", "played")]

bad_total = 0
for cutoff in [(2005, 7), (2012, 3), (2019, 12), (2023, 1), (2025, 9), (2026, 6)]:
    start = sch[(sch.season == cutoff[0]) & (sch.week == cutoff[1])].gameday.min()
    mask = sch.gameday >= start
    gids = set(sch.game_id[mask])
    s2 = sch.copy()
    for c in ["home_score", "away_score", "result"]:
        s2.loc[mask, c] = np.nan
    s2.loc[mask, "played"] = False
    o2 = o[~o.game_id.isin(gids)]
    qb2 = qb[~qb.game_id.isin(gids)]
    trunc = build_model_table(s2, o2, qb2)

    wk_ids = sch.game_id[(sch.season == cutoff[0]) & (sch.week == cutoff[1])]
    a = full[full.game_id.isin(wk_ids)].set_index("game_id")[feat_cols].sort_index()
    b = trunc[trunc.game_id.isin(wk_ids)].set_index("game_id")[feat_cols].sort_index()
    diff = ~np.isclose(a.values.astype(float), b.values.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
    n_bad = int(diff.sum())
    bad_total += n_bad
    print(f"cutoff {cutoff}: {len(a)} games x {len(feat_cols)} features -> {n_bad} mismatching cells", flush=True)
    if n_bad:
        cols_bad = list(a.columns[diff.any(axis=0)])
        mx = np.nanmax(np.abs(a.values.astype(float) - b.values.astype(float)))
        print("   columns:", sorted({c.split('__')[1] if '__' in c else c for c in cols_bad}), "| max abs diff:", mx)

print("TOTAL MISMATCHES:", bad_total)
