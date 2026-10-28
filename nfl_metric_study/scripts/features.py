"""Leakage-free pre-game feature builder.

build_model_table(sch, off_metrics, qb_lines) -> one row per game with home (h_) / away (a_) team features
that use ONLY information available before that game's week starts (plus pre-game context: rest, venue,
weather forecast fields, announced starting QB).

Nothing from betting markets is used.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/6ded8384-9460-54ac-9d03-cde9877e6939/scratchpad/scripts")
import numpy as np
import pandas as pd
from common import *

RATING_CFGS = {          # name: (half-life in calendar weeks, ridge lambda)
    "L20": (20, 3.0),    # tuned on 2004-2014
    "S6": (6, 3.0),      # short-term form
    "H80": (80, 3.0),    # long-run / historical
}

# Approximate home-city coordinates (lat, lon) and time-zone offset from US Eastern, by ORIGINAL schedule code
GEO = {
    "ARI": (33.45, -112.07, -2), "ATL": (33.75, -84.39, 0), "BAL": (39.29, -76.61, 0), "BUF": (42.89, -78.88, 0),
    "CAR": (35.23, -80.84, 0), "CHI": (41.88, -87.63, -1), "CIN": (39.10, -84.51, 0), "CLE": (41.50, -81.69, 0),
    "DAL": (32.78, -96.80, -1), "DEN": (39.74, -104.99, -2), "DET": (42.33, -83.05, 0), "GB": (44.51, -88.02, -1),
    "HOU": (29.76, -95.37, -1), "IND": (39.77, -86.16, 0), "JAX": (30.33, -81.66, 0), "KC": (39.10, -94.58, -1),
    "LV": (36.17, -115.14, -3), "OAK": (37.80, -122.27, -3), "LAC": (34.05, -118.24, -3), "SD": (32.72, -117.16, -3),
    "LA": (34.05, -118.24, -3), "LAR": (34.05, -118.24, -3), "STL": (38.63, -90.20, -1), "MIA": (25.76, -80.19, 0),
    "MIN": (44.98, -93.27, -1), "NE": (42.09, -71.26, 0), "NO": (29.95, -90.07, -1), "NYG": (40.81, -74.07, 0),
    "NYJ": (40.81, -74.07, 0), "PHI": (39.95, -75.17, 0), "PIT": (40.44, -80.00, 0), "SEA": (47.61, -122.33, -3),
    "SF": (37.77, -122.42, -3), "TB": (27.95, -82.46, 0), "TEN": (36.16, -86.78, -1), "WAS": (38.91, -77.04, 0),
}


def haversine_miles(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------------------------------------
def team_history_features(L):
    """Record-based features, computed per team strictly from earlier games."""
    L = L.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    L["pl"] = L.played.astype(float)
    for c in ["win", "margin", "pts", "pts_allowed"]:
        L[c + "_f"] = L[c].where(L.played, 0.0).fillna(0.0)
    g = L.groupby(["team", "season"], sort=False)
    for c in ["pl", "win_f", "margin_f", "pts_f", "pts_allowed_f"]:
        L[c + "_ytd"] = g[c].cumsum() - L[c]                    # exclusive cumulative sum within season
    k_w, k_pd = 3.0, 4.0
    L["ytd_g"] = L.pl_ytd
    L["ytd_win"] = (L.win_f_ytd + 0.5 * k_w) / (L.pl_ytd + k_w)
    L["ytd_pd"] = L.margin_f_ytd / (L.pl_ytd + k_pd)
    ps, pa = (L.pts_f_ytd + 22 * 3) / (L.pl_ytd + 3), (L.pts_allowed_f_ytd + 22 * 3) / (L.pl_ytd + 3)
    L["ytd_pyth"] = ps ** 2.37 / (ps ** 2.37 + pa ** 2.37)

    # prior seasons (regular season only)
    reg = L[(L.game_type == "REG") & L.played]
    S = reg.groupby(["team", "season"]).agg(g=("pl", "sum"), w=("win_f", "sum"), pdiff=("margin_f", "sum")).reset_index()
    for lag in (1, 2, 3):
        Sl = S.copy()
        Sl["season"] = Sl.season + lag
        Sl = Sl.rename(columns={"g": f"g{lag}", "w": f"w{lag}", "pdiff": f"pd{lag}"})
        L = L.merge(Sl, on=["team", "season"], how="left")
    for c in ["g1", "w1", "pd1", "g2", "w2", "pd2", "g3", "w3", "pd3"]:
        L[c] = L[c].fillna(0.0)
    L["prev1_win"] = (L.w1 + 1.5) / (L.g1 + 3)
    L["prev1_pd"] = L.pd1 / (L.g1 + 3)
    gg, ww, pp = L.g1 + L.g2 + L.g3, L.w1 + L.w2 + L.w3, L.pd1 + L.pd2 + L.pd3
    L["prev3_win"] = (ww + 3.0) / (gg + 6)
    L["prev3_pd"] = pp / (gg + 6)

    # short-run results form (unadjusted), within season
    g = L.groupby(["team", "season"], sort=False)
    sh = g["margin"].shift(1)
    L["last3_pd"] = sh.groupby([L.team, L.season]).transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0.0)
    L["last3_win"] = (g["win"].shift(1).groupby([L.team, L.season])
                      .transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0.5))
    L["last1_margin"] = sh.fillna(0.0)
    L["last1_margin"] = L["last1_margin"].clip(-30, 30)

    # coach tenure (games with this team before this one)
    L["coach_tenure"] = L.groupby(["team", "coach"], sort=False).cumcount().astype(float)
    return L


# ----------------------------------------------------------------------------------------------------------
def qb_features(L, qb_lines, sch, hl_weeks=100.0, k=200.0):
    ql = qb_lines.merge(sch[["game_id", "gameday", "played"]], on="game_id")
    ql = ql[ql.played].sort_values("gameday")
    ql["season"] = ql.gameday.dt.year - (ql.gameday.dt.month <= 2).astype(int)
    lg = ql.groupby("season").apply(lambda d: d.qb_epa_sum.sum() / d.qb_db.sum(), include_groups=False)
    by_qb = {q: (d.gameday.values.astype("datetime64[D]"), d.qb_db.values.astype(float), d.qb_epa_sum.values)
             for q, d in ql.groupby("qb_id")}
    ln2 = np.log(2)

    def rate(qid, date, season):
        mu = lg.get(season - 1, 0.0)
        if qid not in by_qb:
            return mu, 0.0
        dts, db, ep = by_qb[qid]
        n = np.searchsorted(dts, date, side="left")           # strictly earlier games only
        if n == 0:
            return mu, 0.0
        w = np.exp(-ln2 * ((date - dts[:n]).astype(float) / 7.0) / hl_weeks)
        return (np.sum(w * ep[:n]) + k * mu) / (np.sum(w * db[:n]) + k), float(db[:n].sum())

    L = L.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    prev_q = L.groupby("team", sort=False)["qb_id"].shift(1)
    r_now, exp_now, r_prev = [], [], []
    for qid, pq, date, season in zip(L.qb_id.values, prev_q.values, L.gameday.values.astype("datetime64[D]"), L.season.values):
        if pd.isna(qid):
            r_now.append(np.nan); exp_now.append(np.nan); r_prev.append(np.nan); continue
        a, e = rate(qid, date, season)
        r_now.append(a); exp_now.append(e)
        r_prev.append(rate(pq, date, season)[0] if (not pd.isna(pq) and pq != qid) else a)
    L["qb_rating"] = r_now
    L["qb_exp"] = np.log1p(exp_now)
    L["qb_changed"] = ((prev_q != L.qb_id) & prev_q.notna() & L.qb_id.notna()).astype(float)
    L["qb_delta"] = L.qb_rating - np.array(r_prev)             # starter vs previous game's starter, as rated today
    L["qb_rookie"] = (np.expm1(L.qb_exp) < 150).astype(float)
    return L


# ----------------------------------------------------------------------------------------------------------
def h2h_features(sch):
    """Home team's results vs this opponent in meetings over the previous 6 seasons (either venue)."""
    hist = {}
    out = []
    for g in sch.itertuples():
        key = tuple(sorted((g.home_team, g.away_team)))
        lst = hist.get(key, [])
        cut = g.gameday - pd.Timedelta(days=6 * 365)
        w = m = n = 0.0
        for d, h, res in lst:
            if d >= cut and d < g.gameday:
                sgn = 1.0 if h == g.home_team else -1.0
                r = res * sgn
                w += 1.0 if r > 0 else (0.5 if r == 0 else 0.0)
                m += float(np.clip(r, -30, 30)); n += 1
        out.append((n, (w + 1.0) / (n + 2.0) - 0.5, m / (n + 3.0)))
        if g.played:
            lst.append((g.gameday, g.home_team, g.result))
            hist[key] = lst
    return pd.DataFrame(out, columns=["h2h_n", "h2h_win", "h2h_margin"], index=sch.index)


# ----------------------------------------------------------------------------------------------------------
def context_features(sch):
    X = pd.DataFrame(index=sch.index)
    X["neutral"] = sch.neutral
    X["div_game"] = sch.div_game.fillna(0).astype(float)
    X["home_rest"] = sch.home_rest.astype(float)
    X["away_rest"] = sch.away_rest.astype(float)
    X["rest_diff"] = (X.home_rest - X.away_rest).clip(-7, 7)
    X["home_bye"] = (X.home_rest >= 11).astype(float)
    X["away_bye"] = (X.away_rest >= 11).astype(float)
    X["home_short"] = (X.home_rest <= 5).astype(float)
    X["away_short"] = (X.away_rest <= 5).astype(float)
    X["home_long"] = ((X.home_rest >= 8) & (X.home_rest < 11)).astype(float)
    X["away_long"] = ((X.away_rest >= 8) & (X.away_rest < 11)).astype(float)
    wd = sch.weekday.fillna("Sunday")
    X["thu"] = (wd == "Thursday").astype(float)
    X["mon"] = (wd == "Monday").astype(float)
    X["sat"] = (wd == "Saturday").astype(float)
    hr = pd.to_datetime(sch.gametime, format="%H:%M", errors="coerce").dt.hour.fillna(13)
    X["prime"] = (hr >= 19).astype(float)
    X["late_window"] = ((hr >= 16) & (hr < 19)).astype(float)

    # weather / venue
    roof = sch.roof.fillna("outdoors")
    X["dome"] = roof.isin(["dome", "closed"]).astype(float)
    t = sch.temp.astype(float)
    month = sch.gameday.dt.month
    # climatology fill for missing outdoor temperatures: mean of EARLIER games for that team-month (no look-ahead)
    cl = pd.DataFrame({"t": t, "team": sch.home_team, "m": month, "d": sch.gameday}).sort_values("d", kind="stable")
    cl["clim"] = cl.groupby(["team", "m"])["t"].transform(lambda s: s.expanding().mean().shift(1))
    fill = cl["clim"].reindex(sch.index).fillna(60.0)
    temp = t.fillna(fill)
    temp = temp.where(X.dome == 0, 70.0)
    X["temp_cold"] = (40 - temp).clip(lower=0)
    X["temp_hot"] = (temp - 80).clip(lower=0)
    wind = sch.wind.astype(float).where(X.dome == 0, 0.0).fillna(7.0)
    X["wind"] = wind.clip(0, 30)
    surf = sch.surface.fillna("grass").str.lower()
    X["turf"] = (~surf.str.contains("grass")).astype(float)

    # travel for the away team
    lat_h = sch.home_team_orig.map(lambda c: GEO.get(c, (np.nan,) * 3)[0]); lon_h = sch.home_team_orig.map(lambda c: GEO.get(c, (np.nan,) * 3)[1])
    lat_a = sch.away_team_orig.map(lambda c: GEO.get(c, (np.nan,) * 3)[0]); lon_a = sch.away_team_orig.map(lambda c: GEO.get(c, (np.nan,) * 3)[1])
    tz_h = sch.home_team_orig.map(lambda c: GEO.get(c, (np.nan,) * 3)[2]); tz_a = sch.away_team_orig.map(lambda c: GEO.get(c, (np.nan,) * 3)[2])
    dist = haversine_miles(lat_a, lon_a, lat_h, lon_h)
    ok = (sch.neutral == 0) & dist.notna()
    X["away_miles"] = dist.where(ok, 0.0).fillna(0.0) / 1000.0
    shift = (tz_h - tz_a).where(ok, 0.0).fillna(0.0)         # + = away team traveled east
    X["away_tz_east"] = shift.clip(lower=0)
    X["away_tz_west"] = (-shift).clip(lower=0)
    away_local = hr + tz_a.fillna(0)
    X["away_body_early"] = ((away_local < 11) & ok).astype(float)   # e.g. West-coast team in 1pm ET game
    return X


# ----------------------------------------------------------------------------------------------------------
def rating_columns(sch, L, cfg_name, hl, lam, keys):
    rr = RidgeRatings(L, sch, list(ALL_RATED), hl, lam)
    fits = rr.fit_all(keys)
    tix = {t: i for i, t in enumerate(rr.teams)}
    metrics = list(ALL_RATED)
    sign = np.array([ALL_RATED[m] for m in metrics], dtype=float)
    n, M = len(sch), len(metrics)
    oh, dh, oa, da = (np.full((n, M), np.nan) for _ in range(4))
    hfa = np.full(n, np.nan)
    for ix, (s, w, h, a) in enumerate(zip(sch.season.values, sch.week.values, sch.home_team.values, sch.away_team.values)):
        f = fits.get((s, w))
        if f is None:
            continue
        i, j = tix[h], tix[a]
        oh[ix], dh[ix], oa[ix], da[ix] = f["off"][i], f["def"][i], f["off"][j], f["def"][j]
        hfa[ix] = f["home"][0]                                  # points-scale home edge estimated at that snapshot
    cols = {}
    for k_, m in enumerate(metrics):
        sg = sign[k_]
        cols[f"{cfg_name}__{m}__off_h"] = sg * oh[:, k_]        # oriented so higher = better for that team
        cols[f"{cfg_name}__{m}__def_h"] = -sg * dh[:, k_]
        cols[f"{cfg_name}__{m}__off_a"] = sg * oa[:, k_]
        cols[f"{cfg_name}__{m}__def_a"] = -sg * da[:, k_]
    cols[f"{cfg_name}__hfa_pts"] = hfa
    return pd.DataFrame(cols, index=sch.index)


# ----------------------------------------------------------------------------------------------------------
def build_model_table(sch, off_metrics, qb_lines, cfgs=RATING_CFGS):
    sch = sch.copy().reset_index(drop=True)
    L = build_long(sch, off_metrics)
    keys = sorted(set(zip(sch.season[sch.season >= 2000], sch.week[sch.season >= 2000])))

    Lh = team_history_features(L)
    Lh = qb_features(Lh, qb_lines, sch)
    tf_cols = ["ytd_g", "ytd_win", "ytd_pd", "ytd_pyth", "prev1_win", "prev1_pd", "prev3_win", "prev3_pd",
               "last3_pd", "last3_win", "last1_margin", "coach_tenure", "qb_rating", "qb_exp", "qb_changed",
               "qb_delta", "qb_rookie"]
    T = Lh[["game_id", "team"] + tf_cols]
    G = sch[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "home_score",
             "away_score", "result", "overtime", "played"]].copy()
    G = G.merge(T.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team": "home_team"}),
                on=["game_id", "home_team"], how="left")
    G = G.merge(T.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team": "away_team"}),
                on=["game_id", "away_team"], how="left")
    G = G.set_index(sch.game_id.map({g: i for i, g in enumerate(sch.game_id)}).reindex(G.game_id).values) if False else G
    G = G.merge(pd.concat([sch[["game_id"]], context_features(sch), h2h_features(sch)], axis=1), on="game_id", how="left")
    for name, (hl, lam) in cfgs.items():
        rc = rating_columns(sch, L, name, hl, lam, keys)
        rc["game_id"] = sch.game_id.values
        G = G.merge(rc, on="game_id", how="left")
    return G


if __name__ == "__main__":
    sch = load_games()
    o = load_offense()
    qb = pd.read_parquet(f"{D}/qb_game.parquet")
    G = build_model_table(sch, o, qb)
    G.to_parquet(f"{D}/model_table.parquet")
    print(G.shape)
    print(G.drop(columns=[c for c in G.columns if "__" in c]).describe().T.round(3).to_string())
