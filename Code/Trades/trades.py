"""Your own trades: import Robinhood realized P&L exports and test them for skill against the prices paid.

    python trades.py import realized_pnl.csv    # load an export into Data/quantum_route.duckdb (re-imports skip duplicates)
    python trades.py report                     # NFL skill-vs-luck summary in the terminal
    python trades.py report --all-sports

The export has one row per closed position: Robinhood has already merged its fills, so a row is a
position, not an order. It has no quantity and no prices, so both are inferred:

- Robinhood charges $0.01 commission per contract per side, so contracts traded = commission / $0.01.
- Proceeds of $0 means it was held and lost; proceeds equal to the contract count means it was held
  and won. Anything else was sold before settlement, assumed as a full round trip (bought N, sold N).
- Entry price = costs / contracts, which is the position's average price across every add.

The skill test treats each position as a zero-edge bet at its own entry price: the contract pays $1
with probability equal to that price, the same contracts and fees as the real trade. Positions on the
same game share one random draw, so a team's moneyline and spread win together and both sides of a
game can't both win. The p-value is the share of simulated zero-edge histories that made at least
the real profit. Positions sold early are simulated as if held, which has the right mean under the
null (a fair price is the expected value of the exit) and more spread, so it makes the test harder
to pass, not easier.

The export has no fill times, so pregame and in-game buys can't be told apart; a model probability
from Quantum Route is a pregame number and is only a fair benchmark for pregame buys.
"""
import hashlib
import math
import re
import sys
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_DIR / "Data" / "quantum_route.duckdb"
MODEL_DIR = PROJECT_DIR / "Data" / "Team_Model"
GAMES_PATH = PROJECT_DIR / "Data" / "games.csv"

COMMISSION_PER_CONTRACT = 0.01

CSV_COLUMNS = {
    "Account Number": "account", "Close Date": "close_date", "Symbol": "symbol",
    "Event Identifier": "event", "Exchange": "exchange", "Costs": "costs", "Proceeds": "proceeds",
    "Gross Pnl": "gross_pnl", "Exchange Fees": "exchange_fees", "Commissions": "commissions",
    "Net Pnl": "net_pnl",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_positions (
    position_id    VARCHAR PRIMARY KEY,  -- hash of the row, so overlapping exports import once
    account        VARCHAR,
    close_date     DATE NOT NULL,
    symbol         VARCHAR NOT NULL,
    event          VARCHAR,
    exchange       VARCHAR,
    costs          DOUBLE,   -- negative, as exported
    proceeds       DOUBLE,
    gross_pnl      DOUBLE,
    exchange_fees  DOUBLE,   -- negative
    commissions    DOUBLE,   -- negative
    net_pnl        DOUBLE,
    source_file    VARCHAR,
    imported_at    TIMESTAMP
);
"""

# Kalshi and ForecastEx team codes -> nflverse codes
NFL_TEAMS = {t: t for t in [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND",
    "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",
    "TB", "TEN", "WAS"]}
NFL_TEAMS.update({"JAC": "JAX", "LAR": "LA", "WSH": "WAS"})

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

PRICE_BUCKETS = [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0001]
PRICE_LABELS = ["<20¢", "20-35¢", "35-50¢", "50-65¢", "65-80¢", "80¢+"]
GAP_BUCKETS = [-1, -0.05, 0, 0.05, 0.10, 1]
GAP_LABELS = ["model below price by 5+", "model 0-5 below", "model 0-5 above", "model 5-10 above",
              "model 10+ above"]


# ---------------------------------------------------------------- storage

def connect(read_only=False):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        return duckdb.connect(str(DB_PATH), read_only=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(SCHEMA).fetchall()
    return con


@contextmanager
def _session(con):
    if con is not None:
        yield con
        return
    con = connect()
    try:
        yield con
    finally:
        con.close()


def read_export(path_or_buffer):
    """A Robinhood realized P&L CSV as a DataFrame with snake_case columns."""
    df = pd.read_csv(path_or_buffer, dtype={"Account Number": str})
    missing = set(CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Not a Robinhood realized P&L export, missing columns: {sorted(missing)}")
    df = df.rename(columns=CSV_COLUMNS)[list(CSV_COLUMNS.values())]
    df["close_date"] = pd.to_datetime(df["close_date"]).dt.date
    for col in ["costs", "proceeds", "gross_pnl", "exchange_fees", "commissions", "net_pnl"]:
        df[col] = pd.to_numeric(df[col]).fillna(0.0)
    return df


def _position_id(row):
    key = f"{row.account}|{row.close_date}|{row.symbol}|{row.costs:.2f}|{row.proceeds:.2f}|{row.net_pnl:.2f}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def import_export(path_or_buffer, source_file=None, con=None):
    """Adds an export's rows to trade_positions; returns (rows in file, rows newly added)."""
    df = read_export(path_or_buffer)
    df.insert(0, "position_id", [_position_id(r) for r in df.itertuples()])
    df = df.drop_duplicates("position_id")
    df["source_file"] = source_file or (str(path_or_buffer) if isinstance(path_or_buffer, (str, Path)) else None)
    df["imported_at"] = datetime.now()
    with _session(con) as con:
        before = con.execute("SELECT count(*) FROM trade_positions").fetchall()[0][0]
        con.register("incoming", df)
        con.execute("INSERT OR IGNORE INTO trade_positions BY NAME SELECT * FROM incoming").fetchall()
        con.unregister("incoming")
        after = con.execute("SELECT count(*) FROM trade_positions").fetchall()[0][0]
    return len(df), after - before


def load_positions(con=None):
    with _session(con) as con:
        return con.execute("SELECT * FROM trade_positions ORDER BY close_date, symbol").df()


# ---------------------------------------------------------------- parsing

def _split_matchup(code):
    """'PITCHI' -> ('PIT', 'CHI'); team codes are 2-3 letters, so try both splits."""
    for i in (2, 3):
        away, home = code[:i], code[i:]
        if away in NFL_TEAMS and home in NFL_TEAMS:
            return NFL_TEAMS[away], NFL_TEAMS[home]
    return None, None


def _kalshi_date(code):
    """'25NOV23' -> date(2025, 11, 23)"""
    m = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2})", code)
    if not m or m.group(2) not in MONTHS:
        return None
    return date(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))


def parse_symbol(symbol):
    """What a contract is about. Returns sport, market (moneyline, spread, total, prop, parlay,
    season_wins, other), game_date, away, home, selection (nflverse team code), line (the team must
    win by more than this / the total must go over it) and side (home, away, over, or None)."""
    out = {"sport": "Other", "market": "other", "game_date": None, "away": None, "home": None,
           "selection": None, "line": np.nan, "side": None}
    m = re.fullmatch(r"NFLGW_(\d{2})(\d{2})(\d{2})([A-Z]+)_([A-Z]+)", symbol)  # ForecastEx moneyline
    if m:
        away, home = _split_matchup(m.group(4))
        selection = NFL_TEAMS.get(m.group(5))
        out.update(sport="NFL", market="moneyline", away=away, home=home, selection=selection,
                   game_date=date(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2))))
        out["side"] = "home" if selection == home else "away" if selection == away else None
        return out

    parts = symbol.split("-")
    series = parts[0]
    if series.startswith("KXNFL"):
        out["sport"] = "NFL"
    elif series.startswith("KXNCAAF"):
        out["sport"] = "College football"
    elif series.startswith(("KXNCAAB", "KXMARMAD")):
        out["sport"] = "College basketball"
    elif series.startswith("KXNBA"):
        out["sport"] = "NBA"
    if out["sport"] != "NFL":
        return out

    if series == "KXNFLWINS":
        out.update(market="season_wins", selection=NFL_TEAMS.get(parts[1]))
        return out
    if len(parts) < 3:
        return out
    event, tail = parts[1], parts[2]
    m = re.fullmatch(r"(\d{2}[A-Z]{3}\d{2})([A-Z]+)", event)
    if m:
        out["game_date"] = _kalshi_date(m.group(1))
        out["away"], out["home"] = _split_matchup(m.group(2))

    if series == "KXNFLGAME":
        out["market"], out["selection"] = "moneyline", NFL_TEAMS.get(tail)
    elif series == "KXNFLSPREAD":
        m = re.fullmatch(r"([A-Z]+?)(\d+)", tail)
        if m:
            out.update(market="spread", selection=NFL_TEAMS.get(m.group(1)), line=int(m.group(2)) + 0.5)
    elif series == "KXNFLTOTAL":
        if tail.isdigit():
            out.update(market="total", line=int(tail) + 0.5, side="over")
        return out
    elif "SGP" in series:
        out["market"] = "parlay"
        return out
    else:
        out["market"] = "prop"
        return out
    sel = out["selection"]
    out["side"] = "home" if sel and sel == out["home"] else "away" if sel and sel == out["away"] else None
    return out


def _nfl_season(d):
    if d is None or pd.isna(d):
        return None
    return d.year if d.month >= 8 else d.year - 1


def enrich(positions, commission=COMMISSION_PER_CONTRACT):
    """Adds the parsed contract, inferred contracts and prices, and how the position ended."""
    df = positions.copy().reset_index(drop=True)
    parsed = pd.DataFrame([parse_symbol(s) for s in df["symbol"]])
    df = pd.concat([df, parsed], axis=1)
    df["cost"] = -df["costs"]
    df["fees"] = -(df["exchange_fees"] + df["commissions"])

    sides = (df["commissions"].abs() / commission).round()
    held_won = (sides > 0) & ((df["proceeds"] - sides).abs() < 0.005)
    held_lost = (sides > 0) & (df["proceeds"].abs() < 0.005)
    df["status"] = np.select([held_won, held_lost], ["won", "lost"], "exited")
    df["contracts"] = np.where(df["status"] == "exited", sides / 2, sides)
    bad = (df["contracts"] <= 0) | ((df["status"] == "exited") & (sides % 2 == 1))
    df.loc[bad, "contracts"] = np.nan
    df["entry_price"] = df["cost"] / df["contracts"]
    df["exit_price"] = np.where(df["status"] == "exited", df["proceeds"] / df["contracts"],
                                np.where(df["status"] == "won", 1.0, 0.0))
    priced = df["entry_price"].between(0.005, 0.995) & df["exit_price"].between(0, 1.0001)
    df.loc[~priced, ["contracts", "entry_price", "exit_price"]] = np.nan
    df["priced"] = priced
    # Contract outcome: known when held; for exits, filled from the schedule when available
    df["outcome"] = np.where(df["status"] == "won", 1.0, np.where(df["status"] == "lost", 0.0, np.nan))

    df["season"] = [_nfl_season(d if d is not None and not pd.isna(d) else c)
                    for d, c in zip(df["game_date"], df["close_date"])]
    game = df["away"].fillna("") + "@" + df["home"].fillna("") + " " + df["game_date"].astype(str)
    has_game = df["away"].notna() & df["home"].notna() & df["game_date"].notna()
    df["game"] = np.where(has_game, game, None)
    # Outcomes that share a random draw: one draw per game for moneylines and spreads, one per
    # game total, and props, parlays and anything unparsed on their own
    on_game_draw = has_game & df["market"].isin(["moneyline", "spread"]) & df["side"].notna()
    df["draw_group"] = np.where(on_game_draw, "game:" + game,
                                np.where(has_game & (df["market"] == "total"), "total:" + game,
                                         "own:" + df["position_id"].astype(str)))
    df["label"] = [_label(r) for r in df.itertuples()]
    return df


def _label(r):
    if r.market == "moneyline" and r.selection:
        other = r.home if r.selection == r.away else r.away
        return f"{r.selection} to beat {other}" if other else f"{r.selection} to win"
    if r.market == "spread" and r.selection:
        return f"{r.selection} by {r.line:g}+"
    if r.market == "total" and r.away:
        return f"{r.away}@{r.home} over {r.line:g}"
    if r.market == "season_wins":
        return f"{r.selection} season wins"
    return r.event if isinstance(r.event, str) else r.symbol


# ---------------------------------------------------------------- Quantum Route context

def _norm_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2)))


def load_schedule():
    """nflverse schedule if the team models have downloaded it (Data/games.csv), else None."""
    if not GAMES_PATH.exists():
        return None
    games = pd.read_csv(GAMES_PATH, usecols=["game_id", "season", "week", "gameday", "away_team",
                                             "home_team", "result"])
    games["gameday"] = pd.to_datetime(games["gameday"]).dt.date
    return games


def load_model_predictions():
    """Quantum Route's pregame margin and margin SD per game from backtest_*.csv (walk-forward,
    preferred) and predictions_*.csv, or None if neither has been run."""
    frames = [pd.read_csv(f) for f in sorted(MODEL_DIR.glob("backtest_*.csv"))]
    frames += [pd.read_csv(f) for f in sorted(MODEL_DIR.glob("predictions_*.csv"))]
    frames = [f for f in frames if {"game_id", "margin", "margin_sd"} <= set(f.columns)]
    if not frames:
        return None
    return pd.concat(frames)[["game_id", "season", "home_team", "away_team", "margin",
                              "margin_sd"]].drop_duplicates("game_id")


def attach_context(df, schedule=None, predictions=None):
    """Adds game_id, outcomes for positions sold early, and the model's probability for each
    moneyline and spread contract (model_prob)."""
    df = df.copy()
    df["game_id"], df["model_prob"] = None, np.nan
    if schedule is not None:
        sched = schedule.copy()
        lookup = {}
        for g in sched.itertuples():
            for delta in (-1, 0, 1):  # Kalshi dates can be a day off the listed gameday
                lookup.setdefault((g.away_team, g.home_team, g.gameday + pd.Timedelta(days=delta)), g)
        rows = [lookup.get((a, h, d)) if d is not None else None
                for a, h, d in zip(df["away"], df["home"], df["game_date"])]
        df["game_id"] = [r.game_id if r is not None else None for r in rows]
        result = pd.Series([r.result if r is not None else np.nan for r in rows], index=df.index)
        signed = np.where(df["side"] == "home", result, -result)  # selected team's margin
        won = np.select([df["market"] == "moneyline", df["market"] == "spread"],
                        [(signed > 0).astype(float), (signed > df["line"]).astype(float)], np.nan)
        won = np.where(result.isna() | df["side"].isna(), np.nan, won)
        df["outcome"] = df["outcome"].fillna(pd.Series(won, index=df.index))

    if predictions is not None:
        if df["game_id"].isna().all():  # no schedule: match on season and teams
            key = predictions.set_index(["season", "away_team", "home_team"])["game_id"]
            key = key[~key.index.duplicated(keep=False)]
            df["game_id"] = [key.get((s, a, h)) for s, a, h in zip(df["season"], df["away"], df["home"])]
        pred = predictions.set_index("game_id")
        margin = df["game_id"].map(pred["margin"])
        sd = df["game_id"].map(pred["margin_sd"])
        signed = np.where(df["side"] == "home", margin, -margin)
        line = np.where(df["market"] == "spread", df["line"], 0.0)
        prob = _norm_cdf((signed - line) / sd)
        usable = df["market"].isin(["moneyline", "spread"]) & df["side"].notna() & margin.notna()
        df["model_prob"] = np.where(usable, prob, np.nan)
    df["gap"] = df["model_prob"] - df["entry_price"]
    return df


# ---------------------------------------------------------------- the skill tests

def simulate_null(df, n_sims=20000, seed=0):
    """Simulated zero-edge P&L per position, shape (n_sims, len(df)). Each position's contract pays
    with probability equal to its entry price; positions in the same draw_group share one uniform
    draw, oriented so home-side contracts win on low draws and away-side ones on high draws."""
    rng = np.random.default_rng(seed)
    groups, index = np.unique(df["draw_group"].to_numpy(), return_inverse=True)
    u = rng.random((n_sims, len(groups)))[:, index]
    p = df["entry_price"].to_numpy()
    away = (df["side"] == "away").to_numpy()
    pays = np.where(away, u > 1 - p, u < p)
    return pays * df["contracts"].to_numpy() - df["cost"].to_numpy() - df["fees"].to_numpy()


def _p_value(sims, actual, mask=None):
    if mask is not None:
        sims, actual = sims[:, mask], actual[mask]
    if sims.shape[1] == 0:
        return np.nan
    totals = sims.sum(axis=1)
    return float((np.sum(totals >= actual.sum() - 1e-9) + 1) / (len(totals) + 1))


def _group_table(df, sims, by, labels=None):
    actual = df["net_pnl"].to_numpy()
    rows = []
    keys = labels if labels is not None else sorted(df[by].dropna().unique())
    for key in keys:
        mask = (df[by] == key).to_numpy()
        if not mask.any():
            continue
        sub = df[mask]
        known = sub["outcome"].notna()
        totals = sims[:, mask].sum(axis=1)
        rows.append({
            by: key, "positions": int(mask.sum()), "games": sub["draw_group"].nunique(),
            "staked": sub["cost"].sum(), "net_pnl": sub["net_pnl"].sum(),
            "roi": sub["net_pnl"].sum() / sub["cost"].sum(),
            "avg_entry": np.average(sub["entry_price"], weights=sub["cost"]),
            "win_rate": sub.loc[known, "outcome"].mean() if known.any() else np.nan,
            "price_implied_win_rate": sub.loc[known, "entry_price"].mean() if known.any() else np.nan,
            "null_mean": totals.mean(), "null_p5": np.percentile(totals, 5),
            "null_p95": np.percentile(totals, 95), "p_value": _p_value(sims, actual, mask),
            # What the bucket should have made if the model's probabilities were the truth
            "model_expected_pnl": ((sub["model_prob"] - sub["entry_price"]) * sub["contracts"] - sub["fees"]).sum()
            if sub["model_prob"].notna().all() else np.nan,
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(df, n_boot=10000, seed=1):
    """95% interval for total net P&L, resampling whole games (draw groups) with replacement."""
    rng = np.random.default_rng(seed)
    per_group = df.groupby("draw_group")["net_pnl"].sum().to_numpy()
    draws = rng.integers(0, len(per_group), size=(n_boot, len(per_group)))
    totals = per_group[draws].sum(axis=1)
    return float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))


def analyze(positions, sports=("NFL",), include_exits=True, n_sims=20000, seed=0,
            schedule=None, predictions=None):
    """Everything the dashboard and the report show. positions is load_positions() or read_export()."""
    df = enrich(positions)
    if sports:
        df = df[df["sport"].isin(sports)]
    unpriced = df[~df["priced"]]
    df = df[df["priced"]]
    if not include_exits:
        df = df[df["status"] != "exited"]
    df = attach_context(df.reset_index(drop=True), schedule, predictions)
    if df.empty:
        raise ValueError("No priced positions to analyze")

    sims = simulate_null(df, n_sims, seed)
    actual = df["net_pnl"].to_numpy()
    totals = sims.sum(axis=1)
    known = df["outcome"].notna()
    ranked = df.sort_values("net_pnl", ascending=False)
    lo, hi = cluster_bootstrap(df)

    summary = {
        "positions": len(df), "games": df["draw_group"].nunique(), "unpriced_positions": len(unpriced),
        "unpriced_net_pnl": float(unpriced["net_pnl"].sum()),
        "exited_positions": int((df["status"] == "exited").sum()),
        "staked": float(df["cost"].sum()), "fees": float(df["fees"].sum()),
        "net_pnl": float(actual.sum()), "roi": float(actual.sum() / df["cost"].sum()),
        "first_date": pd.Timestamp(df["close_date"].min()).date(),
        "last_date": pd.Timestamp(df["close_date"].max()).date(),
        "wins": int((df["outcome"] == 1).sum()), "losses": int((df["outcome"] == 0).sum()),
        "expected_wins": float(df.loc[known, "entry_price"].sum()),
        "avg_entry": float(np.average(df["entry_price"], weights=df["cost"])),
        "null_mean": float(totals.mean()), "null_sd": float(totals.std()),
        "null_p5": float(np.percentile(totals, 5)), "null_p95": float(np.percentile(totals, 95)),
        "p_value": _p_value(sims, actual), "bootstrap_low": lo, "bootstrap_high": hi,
        "top5_pnl": float(ranked["net_pnl"].head(5).sum()),
        "model_matched": int(df["model_prob"].notna().sum()),
    }
    summary["z"] = (summary["net_pnl"] - summary["null_mean"]) / summary["null_sd"]

    df["price_bucket"] = pd.cut(df["entry_price"], PRICE_BUCKETS, labels=PRICE_LABELS, right=False).astype(str)
    df["gap_bucket"] = pd.cut(df["gap"], GAP_BUCKETS, labels=GAP_LABELS, right=False).astype(str)
    df.loc[df["gap"].isna(), "gap_bucket"] = None

    # How much of the profit survives dropping the biggest winners, next to the same share in
    # zero-edge histories (an underdog bettor's profit is lopsided by construction)
    order = np.argsort(-sims, axis=1)
    concentration = []
    for k in (1, 3, 5, 10):
        if k >= len(df):
            break
        sim_rest = totals - np.take_along_axis(sims, order[:, :k], axis=1).sum(axis=1)
        rest = actual.sum() - ranked["net_pnl"].head(k).sum()
        concentration.append({"drop_top": k, "remaining_pnl": rest,
                              "null_remaining_median": float(np.median(sim_rest)),
                              "share_of_null_at_least": float(np.mean(sim_rest >= rest))})

    return {
        "summary": summary, "positions": df, "unpriced": unpriced, "null_totals": totals,
        "by_price": _group_table(df, sims, "price_bucket", PRICE_LABELS),
        "by_gap": _group_table(df, sims, "gap_bucket", GAP_LABELS) if df["gap"].notna().any() else pd.DataFrame(),
        "by_season": _group_table(df, sims, "season"),
        "by_market": _group_table(df, sims, "market"),
        "concentration": pd.DataFrame(concentration),
        "top": ranked.head(10)[["close_date", "label", "contracts", "entry_price", "status", "net_pnl"]],
    }


def verdict(summary):
    p = summary["p_value"]
    if p < 0.01:
        return "Strong evidence of an edge over the prices you paid."
    if p < 0.05:
        return "Evidence of an edge over the prices you paid, at the usual 5% bar."
    if p < 0.2:
        return "Promising, but a zero-edge trader at your prices gets here often enough that it could be luck."
    return "Consistent with luck: zero-edge trading at your prices does this well often."


def report(sports=("NFL",)):
    pd.set_option("display.width", 200)
    result = analyze(load_positions(), sports=sports, schedule=load_schedule(),
                     predictions=load_model_predictions())
    s = result["summary"]
    print(f"{s['positions']} positions on {s['games']} independent outcomes, {s['first_date']} to {s['last_date']}")
    print(f"net P&L ${s['net_pnl']:+.2f} on ${s['staked']:.2f} staked ({s['roi']:+.1%}), fees ${s['fees']:.2f}")
    print(f"held to settlement: {s['wins']} won, {s['losses']} lost; prices implied {s['expected_wins']:.1f} wins")
    print(f"zero-edge trader at your prices: mean ${s['null_mean']:+.2f}, 90% range "
          f"${s['null_p5']:+.2f} to ${s['null_p95']:+.2f}")
    print(f"p-value {s['p_value']:.3f} (z {s['z']:+.2f}); game-bootstrap 95% interval for your total "
          f"${s['bootstrap_low']:+.2f} to ${s['bootstrap_high']:+.2f}")
    print(verdict(s))
    for name in ["by_season", "by_price", "by_gap", "by_market", "concentration"]:
        if len(result[name]):
            print(f"\n{name}\n{result[name].dropna(axis=1, how='all').round(3).to_string(index=False)}")
    if s["unpriced_positions"]:
        print(f"\n{s['unpriced_positions']} positions left out: contracts couldn't be inferred from fees")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command == "import" and len(sys.argv) > 2:
        for path in sys.argv[2:]:
            rows, added = import_export(path, source_file=Path(path).name)
            print(f"{path}: {rows} rows, {added} new")
    elif command == "report":
        report(sports=None if "--all-sports" in sys.argv else ("NFL",))
    else:
        sys.exit(__doc__)
