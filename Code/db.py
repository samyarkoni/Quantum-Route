"""SQL store for backtest history: one DuckDB file at Data/quantum_route.duckdb.

Every backtest run gets a row in backtest_runs and one row per game in backtest_predictions, so
runs can be compared over time instead of overwriting backtest_<season>.csv. The views at the
bottom score each run against the closing lines in SQL; the dashboard reads them directly.

    python db.py import        # load existing Data/Team_Model/backtest_*.csv files as runs
    python db.py history       # list recorded runs with their headline scores
"""
import math
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "Data" / "quantum_route.duckdb"
BACKTEST_DIR = PROJECT_DIR / "Data" / "Team_Model"

PREDICTION_COLUMNS = [
    "game_id", "season", "week", "home_team", "away_team", "spread_line", "total_line",
    "home_moneyline", "away_moneyline", "result", "total", "home_pts", "away_pts", "margin",
    "margin_sd", "total_pred", "p_home_win", "p_home_cover", "model_win_prob", "market_win_prob",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id       VARCHAR PRIMARY KEY,
    season       INTEGER NOT NULL,
    first_week   INTEGER,
    started_at   TIMESTAMP NOT NULL,
    finished_at  TIMESTAMP,
    status       VARCHAR NOT NULL,          -- running | finished | imported
    git_commit   VARCHAR,
    git_dirty    BOOLEAN,
    note         VARCHAR
);

CREATE TABLE IF NOT EXISTS backtest_predictions (
    run_id           VARCHAR NOT NULL,
    game_id          VARCHAR NOT NULL,
    season           INTEGER,
    week             INTEGER,
    home_team        VARCHAR,
    away_team        VARCHAR,
    spread_line      DOUBLE,   -- home minus away, > 0 means home favored
    total_line       DOUBLE,
    home_moneyline   DOUBLE,
    away_moneyline   DOUBLE,
    result           DOUBLE,   -- actual home minus away
    total            DOUBLE,
    home_pts         DOUBLE,
    away_pts         DOUBLE,
    margin           DOUBLE,   -- model's predicted home minus away
    margin_sd        DOUBLE,
    total_pred       DOUBLE,
    p_home_win       DOUBLE,
    p_home_cover     DOUBLE,
    model_win_prob   DOUBLE,   -- from margin and margin_sd, as in run_backtest.score
    market_win_prob  DOUBLE,   -- de-vigged closing moneyline
    PRIMARY KEY (run_id, game_id)
);

-- One row per graded game with the quantities every score needs.
CREATE OR REPLACE VIEW backtest_graded AS
SELECT
    p.*,
    margin - spread_line AS edge,
    abs(margin - spread_line) AS abs_edge,
    CASE WHEN result = 0 THEN 0.5 WHEN result > 0 THEN 1.0 ELSE 0.0 END AS home_won,
    result <> spread_line AS ats_graded,
    (result > spread_line) = (margin - spread_line > 0) AS covered
FROM backtest_predictions p;

-- Headline scores per run, matching run_backtest.score with slope 1.
CREATE OR REPLACE VIEW run_summary AS
SELECT
    r.run_id, r.season, r.status, r.started_at, r.finished_at, r.git_commit, r.git_dirty, r.note,
    count(g.game_id)                                        AS games,
    max(g.week)                                             AS last_week,
    avg(abs(result - margin))                               AS mae_model,
    avg(abs(result - spread_line))                          AS mae_market,
    corr(margin, spread_line)                               AS corr_model_market,
    avg(power(model_win_prob - home_won, 2))                AS brier_model,
    avg(power(market_win_prob - home_won, 2))               AS brier_market,
    avg(abs(total - total_pred))                            AS total_mae_model,
    avg(abs(total - total_line))                            AS total_mae_market,
    count(*) FILTER (ats_graded AND abs_edge > 0 AND covered)       AS ats_wins,
    count(*) FILTER (ats_graded AND abs_edge > 0 AND NOT covered)   AS ats_losses,
    count(*) FILTER (ats_graded AND abs_edge > 3 AND covered)       AS ats3_wins,
    count(*) FILTER (ats_graded AND abs_edge > 3 AND NOT covered)   AS ats3_losses
FROM backtest_runs r
LEFT JOIN backtest_graded g USING (run_id)
GROUP BY ALL;

-- Against-the-spread record by size of the model's disagreement with the line.
CREATE OR REPLACE VIEW ats_by_edge AS
SELECT
    run_id,
    CASE WHEN abs_edge <= 1.5 THEN '0-1.5' WHEN abs_edge <= 3 THEN '1.5-3'
         WHEN abs_edge <= 6 THEN '3-6' ELSE '6+' END       AS edge_bucket,
    count(*) FILTER (covered)                               AS wins,
    count(*) FILTER (NOT covered)                           AS losses,
    avg(covered::DOUBLE)                                    AS cover_rate
FROM backtest_graded
WHERE ats_graded AND abs_edge > 0
GROUP BY ALL;

-- Weekly accuracy, for spotting where a run drifts from the market.
CREATE OR REPLACE VIEW weekly_summary AS
SELECT
    run_id, season, week,
    count(*)                                                AS games,
    avg(abs(result - margin))                               AS mae_model,
    avg(abs(result - spread_line))                          AS mae_market,
    avg(covered::DOUBLE) FILTER (ats_graded AND abs_edge > 0) AS cover_rate
FROM backtest_graded
GROUP BY ALL;
"""


def connect(read_only=False):
    """Connection with the schema in place. read_only connections skip creating it.
    DuckDB lets only one process write at a time, so writers should connect, write and close
    (the functions below do that when not handed a connection) to keep the dashboard readable."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        return duckdb.connect(str(DB_PATH), read_only=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(SCHEMA)
    return con


@contextmanager
def _session(con):
    """Uses con if given, otherwise opens a connection and closes it afterwards."""
    if con is not None:
        yield con
        return
    con = connect()
    try:
        yield con
    finally:
        con.close()


def _implied_probability(moneyline):
    moneyline = np.asarray(moneyline, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):  # np.where evaluates both branches
        return np.where(moneyline < 0, -moneyline / (-moneyline + 100), 100 / (moneyline + 100))


def _with_probabilities(df):
    """Adds model_win_prob and market_win_prob the same way run_backtest.score computes them."""
    df = df.copy()
    home_p = _implied_probability(df["home_moneyline"])
    away_p = _implied_probability(df["away_moneyline"])
    df["market_win_prob"] = home_p / (home_p + away_p)
    z = (0 - df["margin"]) / (df["margin_sd"] * np.sqrt(2))
    df["model_win_prob"] = 1 - 0.5 * (1 + np.vectorize(math.erf)(z))
    for column in PREDICTION_COLUMNS:
        if column not in df:
            df[column] = np.nan
    return df[PREDICTION_COLUMNS]


def _git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "Code"], cwd=PROJECT_DIR,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def start_run(season, first_week=1, note=None, status="running", con=None):
    """Registers a new run and returns its id."""
    run_id = f"{season}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    commit, dirty = _git_state()
    with _session(con) as con:
        con.execute("INSERT INTO backtest_runs VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)",
                    [run_id, season, first_week, datetime.now(), status, commit, dirty, note])
    return run_id


def save_predictions(run_id, rows, con=None):
    """Upserts game predictions (a DataFrame or list of dicts) for a run."""
    df = _with_probabilities(pd.DataFrame(rows))
    df.insert(0, "run_id", run_id)
    with _session(con) as con:
        con.register("incoming", df)
        con.execute("INSERT OR REPLACE INTO backtest_predictions BY NAME SELECT * FROM incoming")
        con.unregister("incoming")


def finish_run(run_id, con=None):
    with _session(con) as con:
        con.execute("UPDATE backtest_runs SET status = 'finished', finished_at = ? WHERE run_id = ?",
                    [datetime.now(), run_id])


def import_csvs(con=None):
    """Loads each Data/Team_Model/backtest_<season>.csv as an 'imported' run, skipping files
    already imported (matched on the note, which records the file name and its mtime)."""
    with _session(con) as con:
        _import_csvs(con)


def _import_csvs(con):
    for path in sorted(BACKTEST_DIR.glob("backtest_*.csv")):
        note = f"imported {path.name} ({datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M})"
        if con.execute("SELECT 1 FROM backtest_runs WHERE note = ?", [note]).fetchone():
            print(f"skip {path.name}: already imported")
            continue
        season = int(path.stem.split("_")[1])
        df = pd.read_csv(path)
        run_id = start_run(season, int(df["week"].min()), note=note, status="imported", con=con)
        save_predictions(run_id, df, con=con)
        con.execute("UPDATE backtest_runs SET finished_at = started_at WHERE run_id = ?", [run_id])
        print(f"{path.name}: {len(df)} games -> run {run_id}")


def history(con=None):
    with _session(con) as con:
        return con.sql("""
            SELECT run_id, status, git_commit AS commit, games,
                   round(mae_model, 2) AS mae_model, round(mae_market, 2) AS mae_market,
                   round(brier_model, 4) AS brier_model, round(brier_market, 4) AS brier_market,
                   ats_wins || '-' || ats_losses AS ats, ats3_wins || '-' || ats3_losses AS "ats_edge>3", note
            FROM run_summary ORDER BY started_at DESC
        """).df()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "history"
    if command == "import":
        import_csvs()
    elif command == "history":
        pd.set_option("display.width", 200)
        print(history().to_string(index=False))
    else:
        sys.exit(__doc__)
