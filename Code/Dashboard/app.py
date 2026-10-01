"""Backtest dashboard over the SQL history in Data/quantum_route.duckdb.

    streamlit run Code/Dashboard/app.py
"""
import sys
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db  # noqa: E402

st.set_page_config(page_title="Quantum Route backtests", layout="wide")


@st.cache_data(ttl=30)
def query(sql, params=None):
    """Runs a read-only query; a short-lived connection keeps the file free for backtest writes."""
    with duckdb.connect(str(db.DB_PATH), read_only=True) as con:
        return con.execute(sql, params or []).df()


if not db.DB_PATH.exists():
    st.title("Quantum Route backtests")
    st.info("No backtest history yet. Run `python Code/Team_Model/run_backtest.py 2025`, or load "
            "existing results with `python Code/db.py import`.")
    st.stop()

try:
    runs = query("SELECT * FROM run_summary ORDER BY started_at DESC")
except duckdb.IOException:
    st.warning("The database is busy with a backtest write. Refresh in a moment.")
    st.stop()

if runs.empty:
    st.info("The history is empty. Run a backtest or `python Code/db.py import`.")
    st.stop()

runs["ats"] = runs["ats_wins"].astype(str) + "-" + runs["ats_losses"].astype(str)
runs["ats_pct"] = runs["ats_wins"] / (runs["ats_wins"] + runs["ats_losses"]).where(lambda n: n > 0)
runs["label"] = (runs["run_id"] + "  ·  " + runs["status"]
                 + runs["note"].fillna("").map(lambda n: f"  ·  {n}" if n else ""))

# Sidebar: which runs to look at
seasons = sorted(runs["season"].unique(), reverse=True)
season = st.sidebar.selectbox("Season", seasons)
season_runs = runs[runs["season"] == season]
run_label = st.sidebar.selectbox("Run", season_runs["label"])
run = season_runs[season_runs["label"] == run_label].iloc[0]
compare_label = st.sidebar.selectbox("Compare with", ["(none)"] + [l for l in season_runs["label"] if l != run_label])

st.title("Quantum Route backtests")

# History across runs
st.subheader("Run history")
chart_runs = runs.melt(id_vars=["run_id", "season", "started_at"], value_vars=["mae_model", "mae_market"],
                       var_name="source", value_name="MAE")
chart_runs["source"] = chart_runs["source"].map({"mae_model": "Model", "mae_market": "Closing line"})
st.altair_chart(
    alt.Chart(chart_runs).mark_line(point=True).encode(
        x=alt.X("started_at:T", title="Run started", axis=alt.Axis(format="%b %d %H:%M")),
        y=alt.Y("MAE:Q", title="Margin MAE (points)", scale=alt.Scale(zero=False)),
        color=alt.Color("source:N", title=None),
        strokeDash=alt.StrokeDash("season:N", title="Season"),
        tooltip=["run_id", "season", "source", alt.Tooltip("MAE:Q", format=".2f")],
    ).properties(height=260),
    use_container_width=True,
)
st.dataframe(
    runs[["run_id", "season", "status", "git_commit", "games", "mae_model", "mae_market", "brier_model",
          "brier_market", "total_mae_model", "total_mae_market", "ats", "ats_pct", "note"]],
    hide_index=True, use_container_width=True,
    column_config={
        "mae_model": st.column_config.NumberColumn(format="%.2f"),
        "mae_market": st.column_config.NumberColumn(format="%.2f"),
        "brier_model": st.column_config.NumberColumn(format="%.4f"),
        "brier_market": st.column_config.NumberColumn(format="%.4f"),
        "total_mae_model": st.column_config.NumberColumn(format="%.2f"),
        "total_mae_market": st.column_config.NumberColumn(format="%.2f"),
        "ats_pct": st.column_config.NumberColumn("ats %", format="%.3f"),
    },
)

# One run in detail
st.subheader(f"Run {run['run_id']}")
cols = st.columns(5)
cols[0].metric("Margin MAE", f"{run['mae_model']:.2f}", f"{run['mae_model'] - run['mae_market']:+.2f} vs line",
               delta_color="inverse")
cols[1].metric("Brier (win prob)", f"{run['brier_model']:.4f}",
               f"{run['brier_model'] - run['brier_market']:+.4f} vs line", delta_color="inverse")
cols[2].metric("Total MAE", f"{run['total_mae_model']:.2f}",
               f"{run['total_mae_model'] - run['total_mae_market']:+.2f} vs line", delta_color="inverse")
cols[3].metric("ATS, all edges", run["ats"], f"{run['ats_pct']:.1%} covered" if pd.notna(run["ats_pct"]) else None,
               delta_color="off")
cols[4].metric("ATS, edge > 3", f"{run['ats3_wins']}-{run['ats3_losses']}")

left, right = st.columns(2)
weekly = query("SELECT * FROM weekly_summary WHERE run_id = ? ORDER BY week", [run["run_id"]])
weekly_long = weekly.melt(id_vars=["week"], value_vars=["mae_model", "mae_market"], var_name="source", value_name="MAE")
weekly_long["source"] = weekly_long["source"].map({"mae_model": "Model", "mae_market": "Closing line"})
left.markdown("**Margin MAE by week**")
left.altair_chart(
    alt.Chart(weekly_long).mark_line(point=True).encode(
        x=alt.X("week:O", title="Week"), y=alt.Y("MAE:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("source:N", title=None), tooltip=["week", "source", alt.Tooltip("MAE:Q", format=".2f")],
    ).properties(height=260),
    use_container_width=True,
)
edges = query("""SELECT edge_bucket, wins, losses, cover_rate FROM ats_by_edge
                 WHERE run_id = ? ORDER BY edge_bucket""", [run["run_id"]])
right.markdown("**ATS cover rate by edge size** (points of disagreement with the line)")
right.altair_chart(
    alt.Chart(edges).mark_bar().encode(
        x=alt.X("edge_bucket:N", title="Edge (points)", sort=None),
        y=alt.Y("cover_rate:Q", title="Cover rate", axis=alt.Axis(format="%")),
        tooltip=["edge_bucket", "wins", "losses", alt.Tooltip("cover_rate:Q", format=".1%")],
    ).properties(height=260)
    + alt.Chart(pd.DataFrame({"y": [0.524]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q"),
    use_container_width=True,
)
right.caption("Dashed line: 52.4%, the break-even cover rate at -110.")

st.markdown("**Games**")
games = query("""
    SELECT week, away_team || ' @ ' || home_team AS game, spread_line, round(margin, 1) AS model_margin,
           round(edge, 1) AS edge, result, covered, round(p_home_win, 3) AS p_home_win,
           round(market_win_prob, 3) AS market_win_prob, total_line, round(total_pred, 1) AS total_pred, total,
           home_team, away_team
    FROM backtest_graded WHERE run_id = ? ORDER BY week, game_id""", [run["run_id"]])
teams = sorted(set(games["home_team"]) | set(games["away_team"]))
f1, f2 = st.columns([2, 1])
team = f1.multiselect("Teams", teams)
min_edge = f2.slider("Minimum |edge|", 0.0, float(max(games["edge"].abs().max(), 1)), 0.0, 0.5)
shown = games[games["edge"].abs() >= min_edge]
if team:
    shown = shown[shown["home_team"].isin(team) | shown["away_team"].isin(team)]
st.dataframe(shown.drop(columns=["home_team", "away_team"]), hide_index=True, use_container_width=True)

# Two runs side by side
if compare_label != "(none)":
    other = season_runs[season_runs["label"] == compare_label].iloc[0]
    st.subheader(f"{run['run_id']} vs {other['run_id']}")
    metrics = ["mae_model", "brier_model", "total_mae_model", "ats_pct"]
    st.dataframe(
        pd.DataFrame({"metric": metrics, run["run_id"]: [run[m] for m in metrics],
                      other["run_id"]: [other[m] for m in metrics],
                      "change": [run[m] - other[m] for m in metrics]}).round(4),
        hide_index=True,
    )
    paired = query("""
        SELECT a.game_id, a.week, a.away_team || ' @ ' || a.home_team AS game,
               a.margin AS margin_a, b.margin AS margin_b, a.result
        FROM backtest_predictions a JOIN backtest_predictions b USING (game_id)
        WHERE a.run_id = ? AND b.run_id = ?""", [run["run_id"], other["run_id"]])
    st.markdown("**Predicted margin per game** (points off the diagonal are games the runs disagree on)")
    st.altair_chart(
        alt.Chart(paired).mark_circle(size=40, opacity=0.6).encode(
            x=alt.X("margin_b:Q", title=f"{other['run_id']} margin"),
            y=alt.Y("margin_a:Q", title=f"{run['run_id']} margin"),
            tooltip=["game", "week", alt.Tooltip("margin_a:Q", format=".1f"),
                     alt.Tooltip("margin_b:Q", format=".1f"), "result"],
        ).properties(height=360),
        use_container_width=True,
    )
