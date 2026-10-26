"""Skill or luck: your own trades tested against the prices you paid.

    streamlit run Code/Dashboard/trades_app.py

Upload a Robinhood realized P&L export in the sidebar (or `python Code/Trades/trades.py import
file.csv`); positions are kept in Data/quantum_route.duckdb, so later exports only add what's new.
"""
import sys
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Trades"))
import trades  # noqa: E402

st.set_page_config(page_title="Quantum Route: skill or luck", layout="wide")
st.title("Skill or luck")

with st.sidebar:
    upload = st.file_uploader("Robinhood realized P&L CSV", type="csv")
    if upload is not None and st.session_state.get("imported") != upload.file_id:
        try:
            rows, added = trades.import_export(upload, source_file=upload.name)
            st.session_state["imported"] = upload.file_id
            st.success(f"{rows} rows read, {added} new")
            st.cache_data.clear()
        except (ValueError, duckdb.IOException) as e:
            st.error(str(e))
    use_sample = st.checkbox("Show the sample export instead", value=not trades.DB_PATH.exists())


@st.cache_data(ttl=30)
def positions(sample):
    if sample:
        df = trades.read_export(trades.Path(trades.__file__).parent / "sample_realized_pnl.csv")
        df.insert(0, "position_id", [trades._position_id(r) for r in df.itertuples()])
        return df
    with duckdb.connect(str(trades.DB_PATH), read_only=True) as con:
        return con.execute("SELECT * FROM trade_positions ORDER BY close_date, symbol").df()


@st.cache_data(ttl=300)
def run(sample, sports, include_exits, n_sims):
    return trades.analyze(positions(sample), sports=sports, include_exits=include_exits, n_sims=n_sims,
                          schedule=trades.load_schedule(), predictions=trades.load_model_predictions())


try:
    raw = positions(use_sample)
except (duckdb.IOException, duckdb.CatalogException):
    st.info("No trades yet. Upload a Robinhood realized P&L export in the sidebar.")
    st.stop()

all_sports = sorted(trades.enrich(raw)["sport"].unique())
with st.sidebar:
    sports = st.multiselect("Sports", all_sports, default=[s for s in ["NFL"] if s in all_sports] or all_sports)
    include_exits = st.checkbox("Include positions sold before settlement", value=True,
                                help="Simulated as if held. Same expected value under the null, more spread.")
    n_sims = st.select_slider("Simulated zero-edge histories", [5000, 20000, 50000], value=20000)

try:
    result = run(use_sample, tuple(sports), include_exits, n_sims)
except ValueError as e:
    st.warning(str(e))
    st.stop()
s = result["summary"]
df = result["positions"]

if use_sample:
    st.caption("Showing the 64-row sample export (anonymized). Upload your full export for the real answer.")

def usd(x, decimals=2, signed=True):
    """Dollars with the $ escaped, since Streamlit reads $...$ as math."""
    sign = "−" if x < 0 else "+" if signed else ""
    return f"{sign}\\${abs(x):,.{decimals}f}"


def stat(col, label, value, note):
    col.metric(label, value)
    col.caption(note)


st.markdown(f"#### {trades.verdict(s)}")
st.write(
    f"A trader with **no edge**, buying the same contracts at your prices and paying the same fees, "
    f"makes at least your {usd(s['net_pnl'])} in **{s['p_value']:.1%}** of {n_sims:,} simulated histories. "
    f"Your {s['positions']} positions cover {s['games']} independent outcomes "
    f"({s['first_date']} to {s['last_date']}); contracts on the same game are simulated together."
)
cols = st.columns(5)
stat(cols[0], "Net P&L", usd(s["net_pnl"]), f"{s['roi']:+.1%} on {usd(s['staked'], 0, False)} staked")
stat(cols[1], "p-value vs your prices", f"{s['p_value']:.3f}", f"z = {s['z']:+.2f}")
stat(cols[2], "Held to the end", f"{s['wins']}-{s['losses']}", f"prices implied {s['expected_wins']:.1f} wins")
stat(cols[3], "Zero-edge top 5%", usd(s["null_p95"], 0),
     f"90% of zero-edge histories land between {usd(s['null_p5'], 0)} and {usd(s['null_p95'], 0)}")
stat(cols[4], "Average entry", f"{s['avg_entry']:.0%}", f"fees {usd(s['fees'], 2, False)}")

left, right = st.columns(2)
null = pd.DataFrame({"pnl": result["null_totals"]})
left.markdown("**Your profit against zero-edge trading at your prices**")
left.altair_chart(
    alt.Chart(null).mark_bar(opacity=0.7).encode(
        x=alt.X("pnl:Q", bin=alt.Bin(maxbins=60), title="Simulated net P&L ($)"),
        y=alt.Y("count():Q", title="Histories"),
    ).properties(height=280)
    + alt.Chart(pd.DataFrame({"x": [s["net_pnl"]]})).mark_rule(color="#d62728", size=2).encode(x="x:Q"),
    use_container_width=True,
)
left.caption(f"Red line: you. Resampling your games instead gives a 95% interval for your total of "
             f"{usd(s['bootstrap_low'], 0)} to {usd(s['bootstrap_high'], 0)}.")

price = result["by_price"]
right.markdown("**Calibration: do your teams win more often than the price says?**")
calib = price.dropna(subset=["win_rate"])
right.altair_chart(
    alt.Chart(calib).mark_circle(color="#1f77b4").encode(
        x=alt.X("price_implied_win_rate:Q", title="Average entry price", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        y=alt.Y("win_rate:Q", title="Actual win rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        size=alt.Size("positions:Q", legend=None, scale=alt.Scale(range=[60, 600])),
        tooltip=["price_bucket", "positions", alt.Tooltip("win_rate:Q", format=".0%"),
                 alt.Tooltip("price_implied_win_rate:Q", format=".0%")],
    ).properties(height=280)
    + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y"),
    use_container_width=True,
)
right.caption("Above the dashed line means those teams won more often than you paid for. Positions sold "
              "early count only when the schedule (Data/games.csv) gives the game's result.")

def dollars(decimals=2, signed=True):
    return lambda x: ("−" if x < 0 else "+" if signed else "") + f"${abs(x):,.{decimals}f}"


money = {"staked": dollars(2, False), "net_pnl": dollars(), "null_mean": dollars(), "null_p5": dollars(0),
         "null_p95": dollars(0), "model_expected_pnl": dollars(), "remaining_pnl": dollars(),
         "null_remaining_median": dollars(), "roi": "{:+.1%}", "avg_entry": "{:.0%}", "win_rate": "{:.0%}",
         "price_implied_win_rate": "{:.0%}", "p_value": "{:.3f}", "share_of_null_at_least": "{:.1%}",
         "entry_price": "{:.2f}", "contracts": "{:,.0f}"}


def table(frame):
    frame = frame.dropna(axis=1, how="all")
    st.dataframe(frame.style.format({k: v for k, v in money.items() if k in frame}, na_rep="–"),
                 hide_index=True, use_container_width=True)


st.subheader("By entry price")
st.write("Each price band gets its own zero-edge simulation. For an underdog strategy, the low bands are "
         "where the edge has to show up.")
table(price)

st.subheader("By gap between Quantum Route and your price")
if result["by_gap"].empty:
    st.info("No model probabilities matched yet. Run the backtest or `betting_board.py predict` for the "
            "seasons you traded, so Data/Team_Model has backtest_*.csv or predictions_*.csv; moneylines and "
            "spreads are then priced by the model's margin and spread.")
else:
    st.write(f"{s['model_matched']} moneyline and spread positions matched a model game. If the gap is real, "
             "bigger gaps should earn more, and actual P&L should track **model_expected_pnl** (what each "
             "band makes if the model's probability is the truth). Fees are about 2¢ a contract, so a gap "
             "smaller than that loses money even when the model is right. The model is pregame, so this is "
             "a fair test only for pregame buys.")
    table(result["by_gap"])

left, right = st.columns(2)
with left:
    st.subheader("By season")
    table(result["by_season"])
with right:
    st.subheader("By market")
    table(result["by_market"])

st.subheader("How much rides on the biggest wins")
st.write("Buying underdogs makes profit lopsided by design, so the comparison is with zero-edge histories "
         "that drop their own top N, not with zero. The last column is the share of those histories that "
         "still do at least as well as you.")
table(result["concentration"])
top = result["top"].copy()
top["close_date"] = pd.to_datetime(top["close_date"]).dt.date
table(top)

st.subheader("Positions")
df["close_date"] = pd.to_datetime(df["close_date"]).dt.date
shown = df[["close_date", "label", "market", "status", "contracts", "entry_price", "exit_price", "outcome",
            "model_prob", "gap", "cost", "fees", "net_pnl", "symbol"]]
st.dataframe(shown.dropna(axis=1, how="all"), hide_index=True, use_container_width=True,
             column_config={c: st.column_config.NumberColumn(format="%.3f")
                            for c in ["entry_price", "exit_price", "model_prob", "gap"]})
if len(result["unpriced"]):
    st.caption(f"Left out: {len(result['unpriced'])} position(s), "
               f"{usd(s['unpriced_net_pnl'])} net, whose contract count couldn't be read from the fees "
               "(usually a partial sale with the rest held): "
               + ", ".join(result["unpriced"]["symbol"]))

with st.expander("What this can't tell yet"):
    st.markdown(
        "- **Pregame vs in-game.** The export has no fill times, so the two can't be split. Robinhood's "
        "monthly statements and trade confirmations have them; an importer for those would allow it.\n"
        "- **Your own probability.** The gap test uses Quantum Route's number. Logging what you thought the "
        "chance was when you bought would let the same test grade your judgement directly.\n"
        "- **Adds at different prices.** A position is its average price; each add isn't judged on its own.\n"
        "- **Closing line value.** Whether the price moved your way after you bought needs market price "
        "history, which the export doesn't have."
    )
