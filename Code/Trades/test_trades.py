"""python -m pytest Code/Trades"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trades

SAMPLE = Path(__file__).resolve().parent / "sample_realized_pnl.csv"


@pytest.fixture
def sample():
    df = trades.read_export(SAMPLE)
    df.insert(0, "position_id", [trades._position_id(r) for r in df.itertuples()])
    return df


def test_parse_kalshi_and_forecastex():
    p = trades.parse_symbol("KXNFLSPREAD-25NOV23PITCHI-CHI7")
    assert (p["market"], p["away"], p["home"], p["selection"], p["side"], p["line"]) == \
        ("spread", "PIT", "CHI", "CHI", "home", 7.5)
    assert p["game_date"] == date(2025, 11, 23)
    p = trades.parse_symbol("KXNFLGAME-25NOV23JACARI-JAC")
    assert (p["away"], p["selection"], p["side"]) == ("JAX", "JAX", "away")
    p = trades.parse_symbol("KXNFLGAME-25DEC29LAATL-LA")  # LA + ATL, not LAA + TL
    assert (p["away"], p["home"]) == ("LA", "ATL")
    p = trades.parse_symbol("NFLGW_121425LACKC_LAC")
    assert (p["market"], p["away"], p["home"], p["side"], p["game_date"]) == \
        ("moneyline", "LAC", "KC", "away", date(2025, 12, 14))
    assert trades.parse_symbol("KXNFLTOTAL-25DEC08PHILAC-41")["line"] == 41.5
    assert trades.parse_symbol("KXNFLWINS-PIT-25B-T8")["market"] == "season_wins"
    assert trades.parse_symbol("KXNFLANYTD-25DEC20PHIWAS-PHIABROWN11")["market"] == "prop"
    assert trades.parse_symbol("KXMARMAD-25R64G5-MISS")["sport"] == "College basketball"


def test_contracts_and_prices_from_fees(sample):
    df = trades.enrich(sample).set_index("symbol")
    nyg = df.loc["KXNFLGAME-25OCT09PHINYG-NYG"]  # $43.20 for 60, settled at $60
    assert (nyg["status"], nyg["contracts"], round(nyg["entry_price"], 2)) == ("won", 60, 0.72)
    la = df.loc["KXNFLGAME-25DEC29LAATL-LA"]
    assert (la["status"], la["contracts"]) == ("lost", 53)
    gb = df.loc["KXNFLSPREAD-25NOV27GBDET-GB7"]  # 40 contract-sides: bought 20 at 31c, sold at 41c
    assert (gb["status"], gb["contracts"], round(gb["exit_price"], 2)) == ("exited", 20, 0.41)
    assert not df.loc["KXNFLSPREAD-25NOV23PITCHI-CHI7", "priced"]  # partial exit, can't be inferred


def test_nfl_totals_match_the_export(sample):
    result = trades.analyze(sample, n_sims=2000)
    s = result["summary"]
    assert round(s["net_pnl"] + s["unpriced_net_pnl"], 2) == 124.81
    assert s["positions"] + s["unpriced_positions"] == 49
    assert 0 < s["p_value"] < 1


def test_same_game_contracts_share_a_draw():
    df = pd.DataFrame({
        "draw_group": ["g", "g", "g"], "side": ["home", "away", "home"],
        "entry_price": [0.6, 0.4, 0.3], "contracts": [1.0, 1.0, 1.0],
        "cost": [0.6, 0.4, 0.3], "fees": [0.0, 0.0, 0.0],
    })
    sims = trades.simulate_null(df, n_sims=5000)
    home_won, away_won, spread_won = sims[:, 0] > 0, sims[:, 1] > 0, sims[:, 2] > 0
    assert not (home_won & away_won).any() and (home_won | away_won).all()
    assert not (spread_won & ~home_won).any()  # covering a spread implies winning
    assert abs(home_won.mean() - 0.6) < 0.03


def test_null_has_zero_edge_before_fees(sample):
    df = trades.enrich(sample)
    df = df[df["priced"]].reset_index(drop=True)
    sims = trades.simulate_null(df, n_sims=40000)
    assert abs(sims.sum(axis=1).mean() + df["fees"].sum()) < 2.0


def test_import_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(trades, "DB_PATH", tmp_path / "qr.duckdb")
    assert trades.import_export(SAMPLE) == (64, 64)
    assert trades.import_export(SAMPLE) == (64, 0)
    assert len(trades.load_positions()) == 64


def test_model_probabilities_and_outcomes_join(sample):
    schedule = pd.DataFrame({
        "game_id": ["2025_12_GB_DET", "2025_13_HOU_IND"], "season": [2025, 2025], "week": [13, 13],
        "gameday": [date(2025, 11, 27), date(2025, 11, 30)], "away_team": ["GB", "HOU"],
        "home_team": ["DET", "IND"], "result": [7.0, 4.0],  # home minus away: DET by 7, IND by 4
    })
    predictions = pd.DataFrame({
        "game_id": ["2025_12_GB_DET", "2025_13_HOU_IND"], "season": [2025, 2025],
        "home_team": ["DET", "IND"], "away_team": ["GB", "HOU"], "margin": [3.0, 0.0], "margin_sd": [13.0, 13.0],
    })
    result = trades.analyze(sample, n_sims=1000, schedule=schedule, predictions=predictions)
    df = result["positions"].set_index("symbol")
    ind = df.loc["NFLGW_113025HOUIND_IND"]  # sold early; the schedule says IND won
    assert ind["outcome"] == 1.0 and ind["model_prob"] == pytest.approx(0.5)
    gb = df.loc["KXNFLSPREAD-25NOV27GBDET-GB7"]  # GB by 7.5+ when the model has DET by 3
    assert gb["outcome"] == 0.0
    assert gb["model_prob"] == pytest.approx(1 - trades._norm_cdf((7.5 + 3) / 13.0)[()], abs=1e-6)
    assert result["summary"]["model_matched"] == 3
    assert len(result["by_gap"]) > 0
