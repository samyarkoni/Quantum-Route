"""Walk-forward score prediction and pregame-adjusted in-game win probability.

This evaluation intentionally excludes betting lines. Team score ratings use only
completed games before each test week's first kickoff; live-game labels are evaluated
out of season and each game receives equal total weight.
"""
from __future__ import annotations

import re
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
TEAM_CODE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TEST_SEASONS = range(2015, 2026)


def load_schedule():
    schedule = nfl.load_schedules(list(range(1999, 2027))).select([
        "game_id", "season", "game_type", "week", "gameday", "home_team", "away_team",
        "home_score", "away_score", "result", "location", "home_qb_id", "away_qb_id",
    ]).to_pandas()
    schedule = schedule[schedule.game_type.eq("REG")].copy()
    schedule["gameday"] = pd.to_datetime(schedule.gameday)
    for side in ("home_team", "away_team"):
        schedule[side] = schedule[side].replace(TEAM_CODE)
    schedule["played"] = schedule.result.notna() & schedule.home_score.notna() & schedule.away_score.notna()
    schedule["neutral"] = schedule.location.eq("Neutral").astype(float)
    schedule["week_start"] = schedule.groupby(["season", "week"]).gameday.transform("min")
    return schedule.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def fit_score_ratings(history, teams, cutoff, half_life_weeks=20.0, window_days=1460, ridge=3.0):
    """Fit weighted opponent-adjusted points-for/points-allowed team ratings."""
    history = history[
        history.played
        & (history.gameday < cutoff)
        & (history.gameday >= cutoff - pd.Timedelta(days=window_days))
    ]
    n_teams = len(teams)
    n_features = 2 + 2 * n_teams
    if len(history) < 30:
        return None

    team_ix = {team: i for i, team in enumerate(teams)}
    n = 2 * len(history)
    X = np.zeros((n, n_features), dtype=float)
    y = np.empty(n, dtype=float)
    dates = history.gameday.to_numpy(dtype="datetime64[D]")
    cutoff_day = np.datetime64(cutoff.date(), "D")
    weights = np.exp(
        -np.log(2) * ((cutoff_day - dates).astype("timedelta64[D]").astype(float) / 7.0)
        / half_life_weeks
    )

    for j, game in enumerate(history.itertuples(index=False)):
        home_row, away_row = 2 * j, 2 * j + 1
        home = team_ix[game.home_team]
        away = team_ix[game.away_team]
        X[home_row, 0] = X[away_row, 0] = 1.0
        X[home_row, 1] = 1.0 - game.neutral
        X[home_row, 2 + home] = 1.0
        X[home_row, 2 + n_teams + away] = 1.0
        X[away_row, 2 + away] = 1.0
        X[away_row, 2 + n_teams + home] = 1.0
        y[home_row], y[away_row] = game.home_score, game.away_score

    row_weights = np.repeat(weights, 2)
    gram = X.T @ (X * row_weights[:, None])
    gram[np.arange(2, n_features), np.arange(2, n_features)] += ridge
    coef = np.linalg.solve(gram, X.T @ (row_weights * y))
    return coef, team_ix


def load_epa_inputs(schedule):
    """Aggregate nflreadpy's supplied EPA by regular-season team game and QB dropback."""
    schedule_dates = schedule[["game_id", "gameday"]].drop_duplicates("game_id")
    schedule_dates = schedule_dates.merge(
        schedule[["game_id", "home_team", "away_team"]].drop_duplicates("game_id"),
        on="game_id",
        validate="one_to_one",
    )
    team_games = []
    qb_games = []
    pbp_columns = [
        "game_id", "season_type", "play_type", "posteam", "epa", "play_deleted",
        "qb_dropback", "passer_player_id",
    ]
    for season in range(1999, 2026):
        pbp = nfl.load_pbp(season).select(pbp_columns).to_pandas()
        pbp = pbp[pbp.season_type.eq("REG") & pbp.epa.notna()]
        if "play_deleted" in pbp:
            pbp = pbp[pbp.play_deleted.ne(1)]
        offensive = pbp[
            pbp.play_type.isin(["pass", "run"])
            & pbp.posteam.notna()
        ]
        if not offensive.empty:
            team = offensive.groupby(["game_id", "posteam"], as_index=False).agg(
                team_epa_sum=("epa", "sum"),
                team_plays=("epa", "count"),
            ).rename(columns={"posteam": "team"})
            team["team_epa_per_play"] = team.team_epa_sum / team.team_plays
            team_games.append(team)

        qb = offensive[
            offensive.qb_dropback.eq(1)
            & offensive.passer_player_id.notna()
        ]
        if qb.empty:
            continue
        grouped = qb.groupby(["game_id", "passer_player_id"], as_index=False).agg(
            qb_epa_sum=("epa", "sum"),
            qb_dropbacks=("qb_dropback", "sum"),
        )
        qb_games.append(grouped)
        print(f"Loaded NFL EPA for {season}: {len(grouped):,} QB-game rows", flush=True)

    if not team_games or not qb_games:
        raise ValueError("nflreadpy returned no regular-season team or QB EPA.")
    team_epa = pd.concat(team_games, ignore_index=True)
    team_epa = team_epa.merge(schedule_dates, on="game_id", how="inner", validate="many_to_one")
    team_epa["team"] = team_epa.team.replace(TEAM_CODE)
    team_epa = team_epa.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)
    qb_history = pd.concat(qb_games, ignore_index=True)
    qb_history = qb_history.merge(
        schedule_dates[["game_id", "gameday"]], on="game_id", how="inner", validate="many_to_one"
    )
    qb_history = qb_history.sort_values(["passer_player_id", "gameday"]).reset_index(drop=True)
    return team_epa, qb_history


def fit_team_epa_ratings(schedule, team_epa, teams, cutoff, half_life_weeks=20.0,
                         window_days=1460, ridge=3.0):
    """Fit opponent-adjusted team offense and defense efficiency from prior EPA/play."""
    history = schedule[
        schedule.played
        & (schedule.gameday < cutoff)
        & (schedule.gameday >= cutoff - pd.Timedelta(days=window_days))
    ][["game_id", "gameday", "home_team", "away_team", "neutral"]]
    if len(history) < 30:
        return None

    home_epa = team_epa[["game_id", "team", "team_epa_per_play", "team_plays"]].rename(
        columns={
            "team": "home_team",
            "team_epa_per_play": "home_epa",
            "team_plays": "home_plays",
        }
    )
    away_epa = team_epa[["game_id", "team", "team_epa_per_play", "team_plays"]].rename(
        columns={
            "team": "away_team",
            "team_epa_per_play": "away_epa",
            "team_plays": "away_plays",
        }
    )
    history = history.merge(home_epa, on=["game_id", "home_team"], how="inner", validate="one_to_one")
    history = history.merge(away_epa, on=["game_id", "away_team"], how="inner", validate="one_to_one")
    if len(history) < 30:
        return None

    team_ix = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    n = 2 * len(history)
    X = np.zeros((n, 2 + 2 * n_teams), dtype=float)
    y = np.empty(n, dtype=float)
    weights = np.empty(n, dtype=float)
    dates = history.gameday.to_numpy(dtype="datetime64[D]")
    cutoff_day = np.datetime64(cutoff.date(), "D")
    age_weeks = (cutoff_day - dates).astype("timedelta64[D]").astype(float) / 7.0
    recency = np.exp(-np.log(2) * age_weeks / half_life_weeks)

    for j, game in enumerate(history.itertuples(index=False)):
        home_row, away_row = 2 * j, 2 * j + 1
        home = team_ix[game.home_team]
        away = team_ix[game.away_team]
        X[home_row, 0] = X[away_row, 0] = 1.0
        X[home_row, 1] = 1.0 - game.neutral
        X[home_row, 2 + home] = 1.0
        X[home_row, 2 + n_teams + away] = 1.0
        X[away_row, 2 + away] = 1.0
        X[away_row, 2 + n_teams + home] = 1.0
        y[home_row], y[away_row] = game.home_epa, game.away_epa
        weights[home_row] = recency[j] * game.home_plays / 62.0
        weights[away_row] = recency[j] * game.away_plays / 62.0

    gram = X.T @ (X * weights[:, None])
    gram[np.arange(2, 2 + 2 * n_teams), np.arange(2, 2 + 2 * n_teams)] += ridge
    coef = np.linalg.solve(gram, X.T @ (weights * y))
    return coef, team_ix


def build_qb_epa_features(schedule, qb_games, half_life_weeks=104.0, prior_dropbacks=200.0):
    """Build leakage-free, recency-weighted QB EPA/dropback priors at each weekly cutoff."""
    cutoff_by_week = schedule.groupby(["season", "week"]).gameday.min().to_dict()
    qb_history = {
        qb_id: (
            group.gameday.to_numpy(dtype="datetime64[D]"),
            group.qb_epa_sum.to_numpy(dtype=float),
            group.qb_dropbacks.to_numpy(dtype=float),
        )
        for qb_id, group in qb_games.groupby("passer_player_id", sort=False)
    }
    ln2 = np.log(2.0)

    def rating(qb_id, cutoff):
        if pd.isna(qb_id) or qb_id not in qb_history:
            return 0.0, 0.0
        dates, epa_sums, dropbacks = qb_history[qb_id]
        cutoff_day = np.datetime64(cutoff.date(), "D")
        n = np.searchsorted(dates, cutoff_day, side="left")
        if n == 0:
            return 0.0, 0.0
        weights = np.exp(
            -ln2 * ((cutoff_day - dates[:n]).astype("timedelta64[D]").astype(float) / 7.0)
            / half_life_weeks
        )
        weighted_dropbacks = np.sum(weights * dropbacks[:n])
        weighted_epa = np.sum(weights * epa_sums[:n])
        shrunk_epa = weighted_epa / (weighted_dropbacks + prior_dropbacks)
        return float(shrunk_epa), float(weighted_dropbacks)

    rows = []
    for game in schedule.itertuples(index=False):
        cutoff = cutoff_by_week[(game.season, game.week)]
        home_epa, home_db = rating(game.home_qb_id, cutoff)
        away_epa, away_db = rating(game.away_qb_id, cutoff)
        rows.append({
            "game_id": game.game_id,
            "season": game.season,
            "week": game.week,
            "home_qb_epa_per_dropback": home_epa,
            "away_qb_epa_per_dropback": away_epa,
            "home_qb_prior_dropbacks": home_db,
            "away_qb_prior_dropbacks": away_db,
            "qb_epa_diff": home_epa - away_epa,
        })
    return pd.DataFrame(rows)


def build_pregame_score_features(schedule, qb_epa_features, team_epa):
    """Generate score forecasts for every game from information available pre-week."""
    teams = sorted(set(schedule.home_team) | set(schedule.away_team))
    snapshots = {}
    for (season, week), games in schedule.groupby(["season", "week"], sort=True):
        cutoff = games.week_start.iloc[0]
        snapshots[(season, week)] = (
            fit_score_ratings(schedule, teams, cutoff),
            fit_team_epa_ratings(schedule, team_epa, teams, cutoff),
        )

    rows = []
    n_teams = len(teams)
    for game in schedule.itertuples(index=False):
        score_fit, epa_fit = snapshots.get((game.season, game.week), (None, None))
        if score_fit is None or epa_fit is None:
            continue
        coef, team_ix = score_fit
        home, away = team_ix[game.home_team], team_ix[game.away_team]
        home_points = coef[0] + coef[1] * (1.0 - game.neutral) + coef[2 + home] + coef[2 + n_teams + away]
        away_points = coef[0] + coef[2 + away] + coef[2 + n_teams + home]
        epa_coef, epa_team_ix = epa_fit
        epa_home, epa_away = epa_team_ix[game.home_team], epa_team_ix[game.away_team]
        epa_home_per_play = (
            epa_coef[0] + epa_coef[1] * (1.0 - game.neutral)
            + epa_coef[2 + epa_home] + epa_coef[2 + n_teams + epa_away]
        )
        epa_away_per_play = (
            epa_coef[0] + epa_coef[2 + epa_away] + epa_coef[2 + n_teams + epa_home]
        )
        rows.append({
            "game_id": game.game_id,
            "season": game.season,
            "week": game.week,
            "pregame_home_points": home_points,
            "pregame_away_points": away_points,
            "pregame_home_margin": home_points - away_points,
            "pregame_total": home_points + away_points,
            "pregame_home_epa_per_play": epa_home_per_play,
            "pregame_away_epa_per_play": epa_away_per_play,
        })
    return pd.DataFrame(rows).merge(
        qb_epa_features, on=["game_id", "season", "week"], how="left", validate="one_to_one"
    )


def score_metrics(actual_home, actual_away, pred_home, pred_away):
    actual_margin = actual_home - actual_away
    pred_margin = pred_home - pred_away
    actual_total = actual_home + actual_away
    pred_total = pred_home + pred_away
    return {
        "home_points_mae": mean_absolute_error(actual_home, pred_home),
        "away_points_mae": mean_absolute_error(actual_away, pred_away),
        "points_mae": mean_absolute_error(
            np.r_[actual_home, actual_away], np.r_[pred_home, pred_away]
        ),
        "points_rmse": mean_squared_error(
            np.r_[actual_home, actual_away], np.r_[pred_home, pred_away]
        ) ** 0.5,
        "margin_mae": mean_absolute_error(actual_margin, pred_margin),
        "margin_rmse": mean_squared_error(actual_margin, pred_margin) ** 0.5,
        "total_mae": mean_absolute_error(actual_total, pred_total),
        "total_rmse": mean_squared_error(actual_total, pred_total) ** 0.5,
    }


def walk_forward_score_eval(schedule, pregame):
    data = schedule.merge(pregame, on=["game_id", "season", "week"], how="inner")
    data = data[data.played & data.season.between(2002, 2025)].copy()
    data = data.sort_values(["season", "week", "gameday"])
    predictions = []

    for season in TEST_SEASONS:
        train = data[data.season < season]
        test = data[data.season == season].copy()
        if train.empty or test.empty:
            continue

        features = train[["pregame_home_points", "pregame_away_points", "neutral", "week"]]
        target = train[["home_score", "away_score"]].astype(float)
        calibrator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        calibrator.fit(features, target)
        test_features = test[["pregame_home_points", "pregame_away_points", "neutral", "week"]]
        calibrated = calibrator.predict(test_features)

        qb_features = [
            "pregame_home_points", "pregame_away_points", "neutral", "week",
            "home_qb_epa_per_dropback", "away_qb_epa_per_dropback",
            "home_qb_prior_dropbacks", "away_qb_prior_dropbacks",
        ]
        qb_calibrator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        qb_calibrator.fit(train[qb_features], target)
        calibrated_qb = qb_calibrator.predict(test[qb_features])

        team_epa_features = [
            "pregame_home_points", "pregame_away_points", "neutral", "week",
            "pregame_home_epa_per_play", "pregame_away_epa_per_play",
        ]
        team_epa_calibrator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        team_epa_calibrator.fit(train[team_epa_features], target)
        calibrated_team_epa = team_epa_calibrator.predict(test[team_epa_features])

        team_epa_qb_features = team_epa_features + [
            "home_qb_epa_per_dropback", "away_qb_epa_per_dropback",
            "home_qb_prior_dropbacks", "away_qb_prior_dropbacks",
        ]
        team_epa_qb_calibrator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        team_epa_qb_calibrator.fit(train[team_epa_qb_features], target)
        calibrated_team_epa_qb = team_epa_qb_calibrator.predict(test[team_epa_qb_features])

        epa_efficiency_features = [
            "pregame_home_epa_per_play", "pregame_away_epa_per_play", "neutral", "week",
        ]
        epa_efficiency_calibrator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        epa_efficiency_calibrator.fit(train[epa_efficiency_features], target)
        calibrated_epa_efficiency = epa_efficiency_calibrator.predict(test[epa_efficiency_features])

        epa_efficiency_qb_features = epa_efficiency_features + [
            "home_qb_epa_per_dropback", "away_qb_epa_per_dropback",
            "home_qb_prior_dropbacks", "away_qb_prior_dropbacks",
        ]
        epa_efficiency_qb_calibrator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        epa_efficiency_qb_calibrator.fit(train[epa_efficiency_qb_features], target)
        calibrated_epa_efficiency_qb = epa_efficiency_qb_calibrator.predict(
            test[epa_efficiency_qb_features]
        )

        baseline_home = train.home_score.mean()
        baseline_away = train.away_score.mean()
        test["raw_home_pred"] = test.pregame_home_points
        test["raw_away_pred"] = test.pregame_away_points
        test["cal_home_pred"] = calibrated[:, 0]
        test["cal_away_pred"] = calibrated[:, 1]
        test["qb_home_pred"] = calibrated_qb[:, 0]
        test["qb_away_pred"] = calibrated_qb[:, 1]
        test["team_epa_home_pred"] = calibrated_team_epa[:, 0]
        test["team_epa_away_pred"] = calibrated_team_epa[:, 1]
        test["team_epa_qb_home_pred"] = calibrated_team_epa_qb[:, 0]
        test["team_epa_qb_away_pred"] = calibrated_team_epa_qb[:, 1]
        test["epa_efficiency_home_pred"] = calibrated_epa_efficiency[:, 0]
        test["epa_efficiency_away_pred"] = calibrated_epa_efficiency[:, 1]
        test["epa_efficiency_qb_home_pred"] = calibrated_epa_efficiency_qb[:, 0]
        test["epa_efficiency_qb_away_pred"] = calibrated_epa_efficiency_qb[:, 1]
        test["qb_margin_adjustment"] = (
            (test.qb_home_pred - test.qb_away_pred)
            - test.pregame_home_margin
        )
        test["team_epa_margin_adjustment"] = (
            (test.epa_efficiency_home_pred - test.epa_efficiency_away_pred)
            - test.pregame_home_margin
        )
        test["team_epa_qb_margin_adjustment"] = (
            (test.epa_efficiency_qb_home_pred - test.epa_efficiency_qb_away_pred)
            - (test.epa_efficiency_home_pred - test.epa_efficiency_away_pred)
        )
        test["mean_home_pred"] = baseline_home
        test["mean_away_pred"] = baseline_away
        predictions.append(test)

    predictions = pd.concat(predictions, ignore_index=True)
    result_rows = []
    for label, home_col, away_col in (
        ("league_mean", "mean_home_pred", "mean_away_pred"),
        ("opponent_adjusted_ratings", "raw_home_pred", "raw_away_pred"),
        ("calibrated_ratings", "cal_home_pred", "cal_away_pred"),
        ("calibrated_ratings_plus_qb_epa", "qb_home_pred", "qb_away_pred"),
        ("calibrated_ratings_plus_team_epa", "team_epa_home_pred", "team_epa_away_pred"),
        (
            "calibrated_ratings_plus_team_and_qb_epa",
            "team_epa_qb_home_pred",
            "team_epa_qb_away_pred",
        ),
        ("team_epa_efficiency_only", "epa_efficiency_home_pred", "epa_efficiency_away_pred"),
        (
            "team_epa_efficiency_plus_qb_epa",
            "epa_efficiency_qb_home_pred",
            "epa_efficiency_qb_away_pred",
        ),
    ):
        metrics = score_metrics(
            predictions.home_score.to_numpy(),
            predictions.away_score.to_numpy(),
            predictions[home_col].to_numpy(),
            predictions[away_col].to_numpy(),
        )
        result_rows.append({"model": label, "n_games": len(predictions), **metrics})

    season_rows = []
    for season, group in predictions.groupby("season"):
        for label, home_col, away_col in (
            ("league_mean", "mean_home_pred", "mean_away_pred"),
            ("opponent_adjusted_ratings", "raw_home_pred", "raw_away_pred"),
            ("calibrated_ratings", "cal_home_pred", "cal_away_pred"),
            ("calibrated_ratings_plus_qb_epa", "qb_home_pred", "qb_away_pred"),
            ("calibrated_ratings_plus_team_epa", "team_epa_home_pred", "team_epa_away_pred"),
            (
                "calibrated_ratings_plus_team_and_qb_epa",
                "team_epa_qb_home_pred",
                "team_epa_qb_away_pred",
            ),
            ("team_epa_efficiency_only", "epa_efficiency_home_pred", "epa_efficiency_away_pred"),
            (
                "team_epa_efficiency_plus_qb_epa",
                "epa_efficiency_qb_home_pred",
                "epa_efficiency_qb_away_pred",
            ),
        ):
            season_rows.append({
                "season": season,
                "model": label,
                "n_games": len(group),
                **score_metrics(
                    group.home_score.to_numpy(),
                    group.away_score.to_numpy(),
                    group[home_col].to_numpy(),
                    group[away_col].to_numpy(),
                ),
            })

    predictions.to_csv(RESULTS / "score_predictions.csv", index=False)
    pd.DataFrame(result_rows).to_csv(RESULTS / "score_metrics.csv", index=False)
    season_metrics = pd.DataFrame(season_rows)
    season_metrics.to_csv(RESULTS / "score_metrics_by_season.csv", index=False)
    return predictions, pd.DataFrame(result_rows), season_metrics


def load_live_plays():
    files = sorted((REPO / "GAMES").glob("[0-9][0-9][0-9][0-9]/**/*.csv"))
    frames = []
    wanted = {
        "game_id", "play_id", "play_type", "posteam", "qtr", "time_on_clock_start",
        "home_team_score", "away_team_score", "down", "yds_to_go", "yardline_100",
    }
    for path in files:
        season_match = re.search(r"/(20\d{2})/", str(path))
        if not season_match or not 2018 <= int(season_match.group(1)) <= 2025:
            continue
        frames.append(pd.read_csv(path, usecols=lambda column: column in wanted, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"No play files for seasons 2018-2025 under {REPO / 'GAMES'}")
    return pd.concat(frames, ignore_index=True)


def build_live_frame(plays, schedule, score_predictions):
    games = schedule[["game_id", "season", "home_team", "away_team", "result"]].merge(
        score_predictions[[
            "game_id", "pregame_home_margin", "qb_margin_adjustment",
            "team_epa_margin_adjustment", "team_epa_qb_margin_adjustment",
        ]], on="game_id", how="inner"
    )
    games = games[games.season.between(2018, 2025) & games.result.notna()]
    plays = plays[plays.game_id.isin(games.game_id)].copy()
    plays = plays.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    plays["posteam"] = plays.posteam.replace(TEAM_CODE)
    plays["qtr"] = pd.to_numeric(plays.qtr, errors="coerce")
    plays["play_id"] = pd.to_numeric(plays.play_id, errors="coerce")
    plays["down"] = pd.to_numeric(plays.down, errors="coerce")
    plays["yds_to_go"] = pd.to_numeric(plays.yds_to_go, errors="coerce")
    plays["yardline_100"] = pd.to_numeric(plays.yardline_100, errors="coerce")
    for col in ("home_team_score", "away_team_score"):
        plays[col] = pd.to_numeric(plays[col], errors="coerce").fillna(0.0)
        plays[f"pre_{col}"] = plays.groupby("game_id", sort=False)[col].shift(1).fillna(0.0)

    clock_parts = plays.time_on_clock_start.astype("string").str.extract(r"^\s*(\d+):(\d{1,2})\s*$")
    quarter_seconds = pd.to_numeric(clock_parts[0], errors="coerce") * 60 + pd.to_numeric(clock_parts[1], errors="coerce")
    plays["seconds_remaining"] = (4 - plays.qtr) * 900 + quarter_seconds
    plays["home_score_diff"] = plays.pre_home_team_score - plays.pre_away_team_score
    plays["home_possession"] = plays.posteam.eq(plays.game_id.map(games.set_index("game_id").home_team)).astype(float)
    plays = plays[
        plays.play_type.isin(["pass", "run", "turnover"])
        & plays.qtr.between(1, 4)
        & plays.seconds_remaining.gt(0)
        & plays.down.between(1, 4)
        & plays.yardline_100.between(1, 99)
    ].copy()

    plays = plays.merge(games, on="game_id", how="inner", suffixes=("", "_schedule"))
    plays = plays[plays.result.ne(0)].copy()
    plays["home_win"] = (plays.result > 0).astype(int)
    plays["remaining_fraction"] = plays.seconds_remaining.clip(0, 3600) / 3600.0
    plays["score_clock"] = plays.home_score_diff * plays.remaining_fraction
    plays["prior_clock"] = plays.pregame_home_margin * plays.remaining_fraction
    plays["qb_epa_clock"] = plays.qb_margin_adjustment * plays.remaining_fraction
    plays["team_epa_prior_margin"] = (
        plays.pregame_home_margin + plays.team_epa_margin_adjustment
    )
    plays["team_epa_prior_clock"] = plays.team_epa_prior_margin * plays.remaining_fraction
    plays["team_epa_qb_prior_margin"] = (
        plays.team_epa_prior_margin + plays.team_epa_qb_margin_adjustment
    )
    plays["team_epa_qb_prior_clock"] = (
        plays.team_epa_qb_prior_margin * plays.remaining_fraction
    )
    feature_columns = [
        "home_score_diff", "seconds_remaining", "remaining_fraction", "home_possession",
        "qtr", "down", "yds_to_go", "yardline_100", "score_clock",
    ]
    prior_columns = feature_columns + ["pregame_home_margin", "prior_clock"]
    prior_qb_columns = prior_columns + ["qb_margin_adjustment", "qb_epa_clock"]
    team_epa_columns = feature_columns + ["team_epa_prior_margin", "team_epa_prior_clock"]
    team_epa_qb_columns = team_epa_columns + [
        "team_epa_qb_prior_margin", "team_epa_qb_prior_clock",
    ]
    plays["game_weight"] = 1.0 / plays.groupby("game_id").game_id.transform("size")
    return (
        plays, feature_columns, prior_columns, prior_qb_columns,
        team_epa_columns, team_epa_qb_columns,
    )


def weighted_probability_metrics(actual, predicted, weights):
    return {
        "logloss": log_loss(actual, np.clip(predicted, 1e-6, 1 - 1e-6), sample_weight=weights),
        "brier": brier_score_loss(actual, predicted, sample_weight=weights),
        "accuracy": np.average((predicted >= 0.5) == actual, weights=weights),
        "n_games": int(round(np.sum(weights))),
    }


def paired_season_test(season_metrics, candidate, reference, metric, family, seed=17, reps=10000):
    paired = season_metrics.pivot(index="season", columns="model", values=metric).dropna()
    difference = (paired[candidate] - paired[reference]).to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(difference, size=(reps, len(difference)), replace=True).mean(axis=1)
    return {
        "family": family,
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "n_seasons": len(difference),
        "mean_candidate_minus_reference": difference.mean(),
        "cluster_bootstrap_ci_low": np.quantile(bootstrap, 0.025),
        "cluster_bootstrap_ci_high": np.quantile(bootstrap, 0.975),
        "season_level_ttest_p": ttest_1samp(difference, 0.0).pvalue,
    }


def walk_forward_live_eval(schedule, score_predictions):
    (
        plays, base_features, prior_features, prior_qb_features,
        team_epa_features, team_epa_qb_features,
    ) = build_live_frame(
        load_live_plays(), schedule, score_predictions
    )
    predictions = []
    for season in range(2020, 2026):
        train = plays[plays.season < season]
        test = plays[plays.season == season].copy()
        if train.empty or test.empty:
            continue

        for label, columns in (
            ("state_only", base_features),
            ("state_plus_pregame_strength", prior_features),
            ("state_plus_pregame_strength_and_qb_epa", prior_qb_features),
            ("state_plus_team_epa_efficiency", team_epa_features),
            ("state_plus_team_epa_efficiency_and_qb_epa", team_epa_qb_features),
        ):
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                l2_regularization=10.0,
                random_state=42,
            )
            model.fit(train[columns], train.home_win, sample_weight=train.game_weight)
            test[f"p_{label}"] = model.predict_proba(test[columns])[:, 1]
        predictions.append(test)

    predictions = pd.concat(predictions, ignore_index=True)
    rows = []
    labels = (
        "state_only",
        "state_plus_pregame_strength",
        "state_plus_pregame_strength_and_qb_epa",
        "state_plus_team_epa_efficiency",
        "state_plus_team_epa_efficiency_and_qb_epa",
    )
    for label in labels:
        rows.append({
            "model": label,
            **weighted_probability_metrics(
                predictions.home_win.to_numpy(),
                predictions[f"p_{label}"].to_numpy(),
                predictions.game_weight.to_numpy(),
            ),
        })
    summary = pd.DataFrame(rows)
    season_rows = []
    for season, group in predictions.groupby("season"):
        for label in labels:
            season_rows.append({
                "season": season,
                "model": label,
                **weighted_probability_metrics(
                    group.home_win.to_numpy(),
                    group[f"p_{label}"].to_numpy(),
                    group.game_weight.to_numpy(),
                ),
            })
    summary.to_csv(RESULTS / "live_win_probability_metrics.csv", index=False)
    pd.DataFrame(season_rows).to_csv(RESULTS / "live_win_probability_by_season.csv", index=False)
    return summary, pd.DataFrame(season_rows)


def main():
    schedule = load_schedule()
    team_epa, qb_games = load_epa_inputs(schedule)
    qb_epa_features = build_qb_epa_features(schedule, qb_games)
    pregame = build_pregame_score_features(schedule, qb_epa_features, team_epa)
    score_predictions, score_summary, score_seasons = walk_forward_score_eval(schedule, pregame)
    live_summary, live_seasons = walk_forward_live_eval(schedule, score_predictions)
    score_tests = [
        paired_season_test(
            score_seasons, "opponent_adjusted_ratings", "league_mean", metric, "score",
        )
        for metric in ("points_mae", "points_rmse", "margin_mae", "margin_rmse", "total_mae", "total_rmse")
    ]
    score_tests.extend(
        paired_season_test(
            score_seasons, "calibrated_ratings_plus_qb_epa", "calibrated_ratings", metric, "score_qb_increment",
        )
        for metric in ("points_mae", "points_rmse", "margin_mae", "margin_rmse", "total_mae", "total_rmse")
    )
    score_tests.extend(
        paired_season_test(
            score_seasons, "calibrated_ratings_plus_team_epa", "calibrated_ratings",
            metric, "score_team_epa_increment",
        )
        for metric in ("points_mae", "points_rmse", "margin_mae", "margin_rmse", "total_mae", "total_rmse")
    )
    score_tests.extend(
        paired_season_test(
            score_seasons, "calibrated_ratings_plus_team_and_qb_epa",
            "calibrated_ratings_plus_team_epa", metric, "score_team_qb_epa_increment",
        )
        for metric in ("points_mae", "points_rmse", "margin_mae", "margin_rmse", "total_mae", "total_rmse")
    )
    score_tests.extend(
        paired_season_test(
            score_seasons, "team_epa_efficiency_only", "calibrated_ratings",
            metric, "score_team_epa_only",
        )
        for metric in ("points_mae", "points_rmse", "margin_mae", "margin_rmse", "total_mae", "total_rmse")
    )
    score_tests.extend(
        paired_season_test(
            score_seasons, "team_epa_efficiency_plus_qb_epa", "team_epa_efficiency_only",
            metric, "score_team_epa_plus_qb",
        )
        for metric in ("points_mae", "points_rmse", "margin_mae", "margin_rmse", "total_mae", "total_rmse")
    )
    score_tests.extend(
        paired_season_test(
            score_seasons, "calibrated_ratings_plus_team_and_qb_epa",
            "calibrated_ratings_plus_qb_epa", metric, "score_team_epa_on_top_of_qb",
        )
        for metric in ("points_mae", "points_rmse", "margin_mae", "margin_rmse", "total_mae", "total_rmse")
    )
    live_tests = [
        paired_season_test(
            live_seasons, "state_plus_pregame_strength", "state_only", metric, "live_win_probability",
        )
        for metric in ("logloss", "brier", "accuracy")
    ]
    live_tests.extend(
        paired_season_test(
            live_seasons,
            "state_plus_pregame_strength_and_qb_epa",
            "state_plus_pregame_strength",
            metric,
            "live_qb_epa_increment",
        )
        for metric in ("logloss", "brier", "accuracy")
    )
    live_tests.extend(
        paired_season_test(
            live_seasons,
            "state_plus_team_epa_efficiency",
            "state_plus_pregame_strength",
            metric,
            "live_team_epa_increment",
        )
        for metric in ("logloss", "brier", "accuracy")
    )
    live_tests.extend(
        paired_season_test(
            live_seasons,
            "state_plus_team_epa_efficiency_and_qb_epa",
            "state_plus_team_epa_efficiency",
            metric,
            "live_team_and_qb_epa_increment",
        )
        for metric in ("logloss", "brier", "accuracy")
    )
    tests = pd.DataFrame(score_tests + live_tests)
    tests.to_csv(RESULTS / "predictability_model_tests.csv", index=False)
    print("=== Walk-forward score prediction (2015-2025; no market inputs) ===")
    print(score_summary.round(3).to_string(index=False))
    print("\n=== Walk-forward in-game home win probability (2020-2025; each game equally weighted) ===")
    print(live_summary.round(4).to_string(index=False))
    print("\nScore metrics by season:\n", score_seasons.round(3).to_string(index=False))
    print("\nLive probability metrics by season:\n", live_seasons.round(4).to_string(index=False))
    print("\nPaired season-level tests (negative deltas favor candidate except accuracy):\n",
          tests.round(4).to_string(index=False))
    print(f"\nWrote predictions and metrics to {RESULTS}")


if __name__ == "__main__":
    main()
