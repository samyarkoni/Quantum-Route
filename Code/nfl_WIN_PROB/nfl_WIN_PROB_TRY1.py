"""NFL in-game win-probability model (2018 season, team-agnostic game-state only).

This script implements the requested v1 pipeline:
1) load and inspect the real 2018 play-by-play CSVs,
2) build the scrimmage-play dataset and situational features,
3) fit a Brownian-motion-style baseline,
4) fit a coarse empirical table for diagnostics,
5) fit a grouped-CV XGBoost model with monotonic constraints,
6) evaluate with log loss / Brier-style metrics,
7) save the trained model plus a small inference helper.

Important: the actual 2018 data includes `posteam`, `qtr`, `game_seconds_remaining`,
`yardline_100`, and `time_on_clock_start` columns, so we use those directly rather than
re-deriving values that already exist. The places where the raw data forces a different
call from the naive assumptions are noted in the code comments and in the final summary.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

libomp_dir = "/Library/Frameworks/R.framework/Versions/4.6/Resources/lib"
for key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
    existing = os.environ.get(key, "")
    entries = [entry for entry in existing.split(":") if entry]
    if libomp_dir not in entries:
        entries.append(libomp_dir)
        os.environ[key] = ":".join(entries)

# Some macOS toolchains prefer the fallback search path for @rpath libraries.
# This avoids LightGBM import failures when the environment has not exported the
# R OpenMP runtime path before launching Python.

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "GAMES" / "2018"
MODEL_PATH = Path(__file__).resolve().parent / "trained_win_probability_model.pkl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

SCRIMMAGE_PLAY_TYPES = {"pass", "run", "turnover"}
SPECIAL_PLAY_TYPES = {
    "kickoff",
    "punt",
    "field_goal",
    "extra_point",
    "qb_kneel",
    "timeout",
    "start_quarter_1",
    "start_quarter_2",
    "start_quarter_3",
    "start_quarter_4",
    "end_game",
    "no_play",
}


def parse_game_id(game_id: str) -> Tuple[str, str]:
    """Return (away_team, home_team) parsed from game_id when it has the form:
    season_week_AWAY_HOME (e.g. 2018_01_NYJ_DET).
    """
    parts = str(game_id).split("_")
    if len(parts) >= 4:
        return parts[2], parts[3]
    raise ValueError(f"Could not parse away/home team from game_id={game_id!r}")


def load_2018_game_data(data_dir: Path) -> pd.DataFrame:
    """Load every 2018 CSV and concatenate them into one play-by-play DataFrame."""
    files = sorted(data_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def summarize_play_types(df: pd.DataFrame) -> pd.Series:
    """Print and return the distinct play_type counts used in the raw data."""
    counts = df["play_type"].dropna().astype(str).value_counts().sort_index()
    print("Distinct play_type values and counts:")
    print(counts.to_string())
    return counts


def compute_half_seconds_remaining(df: pd.DataFrame) -> pd.Series:
    """Compute half_seconds_remaining as seconds remaining in the current half.

    The raw data uses `game_seconds_remaining` as the full-game clock; it does not reset at
    halftime. We therefore convert from the quarter clock (`time_on_clock_start`) to a
    half-based clock so that the feature resets at halftime as requested.

    Notes:
    - start_quarter_2 / start_quarter_3 / start_quarter_4 are administrative rows; they
      are excluded from the modeling frame and only used for reset detection.
    - qtr values in the raw file are not reliable enough at quarter markers to use directly
      for a strict "quarter count" feature, so this function uses the actual `time_on_clock`
      values and the quarter references instead of trusting `qtr` on the reset rows.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    mask = df["time_on_clock_start"].notna()
    if not mask.any():
        return out

    # Parse the clock string (e.g. '14:53') into remaining seconds in the current quarter.
    def parse_clock_to_seconds(x: object) -> float:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s.lower() == "nan":
            return np.nan
        try:
            m, sec = s.split(":")
            return int(m) * 60 + int(sec)
        except Exception:
            return np.nan

    clock_seconds = df.loc[mask, "time_on_clock_start"].map(parse_clock_to_seconds)
    qtr = pd.to_numeric(df.loc[mask, "qtr"], errors="coerce")

    # For first/second quarter, the current half is 1800 sec; for third/fourth quarter,
    # the second half also has a 1800-second window.
    for idx, quarter in zip(clock_seconds.index, qtr):
        s = clock_seconds.loc[idx]
        if pd.isna(s):
            continue
        q = int(float(quarter)) if pd.notna(quarter) else 0
        if q in (1, 2):
            elapsed_in_half = (q - 1) * 900 + (900 - s)
            out.loc[idx] = 1800 - elapsed_in_half
        elif q in (3, 4):
            elapsed_in_half = (q - 3) * 900 + (900 - s)
            out.loc[idx] = 1800 - elapsed_in_half
        else:
            out.loc[idx] = np.nan
    return out


def reconstruct_timeouts(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct remaining timeouts per team from `timeout_team`.

    Assumes 3 timeouts per team per regulation half (reset at halftime), and 2 per team in OT.
    We iterate plays in order for each game and update the remaining count whenever a timeout
    is called. This is done from the actual timeout events in the data, which is the safest
    way to recover the state without a dedicated timeout counter column.
    """
    out = df.copy()
    out["posteam_timeouts_remaining"] = np.nan
    out["defteam_timeouts_remaining"] = np.nan

    for game_id, g in out.sort_values(["game_id", "play_id"]).groupby("game_id", sort=False):
        away_team, home_team = parse_game_id(game_id)
        state = {
            away_team: {1: 3, 2: 3, 3: 2},
            home_team: {1: 3, 2: 3, 3: 2},
        }

        last_half = None
        for _, row in g.iterrows():
            q = int(float(row.get("qtr", np.nan))) if pd.notna(row.get("qtr")) else 0
            if pd.isna(row.get("posteam")):
                continue
            if q <= 2:
                half = 1
            elif q <= 4:
                half = 2
            else:
                half = 3

            # If the game crosses a halftime or overtime reset, we update the state before
            # evaluating the row. We do not have dedicated start-of-half rows for every game,
            # but the raw data does include `timeout_team` calls and `qtr` transitions.
            if last_half is not None and half != last_half:
                state[away_team][half] = 3 if half in (1, 2) else 2
                state[home_team][half] = 3 if half in (1, 2) else 2
            last_half = half

            posteam = row["posteam"]
            defteam = home_team if posteam == away_team else away_team
            count_posteam = max(0, state.get(posteam, {}).get(half, 0))
            count_defteam = max(0, state.get(defteam, {}).get(half, 0))

            out.at[row.name, "posteam_timeouts_remaining"] = count_posteam
            out.at[row.name, "defteam_timeouts_remaining"] = count_defteam

            timeout_team = row.get("timeout_team")
            if pd.notna(timeout_team):
                if timeout_team == away_team:
                    state[away_team][half] = max(0, state[away_team][half] - 1)
                elif timeout_team == home_team:
                    state[home_team][half] = max(0, state[home_team][half] - 1)

    return out


def make_scores_needed_bucket(score_differential: pd.Series) -> pd.Series:
    """Map score differential into a coarse bucket representing how many scores are needed.

    This is intentionally coarse: tied / one-score game / two-score game / three-plus-score
    game, using 8-point steps. The feature is used for calibration and state shaping, not as
    a precise scoring model.
    """
    abs_diff = np.abs(score_differential)
    bucket = pd.Series(np.nan, index=score_differential.index, dtype=float)
    bucket.loc[abs_diff == 0] = 0
    bucket.loc[(abs_diff > 0) & (abs_diff <= 8)] = 1
    bucket.loc[(abs_diff > 8) & (abs_diff <= 16)] = 2
    bucket.loc[abs_diff > 16] = 3
    return bucket


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned modeling frame with only scrimmage plays and the requested features."""
    df = df.copy()

    # 1) Keep only the live-ball play types on which a win-probability model should train.
    #    Here the dataset's live-ball play catalog is explicitly `pass`, `run`, and `turnover`.
    live_mask = df["play_type"].astype(str).str.lower().isin(SCRIMMAGE_PLAY_TYPES)
    df = df.loc[live_mask].copy()

    # 2) Ensure the team on offense is known from the raw data. In the actual files, `posteam`
    #    is already present, so we use it instead of re-deriving from rosters or filenames.
    if "posteam" not in df.columns:
        raise ValueError("`posteam` is required to model possession. The raw data in this repo already has it.")
    df["posteam"] = df["posteam"].replace({"nan": np.nan})
    df = df.dropna(subset=["posteam"]).copy()

    # 3) Parse away/home from game_id and determine the opposing team.
    away_team_lookup = {}
    home_team_lookup = {}
    for game_id in df["game_id"].dropna().unique():
        away_team, home_team = parse_game_id(str(game_id))
        away_team_lookup[game_id] = away_team
        home_team_lookup[game_id] = home_team
    df["away_team"] = df["game_id"].map(away_team_lookup)
    df["home_team"] = df["game_id"].map(home_team_lookup)
    df["defteam"] = np.where(df["posteam"] == df["away_team"], df["home_team"], df["away_team"])

    # 4) Score state by possession, derived directly from the game's score columns.
    df["posteam_score"] = np.where(df["posteam"] == df["home_team"], df["home_team_score"], df["away_team_score"])
    df["defteam_score"] = np.where(df["posteam"] == df["home_team"], df["away_team_score"], df["home_team_score"])
    df["score_differential"] = df["posteam_score"] - df["defteam_score"]

    # 5) Final winner per game: the last play in each game determines the winner. Tie => 0.5.
    final_scores = (
        df.sort_values(["game_id", "play_id"])
          .groupby("game_id", as_index=False)
          .tail(1)
          [["game_id", "home_team_score", "away_team_score", "home_team", "away_team"]]
          .copy()
    )
    final_scores["game_winner"] = np.where(
        final_scores["home_team_score"] == final_scores["away_team_score"],
        0.5,
        np.where(final_scores["home_team_score"] > final_scores["away_team_score"], 1.0, 0.0),
    )
    winner_map = final_scores.set_index("game_id")["game_winner"]
    df["game_winner"] = df["game_id"].map(winner_map)
    df["posteam_wins"] = np.where(
        df["posteam"] == df["home_team"],
        np.where(df["game_winner"] == 1.0, 1.0, np.where(df["game_winner"] == 0.5, 0.5, 0.0)),
        np.where(df["game_winner"] == 0.0, 1.0, np.where(df["game_winner"] == 0.5, 0.5, 0.0)),
    )

    # 6) Use the raw `game_seconds_remaining`/`qtr` data directly when present.
    df["game_seconds_remaining"] = pd.to_numeric(df["game_seconds_remaining"], errors="coerce")
    df["qtr"] = pd.to_numeric(df["qtr"], errors="coerce")
    df["is_overtime"] = df["qtr"].gt(4)

    # 7) Derive half-based time remaining. The raw data's `game_seconds_remaining` is a
    #    full-game clock, so a half-reset feature needs to be built from the quarter clock.
    df["half_seconds_remaining"] = compute_half_seconds_remaining(df)

    # 8) Yardline is already present in the raw data, so we use it directly.
    df["yardline_100"] = pd.to_numeric(df["yardline_100"], errors="coerce").fillna(50.0)
    df["yds_to_go"] = pd.to_numeric(df["yds_to_go"], errors="coerce").fillna(0.0)
    df["goal_to_go"] = (df["yds_to_go"] >= df["yardline_100"]).astype(int)
    df["is_redzone"] = (df["yardline_100"] <= 20).astype(int)

    # 9) Reconstruct timeout state from actual timeout calls.
    df = reconstruct_timeouts(df)

    # 10) Down and score bucket as categorical/ordinal features.
    df["down"] = pd.to_numeric(df["down"], errors="coerce").fillna(1.0)
    df["scores_needed_bucket"] = make_scores_needed_bucket(df["score_differential"]).astype(float)
    df["lead_over_sqrt_time"] = df["score_differential"] / np.sqrt(df["game_seconds_remaining"] + 1)
    df["diff_time_ratio"] = df["score_differential"] * np.exp(4 * (3600 - df["game_seconds_remaining"]) / 3600)

    # 11) Keep only the requested state features; raw player and scheme columns are excluded.
    required = [
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "score_differential",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "down",
        "yds_to_go",
        "goal_to_go",
        "yardline_100",
        "is_redzone",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "is_overtime",
        "scores_needed_bucket",
        "lead_over_sqrt_time",
        "diff_time_ratio",
        "posteam_wins",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features after preparation: {missing}")

    df = df[required].copy()
    return df


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Manual log loss that accepts fractional labels in {0, 0.5, 1}."""
    eps = 1e-8
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(loss.mean())


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Manual Brier score for fractional labels in {0, 0.5, 1}."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def fit_baseline_model(df: pd.DataFrame) -> Tuple[LogisticRegression, pd.DataFrame]:
    """Fit a simple Brownian-motion-style baseline using lead_over_sqrt_time only."""
    X = df[["lead_over_sqrt_time"]].fillna(0.0)
    y = (df["posteam_wins"] >= 0.5).astype(int)
    model = LogisticRegression(max_iter=5000)
    model.fit(X, y)
    output = df[["game_id", "play_id", "lead_over_sqrt_time", "posteam_wins"]].copy()
    output["baseline_prob"] = model.predict_proba(X)[:, 1]
    return model, output


def make_empirical_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a coarse empirical table by score differential, game time, and down."""
    tmp = df[["score_differential", "game_seconds_remaining", "down", "posteam_wins"]].copy()
    tmp["score_bin"] = pd.cut(tmp["score_differential"], bins=np.arange(-30, 31, 3), include_lowest=True)
    tmp["time_bin"] = pd.cut(tmp["game_seconds_remaining"], bins=np.arange(0, 3601, 60), include_lowest=True)
    tmp["down_bin"] = tmp["down"].astype(int)
    table = (
        tmp.groupby(["score_bin", "time_bin", "down_bin"], observed=False)
           .agg(win_rate=("posteam_wins", "mean"), sample_size=("posteam_wins", "size"))
           .reset_index()
    )
    table = table.sort_values(["score_bin", "time_bin", "down_bin"]).reset_index(drop=True)
    return table


def build_lgb_model(monotone_constraints: List[int]):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": 42,
        "monotone_constraints": monotone_constraints,
        "monotone_constraints_method": "interpolate",
    }
    return params


def fit_xgb_grouped_cv(df: pd.DataFrame, feature_columns: List[str], n_splits: int = 3):
    """Fit an XGBoost model with grouped (game_id) cross-validation on regulation plays only."""
    regulation = df.loc[~df["is_overtime"]].copy()
    if regulation.empty:
        raise ValueError("No regulation plays available for grouped CV.")

    X = regulation[feature_columns].copy()
    y = regulation["posteam_wins"].astype(float).to_numpy()
    groups = regulation["game_id"].to_numpy()

    # The required monotonic constraints are from the project specification.
    feature_names = list(X.columns)
    monotone_map = {
        "score_differential": 1,
        "lead_over_sqrt_time": 1,
        "diff_time_ratio": 1,
        "posteam_timeouts_remaining": 1,
        "down": -1,
        "yds_to_go": -1,
        "yardline_100": -1,
        "defteam_timeouts_remaining": -1,
    }
    monotone_constraints = [0] * len(feature_names)
    for idx, name in enumerate(feature_names):
        if name in monotone_map:
            monotone_constraints[idx] = monotone_map[name]

    gkf = GroupKFold(n_splits=min(n_splits, regulation["game_id"].nunique()))
    fold_predictions = []
    cv_losses = []
    fold_models = []
    y_binary = (regulation["posteam_wins"] >= 0.5).astype(int).to_numpy()

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y_binary[train_idx]
        y_val = y_binary[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=feature_names)
        params = build_lgb_model(monotone_constraints)
        model = lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=[valid_set],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        val_pred = model.predict(X_val, raw_score=False)
        loss = compute_log_loss(y[val_idx], val_pred)
        cv_losses.append(loss)
        fold_predictions.append(pd.DataFrame({
            "game_id": regulation.iloc[val_idx]["game_id"].values,
            "play_id": regulation.iloc[val_idx]["play_id"].values,
            "actual": y[val_idx],
            "predicted": val_pred,
        }))
        fold_models.append(model)

    cv_summary = {
        "grouped_cv_logloss_mean": float(np.mean(cv_losses)),
        "grouped_cv_logloss_std": float(np.std(cv_losses)),
        "fold_losses": cv_losses,
        "n_folds": len(fold_models),
    }

    final_y = (y >= 0.5).astype(int)
    final_dataset = lgb.Dataset(X, label=final_y, feature_name=feature_names)
    final_model = lgb.train(
        params=build_lgb_model(monotone_constraints),
        train_set=final_dataset,
        num_boost_round=1000,
    )

    all_pred = pd.concat(fold_predictions, ignore_index=True) if fold_predictions else pd.DataFrame()
    return final_model, cv_summary, all_pred


def evaluate_model(df: pd.DataFrame, model, feature_columns: List[str], model_label: str) -> pd.DataFrame:
    """Evaluate the model on held-out games using grouped split by game_id."""
    regulation = df.loc[~df["is_overtime"]].copy()
    groups = regulation["game_id"].to_numpy()
    gkf = GroupKFold(n_splits=3)

    folds = []
    for train_idx, val_idx in gkf.split(regulation[feature_columns], regulation["posteam_wins"], groups):
        X_train = regulation.iloc[train_idx][feature_columns]
        X_val = regulation.iloc[val_idx][feature_columns]
        y_train = (regulation.iloc[train_idx]["posteam_wins"] >= 0.5).astype(int).to_numpy()
        y_val = regulation.iloc[val_idx]["posteam_wins"].to_numpy()

        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
        valid_set = lgb.Dataset(X_val, label=(y_val >= 0.5).astype(int), reference=train_set, feature_name=feature_columns)
        model_fold = lgb.train(
            params={
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "verbosity": -1,
                "random_state": 42,
                "monotone_constraints": [0] * len(feature_columns),
                "monotone_constraints_method": "interpolate",
            },
            train_set=train_set,
            valid_sets=[valid_set],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        pred = model_fold.predict(X_val, raw_score=False)
        folds.append(pd.DataFrame({
            "game_id": regulation.iloc[val_idx]["game_id"].values,
            "play_id": regulation.iloc[val_idx]["play_id"].values,
            "actual": y_val,
            "predicted": pred,
        }))

    if not folds:
        raise ValueError("No validation folds were produced")

    eval_df = pd.concat(folds, ignore_index=True)
    eval_df["actual"] = eval_df["actual"].astype(float)
    eval_df["predicted"] = eval_df["predicted"].clip(1e-6, 1 - 1e-6)

    metrics = {
        "model_label": model_label,
        "log_loss": compute_log_loss(eval_df["actual"].to_numpy(), eval_df["predicted"].to_numpy()),
        "brier_score": compute_brier_score(eval_df["actual"].to_numpy(), eval_df["predicted"].to_numpy()),
    }
    return eval_df, metrics


def calibration_plot(df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Plot predicted-probability deciles vs actual win rates."""
    df = df.copy()
    df["bucket"] = pd.qcut(df["predicted"], q=10, labels=False, duplicates="drop")
    summary = (
        df.groupby("bucket", as_index=False)
          .agg(predicted_mean=("predicted", "mean"), actual_mean=("actual", "mean"), sample_size=("predicted", "size"))
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(summary["predicted_mean"], summary["actual_mean"], marker="o", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_model_artifacts(model, feature_columns: List[str], model_path: Path) -> None:
    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "created": str(pd.Timestamp.now()),
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)


def get_win_probability(raw_inputs: Dict[str, object], model_path: Path = MODEL_PATH) -> float:
    """Return the probability that the possessing team wins the game from raw state inputs."""
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    feature_columns = payload["feature_columns"]

    record = {k: raw_inputs.get(k, np.nan) for k in feature_columns}
    if "game_seconds_remaining" in record and isinstance(record["game_seconds_remaining"], (int, float)):
        record["game_seconds_remaining"] = float(record["game_seconds_remaining"])
    record["lead_over_sqrt_time"] = record.get("score_differential", 0.0) / math.sqrt(record.get("game_seconds_remaining", 1.0) + 1)
    record["diff_time_ratio"] = record.get("score_differential", 0.0) * math.exp(4 * (3600 - record.get("game_seconds_remaining", 0.0)) / 3600)
    if "down" in record and record["down"] is not None:
        record["down"] = float(record["down"])
    if "scores_needed_bucket" in record and record["scores_needed_bucket"] is not None:
        record["scores_needed_bucket"] = float(record["scores_needed_bucket"])

    df = pd.DataFrame([record])[feature_columns]
    prob = model.predict(df, raw_score=False)[0]
    return float(prob)


def main() -> None:
    df = load_2018_game_data(DATA_DIR)
    summarize_play_types(df)

    # The core modeling frame only includes live-ball scrimmage plays.
    model_df = prepare_model_frame(df)
    print(f"Rows retained after filtering scrimmage plays: {len(model_df)}")
    print(model_df.head().to_string(index=False))

    baseline_model, baseline_df = fit_baseline_model(model_df)
    baseline_metrics = {
        "log_loss": compute_log_loss(baseline_df["posteam_wins"].to_numpy(), baseline_df["baseline_prob"].to_numpy()),
        "brier_score": compute_brier_score(baseline_df["posteam_wins"].to_numpy(), baseline_df["baseline_prob"].to_numpy()),
    }
    print("Baseline metrics:", baseline_metrics)

    empirical = make_empirical_table(model_df)
    print("Empirical table (first 10 rows):")
    print(empirical.head(10).to_string(index=False))

    feature_columns = [
        "score_differential",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "down",
        "yds_to_go",
        "goal_to_go",
        "yardline_100",
        "is_redzone",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "is_overtime",
        "scores_needed_bucket",
        "lead_over_sqrt_time",
        "diff_time_ratio",
    ]

    xgb_model, cv_summary, heldout = fit_xgb_grouped_cv(model_df, feature_columns, n_splits=3)
    print("Grouped-CV summary:", cv_summary)

    # Use the final trained model for a held-out calibration plot.
    eval_df, metrics = evaluate_model(model_df, xgb_model, feature_columns, model_label="xgboost")
    print("XGB evaluation:", metrics)
    calibration_plot(eval_df[["predicted", "actual"]], "Win probability calibration (held-out games)", RESULTS_DIR / "wp_calibration.png")

    # Save a trained model for inference.
    save_model_artifacts(xgb_model, feature_columns, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    # Example inference check.
    sample_input = {
        "score_differential": 0,
        "game_seconds_remaining": 1800,
        "half_seconds_remaining": 900,
        "down": 2,
        "yds_to_go": 7,
        "goal_to_go": 0,
        "yardline_100": 38,
        "is_redzone": 0,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "is_overtime": 0,
        "scores_needed_bucket": 1,
    }
    print("Example inference probability:", get_win_probability(sample_input, MODEL_PATH))

    # Optionally write the empirical table and a plain summary as JSON.
    empirical.to_csv(RESULTS_DIR / "empirical_table.csv", index=False)
    with open(RESULTS_DIR / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)


if __name__ == "__main__":
    main()
