import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Drive_Model"))
from plays import add_features, add_game_state, add_possession, load_plays  # noqa: E402

# Next score in the half from the offense's point of view. Touchdowns count 7 (6 + typical PAT).
NEXT_SCORES = ["td", "fg", "safety", "opp_td", "opp_fg", "opp_safety", "no_score"]
NEXT_SCORE_POINTS = np.array([7, 3, 2, -7, -3, -2, 0])

# Knots for the piecewise-linear yardline and clock terms
YARDLINE_KNOTS = [5, 10, 20, 35, 50, 65, 80, 90]
LATE_HALF_SECONDS = 120


def prepare_plays(seasons=None):
    """All plays with drive features, clock/half, cleaned scores, and possession team."""
    plays = load_plays()
    if seasons is not None:
        plays = plays[plays["season"].isin(seasons)]
    return add_possession(add_game_state(add_features(plays)))


def label_next_score(plays):
    """For each play, the first score at or after it in the same half (the score columns are the
    score after the play, so the play's own score counts). Adds next_score as a NEXT_SCORES label."""
    plays = plays.copy()
    game = plays["game_id"]
    home_pts = plays["home_score"].groupby(game).diff().fillna(plays["home_score"])
    away_pts = plays["away_score"].groupby(game).diff().fillna(plays["away_score"])
    scored = (home_pts > 0) | (away_pts > 0)

    # Separately-recorded conversions (1s, 2s right after a 6) never come first, so any amount
    # other than 3 or 2 is a touchdown
    amount = np.maximum(home_pts, away_pts)
    kind = np.select([amount == 3, amount == 2], ["fg", "safety"], default="td")
    event_kind = pd.Series(np.where(scored, kind, None), index=plays.index)
    event_home = pd.Series(np.where(scored, home_pts > away_pts, np.nan), index=plays.index)

    # Look forward within the half: backfill the next event onto every play before it
    half = [game, plays["half"]]
    next_kind = event_kind.groupby(half).bfill()
    next_home = event_home.groupby(half).bfill()

    own = next_home == plays["posteam_is_home"].astype(float)
    plays["next_score"] = np.where(next_kind.isna(), "no_score", np.where(own, next_kind, "opp_" + next_kind))
    plays.loc[plays["posteam_is_home"].isna(), "next_score"] = None
    return plays


def ep_features(yardline, down, yds_to_go, half_seconds_left):
    """Design matrix. yardline is yards from the opponent's end zone (1-99)."""
    y = np.asarray(yardline, dtype=float)
    down = np.asarray(down, dtype=float)
    ytg = np.clip(np.asarray(yds_to_go, dtype=float), 1, None)
    secs = np.asarray(half_seconds_left, dtype=float)
    late = np.clip(LATE_HALF_SECONDS - secs, 0, None) / LATE_HALF_SECONDS  # 0 until 2:00, 1 at 0:00

    columns = [np.ones_like(y), y / 100]
    columns += [np.clip(y - k, 0, None) / 100 for k in YARDLINE_KNOTS]
    columns += [down == d for d in (2, 3, 4)]
    columns += [np.log(ytg), np.log(ytg) * (down == 3), np.log(ytg) * (down == 4), ytg >= y]
    columns += [secs / 1800, late, late * y / 100]
    return np.column_stack(columns).astype(float)


def _softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


class ExpectedPointsModel:
    """Multinomial logistic regression for the next score in the half, fit by Newton-Raphson.
    Expected points = sum over outcomes of P(outcome) * points."""

    def __init__(self, ridge=1.0, iterations=30, tol=1e-8):
        self.ridge = ridge
        self.iterations = iterations
        self.tol = tol

    @staticmethod
    def training_rows(plays):
        """Scrimmage plays in regulation with a labeled next score."""
        keep = (plays["is_scrimmage"] & (plays["half"] <= 2) & plays["next_score"].notna()
                & plays["half_seconds_left"].notna() & plays["starting_yard"].between(1, 99))
        return plays[keep]

    def fit(self, plays):
        rows = self.training_rows(plays)
        X = ep_features(rows["starting_yard"], rows["down"], rows["yds_to_go"], rows["half_seconds_left"])
        Y = (rows["next_score"].to_numpy()[:, None] == np.array(NEXT_SCORES)[None, :]).astype(float)

        n, f = X.shape
        k = len(NEXT_SCORES) - 1  # no_score is the reference class
        W = np.zeros((f, k))
        penalty = np.full(f, self.ridge)
        penalty[0] = 0  # don't shrink intercepts
        penalty = np.tile(penalty, k)

        def objective(W):
            logits = np.column_stack([X @ W, np.zeros(n)])
            log_norm = logits.max(axis=1) + np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1))
            return (Y * logits).sum() - log_norm.sum() - 0.5 * (penalty * W.T.ravel() ** 2).sum()

        current = objective(W)
        for _ in range(self.iterations):
            P = _softmax(np.column_stack([X @ W, np.zeros(n)]))[:, :k]
            gradient = (X.T @ (Y[:, :k] - P)).T.ravel() - penalty * W.T.ravel()
            hessian = np.empty((k * f, k * f))
            for a in range(k):
                for b in range(a, k):
                    weight = P[:, a] * ((a == b) - P[:, b])
                    block = X.T @ (X * weight[:, None])
                    hessian[a * f:(a + 1) * f, b * f:(b + 1) * f] = block
                    hessian[b * f:(b + 1) * f, a * f:(a + 1) * f] = block
            hessian[np.diag_indices_from(hessian)] += penalty
            step = np.linalg.solve(hessian, gradient).reshape(k, f).T
            # Halve the step until it improves the fit; full Newton steps can overshoot on rare classes
            while (candidate := objective(W + step)) < current and np.abs(step).max() > self.tol:
                step /= 2
            W, current = W + step, candidate
            if np.abs(step).max() < self.tol:
                break
        self.W = W
        return self

    def predict_proba(self, yardline, down, yds_to_go, half_seconds_left):
        X = ep_features(yardline, down, yds_to_go, half_seconds_left)
        return _softmax(np.column_stack([X @ self.W, np.zeros(len(X))]))

    def expected_points(self, yardline, down, yds_to_go, half_seconds_left):
        return self.predict_proba(yardline, down, yds_to_go, half_seconds_left) @ NEXT_SCORE_POINTS


def add_epa(plays, model):
    """ep before each regulation scrimmage play, and epa = ep_after - ep. ep_after is the net points
    (for the offense) scored before the next scrimmage play if any were, otherwise that play's ep,
    sign flipped if the ball changed hands. The last play of a half is followed by ep 0."""
    plays = plays.copy()
    rows = model.training_rows(plays).index
    plays["ep"] = np.nan
    r = plays.loc[rows]
    plays.loc[rows, "ep"] = model.expected_points(r["starting_yard"], r["down"], r["yds_to_go"], r["half_seconds_left"])

    # Score just before each row, and at the end of each half
    game = plays["game_id"]
    home_before = plays["home_score"].groupby(game).shift(fill_value=0)
    away_before = plays["away_score"].groupby(game).shift(fill_value=0)
    half_key = [game, plays["half"]]
    home_end = plays["home_score"].groupby(half_key).transform("last")
    away_end = plays["away_score"].groupby(half_key).transform("last")

    s = plays.loc[rows].assign(home_before=home_before[rows], away_before=away_before[rows],
                               home_end=home_end[rows], away_end=away_end[rows])
    by_half = s.groupby(["game_id", "half"], sort=False)
    has_next = by_half["ep"].shift(-1).notna()
    home_after = by_half["home_before"].shift(-1).where(has_next, s["home_end"])
    away_after = by_half["away_before"].shift(-1).where(has_next, s["away_end"])

    home_sign = np.where(s["posteam_is_home"].astype(bool), 1, -1)
    points = home_sign * ((home_after - s["home_before"]) - (away_after - s["away_before"]))
    same_team = by_half["posteam"].shift(-1) == s["posteam"]
    ep_after = np.where(same_team, 1, -1) * by_half["ep"].shift(-1).fillna(0)

    # A score ends the next-score chain the model was fit on, so the kickoff after it isn't counted
    plays.loc[rows, "ep_after"] = np.where(points != 0, points, ep_after)
    plays["epa"] = plays["ep_after"] - plays["ep"]
    plays.loc[plays["posteam"].isna(), ["ep", "ep_after", "epa"]] = np.nan
    return plays
