#pip install pyarrow

import numpy as np

from plays import distance_bucket, field_zone


def _situation(down, yds_to_go, yardline):
    return int(down), str(distance_bucket(yds_to_go)), str(field_zone(yardline))


class PlayCallModel:
    """Team layer: what the offense calls in each situation (run/pass on 1st-3rd,
    plus punt/field goal on 4th). League-wide for now; fit on one team's plays for team tendencies."""

    def __init__(self, plays, min_count=30):
        calls = plays[plays["is_scrimmage"] & plays["call"].notna()]
        calls = calls[(calls["down"] == 4) | calls["call"].isin(["run", "pass"])]
        keys = {
            "situation": [calls["down"].astype(int), distance_bucket(calls["yds_to_go"]), field_zone(calls["starting_yard"])],
            "distance": [calls["down"].astype(int), distance_bucket(calls["yds_to_go"])],
            "down": [calls["down"].astype(int)],
        }
        self.tables = {}
        for level, cols in keys.items():
            counts = calls.groupby(cols)["call"].value_counts().unstack(fill_value=0)
            counts = counts[counts.sum(axis=1) >= min_count]
            self.tables[level] = {
                key if isinstance(key, tuple) else (key,): (row.index[row > 0].to_numpy(), (row[row > 0] / row.sum()).to_numpy())
                for key, row in counts.iterrows()
            }

    def probabilities(self, down, yds_to_go, yardline):
        """(calls, probabilities) for the most specific situation with enough data."""
        d, dist, zone = _situation(down, yds_to_go, yardline)
        for level, key in (("situation", (d, dist, zone)), ("distance", (d, dist)), ("down", (d,))):
            if key in self.tables[level]:
                return self.tables[level][key]
        raise KeyError(f"No play-call data for down {down}")

    def sample(self, down, yds_to_go, yardline, rng):
        options, probs = self.probabilities(down, yds_to_go, yardline)
        return options[rng.choice(len(options), p=probs)]


class EmpiricalYardsModel:
    """Player layer placeholder: draws yards gained (and turnovers) from historical plays in
    the same situation. A player-based model plugs into the simulator by exposing the same sample()."""

    def __init__(self, plays, min_count=50):
        gains = plays[plays["is_scrimmage"] & plays["call"].isin(["run", "pass"])]
        self.yards = np.where(gains["is_turnover"], 0, gains["yards_gained"].fillna(0)).astype(int)
        self.turnover = gains["is_turnover"].to_numpy()
        down = gains["down"].astype(int).to_numpy()
        dist = distance_bucket(gains["yds_to_go"])
        zone = field_zone(gains["starting_yard"])
        call = gains["call"].to_numpy()

        self.pools = {}
        for level, cols in {
            "situation": (call, down, dist, zone),
            "distance": (call, dist, zone),
            "field": (call, zone),
            "call": (call,),
        }.items():
            keys = list(zip(*cols))
            index = {}
            for i, key in enumerate(keys):
                index.setdefault(key, []).append(i)
            self.pools[level] = {key: np.array(idx) for key, idx in index.items() if len(idx) >= min_count}

    def pool(self, call, down, yds_to_go, yardline):
        """Row indices of the historical plays drawn from in this situation."""
        d, dist, zone = _situation(down, yds_to_go, yardline)
        for level, key in (("situation", (call, d, dist, zone)), ("distance", (call, dist, zone)),
                           ("field", (call, zone)), ("call", (call,))):
            if key in self.pools[level]:
                return self.pools[level][key]
        raise KeyError(f"No yardage data for {call}")

    def sample(self, call, down, yds_to_go, yardline, rng):
        """Returns (yards_gained, turnover)."""
        idx = self.pool(call, down, yds_to_go, yardline)
        i = idx[rng.integers(len(idx))]
        return int(self.yards[i]), bool(self.turnover[i])


class FieldGoalModel:
    """Make probability as a logistic function of kick distance (line of scrimmage + 17)."""

    def __init__(self, plays, iterations=25):
        kicks = plays[plays["is_scrimmage"] & plays["play_type"].eq("field_goal")]
        distance = kicks["starting_yard"].to_numpy() + 17
        made = kicks["fg_made"].to_numpy().astype(float)

        X = np.column_stack([np.ones_like(distance, dtype=float), distance])
        beta = np.zeros(2)
        for _ in range(iterations):  # Newton-Raphson for logistic regression
            p = 1 / (1 + np.exp(-X @ beta))
            gradient = X.T @ (made - p)
            hessian = X.T @ (X * (p * (1 - p))[:, None])
            beta += np.linalg.solve(hessian, gradient)
        self.beta = beta

    def make_probability(self, yardline):
        return 1 / (1 + np.exp(-(self.beta[0] + self.beta[1] * (yardline + 17))))
