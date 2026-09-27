from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriveResult:
    points: float
    yards: int
    plays: int
    outcome: str
    last_snap: int = 0  # yardline of the drive's final snap (punt / kick spot, or last play)


class DriveSimulator:
    """Markov chain over (down, yards to go, yardline). Each step asks the play-call model
    what to run and the yards model what it gains, until the drive ends."""

    def __init__(self, play_calls, yards_model, field_goals, td_points, defensive_td_rate, max_plays=40):
        self.play_calls = play_calls
        self.yards_model = yards_model
        self.field_goals = field_goals
        self.td_points = td_points
        self.defensive_td_rate = defensive_td_rate
        self.max_plays = max_plays

    def simulate(self, yardline, rng, down=1, yds_to_go=None, first_call=None, max_plays=None):
        """first_call forces the opening play (e.g. "run") before normal play calling takes over.
        max_plays overrides the default cap, e.g. with the plays left before the half ends."""
        yds_to_go = min(10, yardline) if yds_to_go is None else yds_to_go
        max_plays = self.max_plays if max_plays is None else max_plays
        total_yards = 0
        snap = yardline

        for n_plays in range(max_plays):
            snap = yardline
            if n_plays == 0 and first_call is not None:
                call = first_call
            else:
                call = self.play_calls.sample(down, yds_to_go, yardline, rng)
            if call == "punt":
                return DriveResult(0, total_yards, n_plays, "punt", snap)
            if call == "field_goal":
                made = rng.random() < self.field_goals.make_probability(yardline)
                return DriveResult(3 if made else 0, total_yards, n_plays, "field_goal" if made else "missed_fg", snap)

            yards, turnover = self.yards_model.sample(call, down, yds_to_go, yardline, rng)
            if turnover:
                if rng.random() < self.defensive_td_rate:
                    return DriveResult(-self.td_points, total_yards, n_plays + 1, "defensive_td", snap)
                return DriveResult(0, total_yards, n_plays + 1, "turnover", snap)

            yards = min(yards, yardline)
            total_yards += yards
            yardline -= yards
            if yardline <= 0:
                return DriveResult(self.td_points, total_yards, n_plays + 1, "touchdown", snap)
            if yardline >= 100:
                return DriveResult(-2, total_yards, n_plays + 1, "safety", snap)

            if yards >= yds_to_go:
                down, yds_to_go = 1, min(10, yardline)
            elif down == 4:
                return DriveResult(0, total_yards, n_plays + 1, "downs", snap)
            else:
                down, yds_to_go = down + 1, yds_to_go - yards

        return DriveResult(0, total_yards, max_plays, "max_plays", snap)

    def expected(self, yardline, n=2000, seed=None, **state):
        """Expected points and yards for a drive starting at this yardline."""
        rng = np.random.default_rng(seed)
        results = pd.DataFrame([vars(self.simulate(yardline, rng, **state)) for _ in range(n)])
        return {
            "start_yardline": yardline,
            "expected_points": results["points"].mean(),
            "expected_yards": results["yards"].mean(),
            "expected_plays": results["plays"].mean(),
            **results["outcome"].value_counts(normalize=True).add_prefix("p_").to_dict(),
        }
