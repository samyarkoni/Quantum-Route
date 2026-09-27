"""Manual runner for the NFL in-game win-probability model.

This model estimates the probability that the team currently on offense wins the game,
conditioned on the current game state.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------------------------
# Manually edit these values for a live game-state estimate.
# All values are in the same units used by the trained model.
# -----------------------------------------------------------------------------

score_differential = 0          # offense score minus defense score
game_seconds_remaining = 1800    # total seconds left in the game ONE QUARTER = 900 SECONDS
half_seconds_remaining = 1800     # seconds left in the current half

down = 1                      # 1,2,3,4
yds_to_go = 10                   # yards to a first down
goal_to_go = 0                  # 1 if offense is at the goal line, else 0
yardline_100 = 75               # distance from own goal line to end zone, 1-100
is_redzone = 1 if yardline_100 <= 20 else 0                  # 1 if yardline_100 <= 20, else 0
posteam_timeouts_remaining = 3
defteam_timeouts_remaining = 3
is_overtime = 0                 # 1 if quarter > 4, else 0

# -----------------------------------------------------------------------------
# Example: if you want to change the state to a trailing offense, just modify the
# variables above before running this script.
# -----------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "trained_win_probability_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run the training script first."
    )

# Import after the variables are defined so the script is easy to edit.
from Code.nfl_WIN_PROB.nfl_WIN_PROB_TRY1 import get_win_probability


if __name__ == "__main__":
    state = {
        "score_differential": score_differential,
        "game_seconds_remaining": game_seconds_remaining,
        "half_seconds_remaining": half_seconds_remaining,
        "down": down,
        "yds_to_go": yds_to_go,
        "goal_to_go": goal_to_go,
        "yardline_100": yardline_100,
        "is_redzone": is_redzone,
        "posteam_timeouts_remaining": posteam_timeouts_remaining,
        "defteam_timeouts_remaining": defteam_timeouts_remaining,
        "is_overtime": is_overtime,
    }

    prob = get_win_probability(state, MODEL_PATH)
    print(f"Probability the team on offense wins the game: {prob:.4f}")
    print(f"Equivalent to {prob * 100:.2f}%")
