import numpy as np
import pandas as pd

from plays import DISTANCE_LABELS, build_drives, distance_bucket
from run_drive_sim import OUTPUT_DIR, build_simulator, load_season_plays

SIMS_PER_BIN = 3000
PLAY_TYPES = ["run", "pass"]


def play_averages(gains):
    """Historical per-play averages for each down / distance / play type bin."""
    first_down = gains["offense_td"] | (~gains["is_turnover"] & (gains["yards_gained"] >= gains["yds_to_go"]))
    return (
        gains.assign(first_down=first_down, yards=gains["yards_gained"].where(~gains["is_turnover"], 0))
        .groupby(["down", "distance", "play_type"], observed=True)
        .agg(
            plays=("yards", "size"),
            avg_yards=("yards", "mean"),
            median_yards=("yards", "median"),
            first_down_rate=("first_down", "mean"),
            turnover_rate=("is_turnover", "mean"),
            td_rate=("offense_td", "mean"),
        )
    )


def simulated_ev(gains, simulator, seed=0):
    """Expected drive points from each bin. Run and pass are simulated from the same sampled
    game states, so the difference is the value of the call rather than of where it was called."""
    rng = np.random.default_rng(seed)
    rows = []
    for (down, distance), bin_plays in gains.groupby(["down", "distance"], observed=True):
        states = bin_plays[["starting_yard", "yds_to_go"]].to_numpy()
        states = states[rng.integers(len(states), size=SIMS_PER_BIN)]

        def mean_points(first_call):
            return np.mean([
                simulator.simulate(int(yardline), rng, down=int(down), yds_to_go=int(ytg), first_call=first_call).points
                for yardline, ytg in states
            ])

        baseline = mean_points(None)
        rows.append({"down": down, "distance": distance, "play_type": "all", "expected_points": baseline, "epa": 0.0})
        for play_type in PLAY_TYPES:
            ev = mean_points(play_type)
            rows.append({"down": down, "distance": distance, "play_type": play_type,
                         "expected_points": ev, "epa": ev - baseline})
    return pd.DataFrame(rows).set_index(["down", "distance", "play_type"])


def main():
    plays = load_season_plays()
    drives = build_drives(plays)
    simulator = build_simulator(plays, drives)

    gains = plays[plays["is_scrimmage"] & plays["call"].isin(PLAY_TYPES)].copy()
    gains["down"] = gains["down"].astype(int)
    gains["play_type"] = gains["call"]
    gains["distance"] = pd.Categorical(distance_bucket(gains["yds_to_go"]), categories=DISTANCE_LABELS, ordered=True)

    averages = play_averages(gains)
    ev = simulated_ev(gains, simulator)

    all_plays = play_averages(gains.assign(play_type="all"))
    table = pd.concat([averages, all_plays]).join(ev).sort_index().reset_index()
    table["play_type"] = pd.Categorical(table["play_type"], categories=["all", *PLAY_TYPES], ordered=True)
    table = table.sort_values(["down", "distance", "play_type"], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "down_distance_ev.csv"
    table.round(4).to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(f"\nexpected_points = average drive points from that bin, given the play call "
          f"(\"all\" = league-average calling); epa = run/pass minus \"all\"")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
