import pandas as pd

from plays import PROJECT_DIR, load_plays, add_features, build_drives
from models import PlayCallModel, EmpiricalYardsModel, FieldGoalModel
from simulate import DriveSimulator

OUTPUT_DIR = PROJECT_DIR / "Data" / "Drive_Model"
SEASONS = range(2018, 2026)
SIMS_PER_YARDLINE = 1000


def load_season_plays():
    plays = add_features(load_plays())
    return plays[plays["season"].isin(SEASONS)]


def build_simulator(plays, drives):
    td_points = drives.loc[drives["outcome"] == "touchdown", "points"].mean()
    turnovers = plays[plays["is_scrimmage"] & plays["is_turnover"]]
    defensive_td_rate = turnovers["defense_score"].mean()
    print(f"Points per TD (with conversion): {td_points:.2f}   Defensive TD rate per turnover: {defensive_td_rate:.3f}")

    return DriveSimulator(
        play_calls=PlayCallModel(plays),
        yards_model=EmpiricalYardsModel(plays),
        field_goals=FieldGoalModel(plays),
        td_points=td_points,
        defensive_td_rate=defensive_td_rate,
    )


def main():
    plays = load_season_plays()
    drives = build_drives(plays)
    simulator = build_simulator(plays, drives)

    ep = pd.DataFrame([simulator.expected(yl, n=SIMS_PER_YARDLINE, seed=yl) for yl in range(1, 100)]).fillna(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ep.to_csv(OUTPUT_DIR / "expected_drive_value.csv", index=False)

    # Validation: simulated vs actual drive results, excluding drives cut off by the clock
    actual = drives[drives["outcome"] != "end_of_half"].copy()
    actual = actual.merge(ep[["start_yardline", "expected_points", "expected_yards"]], on="start_yardline")
    actual["start_bin"] = pd.cut(actual["start_yardline"], range(0, 101, 10))
    check = actual.groupby("start_bin", observed=True).agg(
        drives=("points", "size"),
        actual_points=("points", "mean"),
        sim_points=("expected_points", "mean"),
        actual_yards=("yards", "mean"),
        sim_yards=("expected_yards", "mean"),
    )
    pd.set_option("display.width", 200)
    print("\nSimulated vs actual, by starting yards from opponent end zone:")
    print(check.round(2).to_string())
    print(f"\nAll drives: actual {actual['points'].mean():.2f} pts / {actual['yards'].mean():.1f} yds, "
          f"simulated {actual['expected_points'].mean():.2f} pts / {actual['expected_yards'].mean():.1f} yds")
    print("\nOwn 25 (75 to go):")
    print(ep.loc[ep["start_yardline"] == 75].round(3).T.to_string(header=False))
    print(f"\nWrote {OUTPUT_DIR / 'expected_drive_value.csv'}")


if __name__ == "__main__":
    main()
