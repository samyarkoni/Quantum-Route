import numpy as np
import pandas as pd

from expected_points import NEXT_SCORE_POINTS, NEXT_SCORES, ExpectedPointsModel, add_epa, label_next_score, prepare_plays
from plays import PROJECT_DIR

OUTPUT_DIR = PROJECT_DIR / "Data" / "Scoring_Model"
SEASONS = range(2018, 2026)
HOLDOUT_SEASON = 2025
EARLY_HALF = 1800  # field-position table is for the start of a half


def holdout_check(plays):
    """Train on every season but one, score the held-out season."""
    train = plays[plays["season"] != HOLDOUT_SEASON]
    test = ExpectedPointsModel.training_rows(plays[plays["season"] == HOLDOUT_SEASON])
    model = ExpectedPointsModel().fit(train)

    probs = model.predict_proba(test["starting_yard"], test["down"], test["yds_to_go"], test["half_seconds_left"])
    actual = test["next_score"].map({s: i for i, s in enumerate(NEXT_SCORES)}).to_numpy()
    log_loss = -np.log(probs[np.arange(len(test)), actual]).mean()
    base_rates = ExpectedPointsModel.training_rows(train)["next_score"].value_counts(normalize=True)
    baseline = -np.log(base_rates.reindex(NEXT_SCORES).to_numpy()[actual]).mean()
    print(f"Held-out {HOLDOUT_SEASON}: log loss {log_loss:.4f} vs {baseline:.4f} for league base rates "
          f"({len(test):,} plays)")

    # Calibration: predicted ep vs the next score that actually happened
    ep = probs @ NEXT_SCORE_POINTS
    check = pd.DataFrame({"ep": ep, "actual": NEXT_SCORE_POINTS[actual]})
    check["ep_bin"] = pd.cut(check["ep"], np.arange(-4, 8))
    print(check.groupby("ep_bin", observed=True).agg(plays=("ep", "size"), predicted=("ep", "mean"),
                                                     actual=("actual", "mean")).round(2).to_string())


def field_table(model):
    """EP at each yardline for common down-and-distance states, early in a half."""
    rows = []
    for yardline in range(1, 100):
        for down, ytg in [(1, 10), (2, 3), (2, 7), (2, 10), (3, 1), (3, 3), (3, 7), (3, 10), (4, 1), (4, 3)]:
            ytg = min(ytg, yardline)
            rows.append({"yardline": yardline, "down": down, "yds_to_go": ytg})
    table = pd.DataFrame(rows)
    probs = model.predict_proba(table["yardline"], table["down"], table["yds_to_go"], np.full(len(table), EARLY_HALF))
    table["ep"] = probs @ NEXT_SCORE_POINTS
    return pd.concat([table, pd.DataFrame(probs, columns=[f"p_{s}" for s in NEXT_SCORES])], axis=1)


def team_profiles(plays):
    """Per-season offense and defense EPA per run/pass play."""
    snaps = plays[plays["call"].isin(["run", "pass"]) & plays["epa"].notna()]

    def summarize(group_col, prefix):
        g = snaps.groupby(["season", group_col])
        out = pd.DataFrame({
            "plays": g.size(),
            "epa_per_play": g["epa"].mean(),
            "pass_epa": snaps[snaps["call"] == "pass"].groupby(["season", group_col])["epa"].mean(),
            "run_epa": snaps[snaps["call"] == "run"].groupby(["season", group_col])["epa"].mean(),
            "pass_rate": g["call"].apply(lambda c: (c == "pass").mean()),
            "success_rate": g["epa"].apply(lambda e: (e > 0).mean()),
        })
        out.index.names = ["season", "team"]
        return out.add_prefix(prefix)

    return summarize("posteam", "off_").join(summarize("defteam", "def_"))


def main():
    plays = label_next_score(prepare_plays(SEASONS))
    pd.set_option("display.width", 200)

    holdout_check(plays)
    model = ExpectedPointsModel().fit(plays)
    plays = add_epa(plays, model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = field_table(model)
    table.round(4).to_csv(OUTPUT_DIR / "ep_by_field.csv", index=False)

    keep = ["game_id", "play_id", "season", "posteam", "defteam", "quarter", "half_seconds_left", "down",
            "yds_to_go", "starting_yard", "play_type", "call", "yards_gained", "next_score", "ep", "epa"]
    plays.loc[plays["ep"].notna(), keep].to_parquet(OUTPUT_DIR / "plays_epa.parquet", index=False)

    profiles = team_profiles(plays)
    profiles.round(4).to_csv(OUTPUT_DIR / "team_profiles.csv")

    first_downs = table[table["down"] == 1].set_index("yardline")["ep"]
    print("\n1st & 10 expected points by yards from the opponent's end zone:")
    print(first_downs.loc[[1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99]].round(2).to_string())

    season = profiles.index.get_level_values("season").max()
    latest = profiles.loc[season]
    cols = ["off_epa_per_play", "off_pass_epa", "off_run_epa", "def_epa_per_play"]
    print(f"\nTop offenses by EPA/play, {season}:")
    print(latest.sort_values("off_epa_per_play", ascending=False)[cols].head(8).round(3).to_string())
    print(f"\nWrote {OUTPUT_DIR}/ep_by_field.csv, plays_epa.parquet, team_profiles.csv")


if __name__ == "__main__":
    main()
