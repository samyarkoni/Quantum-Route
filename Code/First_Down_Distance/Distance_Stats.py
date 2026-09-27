import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
GAMES_DIR = PROJECT_DIR / "GAMES"
OUTPUT_DIR = PROJECT_DIR / "Data" / "First_Down_Distance"

PLAY_TYPES = ["run", "pass"]
DOWNS = [1, 2, 3, 4]

# (label, min yards to go, max yards to go) -- inclusive bounds, None = no upper limit
DISTANCE_RANGES = [
    ("1-3", 0, 3),
    ("4-7", 4, 7),
    ("8-12", 8, 12),
    ("13-20", 13, 20),
    ("21+", 21, None),
]

DOWN_NAMES = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def load_plays():
    files = sorted(GAMES_DIR.rglob("*.csv"))
    plays = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    return plays[plays["play_type"].isin(PLAY_TYPES) & plays["down"].isin(DOWNS)]


def main():
    plays = load_plays()
    summary = []

    for down in DOWNS:
        down_dir = OUTPUT_DIR / f"{DOWN_NAMES[down]}_down"
        down_dir.mkdir(parents=True, exist_ok=True)

        for label, low, high in DISTANCE_RANGES:
            in_range = (plays["down"] == down) & (plays["yds_to_go"] >= low)
            if high is not None:
                in_range &= plays["yds_to_go"] <= high

            for play_type in PLAY_TYPES:
                subset = plays[in_range & (plays["play_type"] == play_type)]
                out_path = down_dir / f"{DOWN_NAMES[down]}_and_{label}_{play_type}.csv"
                subset.to_csv(out_path, index=False)
                summary.append((DOWN_NAMES[down], label, play_type, len(subset)))

    summary = pd.DataFrame(summary, columns=["down", "yds_to_go", "play_type", "plays"])
    print(summary.pivot_table(index=["down", "yds_to_go"], columns="play_type", values="plays", sort=False))
    print(f"\nWrote {len(summary)} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
