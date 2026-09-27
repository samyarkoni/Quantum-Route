"""Create one yearly play-participation CSV per NFL team.

The input files are expected under ``GAMES/<season>/<week>/*.csv``.  A game
CSV is written to both participating teams' yearly output files, with team
metadata added to each row.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
from typing import TextIO


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / "GAMES"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "TEAM_PARTICIPATION_STATS"
GAME_ID_PATTERN = re.compile(
    r"^(?P<season>\d{4})_(?P<week>\d{1,2})_(?P<first_team>[A-Z]{2,3})_(?P<second_team>[A-Z]{2,3})$"
)


def _game_metadata(game_id: str, season: str, week: str) -> tuple[str, str]:
    match = GAME_ID_PATTERN.fullmatch(game_id)
    if match is None:
        raise ValueError(
            f"Unsupported game_id {game_id!r}; expected "
            "'YYYY_WW_HOME_AWAY' with NFL team abbreviations"
        )
    if match["season"] != season:
        raise ValueError(
            f"Game {game_id!r} is in a file under season {season}, "
            f"but its game_id says {match['season']}"
        )
    if int(match["week"]) != int(week):
        raise ValueError(
            f"Game {game_id!r} is in a file under week {week}, "
            f"but its game_id says {match['week']}"
        )
    return match["first_team"], match["second_team"]


def _open_team_files(
    teams: set[str],
    season: str,
    output_dir: Path,
    fieldnames: list[str],
) -> dict[str, TextIO]:
    handles: dict[str, TextIO] = {}
    try:
        for team in sorted(teams):
            team_dir = output_dir / team
            team_dir.mkdir(parents=True, exist_ok=True)
            handle = (team_dir / f"{season}.csv").open(
                "w", newline="", encoding="utf-8"
            )
            csv.writer(handle).writerow(fieldnames)
            handles[team] = handle
    except Exception:
        for handle in handles.values():
            handle.close()
        raise
    return handles


def generate_team_participation_stats(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, int]:
    """Generate yearly team files and return the number of rows written per team."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    game_files = sorted(input_path.glob("*/*/*.csv"))
    if not game_files:
        raise FileNotFoundError(f"No game CSV files found under {input_path}")

    rows_by_team_year: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    source_columns: list[str] | None = None

    for game_file in game_files:
        season = game_file.parent.parent.name
        week = game_file.parent.name
        if not season.isdigit() or not week.isdigit():
            raise ValueError(f"Expected season/week directories, got {game_file}")

        with game_file.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or "game_id" not in reader.fieldnames:
                raise ValueError(f"{game_file} must contain a game_id column")
            if source_columns is None:
                source_columns = list(reader.fieldnames)
            elif list(reader.fieldnames) != source_columns:
                raise ValueError(
                    f"{game_file} has a different header from the other game files"
                )

            game_teams: tuple[str, str] | None = None
            for row_number, row in enumerate(reader, start=2):
                game_id = (row.get("game_id") or "").strip()
                if not game_id:
                    raise ValueError(f"{game_file}:{row_number} has an empty game_id")
                first_team, second_team = _game_metadata(game_id, season, week)
                if game_teams is None:
                    game_teams = (first_team, second_team)
                elif game_teams != (first_team, second_team):
                    raise ValueError(
                        f"{game_file}:{row_number} contains more than one game_id"
                    )

                for team, opponent, role in (
                    (first_team, second_team, "home"),
                    (second_team, first_team, "away"),
                ):
                    enriched = dict(row)
                    enriched.update(
                        {
                            "season": season,
                            "week": week.zfill(2),
                            "team": team,
                            "opponent": opponent,
                            "team_role": role,
                        }
                    )
                    rows_by_team_year[(team, season)].append(enriched)

    if source_columns is None:
        raise ValueError("No rows found in game CSV files")

    metadata_columns = ["season", "week", "team", "opponent", "team_role"]
    fieldnames = metadata_columns + [
        column for column in source_columns if column not in metadata_columns
    ]
    counts: dict[str, int] = {}
    for (team, season), rows in sorted(rows_by_team_year.items()):
        team_dir = output_path / team
        team_dir.mkdir(parents=True, exist_ok=True)
        destination = team_dir / f"{season}.csv"
        with destination.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        counts[f"{team}/{season}"] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    counts = generate_team_participation_stats(args.input_dir, args.output_dir)
    print(f"Wrote {len(counts)} team-season CSV files.")
    print(f"Wrote {sum(counts.values()):,} team play rows.")


if __name__ == "__main__":
    main()