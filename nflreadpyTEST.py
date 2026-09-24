"""Simple nflreadpy showcase.

This script demonstrates the core features documented by nflreadpy:
- loading play-by-play data
- loading player and team stats
- loading schedules
- checking current season/week
- using Polars DataFrames and optional pandas conversion
"""

from collections.abc import Iterable
import csv
from functools import lru_cache
from pathlib import Path
import re

import pyarrow, pandas
import nflreadpy as nfl


@lru_cache(maxsize=None)
def _load_participation(season: int):
    return nfl.load_participation(seasons=[season])


@lru_cache(maxsize=None)
def _load_pbp(season: int):
    return nfl.load_pbp(seasons=[season])


@lru_cache(maxsize=None)
def _load_schedule(season: int):
    return nfl.load_schedules(seasons=[season])


@lru_cache(maxsize=None)
def _load_player_names(season: int) -> dict[str, str]:
    """Build a historical ID-to-name mapping from one season's participation data."""
    participation = _load_participation(season)
    player_names = {}
    columns = set(participation.columns)
    name_pairs = [
        ("offense_players", "offense_names"),
        ("defense_players", "defense_names"),
    ]
    available_pairs = [
        (ids_field, names_field)
        for ids_field, names_field in name_pairs
        if ids_field in columns and names_field in columns
    ]
    if not available_pairs:
        players = nfl.load_players()
        return {
            row["gsis_id"]: row["display_name"]
            for row in players.select(["gsis_id", "display_name"]).to_dicts()
            if row["gsis_id"] and row["display_name"]
        }

    for row in participation.select(
        [field for pair in available_pairs for field in pair]
    ).to_dicts():
        for ids_field, names_field in available_pairs:
            ids = _split_field(row[ids_field])
            names = _split_field(row[names_field])
            for player_id, name in zip(ids, names):
                player_names.setdefault(player_id, name)
    return player_names


def map_player_ids_to_names(season: int, player_ids: Iterable[str]) -> list[str]:
    """Return historical player names in the same order as the supplied IDs."""
    player_names = _load_player_names(season)
    return [player_names.get(player_id, "") for player_id in player_ids]


def _split_field(value: str | None) -> list[str]:
    """Split semicolon-delimited participation fields into a Python list."""
    if value is None:
        return []
    return [item.strip() for item in str(value).split(";") if item and item.strip()]


def _format_clock(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    whole_seconds = max(0, int(seconds))
    return f"{whole_seconds // 60}:{whole_seconds % 60:02d}"


def _play_category(pbp_row: dict) -> str | None:
    """Normalize timeout, turnover, and punt events into explicit categories."""
    description = str(pbp_row.get("desc") or "").strip()
    if description == "GAME":
        return "start_quarter_1"
    if description.startswith("END QUARTER"):
        quarter_match = re.search(r"END QUARTER\s+(\d+)", description)
        if quarter_match:
            return f"start_quarter_{int(quarter_match.group(1)) + 1}"
    if description == "END GAME":
        return "end_game"
    if pbp_row.get("timeout_team"):
        return "timeout"
    if pbp_row.get("interception") or pbp_row.get("fumble_lost"):
        return "turnover"
    if pbp_row.get("punt_attempt"):
        return "punt"
    return pbp_row.get("play_type")


def get_game_lineups_for_every_play(season: int, game_id: str):
    """Return all offense/defense lineups for every play in a single game.

    The nflreadpy participation table contains the exact per-play player lists for
    both teams. This function converts the semicolon-delimited fields into ordered
    lineup dictionaries for each play.
    """
    participation = _load_participation(season)
    pbp = _load_pbp(season)
    pbp_game = pbp.filter(pbp["game_id"] == game_id)
    pbp_rows = pbp_game.sort("play_id").to_dicts()

    if not pbp_rows:
        raise ValueError(f"No play-by-play data found for season {season} and game_id {game_id!r}")

    participation_rows = participation.filter(
        participation["nflverse_game_id"] == game_id
    ).to_dicts()
    participation_by_play = {
        int(row["play_id"]): row for row in participation_rows
    }

    def build_lineup(row, team_type: str):
        ids = _split_field(row.get(f"{team_type}_players"))
        names = _split_field(row.get(f"{team_type}_names"))
        positions = _split_field(row.get(f"{team_type}_positions"))
        numbers = _split_field(row.get(f"{team_type}_numbers"))

        lineup = []
        for index, player_id in enumerate(ids):
            lineup.append(
                {
                    "player_id": player_id,
                    "name": names[index] if index < len(names) else None,
                    "position": positions[index] if index < len(positions) else None,
                    "number": numbers[index] if index < len(numbers) else None,
                }
            )
        return lineup

    results = []
    for pbp_row in pbp_rows:
        play_id = int(pbp_row["play_id"])
        row = participation_by_play.get(play_id, {})

        offense_team = (pbp_row or {}).get("posteam") or row.get("possession_team")
        home_team = (pbp_row or {}).get("home_team")
        away_team = (pbp_row or {}).get("away_team")

        if offense_team and home_team and away_team:
            defense_team = away_team if offense_team == home_team else home_team
        else:
            defense_team = row.get("possession_team")

        results.append(
            {
                "play_id": play_id,
                "offense_team": offense_team,
                "defense_team": defense_team,
                "offense": build_lineup(row, "offense"),
                "defense": build_lineup(row, "defense"),
                "offense_personnel": row.get("offense_personnel"),
                "defense_personnel": row.get("defense_personnel"),
                "defense_positions": row.get("defense_positions"),
                "n_defense": row.get("n_defense"),
                "defenders_in_box": row.get("defenders_in_box"),
                "number_of_pass_rushers": row.get("number_of_pass_rushers"),
                "defense_man_zone_type": row.get("defense_man_zone_type"),
                "defense_coverage_type": row.get("defense_coverage_type"),
                "was_pressure": row.get("was_pressure"),
                "players_on_play": _split_field(row.get("players_on_play")),
            }
        )

    return results


def get_game_lineups_with_yards(season: int, game_id: str):
    """Return each play with its lineup and net yards gained/lost.

    This is useful for checking whether a particular personnel grouping or formation
    was associated with a big gain, a loss, or a no-gain play.
    """
    lineups = get_game_lineups_for_every_play(season, game_id)
    pbp = _load_pbp(season)
    pbp = pbp.filter(pbp["game_id"] == game_id)

    pbp_rows = pbp.sort("play_id").to_dicts()
    pbp_by_play = {int(row["play_id"]): row for row in pbp_rows}
    next_pbp_by_play = {
        int(row["play_id"]): next_row
        for row, next_row in zip(pbp_rows, pbp_rows[1:])
    }

    enriched = []
    for play in lineups:
        play_id = play["play_id"]
        pbp_row = pbp_by_play.get(play_id, {})
        next_pbp_row = next_pbp_by_play.get(play_id)
        starting_clock_seconds = pbp_row.get("quarter_seconds_remaining")
        if next_pbp_row and next_pbp_row.get("qtr") == pbp_row.get("qtr"):
            ending_clock_seconds = next_pbp_row.get("quarter_seconds_remaining")
        elif starting_clock_seconds is not None:
            ending_clock_seconds = 0
        else:
            ending_clock_seconds = None
        time_elapsed = (
            starting_clock_seconds - ending_clock_seconds
            if starting_clock_seconds is not None and ending_clock_seconds is not None
            else None
        )
        starting_yard = pbp_row.get("yardline_100")
        yards_gained = pbp_row.get("yards_gained")
        ending_yard = (
            starting_yard - yards_gained
            if starting_yard is not None and yards_gained is not None
            else None
        )
        enriched.append(
            {
                **play,
                "yards_gained": yards_gained,
                "starting_yard": starting_yard,
                "ending_yard": ending_yard,
                "time_of_play": time_elapsed,
                "time_on_clock_start": _format_clock(starting_clock_seconds),
                "time_on_clock_end": _format_clock(ending_clock_seconds),
                "timeout_team": pbp_row.get("timeout_team"),
                "home_team_score": pbp_row.get("total_home_score"),
                "away_team_score": pbp_row.get("total_away_score"),
                "down": pbp_row.get("down"),
                "yds_to_go": pbp_row.get("ydstogo"),
                "defense_personnel": play["defense_personnel"],
                "defense_positions": play["defense_positions"],
                "n_defense": play["n_defense"],
                "defenders_in_box": play["defenders_in_box"],
                "number_of_pass_rushers": play["number_of_pass_rushers"],
                "defense_man_zone_type": play["defense_man_zone_type"],
                "defense_coverage_type": play["defense_coverage_type"],
                "was_pressure": play["was_pressure"],
                "result": pbp_row.get("desc"),
                "play_type": _play_category(pbp_row),
                "down": pbp_row.get("down"),
                "yards_to_go": pbp_row.get("ydstogo"),
            }
        )

    return enriched


def write_games_to_csv(
    season: int,
    game_ids: Iterable[str],
    output_path: str | Path,
) -> Path:
    """Write ordered play participation and yardage data for the supplied games.

    Player IDs are stored as semicolon-delimited strings so each CSV row remains
    one play while retaining all players on offense and defense.
    """
    rows = []
    for game_id in game_ids:
        for play in get_game_lineups_with_yards(season, game_id):
            offense_player_ids = [player["player_id"] for player in play["offense"]]
            defense_player_ids = [player["player_id"] for player in play["defense"]]
            rows.append(
                {
                    "game_id": game_id,
                    "play_id": play["play_id"],
                    "play_type": play["play_type"],
                    "yards_gained": play["yards_gained"],
                    "starting_yard": play["starting_yard"],
                    "ending_yard": play["ending_yard"],
                    "time_of_play": play["time_of_play"],
                    "time_on_clock_start": play["time_on_clock_start"],
                    "time_on_clock_end": play["time_on_clock_end"],
                    "timeout_team": play["timeout_team"],
                    "home_team_score": play["home_team_score"],
                    "away_team_score": play["away_team_score"],
                    "down": play["down"],
                    "yds_to_go": play["yds_to_go"],
                    "defense_personnel": play["defense_personnel"],
                    "defense_positions": ";".join(
                        player["position"] or "" for player in play["defense"]
                    ),
                    "n_defense": play["n_defense"],
                    "defenders_in_box": play["defenders_in_box"],
                    "number_of_pass_rushers": play["number_of_pass_rushers"],
                    "defense_man_zone_type": play["defense_man_zone_type"],
                    "defense_coverage_type": play["defense_coverage_type"],
                    "was_pressure": play["was_pressure"],
                    "offense_player_ids": ";".join(offense_player_ids),
                    "offense_player_names": ";".join(
                        map_player_ids_to_names(season, offense_player_ids)
                    ),
                    "defense_player_ids": ";".join(defense_player_ids),
                    "defense_player_names": ";".join(
                        map_player_ids_to_names(season, defense_player_ids)
                    ),
                }
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "game_id",
                "play_id",
                "play_type",
                "yards_gained",
                "starting_yard",
                "ending_yard",
                "time_of_play",
                "time_on_clock_start",
                "time_on_clock_end",
                "timeout_team",
                "home_team_score",
                "away_team_score",
                "down",
                "yds_to_go",
                "defense_personnel",
                "defense_positions",
                "n_defense",
                "defenders_in_box",
                "number_of_pass_rushers",
                "defense_man_zone_type",
                "defense_coverage_type",
                "was_pressure",
                "offense_player_ids",
                "offense_player_names",
                "defense_player_ids",
                "defense_player_names",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return destination


def game_csv_filename(season: int, game_id: str) -> str:
    """Return a CSV filename with the home team before the away team."""
    game_rows = _load_pbp(season).filter(_load_pbp(season)["game_id"] == game_id)
    if game_rows.height == 0:
        raise ValueError(f"No play-by-play data found for season {season} and game_id {game_id!r}")

    game = game_rows.head(1).to_dicts()[0]
    week = game_id.split("_")[1]
    schedule_data = _load_schedule(season)
    schedule = schedule_data.filter(schedule_data["game_id"] == game_id)
    if schedule.height == 0:
        raise ValueError(f"No schedule data found for season {season} and game_id {game_id!r}")

    weekday = schedule.head(1).to_dicts()[0]["weekday"]
    return (
        f"nfl_plays_{season}_{week}_{weekday}_"
        f"{game['home_team']}_{game['away_team']}.csv"
    )


def game_csv_path(
    season: int,
    game_id: str,
    root_directory: str | Path = "games",
) -> Path:
    """Return the year/week folder path for a game's CSV."""
    week = game_id.split("_")[1]
    return (
        Path(root_directory)
        / str(season)
        / week
        / game_csv_filename(season, game_id)
    )


def export_season_games(
    season: int,
    root_directory: str | Path = "games",
) -> list[Path]:
    """Export every scheduled game for a season into its year/week folder."""
    schedule = _load_schedule(season)
    game_ids = schedule["game_id"].to_list()
    exported_paths = []

    for game_id in game_ids:
        output_path = game_csv_path(season, game_id, root_directory)
        exported_paths.append(
            write_games_to_csv(season, [game_id], output_path)
        )

    return exported_paths


def print_frame_summary(name: str, frame) -> None:
    """Print a compact summary of a Polars DataFrame."""
    print(f"\n{name}")
    print("- shape:", frame.shape)
    print("- columns:", frame.columns[:10])
    print(frame.head(3))


def main() -> None:
    # Current NFL season/week helpers
    print("Current season:", nfl.get_current_season())
    print("Current week:", nfl.get_current_week())

    # Small, fast examples using a recent season
    pbp = nfl.load_pbp(seasons=[2023])
    schedules = nfl.load_schedules(seasons=[2023])
    player_stats = nfl.load_player_stats(seasons=[2023], summary_level="reg")
    team_stats = nfl.load_team_stats(seasons=[2023], summary_level="reg")

    print_frame_summary("Play-by-play (2023)", pbp)
    print_frame_summary("Schedules (2023)", schedules)
    print_frame_summary("Player stats (2023 regular season)", player_stats)
    print_frame_summary("Team stats (2023 regular season)", team_stats)

    # Optional: convert to pandas if desired
    pbp_df = pbp.to_pandas()
    print("\nPolars -> pandas conversion works:")
    print(pbp_df.head(2))

    # Cache utilities
    print("\nClearing cache is available if you want to refresh downloads:")
    print("nfl.clear_cache()")

    # Example: get every lineup for a single game
    print("\nExample lineup extraction for one game:")
    game_lineups = get_game_lineups_for_every_play(2023, "2023_18_LA_SF")
    print(f"Found {len(game_lineups)} plays for game 2023_18_LA_SF")
    print(game_lineups[0])

    # Example: inspect the same play with yards gained/lost included
    print("\nExample lineup + yardage extraction:")
    game_lineups_with_yards = get_game_lineups_with_yards(2023, "2023_18_LA_SF")
    print(game_lineups_with_yards[0])

    # Export one or more games to a play-by-play CSV.
    game_id = "2023_18_LA_SF"
    csv_path = write_games_to_csv(
        2023,
        [game_id],
        game_csv_path(2023, game_id),
    )
    print(f"Wrote play CSV to {csv_path}")


if __name__ == "__main__":
    main()