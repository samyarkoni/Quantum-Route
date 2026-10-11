import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nfl_RAPM import (
    _low_leverage_band,
    eligible_participants,
    fit_rapm,
    load_plays,
)


FIELDS = [
    "season", "game_id", "play_id", "play_type", "yards_gained", "starting_yard",
    "down", "yds_to_go", "offense_player_ids", "offense_player_names",
    "defense_player_ids", "defense_player_names", "rusher_position",
    "rushing_player_type", "qtr", "game_seconds_remaining",
    "time_on_clock_start", "home_team_score", "away_team_score",
]


def row(play_id, play_type="run", yards="4", offense="o1;o2", defense="d1;d2",
        season="2024", **extra):
    values = {field: "" for field in FIELDS}
    values.update({
        "season": season, "game_id": f"{season}_01_A_B", "play_id": str(play_id),
        "play_type": play_type,
        "yards_gained": yards, "starting_yard": "50", "down": "1", "yds_to_go": "10",
        "offense_player_ids": offense, "offense_player_names": "Off One;Off Two",
        "defense_player_ids": defense, "defense_player_names": "Def One;Def Two",
        "qtr": "4", "time_on_clock_start": "15:00",
        "home_team_score": "0", "away_team_score": "0",
    })
    values.update(extra)
    return values


class FootballRAPMTests(unittest.TestCase):
    def write(self, rows):
        directory = Path(tempfile.mkdtemp())
        with (directory / "plays.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return directory

    def test_semicolon_lists_and_duplicate_conflict(self):
        directory = self.write([row("1"), row("1", yards="9")])
        plays, diagnostics = load_plays(directory / "plays.csv")
        self.assertEqual(len(plays), 1)
        self.assertTrue(diagnostics.conflicts)

    def test_special_teams_and_malformed_participation_are_excluded(self):
        with patch("nfl_RAPM.load_roster_metadata", return_value={}):
            result = fit_rapm(self.write([
                row("1", "kickoff"), row("2", offense="", defense="d1;d2"),
                row("3", "pass", yards="8"),
            ]), season=2024)
        self.assertEqual({item["play_type"] for item in result.rows}, {"pass"})
        self.assertEqual(result.diagnostics.excluded_play_types["kickoff"], 1)

    def test_qb_run_policy_and_all_listed_option(self):
        qb = row("1", offense="qb;o1", rusher_position="QB", rushing_player_type="QB")
        rb = row("2", offense="qb;o1", rusher_position="RB", rushing_player_type="RB")
        offense, _ = eligible_participants(qb, quarterback_ids={"qb"})
        self.assertEqual({pid for pid, _ in offense}, {"qb", "o1"})
        offense, _ = eligible_participants(rb, quarterback_ids={"qb"})
        self.assertEqual({pid for pid, _ in offense}, {"o1"})
        offense, _ = eligible_participants(rb, "all_listed")
        self.assertEqual({pid for pid, _ in offense}, {"qb", "o1"})

    def test_sign_convention_and_separate_models(self):
        with patch("nfl_RAPM.load_roster_metadata", return_value={}):
            result = fit_rapm(self.write([
                row("1", "run", yards="10"), row("2", "run", yards="8"),
                row("3", "pass", yards="5"), row("4", "pass", yards="7"),
            ]), ridge=1, season=2024)
        offense = [x for x in result.rows if x["side"] == "offense" and x["model"] == "rushing"]
        defense = [x for x in result.rows if x["side"] == "defense" and x["model"] == "rushing"]
        self.assertTrue(offense and defense)
        self.assertEqual({x["model"] for x in result.rows}, {"rushing", "passing"})
        self.assertIsInstance(offense[0]["estimated_yards_per_play"], float)

    def test_single_season_filter_and_output_label(self):
        with patch("nfl_RAPM.load_roster_metadata", return_value={}):
            result = fit_rapm(self.write([
                row("1", yards="10", season="2024"),
                row("2", yards="2", season="2023"),
            ]), season=2024)
        self.assertTrue(result.rows)
        self.assertEqual({item["season"] for item in result.rows}, {2024})
        self.assertEqual(result.diagnostics.excluded_other_seasons, 1)

    def test_roster_team_and_position_are_exported_by_player_id(self):
        metadata = {
            "o1": {"team": "ATL", "position": "RB"},
            "o2": {"team": "ATL", "position": "WR"},
            "d1": {"team": "PIT", "position": "LB"},
            "d2": {"team": "PIT", "position": "CB"},
        }
        with patch("nfl_RAPM.load_roster_metadata", return_value=metadata):
            result = fit_rapm(self.write([row("1")]), season=2024)
        player = next(item for item in result.rows if item["player_id"] == "o1")
        self.assertEqual((player["team"], player["position"]), ("ATL", "RB"))
        self.assertTrue(player["player_metadata_found"])

    def test_low_leverage_thresholds_and_more_than_30_minutes(self):
        self.assertIsNone(_low_leverage_band(1801))
        self.assertEqual(_low_leverage_band(1800), ("30:00-15:00", 21))
        self.assertEqual(_low_leverage_band(1799), ("30:00-15:00", 21))
        self.assertEqual(_low_leverage_band(899), ("15:00-10:00", 17))
        self.assertEqual(_low_leverage_band(599), ("10:00-5:00", 14))
        self.assertEqual(_low_leverage_band(299), ("5:00-3:00", 10))
        self.assertEqual(_low_leverage_band(179), ("3:00-0:00", 9))

    def test_filter_uses_preplay_score_and_reports_exclusions(self):
        touchdown = row(
            "1", season="2024", qtr="4", time_on_clock_start="02:00",
            home_team_score="21", away_team_score="0",
        )
        later = row(
            "2", season="2024", qtr="4", time_on_clock_start="01:50",
            home_team_score="21", away_team_score="0",
        )
        with patch("nfl_RAPM.load_roster_metadata", return_value={}):
            result = fit_rapm(self.write([touchdown, later]), season=2024)
        self.assertEqual(result.diagnostics.excluded_low_leverage_plays, 1)
        self.assertEqual(result.diagnostics.low_leverage_by_band["3:00-0:00"], 1)
        self.assertTrue(result.rows)
        self.assertEqual({entry["model_plays"] for entry in result.rows}, {1})


if __name__ == "__main__":
    unittest.main()
