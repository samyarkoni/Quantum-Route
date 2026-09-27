import csv
import tempfile
import unittest
from pathlib import Path

from nfl_RAPM import eligible_participants, fit_rapm, load_plays


FIELDS = [
    "season", "game_id", "play_id", "play_type", "yards_gained", "starting_yard",
    "down", "yds_to_go", "offense_player_ids", "offense_player_names",
    "defense_player_ids", "defense_player_names", "rusher_position",
    "rushing_player_type",
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
        result = fit_rapm(self.write([
            row("1", "kickoff"), row("2", offense="", defense="d1;d2"),
            row("3", "pass", yards="8"),
        ]))
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
        result = fit_rapm(self.write([
            row("1", "run", yards="10"), row("2", "run", yards="8"),
            row("3", "pass", yards="5"), row("4", "pass", yards="7"),
        ]), ridge=1)
        offense = [x for x in result.rows if x["side"] == "offense" and x["model"] == "rushing"]
        defense = [x for x in result.rows if x["side"] == "defense" and x["model"] == "rushing"]
        self.assertTrue(offense and defense)
        self.assertEqual({x["model"] for x in result.rows}, {"rushing", "passing"})
        self.assertIsInstance(offense[0]["estimated_yards_per_play"], float)

    def test_single_season_filter_and_output_label(self):
        result = fit_rapm(self.write([
            row("1", yards="10", season="2024"),
            row("2", yards="2", season="2023"),
        ]), season=2024)
        self.assertTrue(result.rows)
        self.assertEqual({item["season"] for item in result.rows}, {2024})
        self.assertEqual(result.diagnostics.excluded_other_seasons, 1)


if __name__ == "__main__":
    unittest.main()
