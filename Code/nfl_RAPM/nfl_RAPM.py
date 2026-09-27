"""Regularized adjusted plus-minus estimates for football participation data.

The model estimates adjusted associations with ``yards_gained``.  It is not a
causal decomposition of a play and should not be interpreted as proof that a
player caused a particular number of yards.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

SCRIMMAGE_TYPES = {"run", "pass", "qb_kneel", "qb_spike"}
SPECIAL_TEAMS_TYPES = {
    "kickoff", "punt", "field_goal", "extra_point", "free_kick",
    "onside_kick", "punt_return", "kickoff_return", "field_goal_blocked",
}
REQUIRED_COLUMNS = {"game_id", "play_id", "play_type", "yards_gained"}
IGNORED_DUPLICATE_COLUMNS = {"team", "opponent", "team_role"}
LEAKY_COLUMNS = {
    "yards_gained", "ending_yard", "home_team_score", "away_team_score",
    "time_of_play", "time_on_clock_start", "time_on_clock_end",
}


@dataclass
class Diagnostics:
    files_read: int = 0
    rows_read: int = 0
    duplicate_plays: int = 0
    conflicts: list[str] = field(default_factory=list)
    malformed_rows: list[str] = field(default_factory=list)
    missing_participation: int = 0
    excluded_play_types: Counter = field(default_factory=Counter)
    excluded_missing_yards: int = 0
    excluded_missing_participants: int = 0
    excluded_other_seasons: int = 0
    player_metadata_missing: list[str] = field(default_factory=list)


@dataclass
class FitResult:
    rows: list[dict[str, object]]
    diagnostics: Diagnostics
    controls: tuple[str, ...]
    ridge: float
    model: str
    convergence_iterations: int
    notes: tuple[str, ...] = ()

    def to_csv(self, path: str | Path) -> None:
        if not self.rows:
            raise ValueError("No estimates to export")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def _split_participants(value: object) -> list[str]:
    """The generator writes semicolon-delimited lists; tolerate common variants."""
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    delimiter = ";" if ";" in text else "|" if "|" in text else ","
    return [part.strip() for part in text.split(delimiter) if part.strip()]


def _participants(row: Mapping[str, object], side: str) -> list[tuple[str, str]]:
    ids = _split_participants(row.get(f"{side}_player_ids"))
    names = _split_participants(row.get(f"{side}_player_names"))
    if len(ids) != len(names):
        return []
    return list(dict.fromkeys(zip(ids, names)))


def load_plays(paths: str | Path | Iterable[str | Path]) -> tuple[list[dict[str, str]], Diagnostics]:
    """Read team files, deduplicate by (game_id, play_id), and report conflicts."""
    if isinstance(paths, (str, Path)):
        if Path(paths).is_dir():
            source_paths = sorted(set(Path(paths).glob("*.csv")) | set(Path(paths).glob("*/*.csv")))
        else:
            source_paths = [Path(paths)]
    else:
        source_paths = [Path(path) for path in paths]
    diagnostics = Diagnostics(files_read=len(source_paths))
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for source_path in source_paths:
        with source_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                raise ValueError(f"{source_path} is missing required columns {REQUIRED_COLUMNS}")
            for line, raw in enumerate(reader, 2):
                diagnostics.rows_read += 1
                row = {key: (value or "").strip() for key, value in raw.items()}
                key = (row["game_id"], row["play_id"])
                if not all(key):
                    diagnostics.malformed_rows.append(f"{source_path}:{line}: missing play key")
                    continue
                previous = unique.get(key)
                if previous is None:
                    unique[key] = row
                    continue
                diagnostics.duplicate_plays += 1
                for column, value in row.items():
                    if column in IGNORED_DUPLICATE_COLUMNS:
                        continue
                    if value != previous.get(column, ""):
                        diagnostics.conflicts.append(
                            f"{key}: {column!r} differs ({previous.get(column)!r} vs {value!r})"
                        )
    return list(unique.values()), diagnostics


def _number(row: Mapping[str, object], column: str) -> float | None:
    try:
        value = float(str(row.get(column, "")).strip())
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _is_qb(row: Mapping[str, object]) -> bool:
    return any("qb" in str(row.get(column, "")).lower()
               for column in ("rusher_position", "rushing_player_type"))


def eligible_participants(
    row: Mapping[str, object],
    policy: str = "field_players",
    quarterback_ids: set[str] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return offense and defense identities under the documented policy."""
    offense = _participants(row, "offense")
    defense = _participants(row, "defense")
    if policy == "all_listed":
        return offense, defense
    if policy != "field_players":
        raise ValueError("policy must be 'field_players' or 'all_listed'")
    if row.get("play_type") in {"run", "qb_kneel", "qb_spike"} and not _is_qb(row):
        qbs = quarterback_ids or set()
        offense = [(pid, name) for pid, name in offense if pid not in qbs]
    return offense, defense


def _control_values(row: Mapping[str, object]) -> list[tuple[str, float]]:
    # All controls are observed before the snap. Scores, clocks, ending_yard,
    # time_of_play, and yards_gained are deliberately not used.
    controls: list[tuple[str, float]] = []
    for column in ("starting_yard", "down", "yds_to_go", "n_defense",
                   "defenders_in_box", "number_of_pass_rushers"):
        value = _number(row, column)
        if value is not None:
            controls.append((column, value))
    return controls


def load_roster_metadata(season: int) -> dict[str, dict[str, str]]:
    """Load season-specific team and position mappings keyed by NFL player ID."""
    try:
        import nflreadpy
    except ImportError as exc:
        raise RuntimeError(
            "Player team and position output requires the nflreadpy package"
        ) from exc

    roster = nflreadpy.load_rosters(seasons=[season])
    required = {"gsis_id", "team", "position"}
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"nflreadpy roster data is missing columns: {sorted(missing)}")

    metadata: dict[str, dict[str, str]] = {}
    for row in roster.select(["gsis_id", "team", "position"]).to_dicts():
        player_id = str(row.get("gsis_id") or "").strip()
        if not player_id:
            continue
        value = {
            "team": str(row.get("team") or "").strip(),
            "position": str(row.get("position") or "").strip(),
        }
        previous = metadata.get(player_id)
        if previous is not None and previous != value:
            raise ValueError(
                f"nflreadpy returned conflicting season roster entries for {player_id}: "
                f"{previous} vs {value}"
            )
        metadata[player_id] = value
    return metadata


def _fit_sparse_ridge(
    rows: Sequence[tuple[float, list[tuple[str, float]]]],
    ridge: float,
    max_iterations: int = 80,
) -> tuple[dict[str, float], dict[str, float], int]:
    names = sorted({name for _, features in rows for name, _ in features})
    index = {name: i for i, name in enumerate(names)}
    columns: list[list[tuple[int, float]]] = [[] for _ in names]
    for r, (_, features) in enumerate(rows):
        for name, value in features:
            columns[index[name]].append((r, value))
    y = np.array([target for target, _ in rows], dtype=float)
    beta = np.zeros(len(names))
    residual = y.copy()
    penalty = np.array([0.0 if name == "intercept" else ridge for name in names])
    for iteration in range(1, max_iterations + 1):
        max_change = 0.0
        for j, column in enumerate(columns):
            if not column:
                continue
            old = beta[j]
            numerator = sum(value * (residual[r] + value * old) for r, value in column)
            denominator = sum(value * value for r, value in column) + penalty[j]
            beta[j] = numerator / denominator if denominator else 0.0
            delta = beta[j] - old
            if delta:
                for r, value in column:
                    residual[r] -= value * delta
                max_change = max(max_change, abs(delta))
        if max_change < 1e-7:
            break
    noise = float(np.sqrt(np.mean(residual * residual))) if len(residual) else float("nan")
    stderr = {}
    for j, column in enumerate(columns):
        information = sum(value * value for _, value in column) + penalty[j]
        stderr[names[j]] = noise / math.sqrt(information) if information else float("nan")
    return dict(zip(names, beta)), stderr, iteration


def fit_rapm(
    paths: str | Path | Iterable[str | Path],
    ridge: float = 10.0,
    policy: str = "field_players",
    quarterback_ids: set[str] | None = None,
    min_plays: int = 1,
    season: int | str | None = None,
) -> FitResult:
    """Fit separate jointly-adjusted rushing and passing yards models.

    Only ``season`` is fitted and every output row is labeled with it. If a
    source omits ``season``, the first component of its standard ``game_id`` is
    used when it is a four-digit year.
    """
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if season is None:
        raise ValueError("season is required to fit and label season-specific estimates")
    plays, diagnostics = load_plays(paths)
    training: dict[str, list[tuple[float, list[tuple[str, float]], dict[str, object]]]] = defaultdict(list)
    control_names = sorted({
        name for row in plays for name, _ in _control_values(row)
    })
    notes = [
        "Coefficients are adjusted associations with yards_gained, not causal effects.",
        "field_players excludes configured quarterback IDs from non-QB runs; absent IDs cannot be inferred from names.",
        "Every listed participant is treated as on the field; the source cannot resolve inactive/listing errors.",
        f"Controls are pre-play numeric fields only: {', '.join(control_names)}."
        if control_names else "No pre-play controls were observed.",
    ]
    for row in plays:
        row_season = str(row.get("season", "")).strip()
        if not row_season:
            game_season = str(row.get("game_id", "")).split("_", 1)[0]
            row_season = game_season if len(game_season) == 4 and game_season.isdigit() else ""
        if season is not None and row_season != str(season):
            diagnostics.excluded_other_seasons += 1
            continue
        play_type = str(row.get("play_type", "")).lower()
        if play_type in SPECIAL_TEAMS_TYPES or play_type not in SCRIMMAGE_TYPES:
            diagnostics.excluded_play_types[play_type] += 1
            continue
        target = _number(row, "yards_gained")
        offense, defense = eligible_participants(row, policy, quarterback_ids)
        if target is None:
            diagnostics.excluded_missing_yards += 1
            continue
        if not offense or not defense:
            diagnostics.missing_participation += 1
            diagnostics.excluded_missing_participants += 1
            continue
        model_type = "rushing" if play_type in {"run", "qb_kneel", "qb_spike"} else "passing"
        features = [("intercept", 1.0)] + [(f"control:{name}", value) for name, value in _control_values(row)]
        features += [(f"offense:{pid}", 1.0) for pid, _ in offense]
        features += [(f"defense:{pid}", -1.0) for pid, _ in defense]
        training[model_type].append((target, features, row))

    output: list[dict[str, object]] = []
    iterations = 0
    for model_type, observations in sorted(training.items()):
        if len(observations) < min_plays:
            continue
        coefficients, errors, iterations = _fit_sparse_ridge(
            [(target, features) for target, features, _ in observations], ridge
        )
        counts = Counter()
        names: dict[tuple[str, str], str] = {}
        for _, _, row in observations:
            offense, defense = eligible_participants(row, policy, quarterback_ids)
            for pid, name in offense:
                counts[("offense", pid)] += 1
                names[("offense", pid)] = name
            for pid, name in defense:
                counts[("defense", pid)] += 1
                names[("defense", pid)] = name
        for (side, pid), count in sorted(counts.items()):
            feature = f"{side}:{pid}"
            output.append({
                "player_id": pid, "player_name": names[(side, pid)], "side": side,
                "team": "", "position": "",
                "season": int(season) if season is not None and str(season).isdigit() else (row_season or ""),
                "model": model_type, "play_type": "run" if model_type == "rushing" else "pass",
                "estimated_yards_per_play": coefficients.get(feature, 0.0) * (1 if side == "offense" else -1),
                "standard_error": errors.get(feature, float("nan")),
                "plays_included": count,
                "model_plays": len(observations), "ridge": ridge,
            })
    if output:
        roster_metadata = load_roster_metadata(int(season))
        for result_row in output:
            metadata = roster_metadata.get(str(result_row["player_id"]))
            if metadata is None:
                diagnostics.player_metadata_missing.append(str(result_row["player_id"]))
                result_row["player_metadata_found"] = False
                continue
            result_row["team"] = metadata["team"]
            result_row["position"] = metadata["position"]
            result_row["player_metadata_found"] = bool(metadata["team"] and metadata["position"])
            if not result_row["player_metadata_found"]:
                diagnostics.player_metadata_missing.append(str(result_row["player_id"]))
    return FitResult(output, diagnostics, tuple(control_names), ridge,
                     "joint sparse ridge", iterations, tuple(notes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--policy", choices=("field_players", "all_listed"), default="field_players")
    args = parser.parse_args()
    fit_rapm(
        args.input_dir, ridge=args.ridge, policy=args.policy, season=args.season
    ).to_csv(args.output)


if __name__ == "__main__":
    main()
