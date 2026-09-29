"""Estimate yards-only football RAPM ratings from play-by-play CSV files."""

from __future__ import annotations

import argparse
import glob
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


LOG = logging.getLogger("football_rapm")
REQUIRED_COLUMNS = {
    "game_id",
    "play_id",
    "play_type",
    "yards_gained",
    "offense_player_ids",
    "offense_player_names",
    "defense_player_ids",
    "defense_player_names",
}
GAME_ID_PATTERN = re.compile(
    r"^(?P<season>\d{4})_(?P<week>\d{1,2})_(?P<home>[A-Za-z]{2,4})_(?P<away>[A-Za-z]{2,4})$"
)
POSITION_GROUPS = {
    "QB": {"QB"},
    "RB": {"RB", "FB", "HB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "OL": {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"},
    "DL": {"DE", "DT", "NT", "EDGE", "DL", "LE", "RE"},
    "LB": {"OLB", "ILB", "MLB", "LB"},
    "DB": {"CB", "S", "FS", "SS", "DB", "SAF"},
    "SPEC": {"K", "P", "LS"},
}


@dataclass
class FitResult:
    name: str
    frame: pd.DataFrame
    columns: list[tuple[str, str, str]]
    coefficients: np.ndarray
    counts: np.ndarray
    group_targets: np.ndarray
    group_keys: list[tuple[str, str, str]]
    lambda_value: float
    cv_rmse: float
    intercept_rmse: float
    baseline_rmse: float
    baseline: np.ndarray
    design: sparse.csr_matrix
    passes: int
    largest_change: float
    bootstrap: dict[tuple[str, str, str], tuple[float, float]]
    joint_coefficients: np.ndarray
    group_priors_enabled: bool


def normalize_yards(yards: float, cap: float = 15.0, shrink: float = 0.25) -> float:
    """Shrink yardage beyond ``cap`` toward the cap without changing its sign."""
    if cap < 0:
        raise ValueError("cap must be non-negative")
    if not 0 <= shrink <= 1:
        raise ValueError("shrink must be between 0 and 1")
    value = float(yards)
    magnitude = abs(value)
    if magnitude <= cap:
        return value
    return math.copysign(cap + shrink * (magnitude - cap), value)


def _resolve_inputs(input_value: str) -> list[Path]:
    candidate = Path(input_value).expanduser()
    if candidate.is_dir():
        paths = sorted(candidate.rglob("*.csv"))
    else:
        matches = glob.glob(input_value, recursive=True)
        paths = sorted(Path(match) for match in matches if Path(match).is_file())
        if not paths and candidate.is_file():
            paths = [candidate]
    if not paths:
        raise FileNotFoundError(f"No CSV files matched --input {input_value!r}")
    return paths


def _list_values(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    candidates = [";", "|", ","]
    delimiter = max(candidates, key=text.count)
    parts = text.split(delimiter) if text.count(delimiter) else text.split()
    return [part.strip() for part in parts if part.strip()]


def _derive_game_fields(frame: pd.DataFrame) -> None:
    parsed = frame["game_id"].astype("string").str.extract(GAME_ID_PATTERN)
    if "season" not in frame:
        frame["season"] = parsed["season"]
    else:
        frame["season"] = frame["season"].fillna(parsed["season"])
    if "week" not in frame:
        frame["week"] = parsed["week"]
    else:
        frame["week"] = frame["week"].fillna(parsed["week"])
    frame["home_team_derived"] = parsed["home"]
    frame["away_team_derived"] = parsed["away"]


def load_plays(input_value: str) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict[str, str]]:
    """Load, coalesce duplicate team files, and collect player names and positions."""
    paths = _resolve_inputs(input_value)
    LOG.info("Reading %d CSV file(s)", len(paths))
    chunks: list[pd.DataFrame] = []
    for path in paths:
        chunk = pd.read_csv(path, dtype="string", low_memory=False)
        missing = REQUIRED_COLUMNS.difference(chunk.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        chunk["_source_file"] = str(path)
        chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True, sort=False)
    for key in ("game_id", "play_id"):
        data[key] = data[key].astype("string").str.strip()
    invalid_keys = data["game_id"].isna() | data["game_id"].eq("") | data["play_id"].isna() | data[
        "play_id"
    ].eq("")
    if invalid_keys.any():
        raise ValueError(f"{int(invalid_keys.sum())} input row(s) have missing game_id or play_id")
    before = len(data)
    data = data.groupby(["game_id", "play_id"], sort=False, dropna=False).first().reset_index()
    LOG.info("Rows before de-duplication: %s; after: %s", f"{before:,}", f"{len(data):,}")
    _derive_game_fields(data)
    seasons = sorted(data["season"].dropna().astype(str).unique().tolist())
    teams: set[str] = set()
    for column in ("team", "posteam", "opponent"):
        if column in data:
            teams.update(data[column].dropna().astype(str).str.strip().unique().tolist())
    LOG.info(
        "Loaded seasons: %s; teams identified: %s; games: %s",
        ", ".join(seasons) or "unknown",
        len(teams),
        data["game_id"].nunique(),
    )
    source_team_count = data["team"].nunique(dropna=True) if "team" in data else 0
    if source_team_count == 1:
        LOG.warning(
            "Only one team's file appears to be supplied. Opponents' players may have "
            "limited game coverage and will be heavily shrunk."
        )

    names: defaultdict[str, Counter[str]] = defaultdict(Counter)
    defender_positions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    rusher_positions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    mismatched_names = 0
    defense_position_rows = 0
    defense_position_mismatches = 0
    for row in data.itertuples(index=False):
        row_data = row._asdict()
        raw_play_type = row_data.get("play_type")
        normalized_play_type = (
            "" if raw_play_type is None or pd.isna(raw_play_type) else str(raw_play_type).strip().lower()
        )
        modeled_play = normalized_play_type in {"run", "rush", "pass"}
        for id_col, name_col in (
            ("offense_player_ids", "offense_player_names"),
            ("defense_player_ids", "defense_player_names"),
        ):
            ids = _list_values(row_data.get(id_col))
            player_names = _list_values(row_data.get(name_col))
            if len(ids) != len(player_names):
                mismatched_names += 1
            else:
                for player_id, player_name in zip(ids, player_names):
                    names[player_id][player_name] += 1
        defense_ids = _list_values(row_data.get("defense_player_ids"))
        positions = _list_values(row_data.get("defense_positions"))
        if modeled_play and defense_ids and "defense_positions" in data.columns:
            defense_position_rows += 1
            if len(positions) != len(defense_ids):
                defense_position_mismatches += 1
            else:
                for player_id, position in zip(defense_ids, positions):
                    defender_positions[player_id][position.upper()] += 1
        rusher_id = row_data.get("rusher_player_id")
        rusher_position = row_data.get("rusher_position")
        if rusher_id is not None and not pd.isna(rusher_id) and str(rusher_id).strip():
            rusher_name = row_data.get("rusher_player_name")
            if rusher_name is not None and not pd.isna(rusher_name) and str(rusher_name).strip():
                names[str(rusher_id).strip()][str(rusher_name).strip()] += 1
        if (
            modeled_play
            and normalized_play_type in {"run", "rush"}
            and rusher_id is not None
            and not pd.isna(rusher_id)
            and str(rusher_id).strip()
        ):
            if rusher_position is not None and not pd.isna(rusher_position):
                rusher_positions[str(rusher_id).strip()][str(rusher_position).strip().upper()] += 1
    if mismatched_names:
        LOG.warning(
            "%d row(s) have unequal player ID/name list lengths; matched entries are retained",
            mismatched_names,
        )
    position_mismatch_rate = (
        defense_position_mismatches / defense_position_rows if defense_position_rows else 0.0
    )
    if defense_position_mismatches:
        LOG.info(
            "Defense position lists align on %.1f%% of populated rows",
            100 * (1 - position_mismatch_rate),
        )
    if position_mismatch_rate > 0.10:
        LOG.warning("Defense position alignment is unreliable; ignoring defense_positions")
        defender_positions.clear()
    best_names = {player_id: counts.most_common(1)[0][0] for player_id, counts in names.items()}
    best_defense_positions = {
        player_id: counts.most_common(1)[0][0]
        for player_id, counts in defender_positions.items()
    }
    best_rusher_positions = {
        player_id: counts.most_common(1)[0][0] for player_id, counts in rusher_positions.items()
    }
    return data, best_names, best_defense_positions, best_rusher_positions


def _load_nflreadpy_positions(player_ids: set[str]) -> dict[str, str]:
    """Map observed GSIS IDs to primary positions using nflreadpy's player database."""
    import nflreadpy as nfl

    players = nfl.load_players()
    required = {"gsis_id", "position"}
    missing = required.difference(players.columns)
    if missing:
        raise ValueError(
            "nflreadpy player database is missing required columns: "
            + ", ".join(sorted(missing))
        )
    selected_columns = ["gsis_id", "position"]
    if "ngs_position" in players.columns:
        selected_columns.append("ngs_position")
    database_positions: dict[str, str] = {}
    for row in players.select(selected_columns).to_dicts():
        raw_id = row.get("gsis_id")
        if raw_id is None or pd.isna(raw_id):
            continue
        player_id = str(raw_id).strip()
        raw_position = row.get("position")
        if raw_position is None or pd.isna(raw_position) or not str(raw_position).strip():
            raw_position = row.get("ngs_position")
        if raw_position is not None and not pd.isna(raw_position):
            position = str(raw_position).strip().upper()
            if player_id and position:
                database_positions[player_id] = position
    matched = player_ids.intersection(database_positions)
    LOG.info(
        "nflreadpy player database supplied positions for %s/%s observed IDs (%.1f%%)",
        f"{len(matched):,}",
        f"{len(player_ids):,}",
        100 * len(matched) / len(player_ids) if player_ids else 100.0,
    )
    return {player_id: database_positions[player_id] for player_id in matched}


def _position_group(position: str | None) -> str:
    """Map the player's roster position independent of which side listed the player."""
    if not position:
        return "UNK"
    normalized = re.sub(r"[^A-Z]", "", str(position).upper())
    for group, positions in POSITION_GROUPS.items():
        if normalized in positions:
            return group
    return "UNK"


def _load_offense_positions(
    position_file: str | None,
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    if not position_file:
        return {}, {}
    positions = pd.read_csv(position_file, dtype="string")
    required = {"player_id", "position"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"--positions CSV is missing: {', '.join(sorted(missing))}")
    global_positions: dict[str, str] = {}
    season_positions: dict[tuple[str, str], str] = {}
    for row in positions.itertuples(index=False):
        raw_id, raw_position = getattr(row, "player_id"), getattr(row, "position")
        if pd.isna(raw_id) or pd.isna(raw_position):
            continue
        player_id = str(raw_id).strip()
        position = str(raw_position).strip().upper()
        if not player_id or not position:
            continue
        if hasattr(row, "season") and not pd.isna(getattr(row, "season")):
            season_positions[(str(getattr(row, "season")), player_id)] = position
        else:
            global_positions[player_id] = position
    return global_positions, season_positions


def _print_data_checks(data: pd.DataFrame) -> None:
    play_types = data["play_type"].fillna("<missing>").str.lower().value_counts(dropna=False)
    LOG.info("Play type counts before filtering:\n%s", play_types.to_string())
    for column in ("rusher_position", "rushing_player_type"):
        if column in data:
            counts = data[column].fillna("<missing>").value_counts(dropna=False)
            LOG.info("%s value counts:\n%s", column, counts.to_string())
        else:
            LOG.info("%s is absent", column)
    play_type = data["play_type"].fillna("").str.lower()
    rushing = play_type.isin({"run", "rush"})
    has_rusher = (
        data["rusher_player_id"].notna() & data["rusher_player_id"].astype("string").str.strip().ne("")
        if "rusher_player_id" in data
        else pd.Series(False, index=data.index)
    )
    offense_missing = data["offense_player_ids"].fillna("").str.strip().eq("")
    defense_missing = data["defense_player_ids"].fillna("").str.strip().eq("")
    off_counts = data["offense_player_ids"].map(lambda value: len(_list_values(value)))
    def_counts = data["defense_player_ids"].map(lambda value: len(_list_values(value)))
    LOG.info("Rushing plays with rusher ID: %.1f%%", 100 * has_rusher[rushing].mean() if rushing.any() else 0)
    if "rusher_player_id" in data:
        rusher_values = data["rusher_player_id"].fillna("").astype(str).str.strip()
        missing_from_lineup = sum(
            bool(is_rush and rusher_id and rusher_id not in _list_values(offense_ids))
            for is_rush, rusher_id, offense_ids in zip(
                rushing, rusher_values, data["offense_player_ids"]
            )
        )
        LOG.info(
            "Rushing plays with a rusher ID absent from offense_player_ids: %d",
            missing_from_lineup,
        )
    else:
        LOG.info("rusher_player_id is absent; no ball-carrier variables can be built")
    LOG.info(
        "Plays missing either player list: %.1f%%",
        100 * (offense_missing | defense_missing).mean() if len(data) else 0,
    )
    non_eleven = (off_counts.ne(11) | def_counts.ne(11)).mean() if len(data) else 0
    LOG.info("Plays with a parsed player count other than 11 on either side: %.1f%%", 100 * non_eleven)


def _score_is_running(data: pd.DataFrame) -> bool:
    if not {"home_team_score", "away_team_score"}.issubset(data.columns):
        return False
    scores = data[["game_id", "home_team_score", "away_team_score"]].copy()
    scores[["home_team_score", "away_team_score"]] = scores[
        ["home_team_score", "away_team_score"]
    ].apply(pd.to_numeric, errors="coerce")
    pairs_per_game = scores.groupby("game_id", dropna=False)[
        ["home_team_score", "away_team_score"]
    ].nunique(dropna=True)
    varying = pairs_per_game.gt(1).any(axis=1)
    if not varying.empty and not varying.any():
        LOG.info("Score fields are constant within each game; treating them as final scores")
        return False
    if varying.any():
        LOG.info("Score fields vary within games; using offense-relative score differential")
        return True
    return False


def _score_differential(data: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=data.index, dtype=float)
    if not {"home_team_score", "away_team_score"}.issubset(data.columns):
        return result
    home_score = pd.to_numeric(data["home_team_score"], errors="coerce")
    away_score = pd.to_numeric(data["away_team_score"], errors="coerce")
    posteam = data["posteam"].astype("string").str.upper() if "posteam" in data else None
    if posteam is not None:
        home = data["home_team_derived"].astype("string").str.upper()
        away = data["away_team_derived"].astype("string").str.upper()
        derived_home = posteam.eq(home)
        derived_away = posteam.eq(away)
        result.loc[derived_home] = (home_score - away_score).loc[derived_home]
        result.loc[derived_away] = (away_score - home_score).loc[derived_away]
    elif "team_role" in data and "team" in data:
        # A team file's role is not necessarily the offense's role; without posteam
        # the score differential cannot safely be oriented toward the offense.
        LOG.warning("Cannot orient score differential without posteam; omitting it")
    return result


def clean_plays(
    data: pd.DataFrame, cap: float, shrink: float, use_score: bool
) -> tuple[pd.DataFrame, dict[str, int], float]:
    """Filter to run/pass plays and add normalized yards while retaining raw yards."""
    _print_data_checks(data)
    play_type = data["play_type"].fillna("").str.strip().str.lower()
    keep_type = play_type.isin({"run", "rush", "pass"})
    missing_play_type = data["play_type"].isna() | play_type.eq("")
    yards = pd.to_numeric(data["yards_gained"], errors="coerce")
    has_yards = yards.notna() & np.isfinite(yards)
    has_players = data["offense_player_ids"].map(lambda value: bool(_list_values(value))) & data[
        "defense_player_ids"
    ].map(lambda value: bool(_list_values(value)))
    dropped = {
        "missing_play_type": int(missing_play_type.sum()),
        "unsupported_play_type": int((~keep_type & ~missing_play_type).sum()),
        "missing_or_nonfinite_yards": int((keep_type & ~has_yards).sum()),
        "missing_player_lists": int((keep_type & has_yards & ~has_players).sum()),
    }
    mismatch_rate = float("nan")
    if {"starting_yard", "ending_yard"}.issubset(data.columns):
        starting = pd.to_numeric(data["starting_yard"], errors="coerce")
        ending = pd.to_numeric(data["ending_yard"], errors="coerce")
        comparable = starting.notna() & ending.notna() & yards.notna()
        if comparable.any():
            mismatch = (starting[comparable] - ending[comparable]).abs().sub(
                yards[comparable].abs()
            ).abs().gt(1.0)
            mismatch_rate = float(mismatch.mean())
            LOG.info(
                "Starting/ending yardage absolute-distance mismatch rate (1-yard tolerance): %.1f%%",
                100 * mismatch_rate,
            )
    LOG.info("Dropped rows: %s", dropped)
    cleaned = data.loc[keep_type & has_yards & has_players].copy()
    cleaned["play_type"] = np.where(
        cleaned["play_type"].str.strip().str.lower().isin({"run", "rush"}), "rush", "pass"
    )
    cleaned["raw_yards"] = yards.loc[cleaned.index].astype(float)
    cleaned["yards"] = cleaned["raw_yards"].map(lambda value: normalize_yards(value, cap, shrink))
    if "season" not in cleaned:
        cleaned["season"] = ""
    cleaned["season"] = cleaned["season"].fillna("").astype(str)
    if use_score:
        cleaned["score_diff"] = _score_differential(cleaned)
    else:
        cleaned["score_diff"] = np.nan
    return cleaned.reset_index(drop=True), dropped, mismatch_rate


def _make_position_maps(
    defense_positions: dict[str, str],
    rusher_positions: dict[str, str],
    user_global: dict[str, str],
    user_season: dict[tuple[str, str], str],
    nflreadpy_positions: dict[str, str],
) -> tuple[dict[str, str], dict[tuple[str, str], str], dict[str, str]]:
    offense_positions = dict(nflreadpy_positions)
    offense_positions.update(rusher_positions)
    offense_positions.update(user_global)
    defense_position_map = dict(nflreadpy_positions)
    defense_position_map.update(defense_positions)
    return offense_positions, dict(user_season), defense_position_map


def _player_position(
    player_id: str,
    season: str,
    side: str,
    offense_positions: dict[str, str],
    season_positions: dict[tuple[str, str], str],
    defense_positions: dict[str, str],
) -> str:
    if side == "defense":
        return defense_positions.get(player_id, "UNK")
    if (season, player_id) in season_positions:
        return season_positions[(season, player_id)]
    return offense_positions.get(player_id, "UNK")


def _make_design(
    frame: pd.DataFrame,
    by_season: bool,
    offense_positions: dict[str, str],
    season_positions: dict[tuple[str, str], str],
    defense_positions: dict[str, str],
    names: dict[str, str],
) -> tuple[
    sparse.csr_matrix,
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    np.ndarray,
    dict[tuple[str, str, str], dict[str, str]],
]:
    """Construct the sparse player matrix and its player/group metadata."""
    records: list[list[tuple[str, str, str]]] = []
    player_metadata: dict[tuple[str, str, str], dict[str, str]] = {}
    columns_seen: set[tuple[str, str, str]] = set()
    row_play_types = frame["play_type"].to_numpy()
    for row_number, row in enumerate(frame.itertuples(index=False)):
        values = row._asdict()
        raw_season = values.get("season")
        season = "" if raw_season is None or pd.isna(raw_season) else str(raw_season)
        suffix = f"@{season}" if by_season else ""
        offense_ids = list(dict.fromkeys(_list_values(values.get("offense_player_ids"))))
        defense_ids = list(dict.fromkeys(_list_values(values.get("defense_player_ids"))))
        rusher = values.get("rusher_player_id")
        carrier_id = str(rusher).strip() if rusher is not None and not pd.isna(rusher) else ""
        if row_play_types[row_number] != "rush":
            carrier_id = ""
        row_columns: list[tuple[str, str, str]] = []
        for side, player_ids in (("offense", offense_ids), ("defense", defense_ids)):
            for player_id in player_ids:
                role = "on_field"
                if side == "offense" and player_id == carrier_id:
                    role = "ball_carrier"
                player_key = player_id + suffix
                column = (side, role, player_key)
                row_columns.append(column)
                columns_seen.add(column)
                position = _player_position(
                    player_id,
                    season,
                    side,
                    offense_positions,
                    season_positions,
                    defense_positions,
                )
                group = _position_group(position)
                player_metadata.setdefault(
                    column,
                    {
                        "player_id": player_id,
                        "player_name": names.get(player_id, ""),
                        "position_group": group,
                        "season": season if by_season else "",
                    },
                )
            if side == "offense" and carrier_id and carrier_id not in player_ids:
                player_key = carrier_id + suffix
                column = ("offense", "ball_carrier", player_key)
                row_columns.append(column)
                columns_seen.add(column)
                position = _player_position(
                    carrier_id,
                    season,
                    side,
                    offense_positions,
                    season_positions,
                    defense_positions,
                )
                player_metadata.setdefault(
                    column,
                    {
                        "player_id": carrier_id,
                        "player_name": names.get(carrier_id, ""),
                        "position_group": _position_group(position),
                        "season": season if by_season else "",
                    },
                )
        records.append(row_columns)
    columns = sorted(columns_seen)
    column_index = {column: index for index, column in enumerate(columns)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, row_columns in enumerate(records):
        for column in row_columns:
            row_indices.append(row_index)
            column_indices.append(column_index[column])
    values = np.fromiter(
        (-1.0 if columns[column][0] == "defense" else 1.0 for column in column_indices),
        dtype=np.float64,
        count=len(column_indices),
    )
    design = sparse.csr_matrix(
        (values, (row_indices, column_indices)), shape=(len(frame), len(columns))
    )
    counts = np.asarray((design != 0).sum(axis=0)).ravel().astype(float)
    group_keys = [
        (
            column[0],
            column[1],
            player_metadata[column]["position_group"],
        )
        for column in columns
    ]
    return design, columns, group_keys, counts, player_metadata


def _baseline_features(frame: pd.DataFrame, scheme_controls: bool) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for column in (
        "down",
        "yds_to_go",
        "yardline_100",
        "qtr",
        "game_seconds_remaining",
    ):
        if column in frame:
            features[column] = pd.to_numeric(frame[column], errors="coerce")
    if "score_diff" in frame and frame["score_diff"].notna().any():
        features["score_diff"] = pd.to_numeric(frame["score_diff"], errors="coerce")
    if scheme_controls:
        for column in ("defense_personnel", "n_defense", "defenders_in_box"):
            if column in frame:
                if column == "defense_personnel":
                    features[column] = frame[column].fillna("<missing>").astype(str)
                else:
                    features[column] = pd.to_numeric(frame[column], errors="coerce")
    if features.empty:
        features["constant"] = 0.0
    categorical = [column for column in features if not pd.api.types.is_numeric_dtype(features[column])]
    if categorical:
        features = pd.get_dummies(features, columns=categorical, dummy_na=True, dtype=float)
    return features.astype(float)


def _split_indices(
    groups: np.ndarray, folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_games = pd.unique(groups)
    if len(unique_games) < 2:
        return []
    splitter = GroupKFold(n_splits=min(folds, len(unique_games)))
    return [(train, test) for train, test in splitter.split(np.zeros(len(groups)), groups=groups)]


def _baseline_oof(
    features: pd.DataFrame, target: np.ndarray, frame: pd.DataFrame, folds: int, seed: int
) -> np.ndarray:
    predictions = np.full(len(target), np.nan, dtype=float)
    split_indices = _split_indices(frame["game_id"].astype(str).to_numpy(), folds)
    if not split_indices:
        LOG.warning("Fewer than two games in a play type; using its mean as baseline")
        return np.full(len(target), float(np.mean(target)) if len(target) else 0.0)
    play_types = frame["play_type"].to_numpy()
    for play_type in np.unique(play_types):
        rows = np.flatnonzero(play_types == play_type)
        row_set = set(rows.tolist())
        for train_all, test_all in split_indices:
            train = np.asarray([idx for idx in train_all if idx in row_set], dtype=int)
            test = np.asarray([idx for idx in test_all if idx in row_set], dtype=int)
            if not len(test):
                continue
            if not len(train):
                predictions[test] = float(np.mean(target[rows]))
                continue
            min_leaf = max(25, min(200, len(train) // 40))
            model = HistGradientBoostingRegressor(
                max_iter=40,
                max_leaf_nodes=7,
                max_depth=3,
                min_samples_leaf=min_leaf,
                l2_regularization=2.0,
                random_state=seed,
            )
            model.fit(features.iloc[train], target[train])
            predictions[test] = model.predict(features.iloc[test])
    missing = np.isnan(predictions)
    if missing.any():
        predictions[missing] = float(np.mean(target))
    return predictions


def _baseline_fit_predict(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    train_frame: pd.DataFrame,
    test_features: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    predictions = np.full(len(test_frame), np.nan, dtype=float)
    train_types = train_frame["play_type"].to_numpy()
    test_types = test_frame["play_type"].to_numpy()
    fallback = float(np.mean(train_target)) if len(train_target) else 0.0
    for play_type in np.unique(test_types):
        train = np.flatnonzero(train_types == play_type)
        test = np.flatnonzero(test_types == play_type)
        if not len(train):
            predictions[test] = fallback
            continue
        min_leaf = max(25, min(200, len(train) // 40))
        model = HistGradientBoostingRegressor(
            max_iter=40,
            max_leaf_nodes=7,
            max_depth=3,
            min_samples_leaf=min_leaf,
            l2_regularization=2.0,
            random_state=seed,
        )
        model.fit(train_features.iloc[train], train_target[train])
        predictions[test] = model.predict(test_features.iloc[test])
    missing = np.isnan(predictions)
    if missing.any():
        predictions[missing] = fallback
    return predictions


def _nested_cv_baselines(
    features: pd.DataFrame,
    target: np.ndarray,
    frame: pd.DataFrame,
    folds: int,
    seed: int,
    type_means_only: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    games = frame["game_id"].astype(str).to_numpy()
    outer_splits = _split_indices(games, folds)
    validation_baselines = []
    play_types = frame["play_type"].to_numpy()
    for fold_number, (train, test) in enumerate(outer_splits):
        train_frame = frame.iloc[train].reset_index(drop=True)
        test_frame = frame.iloc[test].reset_index(drop=True)
        train_target = target[train]
        if type_means_only:
            means = (
                pd.DataFrame({"play_type": play_types[train], "target": train_target})
                .groupby("play_type")["target"]
                .mean()
            )
            fallback = float(np.mean(train_target))
            train_baseline = np.asarray(
                [means.get(kind, fallback) for kind in play_types[train]], dtype=float
            )
            test_baseline = np.asarray(
                [means.get(kind, fallback) for kind in play_types[test]], dtype=float
            )
        else:
            train_features = features.iloc[train].reset_index(drop=True)
            test_features = features.iloc[test].reset_index(drop=True)
            train_baseline = _baseline_oof(
                train_features,
                train_target,
                train_frame,
                folds,
                seed + fold_number,
            )
            test_baseline = _baseline_fit_predict(
                train_features,
                train_target,
                train_frame,
                test_features,
                test_frame,
                seed + fold_number,
            )
        validation_baselines.append((train, test, train_baseline, test_baseline))
    return validation_baselines


def _baseline_predictions(
    features: pd.DataFrame, target: np.ndarray, frame: pd.DataFrame, folds: int, seed: int
) -> np.ndarray:
    return _baseline_oof(features, target, frame, folds, seed)


def _initial_type_mean(target: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    means = frame.assign(_target=target).groupby("play_type")["_target"].mean()
    return frame["play_type"].map(means).to_numpy(dtype=float)


def _ridge_fit(
    design: sparse.csr_matrix,
    target: np.ndarray,
    alpha: float,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, float]:
    estimator = Ridge(alpha=float(alpha), fit_intercept=fit_intercept, solver="lsqr")
    estimator.fit(design, target)
    return np.asarray(estimator.coef_).ravel(), float(estimator.intercept_)


def _cross_validated_rmse(
    design: sparse.csr_matrix,
    target: np.ndarray,
    prior: np.ndarray,
    validation_baselines: Sequence[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    alpha: float,
) -> float:
    if not validation_baselines:
        return float("nan")
    squared_error = 0.0
    observations = 0
    for train, test, train_baseline, test_baseline in validation_baselines:
        centered_target = (
            target[train]
            - train_baseline
            - np.asarray(design[train] @ prior).ravel()
        )
        beta, intercept = _ridge_fit(design[train], centered_target, alpha)
        prediction = (
            test_baseline
            + np.asarray(design[test] @ (prior + beta)).ravel()
            + intercept
        )
        squared_error += float(np.square(target[test] - prediction).sum())
        observations += len(test)
    return math.sqrt(squared_error / observations) if observations else float("nan")


def _intercept_cv_rmse(target: np.ndarray, frame: pd.DataFrame, folds: int) -> float:
    splits = _split_indices(frame["game_id"].astype(str).to_numpy(), folds)
    if not splits:
        return float("nan")
    squared_error = 0.0
    observations = 0
    types = frame["play_type"].to_numpy()
    for train, test in splits:
        means = pd.DataFrame({"type": types[train], "y": target[train]}).groupby("type")["y"].mean()
        prediction = np.asarray([means.get(kind, float(np.mean(target[train]))) for kind in types[test]])
        squared_error += float(np.square(target[test] - prediction).sum())
        observations += len(test)
    return math.sqrt(squared_error / observations) if observations else float("nan")


def _group_means(
    coefficients: np.ndarray,
    counts: np.ndarray,
    group_keys: Sequence[tuple[str, str, str]],
) -> np.ndarray:
    accum: defaultdict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for beta, count, group in zip(coefficients, counts, group_keys):
        accum[group][0] += float(beta) * float(count)
        accum[group][1] += float(count)
    means = {
        group: total / count if count else 0.0
        for group, (total, count) in accum.items()
    }
    return np.asarray([means[group] for group in group_keys], dtype=float)


def _alternate_sides(
    design: sparse.csr_matrix,
    target: np.ndarray,
    baseline: np.ndarray,
    columns: Sequence[tuple[str, str, str]],
    counts: np.ndarray,
    group_keys: Sequence[tuple[str, str, str]],
    alpha: float,
    max_passes: int,
    tolerance: float,
    group_priors: bool,
    initial_coefficients: np.ndarray | None = None,
    log_name: str = "fit",
    cv_rmse: float = float("nan"),
    baseline_refit: Callable[[np.ndarray, int], np.ndarray] | None = None,
    log_iterations: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Alternate offense and defense ridge blocks using one fixed penalty."""
    sides = np.asarray([column[0] for column in columns])
    offense_indices = np.flatnonzero(sides == "offense")
    defense_indices = np.flatnonzero(sides == "defense")
    offense = design[:, offense_indices].tocsr()
    defense = design[:, defense_indices].tocsr()
    offense_counts = counts[offense_indices]
    defense_counts = counts[defense_indices]
    offense_groups = [group_keys[index] for index in offense_indices]
    defense_groups = [group_keys[index] for index in defense_indices]
    coefficients = (
        np.zeros(design.shape[1], dtype=float)
        if initial_coefficients is None
        else np.asarray(initial_coefficients, dtype=float).copy()
    )
    offense_beta = coefficients[offense_indices].copy()
    defense_beta = coefficients[defense_indices].copy()
    offense_prior = (
        _group_means(offense_beta, offense_counts, offense_groups)
        if group_priors
        else np.zeros(len(offense_indices), dtype=float)
    )
    defense_prior = (
        _group_means(defense_beta, defense_counts, defense_groups)
        if group_priors
        else np.zeros(len(defense_indices), dtype=float)
    )
    previous_offense_deviation: np.ndarray | None = None
    previous_defense_deviation: np.ndarray | None = None
    largest_change = float("inf")
    passes = 0
    for pass_number in range(max_passes):
        offense_residual = (
            target
            - baseline
            - np.asarray(defense @ defense_beta).ravel()
            - np.asarray(offense @ offense_prior).ravel()
        )
        offense_deviation, _ = _ridge_fit(
            offense, offense_residual, alpha, fit_intercept=False
        )
        offense_beta = offense_prior + offense_deviation
        if group_priors:
            offense_prior = _group_means(offense_beta, offense_counts, offense_groups)
        else:
            offense_prior.fill(0.0)

        defense_residual = (
            target
            - baseline
            - np.asarray(offense @ offense_beta).ravel()
            - np.asarray(defense @ defense_prior).ravel()
        )
        defense_deviation, _ = _ridge_fit(
            defense, defense_residual, alpha, fit_intercept=False
        )
        defense_beta = defense_prior + defense_deviation
        if group_priors:
            defense_prior = _group_means(defense_beta, defense_counts, defense_groups)
        else:
            defense_prior.fill(0.0)

        offense_deviation = offense_beta - offense_prior
        defense_deviation = defense_beta - defense_prior
        changes: list[float] = []
        if previous_offense_deviation is not None and len(offense_deviation):
            changes.append(
                float(np.max(np.abs(offense_deviation - previous_offense_deviation)))
            )
        if previous_defense_deviation is not None and len(defense_deviation):
            changes.append(
                float(np.max(np.abs(defense_deviation - previous_defense_deviation)))
            )
        largest_change = max(changes) if changes else float("inf")
        passes = pass_number + 1
        if log_iterations:
            LOG.info(
                "%s pass %d (offense/defense alternation): lambda=%.5g CV RMSE=%.4f "
                "largest deviation change=%.6f",
                log_name,
                passes + 1,
                alpha,
                cv_rmse,
                largest_change,
            )
        previous_offense_deviation = offense_deviation.copy()
        previous_defense_deviation = defense_deviation.copy()
        coefficients[offense_indices] = offense_beta
        coefficients[defense_indices] = defense_beta
        if baseline_refit is not None:
            residual = target - np.asarray(design @ coefficients).ravel()
            baseline = baseline_refit(residual, pass_number)
        if passes > 1 and largest_change < tolerance:
            break
    coefficients[offense_indices] = offense_beta
    coefficients[defense_indices] = defense_beta
    targets = np.zeros_like(coefficients)
    targets[offense_indices] = offense_prior
    targets[defense_indices] = defense_prior
    return coefficients, targets, baseline, passes, largest_change


def fit_model(
    name: str,
    frame: pd.DataFrame,
    design: sparse.csr_matrix,
    columns: list[tuple[str, str, str]],
    group_keys: list[tuple[str, str, str]],
    counts: np.ndarray,
    features: pd.DataFrame,
    lambdas: Sequence[float],
    folds: int,
    max_passes: int,
    tolerance: float,
    group_priors: bool,
    seed: int,
) -> FitResult:
    """Fit one joint ridge model, then alternate offense/defense blocks."""
    y = frame["yards"].to_numpy(dtype=float)
    initial_baseline = _initial_type_mean(y, frame)
    alpha = float(lambdas[len(lambdas) // 2])
    intercept_rmse = _intercept_cv_rmse(y, frame, folds)
    baseline_only = _baseline_predictions(features, y, frame, folds, seed)
    baseline_rmse = float(np.sqrt(np.mean(np.square(y - baseline_only))))
    validation_baselines = _nested_cv_baselines(
        features, y, frame, folds, seed, type_means_only=True
    )
    cv_scores = [
        _cross_validated_rmse(
            design, y, np.zeros(design.shape[1]), validation_baselines, candidate
        )
        for candidate in lambdas
    ]
    finite_scores = [
        (score, float(candidate))
        for score, candidate in zip(cv_scores, lambdas)
        if np.isfinite(score)
    ]
    if finite_scores:
        final_cv, alpha = min(finite_scores)
    else:
        final_cv = float("nan")
        LOG.warning("%s: grouped CV unavailable; using lambda %.5g", name, alpha)
    if alpha in {float(lambdas[0]), float(lambdas[-1])}:
        LOG.warning(
            "%s selected an edge lambda (%.5g); consider widening --lambda-grid",
            name,
            alpha,
        )
    joint_deviations, joint_intercept = _ridge_fit(
        design, y - initial_baseline, alpha, fit_intercept=True
    )
    joint_coefficients = joint_deviations.copy()
    baseline = initial_baseline + joint_intercept
    LOG.info(
        "%s pass 1 (joint fit): lambda=%.5g CV RMSE=%.4f largest deviation change=n/a",
        name,
        alpha,
        final_cv,
    )
    if max_passes > 1:
        def refit_baseline(residual: np.ndarray, pass_number: int) -> np.ndarray:
            return _baseline_predictions(
                features, residual, frame, folds, seed + pass_number + 1
            )

        coefficients, targets, baseline, alternations, largest_change = _alternate_sides(
            design,
            y,
            baseline,
            columns,
            counts,
            group_keys,
            alpha,
            max_passes - 1,
            tolerance,
            group_priors,
            initial_coefficients=joint_coefficients,
            log_name=name,
            cv_rmse=final_cv,
            baseline_refit=refit_baseline,
        )
    else:
        coefficients = joint_coefficients
        targets = np.zeros_like(coefficients)
        alternations = 0
        largest_change = float("inf")
    passes = 1 + alternations
    return FitResult(
        name=name,
        frame=frame,
        columns=columns,
        coefficients=coefficients,
        counts=counts,
        group_targets=targets,
        group_keys=group_keys,
        lambda_value=alpha,
        cv_rmse=final_cv,
        intercept_rmse=intercept_rmse,
        baseline_rmse=baseline_rmse,
        baseline=baseline,
        design=design,
        passes=passes,
        largest_change=largest_change,
        bootstrap={},
        joint_coefficients=joint_coefficients,
        group_priors_enabled=group_priors,
    )


def _bootstrap_fit(
    result: FitResult, samples: int, seed: int
) -> dict[tuple[str, str, str], tuple[float, float]]:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    games = result.frame["game_id"].astype(str).to_numpy()
    unique_games = np.unique(games)
    if len(unique_games) < 2:
        LOG.warning("%s: bootstrap requires multiple games; skipping", result.name)
        return {}
    estimates: defaultdict[int, list[float]] = defaultdict(list)
    for _ in range(samples):
        selected = rng.choice(unique_games, size=len(unique_games), replace=True)
        rows = np.concatenate([np.flatnonzero(games == game) for game in selected])
        target = result.frame["yards"].to_numpy(dtype=float)[rows]
        matrix = result.design[rows]
        baseline = result.baseline[rows]
        sampled_counts = np.asarray((matrix != 0).sum(axis=0)).ravel().astype(float)
        full, targets, _, _, _ = _alternate_sides(
            matrix,
            target,
            baseline,
            result.columns,
            sampled_counts,
            result.group_keys,
            result.lambda_value,
            max_passes=6,
            tolerance=0.005,
            group_priors=result.group_priors_enabled,
            log_name=f"{result.name} bootstrap",
            log_iterations=False,
        )
        for index, estimate in enumerate(full - targets):
            estimates[index].append(float(estimate))
    return {
        result.columns[index]: tuple(np.percentile(values, [10, 90]).astype(float))
        for index, values in estimates.items()
        if values
    }


def _format_outputs(
    result: FitResult, metadata: dict[tuple[str, str, str], dict[str, str]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (side, role, player_key) in enumerate(result.columns):
        if side == "defense" and role != "on_field":
            continue
        player = metadata[(side, role, player_key)]
        group = player["position_group"]
        deviation = float(result.coefficients[index] - result.group_targets[index])
        q10, q90 = result.bootstrap.get((side, role, player_key), (np.nan, np.nan))
        rows.append(
            {
                "player_id": player["player_id"],
                "player_name": player["player_name"],
                "team": "",
                "position_group": group,
                "side": side,
                "role": role,
                "season": player["season"],
                "n_plays": int(result.counts[index]),
                "rapm": float(result.coefficients[index]),
                "rapm_per_100": float(result.coefficients[index] * 100),
                "rapm_vs_group_avg": deviation,
                "rank_in_group": np.nan,
                "p10": q10,
                "p90": q90,
                "_player_key": player_key,
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    rank_keys = ["side", "role", "position_group"]
    output["rank_in_group"] = output.groupby(rank_keys)["rapm_vs_group_avg"].rank(
        method="min", ascending=False
    ).astype(int)
    output.drop(columns=["_player_key"], inplace=True)
    if not result.frame.empty and "posteam" in result.frame:
        player_teams: defaultdict[str, list[str]] = defaultdict(list)
        for row in result.frame.itertuples(index=False):
            values = row._asdict()
            posteam = values.get("posteam")
            if posteam is None or pd.isna(posteam) or not str(posteam).strip():
                continue
            offense_team = str(posteam).strip()
            home, away = values.get("home_team_derived"), values.get("away_team_derived")
            if home is not None and not pd.isna(home) and offense_team.upper() == str(home).upper():
                defense_team = away
            elif away is not None and not pd.isna(away) and offense_team.upper() == str(away).upper():
                defense_team = home
            else:
                defense_team = ""
            for player_id in _list_values(values.get("offense_player_ids")):
                player_teams[player_id].append(offense_team)
            rusher_id = values.get("rusher_player_id")
            if rusher_id is not None and not pd.isna(rusher_id) and str(rusher_id).strip():
                player_teams[str(rusher_id).strip()].append(offense_team)
            for player_id in _list_values(values.get("defense_player_ids")):
                if (
                    defense_team is not None
                    and not pd.isna(defense_team)
                    and str(defense_team).strip()
                ):
                    player_teams[player_id].append(str(defense_team))
        team_for_player = {
            player_id: Counter(player_teams[player_id]).most_common(1)[0][0]
            for player_id in player_teams
            if player_teams[player_id]
        }
        output["team"] = output["player_id"].map(team_for_player).fillna("")
    return output


def _make_qb_view(
    pass_result: FitResult,
    rush_result: FitResult,
    metadata: dict[tuple[str, str, str], dict[str, str]],
    pass_output: pd.DataFrame,
    rush_output: pd.DataFrame,
) -> pd.DataFrame:
    quarterbacks = {
        (item["player_id"], item["season"]): item
        for (side, _, _), item in metadata.items()
        if side == "offense" and item["position_group"] == "QB"
    }
    pass_values = {
        (row.player_id, row.season): row
        for row in pass_output.itertuples(index=False)
        if row.side == "offense" and row.role == "on_field" and row.position_group == "QB"
    }
    rush_values = {
        (row.player_id, row.season): row
        for row in rush_output.itertuples(index=False)
        if row.side == "offense" and row.role == "ball_carrier" and row.position_group == "QB"
    }
    output = []
    for (player_id, season), player in quarterbacks.items():
        pass_row = pass_values.get((player_id, season))
        rush_row = rush_values.get((player_id, season))
        pass_count = int(pass_row.n_plays) if pass_row else 0
        run_count = int(rush_row.n_plays) if rush_row else 0
        pass_deviation = float(pass_row.rapm_vs_group_avg) if pass_row else 0.0
        run_deviation = float(rush_row.rapm_vs_group_avg) if rush_row else 0.0
        total = pass_count + run_count
        combined = (
            (pass_deviation * pass_count + run_deviation * run_count) / total if total else np.nan
        )
        output.append(
            {
                "player_id": player_id,
                "player_name": player["player_name"],
                "season": season,
                "pass_rapm": float(pass_row.rapm) if pass_row else np.nan,
                "n_pass_plays": pass_count,
                "qb_run_rapm": float(rush_row.rapm) if rush_row else np.nan,
                "n_qb_runs": run_count,
                "combined_qb_rapm": combined,
            }
        )
    return pd.DataFrame(output)


def _print_leaderboards(outputs: dict[str, pd.DataFrame], min_plays: int, min_carries: int) -> None:
    for fit_name, table in outputs.items():
        if table.empty:
            continue
        for (side, role, group), subset in table.groupby(
            ["side", "role", "position_group"]
        ):
            threshold = min_carries if role == "ball_carrier" else min_plays
            qualified = subset[subset["n_plays"] >= threshold].sort_values(
                "rapm_vs_group_avg", ascending=False
            )
            if qualified.empty:
                continue
            LOG.info(
                "\n%s | %s %s %s leaderboard (minimum %d plays)",
                fit_name,
                side.upper(),
                role,
                group,
                threshold,
            )
            print(
                qualified.head(15)[
                    ["player_name", "player_id", "team", "n_plays", "rapm_vs_group_avg"]
                ].to_string(index=False)
            )
            LOG.info("%s | %s %s %s bottom 15", fit_name, side.upper(), role, group)
            print(
                qualified.tail(15).sort_values("rapm_vs_group_avg")[
                    ["player_name", "player_id", "team", "n_plays", "rapm_vs_group_avg"]
                ].to_string(index=False)
            )


def _split_half_reliability(
    result: FitResult, folds: int, seed: int, minimum: int
) -> dict[tuple[str, str, str], float]:
    games = np.unique(result.frame["game_id"].astype(str))
    if len(games) < 4:
        return {}
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(games)
    halves = [set(shuffled[: len(shuffled) // 2]), set(shuffled[len(shuffled) // 2 :])]
    deviations: list[np.ndarray] = []
    counts_by_half: list[np.ndarray] = []
    for half in halves:
        rows = np.flatnonzero(result.frame["game_id"].astype(str).isin(half).to_numpy())
        matrix = result.design[rows]
        y = result.frame["yards"].to_numpy(dtype=float)[rows]
        counts = np.asarray((matrix != 0).sum(axis=0)).ravel().astype(float)
        coefficients, targets, _, _, _ = _alternate_sides(
            matrix,
            y,
            result.baseline[rows],
            result.columns,
            counts,
            result.group_keys,
            result.lambda_value,
            max_passes=6,
            tolerance=0.005,
            group_priors=result.group_priors_enabled,
            log_name=f"{result.name} split-half",
            log_iterations=False,
        )
        deviations.append(coefficients - targets)
        counts_by_half.append(counts)
    reliability: dict[tuple[str, str, str], float] = {}
    groups = sorted(set(result.group_keys))
    for group in groups:
        indices = np.asarray([i for i, key in enumerate(result.group_keys) if key == group])
        eligible = indices[
            (counts_by_half[0][indices] >= minimum) & (counts_by_half[1][indices] >= minimum)
        ]
        if len(eligible) >= 3:
            correlation = np.corrcoef(deviations[0][eligible], deviations[1][eligible])[0, 1]
            reliability[group] = float(correlation) if np.isfinite(correlation) else float("nan")
    return reliability


def _write_diagnostics(
    outdir: Path,
    data: pd.DataFrame,
    dropped: dict[str, int],
    results: Sequence[FitResult],
    reliability: dict[str, dict[tuple[str, str, str], float]],
    yardage_mismatch_rate: float,
) -> None:
    lines = [
        "Football RAPM diagnostics",
        f"Rows after de-duplication: {len(data):,}",
        f"Games: {data['game_id'].nunique():,}",
        f"Seasons: {', '.join(sorted(data['season'].dropna().astype(str).unique()))}",
        "Teams: "
        + ", ".join(
            sorted(
                {
                    str(team).strip()
                    for column in ("team", "posteam", "opponent")
                    if column in data
                    for team in data[column].dropna().unique()
                    if str(team).strip()
                }
            )
        ),
        "Drop counts: " + ", ".join(f"{key}={value:,}" for key, value in dropped.items()),
        "Starting/ending yardage absolute-distance mismatch rate (1-yard tolerance): "
        + (f"{yardage_mismatch_rate:.4%}" if np.isfinite(yardage_mismatch_rate) else "unavailable"),
    ]
    for result in results:
        lines.extend(
            [
                "",
                f"[{result.name}]",
                f"Plays: {len(result.frame):,}",
                f"Distinct player IDs: {len({column[2] for column in result.columns}):,}",
                f"Players/columns: {len(result.columns):,}",
                f"Chosen lambda: {result.lambda_value:g}",
                f"CV RMSE (intercept-only): {result.intercept_rmse:.6f}",
                f"CV RMSE (baseline-only): {result.baseline_rmse:.6f}",
                f"CV RMSE (full model, pass-1 joint fit): {result.cv_rmse:.6f}",
                f"Passes: {result.passes}",
                f"Largest within-group deviation change: {result.largest_change:.8f}",
            ]
        )
        coefficient_stats: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
        deviation_stats: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
        for coefficient, group_target, group in zip(
            result.coefficients, result.group_targets, result.group_keys
        ):
            coefficient_stats[group].append(float(coefficient))
            deviation_stats[group].append(float(coefficient - group_target))
        lines.append("Coefficient mean and SD by group:")
        for group, values in sorted(coefficient_stats.items()):
            lines.append(
                f"  {group}: mean={np.mean(values):.6f}, sd={np.std(values):.6f}, n={len(values)}"
            )
        lines.append("Within-group deviation mean and SD:")
        for group, values in sorted(deviation_stats.items()):
            lines.append(
                f"  {group}: mean={np.mean(values):.6f}, sd={np.std(values):.6f}, n={len(values)}"
            )
        lines.append("Split-half reliability:")
        lines.extend(
            f"  {group}: {value:.6f}" for group, value in sorted(reliability[result.name].items())
        )
    (outdir / "diagnostics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _play_expectations(result: FitResult) -> pd.DataFrame:
    """Return side-specific expectations and residuals for each modeled play."""
    sides = np.asarray([column[0] for column in result.columns])
    offense_columns = np.flatnonzero(sides == "offense")
    defense_columns = np.flatnonzero(sides == "defense")
    offense_contribution = np.asarray(
        result.design[:, offense_columns] @ result.coefficients[offense_columns]
    ).ravel()
    defense_contribution = np.asarray(
        result.design[:, defense_columns] @ result.coefficients[defense_columns]
    ).ravel()
    actual = result.frame["yards"].to_numpy(dtype=float)
    output = pd.DataFrame(
        {
            "fit": result.name,
            "game_id": result.frame["game_id"].astype(str),
            "play_id": result.frame["play_id"].astype(str),
            "play_type": result.frame["play_type"].astype(str),
            "actual_yards_gained": result.frame["raw_yards"].to_numpy(dtype=float),
            "normalized_yards": actual,
            "offense_expectation": result.baseline + defense_contribution,
            "defense_expectation": result.baseline + offense_contribution,
            "offense_residual": actual - result.baseline - defense_contribution,
            "defense_residual": result.baseline
            + offense_contribution
            - actual,
        }
    )
    return output


def run_analysis(args: argparse.Namespace) -> None:
    data, names, defender_positions, rusher_positions = load_plays(args.input)
    observed_player_ids = {
        player_id
        for column in ("offense_player_ids", "defense_player_ids")
        for value in data[column]
        for player_id in _list_values(value)
    }
    nflreadpy_positions = (
        _load_nflreadpy_positions(observed_player_ids)
        if not args.no_nflreadpy_positions
        else {}
    )
    score_running = _score_is_running(data)
    cleaned, dropped, yardage_mismatch_rate = clean_plays(
        data, args.cap, args.shrink, score_running
    )
    if args.by_season and cleaned["season"].eq("").all():
        raise ValueError("--by-season requires a season column or derivable season in game_id")
    offense_global, offense_season = _load_offense_positions(args.positions)
    offense_positions, season_positions, defense_position_map = _make_position_maps(
        defender_positions,
        rusher_positions,
        offense_global,
        offense_season,
        nflreadpy_positions,
    )
    lambda_grid = args.lambda_grid
    if not lambda_grid or any(value <= 0 for value in lambda_grid):
        raise ValueError("--lambda-grid must contain positive values")
    model_specs = (
        ("rapm_rush", cleaned[cleaned["play_type"] == "rush"].reset_index(drop=True)),
        ("rapm_pass", cleaned[cleaned["play_type"] == "pass"].reset_index(drop=True)),
        ("rapm_all_plays", cleaned.reset_index(drop=True)),
    )
    results: list[FitResult] = []
    metadata: dict[tuple[str, str, str], dict[str, str]] = {}
    for name, frame in model_specs:
        if frame.empty:
            LOG.warning("%s has no eligible plays; writing an empty file", name)
            design = sparse.csr_matrix((0, 0))
            result = FitResult(
                name, frame, [], np.zeros(0), np.zeros(0), np.zeros(0), [], float(lambda_grid[0]),
                float("nan"), float("nan"), float("nan"), np.zeros(0), design, 0, 0.0, {},
                np.zeros(0), not args.no_group_priors,
            )
        else:
            design, columns, groups, counts, fit_metadata = _make_design(
                frame,
                args.by_season,
                offense_positions,
                season_positions,
                defense_position_map,
                names,
            )
            metadata.update(fit_metadata)
            features = _baseline_features(frame, args.scheme_controls)
            result = fit_model(
                name,
                frame,
                design,
                columns,
                groups,
                counts,
                features,
                lambda_grid,
                args.folds,
                args.max_passes,
                args.tol,
                not args.no_group_priors,
                args.seed,
            )
            result.bootstrap = _bootstrap_fit(result, args.bootstrap, args.seed + 100)
        results.append(result)
    result_by_name = {result.name: result for result in results}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}
    for name, result in result_by_name.items():
        output = _format_outputs(result, metadata)
        if args.by_season and "season" not in output:
            output["season"] = pd.Series(dtype=str)
        columns = [
            "player_id",
            "player_name",
            "team",
            "position_group",
            "side",
            "role",
        ]
        if args.by_season:
            columns.append("season")
        columns.extend(
            [
                "n_plays",
                "rapm",
                "rapm_per_100",
                "rapm_vs_group_avg",
                "rank_in_group",
            ]
        )
        if args.bootstrap > 0:
            columns.extend(["p10", "p90"])
        for column in columns:
            if column not in output:
                output[column] = pd.Series(dtype=float)
        output[columns].to_csv(outdir / f"{name}.csv", index=False)
        outputs[name] = output
    if args.export_expectations:
        expectations = pd.concat(
            [_play_expectations(result) for result in results if not result.frame.empty],
            ignore_index=True,
        )
        expectations.to_csv(outdir / "play_expectations.csv", index=False)
    qb_view = _make_qb_view(
        result_by_name["rapm_pass"],
        result_by_name["rapm_rush"],
        metadata,
        outputs["rapm_pass"],
        outputs["rapm_rush"],
    )
    qb_columns = [
        "player_id",
        "player_name",
        *(["season"] if args.by_season else []),
        "pass_rapm",
        "n_pass_plays",
        "qb_run_rapm",
        "n_qb_runs",
        "combined_qb_rapm",
    ]
    for column in qb_columns:
        if column not in qb_view:
            qb_view[column] = pd.Series(dtype=float)
    qb_view[qb_columns].to_csv(outdir / "rapm_qb.csv", index=False)
    reliability = {
        result.name: _split_half_reliability(
            result, args.folds, args.seed + 200, args.min_plays
        )
        for result in results
    }
    _write_diagnostics(
        outdir, data, dropped, results, reliability, yardage_mismatch_rate
    )
    _print_leaderboards(outputs, args.min_plays, args.min_carries)
    LOG.info("Wrote RAPM outputs to %s", outdir.resolve())


def _run_selftest() -> None:
    """Exercise normalization, sign recovery, and deterministic seeded fitting."""
    expected = {10: 10, 15: 15, 23: 17, 55: 25, -18: -15.75, -40: -21.25}
    for value, result in expected.items():
        assert normalize_yards(value) == result, (value, normalize_yards(value), result)
    seed = 731
    rng = np.random.default_rng(seed)
    n_games, plays_per_game, n_teams = 60, 500, 24
    offense_roster_size, defense_roster_size = 22, 22
    off_effect = rng.normal(0, 0.45, size=(n_teams, offense_roster_size))
    carrier_effect = off_effect + rng.normal(0, 0.12, size=off_effect.shape)
    def_effect = rng.normal(0, 0.45, size=(n_teams, defense_roster_size))
    rows: list[dict[str, object]] = []
    truth_off_on: dict[str, float] = {}
    truth_off_carrier: dict[str, float] = {}
    truth_def: dict[str, float] = {}
    offense_positions: dict[str, str] = {}
    defense_positions: dict[str, str] = {}
    offense_roster_positions = ["QB"] * 2 + ["RB"] * 4 + ["WR"] * 6 + ["TE"] * 3 + ["T"] * 7
    defense_roster_positions = ["DL"] * 8 + ["LB"] * 6 + ["DB"] * 8
    for team in range(n_teams):
        for slot, position in enumerate(offense_roster_positions):
            player_id = f"O{team}_{slot}"
            truth_off_on[player_id] = off_effect[team, slot]
            truth_off_carrier[player_id] = carrier_effect[team, slot]
            offense_positions[player_id] = position
        for slot, position in enumerate(defense_roster_positions):
            player_id = f"D{team}_{slot}"
            truth_def[f"D{team}_{slot}"] = def_effect[team, slot]
            defense_positions[player_id] = position
    for game in range(n_games):
        offense_team = game % n_teams
        defense_team = (game * 7 + 3) % n_teams
        for play in range(plays_per_game):
            is_rush = rng.random() < 0.42
            chosen_offense = [
                int(rng.integers(0, 2)),
                int(rng.integers(2, 6)),
                *rng.choice(np.arange(6, 12), size=3, replace=False).tolist(),
                int(rng.integers(12, 15)),
                *rng.choice(np.arange(15, 22), size=5, replace=False).tolist(),
            ]
            chosen_defense = [
                *rng.choice(np.arange(0, 8), size=4, replace=False).tolist(),
                *rng.choice(np.arange(8, 14), size=3, replace=False).tolist(),
                *rng.choice(np.arange(14, 22), size=4, replace=False).tolist(),
            ]
            off_ids = [f"O{offense_team}_{slot}" for slot in chosen_offense]
            def_ids = [f"D{defense_team}_{slot}" for slot in chosen_defense]
            carrier_slot = None
            if is_rush:
                carrier_slot = (
                    chosen_offense[0]
                    if rng.random() < 0.20
                    else chosen_offense[1]
                )
            offense_contribution = sum(
                carrier_effect[offense_team, slot]
                if slot == carrier_slot
                else off_effect[offense_team, slot]
                for slot in chosen_offense
            )
            defense_contribution = sum(def_effect[defense_team, slot] for slot in chosen_defense)
            if is_rush:
                yards = 4.2 + offense_contribution - defense_contribution
            else:
                yards = 6.0 + offense_contribution - defense_contribution
            situation = float(play % 3 == 0)
            yards += 2.0 * situation + rng.standard_t(df=4) * 1.2
            rows.append(
                {
                    "game_id": f"2024_{game + 1:02d}_H{game:02d}_A{game:02d}",
                    "play_id": str(play),
                    "play_type": "rush" if is_rush else "pass",
                    "yards": normalize_yards(yards),
                    "season": "2024",
                    "offense_player_ids": ";".join(off_ids),
                    "defense_player_ids": ";".join(def_ids),
                    "rusher_player_id": f"O{offense_team}_{carrier_slot}" if is_rush else "",
                    "situation": situation,
                }
            )
    frame = pd.DataFrame(rows)
    results_by_kind: dict[str, FitResult] = {}
    lambda_grid = (10.0, 100.0, 1000.0)
    for model_name, model_frame in (
        ("rush", frame[frame["play_type"] == "rush"].reset_index(drop=True)),
        ("pass", frame[frame["play_type"] == "pass"].reset_index(drop=True)),
        ("all", frame.reset_index(drop=True)),
    ):
        design, columns, groups, counts, metadata = _make_design(
            model_frame,
            False,
            offense_positions,
            {},
            defense_positions,
            {player_id: player_id for player_id in (*offense_positions, *defense_positions)},
        )
        features = pd.DataFrame({"situation": model_frame["situation"]})
        result = fit_model(
            model_name,
            model_frame,
            design,
            columns,
            groups,
            counts,
            features,
            lambda_grid,
            folds=3,
            max_passes=2,
            tolerance=0.005,
            group_priors=True,
            seed=seed,
        )
        estimates = {
            (column[1], metadata[column]["player_id"]): result.coefficients[i]
            for i, column in enumerate(columns)
        }
        if model_name == "all":
            offense_ids = [
                (column[1], metadata[column]["player_id"])
                for column in columns
                if column[0] == "offense"
            ]
            offense_truth = [
                truth_off_carrier[player_id] if role == "ball_carrier" else truth_off_on[player_id]
                for role, player_id in offense_ids
            ]
            offense_estimates = [estimates[key] for key in offense_ids]
            defense_ids = [
                metadata[column]["player_id"]
                for column in columns
                if column[0] == "defense"
            ]
            offense_corr = np.corrcoef(
                offense_truth,
                offense_estimates,
            )[0, 1]
            defense_corr = np.corrcoef(
                [truth_def[player_id] for player_id in defense_ids],
                [estimates[("on_field", player_id)] for player_id in defense_ids],
            )[0, 1]
            assert offense_corr > 0.5, f"offense sign/recovery correlation too low: {offense_corr:.3f}"
            assert defense_corr > 0.5, f"defense sign/recovery correlation too low: {defense_corr:.3f}"
        results_by_kind[model_name] = result
    pooled = results_by_kind["all"]
    design, columns, groups, counts, _ = _make_design(
        pooled.frame,
        False,
        offense_positions,
        {},
        defense_positions,
        {player_id: player_id for player_id in (*offense_positions, *defense_positions)},
    )
    repeat = fit_model(
        "all-repeat",
        pooled.frame,
        design,
        columns,
        groups,
        counts,
        pd.DataFrame({"situation": pooled.frame["situation"]}),
        lambda_grid,
        folds=3,
        max_passes=2,
        tolerance=0.005,
        group_priors=True,
        seed=seed,
    )
    assert np.array_equal(pooled.coefficients, repeat.coefficients), "fixed-seed fit did not reproduce"
    pooled_y = pooled.frame["yards"].to_numpy(dtype=float)
    initial_baseline = _initial_type_mean(pooled_y, pooled.frame)
    joint_coefficients, joint_intercept = _ridge_fit(
        design,
        pooled_y - initial_baseline,
        pooled.lambda_value,
        fit_intercept=True,
    )
    fixed_baseline = initial_baseline + joint_intercept
    alternating, _, _, equivalence_passes, _ = _alternate_sides(
        design,
        pooled_y,
        fixed_baseline,
        columns,
        counts,
        groups,
        pooled.lambda_value,
        max_passes=40,
        tolerance=0.001,
        group_priors=False,
        initial_coefficients=joint_coefficients,
        log_name="equivalence self-test",
    )
    observed = counts >= 30
    equivalence_correlation = float(
        np.corrcoef(joint_coefficients[observed], alternating[observed])[0, 1]
    )
    assert equivalence_passes > 1, "block alternation converged without an update"
    assert equivalence_correlation > 0.98, (
        "joint and alternating ridge coefficients diverged: "
        f"{equivalence_correlation:.3f}"
    )
    print(
        "Joint/alternating coefficient correlation: "
        f"{equivalence_correlation:.4f} after {equivalence_passes} block passes."
    )
    assert all(np.isfinite(result.coefficients).all() for result in results_by_kind.values())
    print(
        "Self-test passed: normalization, three fits, sign recovery, reproducibility, "
        "and joint/alternating equivalence."
    )


def _parse_lambda_grid(value: str) -> list[float]:
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("lambda grid range must be START:STOP:COUNT")
        try:
            start, stop = float(parts[0]), float(parts[1])
            count = int(parts[2])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        if start <= 0 or stop <= 0 or count < 2:
            raise argparse.ArgumentTypeError("lambda range needs positive bounds and count >= 2")
        return np.logspace(np.log10(start), np.log10(stop), count).tolist()
    try:
        values = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("lambda grid values must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="CSV path, glob pattern, or directory containing CSVs")
    parser.add_argument("--outdir", default="rapm_output", help="Output directory")
    parser.add_argument("--positions", help="Optional offense player position CSV")
    parser.add_argument(
        "--no-nflreadpy-positions",
        action="store_true",
        help="Do not enrich player positions from nflreadpy.load_players()",
    )
    parser.add_argument("--by-season", action="store_true", help="Estimate separate player-season effects")
    parser.add_argument("--cap", type=float, default=15.0, help="Yardage cap threshold (default: 15)")
    parser.add_argument("--shrink", type=float, default=0.25, help="Beyond-cap shrink factor (default: 0.25)")
    parser.add_argument("--folds", type=int, default=5, help="Game-grouped cross-validation folds")
    parser.add_argument(
        "--lambda-grid",
        type=_parse_lambda_grid,
        default=np.logspace(1, 5, 9).tolist(),
        help="Comma-separated lambda values or START:STOP:COUNT (default: 10:100000:9)",
    )
    parser.add_argument("--max-passes", type=int, default=6)
    parser.add_argument("--tol", type=float, default=0.005)
    parser.add_argument("--no-group-priors", action="store_true")
    parser.add_argument("--scheme-controls", action="store_true")
    parser.add_argument("--min-plays", type=int, default=100)
    parser.add_argument("--min-carries", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=0, help="Game-cluster bootstrap replicates")
    parser.add_argument(
        "--export-expectations",
        action="store_true",
        help="Write per-play expectations and performance residuals for the three fits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selftest", action="store_true", help="Run the simulated-data self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        _run_selftest()
        return 0
    if not args.input:
        parser.error("--input is required unless --selftest is used")
    if args.folds < 2 or args.max_passes < 1 or args.bootstrap < 0:
        parser.error("--folds must be >= 2, --max-passes >= 1, and --bootstrap >= 0")
    if args.min_plays < 0 or args.min_carries < 0 or args.tol < 0:
        parser.error("play thresholds and --tol must be non-negative")
    if args.cap < 0 or not 0 <= args.shrink <= 1:
        parser.error("--cap must be non-negative and --shrink must be between 0 and 1")
    run_analysis(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        LOG.error("%s", error)
        raise SystemExit(2) from error
