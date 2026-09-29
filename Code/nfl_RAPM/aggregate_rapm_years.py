"""Aggregate season-level RAPM exports into time-decayed player ratings."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUALIFYING_PLAYS = 200.0
QUALIFIED_SAMPLE_CAP = 3.0
UNQUALIFIED_WEIGHT_CAP = 0.25
HALF_LIVES = (1.0, 1.5, 2.0, 3.0, 4.0)
SAMPLE_SCHEMES = ("linear_cap3", "linear_cap1", "sqrt_cap3")
SIDES = ("offense", "defense")


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {_key(column): column for column in columns}
    return next((lookup[alias] for alias in aliases if alias in lookup), None)


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_side(value: object) -> str | None:
    side = _clean_text(value).lower()
    if side in {"offense", "offence", "off", "o"}:
        return "offense"
    if side in {"defense", "defence", "def", "d"}:
        return "defense"
    return None


def _metric_per_play(value: object, per_100: bool) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed) or not np.isfinite(float(parsed)):
        return float("nan")
    number = float(parsed)
    return number / 100.0 if per_100 else number


def _season_file(folder: Path) -> Path:
    preferred = folder / "rapm_all_plays.csv"
    if preferred.is_file():
        return preferred
    alternatives = sorted(folder.glob("*all*plays*.csv"))
    if alternatives:
        return alternatives[0]
    raise FileNotFoundError(
        f"{folder} has no rapm_all_plays.csv; this aggregator needs the pooled "
        "season export, not the play-by-play source files"
    )


def _read_season(folder: Path, season: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _season_file(folder)
    table = pd.read_csv(path, dtype="string", low_memory=False)
    columns = list(table.columns)
    id_col = _find_column(columns, ("playerid", "gsisid", "nflgsisid"))
    name_col = _find_column(columns, ("playername", "displayname", "name"))
    position_col = _find_column(
        columns, ("position", "positiongroup", "pos", "rosterposition")
    )
    team_col = _find_column(columns, ("mostrecentteam", "team"))
    side_col = _find_column(columns, ("side", "unit"))
    role_col = _find_column(columns, ("role", "playerrole"))
    plays_col = _find_column(
        columns, ("nplays", "plays", "playcount", "qualifyingplays")
    )
    rapm_col = _find_column(columns, ("rapm", "rapmperplay", "finalrapm"))
    rapm_per_100_col = _find_column(columns, ("rapmper100", "rapmper100plays"))

    if plays_col is None:
        raise ValueError(f"{path} has no recognized play-count column")
    has_component_columns = any(
        _find_column(columns, aliases) is not None
        for aliases in (
            ("orapm", "offenserapm"),
            ("drapm", "defenserapm"),
        )
    )
    if rapm_col is None and rapm_per_100_col is None and not has_component_columns:
        raise ValueError(f"{path} has no recognized RAPM value column")
    if id_col is None and name_col is None:
        raise ValueError(f"{path} has neither a player ID nor a player name column")

    metric_col = rapm_col or rapm_per_100_col
    metric_per_100 = rapm_col is None
    if id_col and name_col:
        grouped_names: defaultdict[str, set[str]] = defaultdict(set)
        for raw_id, raw_name in zip(table[id_col], table[name_col]):
            player_id, player_name = _clean_text(raw_id), _clean_text(raw_name)
            if player_id and player_name:
                grouped_names[player_id].add(player_name)
        name_conflicts = sum(len(names) > 1 for names in grouped_names.values())
    else:
        name_conflicts = 0

    records: list[dict[str, Any]] = []
    missing_identity_rows = 0
    invalid_side_rows = 0
    missing_value_rows = 0
    for row_number, row in table.iterrows():
        player_id = _clean_text(row[id_col]) if id_col else ""
        player_name = _clean_text(row[name_col]) if name_col else ""
        if not player_id and not player_name:
            missing_identity_rows += 1
            continue
        position = _clean_text(row[position_col]) if position_col else ""
        team = _clean_text(row[team_col]) if team_col else ""
        role = _clean_text(row[role_col]) if role_col else ""
        plays = pd.to_numeric(pd.Series([row[plays_col]]), errors="coerce").iloc[0]
        if pd.isna(plays) or not np.isfinite(float(plays)) or float(plays) < 0:
            raise ValueError(
                f"{path} row {row_number + 2} has an invalid play count: "
                f"{row[plays_col]!r}"
            )
        if side_col:
            side = _normalize_side(row[side_col])
            if side is None:
                invalid_side_rows += 1
                continue
            value_column = rapm_col or rapm_per_100_col
            value = _metric_per_play(row[value_column], metric_per_100)
            if np.isnan(value):
                missing_value_rows += 1
                continue
            records.append(
                {
                    "season": season,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "team": team,
                    "side": side,
                    "role": role,
                    "plays": float(plays),
                    "observed_rapm": value,
                    "source": str(path),
                    "source_metric": metric_col,
                    "converted_from_per_100": metric_per_100,
                }
            )
            continue

        for side, aliases in (
            ("offense", ("orapm", "offenserapm")),
            ("defense", ("drapm", "defenserapm")),
        ):
            value_col = _find_column(columns, aliases)
            if value_col is None:
                continue
            side_plays_col = _find_column(
                columns,
                (
                    f"{side}nplays",
                    f"{side}plays",
                    f"{side}playcount",
                ),
            )
            side_plays = plays
            if side_plays_col:
                parsed_plays = pd.to_numeric(
                    pd.Series([row[side_plays_col]]), errors="coerce"
                ).iloc[0]
                if not pd.isna(parsed_plays):
                    side_plays = float(parsed_plays)
            value = _metric_per_play(row[value_col], "per100" in _key(value_col))
            if np.isnan(value):
                missing_value_rows += 1
                continue
            records.append(
                {
                    "season": season,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "team": team,
                    "side": side,
                    "role": "",
                    "plays": side_plays,
                    "observed_rapm": value,
                    "source": str(path),
                    "source_metric": value_col,
                    "converted_from_per_100": "per100" in _key(value_col),
                }
            )
    if not records:
        raise ValueError(f"{path} has no usable player RAPM records")
    diagnostics = {
        "season": season,
        "source": str(path),
        "rows": len(table),
        "records": len(records),
        "player_count": len({r["player_id"] or r["player_name"] for r in records}),
        "missing_identity_rows": missing_identity_rows,
        "invalid_side_rows": invalid_side_rows,
        "missing_value_rows": missing_value_rows,
        "conflicting_names_for_id": name_conflicts,
        "metric_column": metric_col,
        "metric_unit_conversion": "divide by 100" if metric_per_100 else "none",
    }
    return records, diagnostics


def discover_seasons(
    input_root: Path, start_year: int | None, end_year: int | None
) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for folder in input_root.glob("rapm_*"):
        match = re.fullmatch(r"rapm_(\d{4})", folder.name)
        if folder.is_dir() and match:
            found[int(match.group(1))] = folder
    if start_year is not None:
        found = {year: folder for year, folder in found.items() if year >= start_year}
    if end_year is not None:
        found = {year: folder for year, folder in found.items() if year <= end_year}
    if not found:
        raise FileNotFoundError(
            f"No rapm_[YEAR] folders found in {input_root} for the selected range"
        )
    return dict(sorted(found.items()))


def collapse_player_seasons(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame(records)
    issues: list[str] = []
    frame["_identity"] = np.where(
        frame["player_id"].ne(""),
        "id:" + frame["player_id"],
        "name:" + frame["player_name"].str.casefold().str.strip(),
    )
    frame["_role_key"] = frame["role"].fillna("")
    duplicate_keys = ["season", "_identity", "side", "_role_key"]
    duplicate_rows = frame.duplicated(duplicate_keys, keep=False)
    if duplicate_rows.any():
        issues.append(
            f"{int(duplicate_rows.sum())} duplicate input row(s) share a "
            "player-season-side-role key; they were play-weighted together."
        )

    def combine(group: pd.DataFrame) -> pd.Series:
        season, identity, side = group.name
        plays = group["plays"].to_numpy(dtype=float)
        values = group["observed_rapm"].to_numpy(dtype=float)
        total_plays = float(plays.sum())
        value = (
            float(np.average(values, weights=plays))
            if total_plays > 0
            else float(np.mean(values))
        )
        for column in ("player_name", "position", "team"):
            candidates = [value for value in group[column] if value]
            if candidates:
                return_value = Counter(candidates).most_common(1)[0][0]
            else:
                return_value = ""
            if column == "player_name":
                player_name = return_value
            elif column == "position":
                position = return_value
            else:
                team = return_value
        return pd.Series(
            {
                "season": int(season),
                "identity": identity,
                "player_id": next((v for v in group["player_id"] if v), ""),
                "player_name": player_name,
                "position": position,
                "team": team,
                "side": side,
                "plays": total_plays,
                "observed_rapm": value,
                "source_roles": ";".join(sorted(set(group["_role_key"]))),
                "input_rows": int(len(group)),
            }
        )

    season_side = (
        frame.groupby(["season", "_identity", "side"], sort=False, dropna=False)
        .apply(combine, include_groups=False)
        .reset_index(drop=True)
    )
    missing_ids = season_side["player_id"].eq("")
    if missing_ids.any():
        issues.append(
            f"{int(missing_ids.sum())} player-season-side aggregate(s) lack stable IDs "
            "and were matched using exact player-name text only."
        )
    name_id_ambiguities = (
        season_side.loc[season_side["player_id"].ne("")]
        .groupby(season_side["player_name"].str.casefold())["player_id"]
        .nunique()
    )
    ambiguous_names = name_id_ambiguities[name_id_ambiguities > 1]
    if not ambiguous_names.empty:
        issues.append(
            f"{len(ambiguous_names)} display name(s) map to multiple player IDs; "
            "those players remain separate because IDs take precedence."
        )
    return season_side, issues


def _league_averages(observations: pd.DataFrame) -> dict[tuple[int, str], float]:
    qualified = observations[observations["plays"] >= QUALIFYING_PLAYS]
    averages = qualified.groupby(["season", "side"])["observed_rapm"].mean()
    return {(int(season), str(side)): float(value) for (season, side), value in averages.items()}


def _sample_weight(plays: float, qualified: bool, scheme: str) -> float:
    if not qualified:
        return min(max(plays, 0.0) / QUALIFYING_PLAYS, UNQUALIFIED_WEIGHT_CAP)
    ratio = max(plays, 0.0) / QUALIFYING_PLAYS
    if scheme == "linear_cap1":
        return min(ratio, 1.0)
    if scheme == "sqrt_cap3":
        return min(math.sqrt(ratio), QUALIFIED_SAMPLE_CAP)
    return min(ratio, QUALIFIED_SAMPLE_CAP)


def _evaluate(
    observations: pd.DataFrame,
    league_averages: dict[tuple[int, str], float],
    target_season: int,
    half_life: float,
    sample_scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    worked = observations.copy()
    worked["qualified"] = worked["plays"] >= QUALIFYING_PLAYS
    worked["status"] = np.where(worked["qualified"], "QUALIFIED", "UNQUALIFIED")
    league_values: list[float] = []
    season_values: list[float] = []
    decays: list[float] = []
    sample_weights: list[float] = []
    combined_weights: list[float] = []
    for row in worked.itertuples(index=False):
        league = league_averages.get((int(row.season), str(row.side)), 0.0)
        league_values.append(league)
        season_values.append(
            float(row.observed_rapm)
            if row.qualified
            else 0.50 * league
        )
        decay = 0.5 ** ((target_season - int(row.season)) / half_life)
        sample = _sample_weight(float(row.plays), bool(row.qualified), sample_scheme)
        decays.append(decay)
        sample_weights.append(sample)
        combined_weights.append(decay * sample)
    worked["season_league_average"] = league_values
    worked["season_value_used"] = season_values
    worked["time_decay_weight"] = decays
    worked["sample_weight"] = sample_weights
    worked["combined_weight"] = combined_weights

    results: list[dict[str, Any]] = []
    for (identity, side), group in worked.groupby(["identity", "side"], sort=False):
        weights = group["combined_weight"].to_numpy(dtype=float)
        values = group["season_value_used"].to_numpy(dtype=float)
        effective_weight = float(weights.sum())
        value = (
            float(np.average(values, weights=weights))
            if effective_weight > 0
            else float("nan")
        )
        qualified_weight = float(
            group.loc[group["qualified"], "combined_weight"].sum()
        )
        unqualified_weight = effective_weight - qualified_weight
        first = group.iloc[0]
        results.append(
            {
                "identity": identity,
                "side": side,
                "value": value,
                "effective_weight": effective_weight,
                "qualified_weight": qualified_weight,
                "unqualified_weight": unqualified_weight,
                "pct_qualified_weight": (
                    100.0 * qualified_weight / effective_weight
                    if effective_weight > 0
                    else float("nan")
                ),
                "pct_unqualified_weight": (
                    100.0 * unqualified_weight / effective_weight
                    if effective_weight > 0
                    else float("nan")
                ),
                "qualified_seasons": int(group.loc[group["qualified"], "season"].nunique()),
                "total_seasons": int(group["season"].nunique()),
                "total_plays": float(group["plays"].sum()),
                "weighted_seasons_used": int((group["combined_weight"] > 0).sum()),
            }
        )
    return worked, pd.DataFrame(results)


def _primary_output(
    observations: pd.DataFrame,
    league_averages: dict[tuple[int, str], float],
    target_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    worked, component_results = _evaluate(
        observations,
        league_averages,
        target_season,
        half_life=2.0,
        sample_scheme="linear_cap3",
    )
    metadata: dict[str, dict[str, Any]] = {}
    for identity, group in observations.groupby("identity", sort=False):
        by_year = group.sort_values("season")
        latest = by_year.iloc[-1]
        positions = [value for value in group["position"] if value]
        teams = [
            value
            for value in group.loc[group["season"].eq(latest["season"]), "team"]
            if value
        ]
        metadata[identity] = {
            "player_id": next((value for value in group["player_id"] if value), ""),
            "player_name": next((value for value in reversed(group["player_name"].tolist()) if value), ""),
            "position": Counter(positions).most_common(1)[0][0] if positions else "",
            "most_recent_team": Counter(teams).most_common(1)[0][0] if teams else "",
            "most_recent_player_season": int(latest["season"]),
        }

    side_lookup = {
        (row.identity, row.side): row
        for row in component_results.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for identity, meta in metadata.items():
        offense = side_lookup.get((identity, "offense"))
        defense = side_lookup.get((identity, "defense"))
        player_rows = observations[observations["identity"] == identity]
        plays_by_side = player_rows.groupby("side")["plays"].sum().to_dict()
        rating_side = max(
            SIDES,
            key=lambda side: (
                float(plays_by_side.get(side, 0.0)),
                int(side == "offense"),
            ),
        )
        rating = offense if rating_side == "offense" else defense
        other_side = "defense" if rating_side == "offense" else "offense"
        other_rating = defense if rating_side == "offense" else offense
        side_rows = player_rows[player_rows["side"] == rating_side]
        other_side_rows = player_rows[player_rows["side"] == other_side]
        qualified_years = set(
            side_rows.loc[
                side_rows["plays"] >= QUALIFYING_PLAYS, "season"
            ].astype(int)
        )
        recent = player_rows[player_rows["season"] == meta["most_recent_player_season"]]
        recent_selected = recent[recent["side"] == rating_side]
        latest_selected = side_rows.sort_values("season").iloc[-1]
        rows.append(
            {
                "player_id": meta["player_id"],
                "player_name": meta["player_name"],
                "position": meta["position"],
                "most_recent_team": meta["most_recent_team"],
                "side": rating_side,
                "final_RAPM": rating.value if rating is not None else np.nan,
                "qualified_seasons": len(qualified_years),
                "total_seasons": int(side_rows["season"].nunique()),
                "total_plays": float(side_rows["plays"].sum()),
                "most_recent_season": int(latest_selected["season"]),
                "most_recent_season_RAPM": (
                    float(recent_selected["observed_rapm"].iloc[0])
                    if not recent_selected.empty
                    else np.nan
                ),
                "weighted_seasons_used": len(
                    set(
                        side_rows.loc[side_rows["plays"] > 0, "season"].astype(int)
                    )
                ),
                "effective_weight": rating.effective_weight if rating is not None else 0.0,
                "percentage_of_weight_from_qualified_seasons": (
                    rating.pct_qualified_weight if rating is not None else np.nan
                ),
                "percentage_of_weight_from_unqualified_seasons": (
                    rating.pct_unqualified_weight if rating is not None else np.nan
                ),
                "other_side_plays_excluded": float(other_side_rows["plays"].sum()),
                "has_both_side_records": bool(not side_rows.empty and not other_side_rows.empty),
                "qualified_other_side_seasons": (
                    other_rating.qualified_seasons if other_rating is not None else 0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["player_name", "player_id"]), worked


def _sensitivity(
    observations: pd.DataFrame,
    league_averages: dict[tuple[int, str], float],
    target_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    baseline: pd.DataFrame | None = None
    for scheme in SAMPLE_SCHEMES:
        for half_life in HALF_LIVES:
            _, values = _evaluate(
                observations, league_averages, target_season, half_life, scheme
            )
            values["half_life_years"] = half_life
            values["sample_weight_scheme"] = scheme
            rows.append(values)
            if half_life == 2.0 and scheme == "linear_cap3":
                baseline = values[["identity", "side", "value"]].rename(
                    columns={"value": "baseline_value"}
                )
    sensitivity = pd.concat(rows, ignore_index=True)
    assert baseline is not None
    comparison = sensitivity.merge(baseline, on=["identity", "side"], how="left")
    comparison["delta_from_primary"] = comparison["value"] - comparison["baseline_value"]
    summaries: list[dict[str, Any]] = []
    for (half_life, scheme, side), group in comparison.groupby(
        ["half_life_years", "sample_weight_scheme", "side"], sort=True
    ):
        valid = group[["value", "baseline_value", "delta_from_primary"]].dropna()
        summaries.append(
            {
                "half_life_years": half_life,
                "sample_weight_scheme": scheme,
                "side": side,
                "players_compared": len(valid),
                "correlation_to_primary": (
                    float(valid["value"].corr(valid["baseline_value"]))
                    if len(valid) > 1
                    else np.nan
                ),
                "median_absolute_change": (
                    float(valid["delta_from_primary"].abs().median())
                    if not valid.empty
                    else np.nan
                ),
                "max_absolute_change": (
                    float(valid["delta_from_primary"].abs().max())
                    if not valid.empty
                    else np.nan
                ),
            }
        )
    return comparison, pd.DataFrame(summaries)


def _examples(worked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    examples: dict[str, str] = {}
    selections: dict[str, str] = {}
    grouped = worked.groupby("identity", sort=False)
    for identity, group in grouped:
        qcount = int(group.loc[group["qualified"], "season"].nunique())
        if qcount >= 2:
            selections.setdefault("multiple_qualified_seasons", identity)
        elif qcount == 1:
            selections.setdefault("one_qualified_season", identity)
        elif qcount == 0:
            selections.setdefault("only_unqualified_seasons", identity)
        qualified_years = sorted(
            set(group.loc[group["qualified"], "season"].astype(int))
        )
        if len(qualified_years) >= 2:
            missing_between = any(
                qualified_years[index + 1] - qualified_years[index] > 1
                and any(
                    year not in set(group["season"].astype(int))
                    for year in range(qualified_years[index] + 1, qualified_years[index + 1])
                )
                for index in range(len(qualified_years) - 1)
            )
            if missing_between:
                selections.setdefault("missing_season_between_qualified", identity)
    rows: list[dict[str, Any]] = []
    for label, identity in selections.items():
        group = grouped.get_group(identity).sort_values(["side", "season"])
        player = group.iloc[0]
        examples[label] = (
            f"{player['player_name']} ({player['player_id'] or identity})"
        )
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "example_case": label,
                    "player_id": row.player_id,
                    "player_name": row.player_name,
                    "side": row.side,
                    "season": row.season,
                    "plays": row.plays,
                    "status": row.status,
                    "observed_RAPM_not_used_if_unqualified": row.observed_rapm,
                    "season_league_average": row.season_league_average,
                    "season_value_used": row.season_value_used,
                    "time_decay_weight": row.time_decay_weight,
                    "sample_weight": row.sample_weight,
                    "combined_weight": row.combined_weight,
                }
            )
    columns = [
        "example_case",
        "player_id",
        "player_name",
        "side",
        "season",
        "plays",
        "status",
        "observed_RAPM_not_used_if_unqualified",
        "season_league_average",
        "season_value_used",
        "time_decay_weight",
        "sample_weight",
        "combined_weight",
    ]
    return pd.DataFrame(rows, columns=columns), examples


def _formula_and_example() -> str:
    qualified_sample = "min(plays / 200, 3.0)"
    unqualified_sample = f"min(plays / 200, {UNQUALIFIED_WEIGHT_CAP:.2f})"
    current_weight = 1.0 * min(400.0 / QUALIFYING_PLAYS, QUALIFIED_SAMPLE_CAP)
    old_weight = (0.5 ** (2.0 / 2.0)) * min(
        120.0 / QUALIFYING_PLAYS, UNQUALIFIED_WEIGHT_CAP
    )
    fallback = 0.5 * 0.10
    weighted_value = (current_weight * 0.20 + old_weight * fallback) / (
        current_weight + old_weight
    )
    return (
        "Formula (computed independently for offense and defense; they are never added):\n"
        "  years_ago = target_season - season\n"
        "  decay_weight = 0.5 ** (years_ago / half_life_years)\n"
        f"  qualified (plays >= {int(QUALIFYING_PLAYS)}): season_value = observed RAPM; "
        f"sample_weight = {qualified_sample}\n"
        "  unqualified: season_value = 0.50 * that season's qualified-player "
        "league mean; "
        f"sample_weight = {unqualified_sample} (a conservative cap on total influence)\n"
        "  if the league mean is zero, the unqualified fallback is zero\n"
        "  combined_weight = decay_weight * sample_weight\n"
        "  final_side_RAPM = sum(combined_weight * season_value) / "
        "sum(combined_weight)\n"
        "  absent player-seasons contribute no observation or weight.\n"
        "Worked example (illustrative offense-side player; target season 2025; "
        "half-life 2 years):\n"
        "  2025: 400 plays, observed RAPM +0.20 => value +0.20, decay 1.000, "
        f"sample 2.000, weight {current_weight:.3f}\n"
        "  2023: 120 plays, observed RAPM is ignored; qualified-season league "
        "+0.10 => value +0.05, decay 0.500, "
        f"sample {min(120 / 200, UNQUALIFIED_WEIGHT_CAP):.3f}, weight {old_weight:.3f}\n"
        f"  final_RAPM (offense side) = (2.000 * 0.20 + {old_weight:.3f} * 0.05) / "
        f"(2.000 + {old_weight:.3f}) = {weighted_value:.6f}\n"
        "  The missing 2024 season contributes nothing. The player's final rating "
        "uses only their selected primary side; the two sides are never added."
    )


def run_selftest() -> None:
    assert _sample_weight(200, True, "linear_cap3") == 1.0
    assert _sample_weight(600, True, "linear_cap3") == 3.0
    assert _sample_weight(1000, True, "linear_cap3") == 3.0
    assert _sample_weight(120, False, "linear_cap3") == UNQUALIFIED_WEIGHT_CAP
    frame = pd.DataFrame(
        [
            {"season": 2025, "identity": "x", "side": "offense", "plays": 400, "observed_rapm": 0.2,
             "player_id": "x", "player_name": "Example", "position": "RB", "team": ""},
            {"season": 2023, "identity": "x", "side": "offense", "plays": 120, "observed_rapm": 999.0,
             "player_id": "x", "player_name": "Example", "position": "RB", "team": ""},
        ]
    )
    league = {(2025, "offense"): 0.1, (2023, "offense"): 0.1}
    worked, result = _evaluate(frame, league, 2025, 2.0, "linear_cap3")
    assert worked["season_value_used"].tolist() == [0.2, 0.05]
    assert np.allclose(worked["time_decay_weight"], [1.0, 0.5])
    expected = (2.0 * 0.2 + 0.125 * 0.05) / 2.125
    assert np.isclose(result.iloc[0]["value"], expected)
    assert _sample_weight(400, True, "linear_cap1") == 1.0
    assert np.isclose(_sample_weight(400, True, "sqrt_cap3"), math.sqrt(2.0))
    player_seasons = pd.DataFrame(
        [
            {
                "season": 2025,
                "identity": "id:two-way-listing",
                "player_id": "two-way-listing",
                "player_name": "Example Player",
                "position": "RB",
                "team": "",
                "side": "offense",
                "plays": 250.0,
                "observed_rapm": 0.2,
            },
            {
                "season": 2025,
                "identity": "id:two-way-listing",
                "player_id": "two-way-listing",
                "player_name": "Example Player",
                "position": "RB",
                "team": "",
                "side": "defense",
                "plays": 300.0,
                "observed_rapm": -0.1,
            },
        ]
    )
    primary, _ = _primary_output(
        player_seasons,
        {(2025, "offense"): 0.0, (2025, "defense"): 0.0},
        2025,
    )
    assert len(primary) == 1
    assert primary.iloc[0]["side"] == "defense"
    assert np.isclose(primary.iloc[0]["final_RAPM"], -0.1)
    assert primary.iloc[0]["other_side_plays_excluded"] == 250.0
    assert "final_ORAPM" not in primary.columns
    assert "final_DRAPM" not in primary.columns
    print("Aggregation self-test passed.")


def _write_summary(
    path: Path,
    seasons: list[int],
    season_diagnostics: list[dict[str, Any]],
    observations: pd.DataFrame,
    final: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    issues: list[str],
    examples: dict[str, str],
) -> None:
    status_counts = Counter()
    player_qualified_counts: Counter[int] = Counter()
    for _, group in observations.groupby("identity", sort=False):
        qualified_years = set(
            group.loc[group["plays"] >= QUALIFYING_PLAYS, "season"].astype(int)
        )
        player_qualified_counts[len(qualified_years)] += 1
    for n in range(3):
        status_counts[n] = player_qualified_counts[n]
    status_counts[3] = sum(
        count for qualified, count in player_qualified_counts.items() if qualified >= 3
    )

    lines = [
        "Time-decayed multi-season RAPM aggregation",
        f"Available seasons: {', '.join(map(str, seasons))}",
        f"Target season: {max(seasons)}",
        "Source: rapm_all_plays.csv in each rapm_[YEAR] directory; no regression rerun.",
        "",
        _formula_and_example(),
        "",
        "Interpretation: every player receives one final_RAPM from their primary "
        "side, selected as the side with the most observed plays across supplied "
        "seasons. Any records on the opposite side are excluded from that player's "
        "final rating and identified in output diagnostics; offense and defense "
        "ratings are never added. Recent seasons receive more weight, qualified "
        "larger samples receive more weight, unqualified seasons use half the "
        "qualified-player league average, and missing seasons contribute nothing. "
        "League averages are calculated separately by season and side; "
        "position-specific averages are not used because the source RAPM is not "
        "explicitly position-normalized. If the league average is centered at "
        "zero, the specified unqualified fallback is also zero.",
        "",
        "Per-season inputs:",
    ]
    for info in season_diagnostics:
        lines.append(
            f"  {info['season']}: {info['records']} records, "
            f"{info['player_count']} distinct IDs/names, column {info['metric_column']}, "
            f"unit conversion: {info['metric_unit_conversion']}"
        )
    lines.extend(
        [
            "",
            "Player count by number of qualified seasons (either side):",
            f"  0: {status_counts[0]}",
            f"  1: {status_counts[1]}",
            f"  2: {status_counts[2]}",
            f"  3+: {status_counts[3]}",
            "",
            "Primary final_RAPM distribution (each player on their primary side):",
            final["final_RAPM"].describe().to_string(),
            "",
            "Sensitivity correlations to the 2-year / linear-cap-3 primary result:",
        ]
    )
    primary_sensitivity = sensitivity_summary[
        (sensitivity_summary["sample_weight_scheme"] == "linear_cap3")
        | (sensitivity_summary["half_life_years"] == 2.0)
    ]
    lines.append(primary_sensitivity.to_string(index=False))
    lines.extend(["", "Example cases found:"])
    if examples:
        lines.extend(f"  {case}: {player}" for case, player in examples.items())
    else:
        lines.append("  No eligible example cases in the selected seasons.")
    missing_example_cases = {
        "multiple_qualified_seasons",
        "missing_season_between_qualified",
    }.difference(examples)
    if missing_example_cases:
        lines.append(
            "  Not available in these inputs: "
            + ", ".join(sorted(missing_example_cases))
            + ". Add more rapm_[YEAR] folders to illustrate longitudinal cases."
        )
    unqualified_primary = final[
        final["percentage_of_weight_from_unqualified_seasons"] > 50
    ]
    lines.extend(
        [
            "",
            f"Players with >50% of their selected-side weight from unqualified "
            f"fallback seasons: {len(unqualified_primary)}",
            "The corresponding player rows are in primarily_unqualified.csv.",
            "",
            "Data-quality issues:",
        ]
    )
    if issues:
        lines.extend(f"  - {issue}" for issue in issues)
    else:
        lines.append("  None detected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("."),
        help="Directory containing rapm_[YEAR] folders (default: current directory)",
    )
    parser.add_argument("--start-year", type=int, help="Inclusive first season")
    parser.add_argument("--end-year", type=int, help="Inclusive last season")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("rapm_multi_year"),
        help="Directory for combined CSVs and audit reports",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show formula and worked example without writing output files",
    )
    parser.add_argument("--selftest", action="store_true", help="Run aggregation checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        run_selftest()
        return 0
    if args.start_year and args.end_year and args.start_year > args.end_year:
        parser.error("--start-year must be <= --end-year")
    seasons = discover_seasons(args.input_root, args.start_year, args.end_year)
    all_records: list[dict[str, Any]] = []
    season_diagnostics: list[dict[str, Any]] = []
    for season, folder in seasons.items():
        records, diagnostics = _read_season(folder, season)
        all_records.extend(records)
        season_diagnostics.append(diagnostics)
    observations, issues = collapse_player_seasons(all_records)
    target_season = max(seasons)
    league_averages = _league_averages(observations)
    for season in seasons:
        for side in SIDES:
            if (season, side) not in league_averages:
                issues.append(
                    f"{season} {side}: no qualified player-seasons to form a league "
                    "average; unqualified fallback is 0."
                )
    if any(info["metric_unit_conversion"] == "divide by 100" for info in season_diagnostics):
        if any(info["metric_unit_conversion"] == "none" for info in season_diagnostics):
            issues.append(
                "Source seasons mixed RAPM-per-play and RAPM-per-100 columns; "
                "per-100 values were divided by 100 before aggregation."
            )
    if not observations["team"].astype(str).str.strip().any():
        issues.append(
            "The yearly RAPM exports contain no team values; most_recent_team is "
            "blank rather than guessed."
        )
    preview = _formula_and_example()
    print(preview)
    if args.preview_only:
        print("\nPreview only: no output files were written.")
        return 0

    final, worked = _primary_output(observations, league_averages, target_season)
    nonempty_ids = final.loc[final["player_id"].ne(""), "player_id"]
    if nonempty_ids.duplicated().any():
        raise ValueError("Internal validation failed: duplicate player IDs in final output")
    sensitivity, sensitivity_summary = _sensitivity(
        observations, league_averages, target_season
    )
    examples, selected_examples = _examples(worked)
    primarily_unqualified = final[
        final["percentage_of_weight_from_unqualified_seasons"] > 50
    ].copy()
    args.outdir.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.outdir / "multi_year_rapm.csv", index=False)
    worked.to_csv(args.outdir / "season_calculations.csv", index=False)
    sensitivity.to_csv(args.outdir / "sensitivity_analysis.csv", index=False)
    sensitivity_summary.to_csv(args.outdir / "sensitivity_correlations.csv", index=False)
    examples.to_csv(args.outdir / "example_calculations.csv", index=False)
    primarily_unqualified.to_csv(
        args.outdir / "primarily_unqualified.csv", index=False
    )
    pd.DataFrame(season_diagnostics).to_csv(
        args.outdir / "input_diagnostics.csv", index=False
    )
    _write_summary(
        args.outdir / "summary.txt",
        list(seasons),
        season_diagnostics,
        observations,
        final,
        sensitivity_summary,
        issues,
        selected_examples,
    )
    print(f"\nWrote aggregation outputs to {args.outdir.resolve()}")
    print(
        "Output files: multi_year_rapm.csv, season_calculations.csv, "
        "sensitivity_analysis.csv, sensitivity_correlations.csv, "
        "example_calculations.csv, primarily_unqualified.csv, "
        "input_diagnostics.csv, summary.txt"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
