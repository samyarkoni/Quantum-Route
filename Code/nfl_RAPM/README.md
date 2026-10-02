# Football RAPM-like yards model

`nfl_RAPM.py` reads the team participation CSVs, filters to one requested season,
deduplicates `(game_id,
play_id)`, reports contradictory duplicate fields, excludes non-scrimmage and
special-teams play types, and fits separate rushing and passing models.

The outcome is strictly `yards_gained`. Each model is one joint ridge-regularized
least-squares fit: offensive participation is coded `+1`, defensive
participation `-1`, so a positive reported offense estimate means more yards and
a positive reported defense estimate means more yards prevented. Controls are
numeric fields observed before the play (`starting_yard`, `down`, `yds_to_go`,
and available personnel/box counts). Outcome, ending-yard, score, and clock
fields are not predictors.

By default, plays are excluded as low leverage when the pre-play score lead is
at least the configured table threshold: 21 points with 30:00–15:00 remaining,
17 with 15:00–10:00, 14 with 10:00–5:00, 10 with 5:00–3:00, and 9 with 3:00–0:00.
More than 30:00 remaining and overtime are outside these bands. Scores are
carried forward from the previous play because the participation files store
the current play's cumulative score; clock time comes from
`game_seconds_remaining` when available, otherwise quarter plus
`time_on_clock_start`. If score or clock state is missing, the play is retained
and counted in `low_leverage_state_missing`. CLI runs print exclusion counts by
band.

The default `field_players` policy treats every listed player as on the field,
but removes configured quarterback IDs from ordinary running-back runs. QB runs
(identified by `rusher_position` or `rushing_player_type` containing `qb`, as
well as `qb_kneel`/`qb_spike`) retain all listed offensive players. The bundled
files do not provide rusher/position or quarterback IDs, so that exception
cannot be inferred from names; pass `quarterback_ids` in Python or use
`all_listed` explicitly. Player IDs, never names, are model identities.

The default ridge strength is `10.0` (yards are not standardized); it is
configurable. `standard_error` is a residual/information approximation and
should be read with play counts. Results are associations adjusted for the
listed controls, not proof of causation. The source also cannot establish that
every listed player was actually on the field or establish divisions.

Example:

```bash
python3 Code/nfl_RAPM/nfl_RAPM.py Code/nfl_RAPM/TEAM_PARTICIPATION_STATS \
  --season 2024 --output results/football_rapm_2024.csv --ridge 10
```

The exported CSV contains one row per player, side, and model for the selected
season. The `season` column identifies the fitted season; rerun with another
`--season` value to create another season's estimates.

The export also includes `team`, `position`, and `player_metadata_found`.
These fields are looked up by GSIS player ID in the selected season's
`nflreadpy.load_rosters` data. Players without a roster match retain blank team
and position fields and are listed in `result.diagnostics.player_metadata_missing`
when using the Python API. The tool does not guess metadata from a player's
display name.
