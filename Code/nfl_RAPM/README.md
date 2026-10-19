# Yards-only football RAPM

`football_rapm.py` fits exactly three sparse ridge models from NFL play-by-play CSVs:
rushing plays, passing plays, and all rushing/passing plays pooled. Coefficients are in
normalized yards per play and positive values favor the player's own side of the ball.
The model adjusts for every listed offensive and defensive player, with rushing carriers
represented by a separate ball-carrier variable.

## Requirements

Python 3.10 or newer with `pandas`, `numpy`, `scipy`, and `scikit-learn`.

## Example

The loader accepts optional play fields in addition to the participation
columns, including `spread_line`, `pos_team` (or `posteam`),
`rusher_player_id`, `rusher_player_name`, `rusher_position`,
`rushing_player_type`, `qtr`, `game_seconds_remaining`, and `yardline_100`.
`pos_team` is normalized to the model's canonical `posteam` field and is used to
orient pre-play score and spread relative to the offense. A conflicting nonblank
`posteam` value takes precedence and is reported. `spread_line` follows the
nflverse convention: positive values favor the home team. The model converts it
to `offense_spread`, positive when the possession team is favored, and uses it as
a pre-game context control; rows without a confidently identified possession
side have a missing spread control. Rusher IDs create ball-carrier effects in
rushing models, including when the rusher is absent from the listed offensive
participants, and `rusher_player_name` supplies the display name when needed.
When these optional fields are absent, their associated features are omitted.
Player positions are looked up by
GSIS ID from `nflreadpy.load_players()`; recognized defensive position labels,
rusher positions, and explicit `--positions` entries take precedence over the
database fallback. Blank or unrecognized labels (including `UNK`) do not replace
a recognized database position. On the current 2025 team CSV player IDs, the
database lookup covered all observed IDs. Position groups follow the player's
roster position even if a participation row lists them on the opposite side;
shrinkage/ranking groups still include side and role. K, P, and LS positions
are shown as `SPEC` rather than `UNK`.

From the repository root, pass the team participation data directory (duplicate
plays in team files are merged automatically):

```sh
python3 Code/nfl_RAPM/football_rapm.py \
  --input 'Code/nfl_RAPM/TEAM_PARTICIPATION_STATS/ARI/2024.csv' \
  --outdir rapm_2024 \
  --positions player_positions.csv
```

For a multi-team season, point to a directory or glob so that each game is represented
by both teams' files:

```sh
python3 Code/nfl_RAPM/football_rapm.py \
  --input 'Code/nfl_RAPM/TEAM_PARTICIPATION_STATS/*/2024.csv' \
  --outdir rapm_2024 \
  --lambda-grid 10:100000:9 \
  --bootstrap 100
```

The optional positions file has `player_id,position` columns and may include a `season`
column for season-specific labels. Run `python3 Code/nfl_RAPM/football_rapm.py --selftest`
to exercise yardage normalization and synthetic offense/defense sign recovery.

## Outputs and options

The output directory contains `rapm_rush.csv`, `rapm_pass.csv`,
`rapm_all_plays.csv`, `rapm_qb.csv`, and `diagnostics.txt`. The full RAPM tables include
all estimated player-side-role coefficients; QB and position leaderboards are reporting
views of these same three fits. Common options include `--by-season`, `--cap`,
`--shrink`, `--folds`, `--lambda-grid`, `--max-passes`, `--tol`,
`--no-group-priors`, `--scheme-controls`, `--min-plays`, `--min-carries`,
`--bootstrap`, and `--seed`. Scheme controls, when requested, use only
`defense_personnel`, `n_defense`, and `defenders_in_box` in the situational baseline.

## Data assumptions and limitations

The loader uses game IDs formatted `season_week_home_away` to fill missing season and
week fields, matching the team-participation files in this repository. Score
differentials are omitted when `posteam` is absent or when score fields are constant
within games, since team-file ownership does not establish which side had possession.
When a required optional field such as rusher IDs is absent, carrier-specific estimates
are unavailable rather than inferred.

One season of football RAPM is noisy. Players who are almost always on the field
together (such as a starting offensive line) cannot be separated reliably and will be
shrunk toward similar values. The response is yards only: incompletions and
interceptions appear as zero-yard plays, while turnovers, penalties, and scoring are
not separately modeled. Ratings adjust for teammates and opponents; they are not proof
of causal ability. Team labels are reported only when the offensive team can be
identified from `posteam`; otherwise that output field is blank.
