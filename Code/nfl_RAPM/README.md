# Yards-only football RAPM

`football_rapm.py` estimates player ratings from NFL play-by-play participation
CSVs. It fits exactly three sparse ridge models: rushing plays, passing plays, and
the pooled set of those plays. Positive offensive ratings mean more yards gained;
positive defensive ratings mean fewer yards allowed, after adjusting for the
opposing players and pre-snap situation.

## Requirements

Python 3.10 or newer with `pandas`, `numpy`, `scipy`, `scikit-learn`, and
`nflreadpy`.

## Inspecting the current team CSVs

The current `TEAM_PARTICIPATION_STATS/*/2025.csv` files share a 31-column schema.
They include season/week and team metadata; play IDs and types; `yards_gained`;
starting/ending yard markers; score, down, and distance data; defensive personnel,
positions, and charting fields; and offense/defense player IDs and names.
`rusher_player_id`, `rusher_position`, `posteam`, `qtr`,
`game_seconds_remaining`, and `yardline_100` are absent from these files. The
loader handles those fields as optional: without rusher IDs it cannot create
ball-carrier estimates, and without `posteam` it cannot orient a running score
differential or reliably label player teams. Player positions are looked up by
GSIS ID from `nflreadpy.load_players()`; recognized defensive position labels,
rusher positions, and explicit `--positions` entries take precedence over the
database fallback. Blank or unrecognized labels (including `UNK`) do not replace
a recognized database position. On the current 2025 team CSV player IDs, the
database lookup covered all observed IDs. Position groups follow the player's
roster position even if a participation row lists them on the opposite side;
shrinkage/ranking groups still include side and role. K, P, and LS positions
are shown as `SPEC` rather than `UNK`.

## Run the analysis

Run from the repository root, supplying all team files for the season. The loader
merges the duplicate play rows shared between teams:

```sh
python3 Code/nfl_RAPM/football_rapm.py \
  --input 'Code/nfl_RAPM/TEAM_PARTICIPATION_STATS/*/2025.csv' \
  --outdir rapm_2025
```

To estimate one shared player rating across all selected seasons, use a directory
or glob spanning those seasons. Use `--by-season` when separate player-season
ratings are desired:

```sh
python3 Code/nfl_RAPM/football_rapm.py \
  --input 'Code/nfl_RAPM/TEAM_PARTICIPATION_STATS/*/*.csv' \
  --outdir rapm_all_seasons \
  --by-season
```

To add game-cluster bootstrap intervals, include a positions file for offense
players, and export play-level expectations for auditing:

```sh
python3 Code/nfl_RAPM/football_rapm.py \
  --input 'Code/nfl_RAPM/TEAM_PARTICIPATION_STATS/*/2025.csv' \
  --outdir rapm_2025 \
  --positions player_positions.csv \
  --bootstrap 100 \
  --export-expectations
```

The optional positions CSV requires `player_id,position` columns and can include
`season` for season-specific labels. `--no-nflreadpy-positions` disables the
database lookup if you want to rely only on labels from the CSVs and any supplied
positions file. Install the required runtime packages with
`python3 -m pip install pandas numpy scipy scikit-learn nflreadpy`. Run the
synthetic checks with:

```sh
python3 Code/nfl_RAPM/football_rapm.py --selftest
```

## Estimation and options

Each fit selects lambda once using game-grouped cross-validation on a joint
offense/defense ridge model. The selected penalty is then frozen while the tool
alternates offensive and defensive ridge updates and refits an out-of-fold
situational baseline. Group-prior shrinkage can be disabled with
`--no-group-priors`; `--scheme-controls` opts into the available pre-snap defensive
scheme controls. The lambda grid accepts explicit values or a log-spaced range,
such as `--lambda-grid 10:100000:9`.

Useful controls include `--folds`, `--max-passes`, `--tol`, `--cap`, `--shrink`,
`--min-plays`, `--min-carries`, and `--seed`. Yardage beyond 15 yards is shrunk
toward 15 by default (`--cap 15 --shrink 0.25`).

The output directory contains `rapm_rush.csv`, `rapm_pass.csv`,
`rapm_all_plays.csv`, `rapm_qb.csv`, and `diagnostics.txt`. With
`--export-expectations`, it also contains `play_expectations.csv`, with expected
yards for each side and offense/defense performance residuals for each of the
three fits. Leaderboards are printed to the console.

## Limitations

One season of football RAPM is noisy. Players who are almost always on the field
together (such as a starting offensive line) cannot be separated reliably and
will be shrunk toward similar values. The response is yards only: incompletions
and interceptions appear as zero-yard plays, while turnovers, penalties, and
scoring are not modeled separately. Ratings adjust for teammates and opponents;
they are not proof of causal ability.
