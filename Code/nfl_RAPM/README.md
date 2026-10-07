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

Each fit selects penalties using game-grouped cross-validation of the reported
pipeline: the same out-of-fold gradient-boosted situational baseline, joint
offense/defense fit, offense/defense alternation, and position-group priors are
used within each validation fold. `CV RMSE (full pipeline, grouped folds)` and
`CV RMSE (baseline-only)` in diagnostics use the same held-out games; the former
scores the player model and baseline together, while the latter scores the
baseline without player effects.

The default penalty candidates share a defense penalty (lambda) and test
offense/defense penalty ratios of `0.5`, `1`, and `2`. The offense penalty is
`lambda * ratio`; the defense penalty is `lambda`. Thus ratios below 1 test
weaker offense shrinkage. Group-prior shrinkage can be disabled with
`--no-group-priors`; `--scheme-controls` opts into the available pre-snap defensive
scheme controls. Both candidate sets are selected together by grouped CV. The
lambda grid accepts explicit values or a log-spaced range,
such as `--lambda-grid 10:750:9`. By default, grouped cross-validation searches
nine logarithmically spaced lambda values from 10 through 750; it selects the
best candidate, so 750 is an upper bound on the default search rather than a
forced lambda. An explicit `--lambda-grid` overrides those defaults. Use
`--offense-penalty-ratios 0.25,0.5,1,2` to override the default ratio candidates.
Result CSVs and diagnostics record the selected offense and defense penalties.

Low-leverage regulation plays are excluded before fitting using the absolute
pre-play lead: at least 21 points from 30:00–15:00 remaining, 17 from 15:00–10:00,
14 from 10:00–5:00, 10 from 5:00–3:00, and 9 from 3:00–0:00. Time beyond 30:00
and overtime are not in these bands. The score fields are cumulative for each
play, so filtering uses the prior play's score rather than a score made by the
current play. Clock uses `game_seconds_remaining` when present, otherwise
quarter and `time_on_clock_start`. When either score or clock state is
unavailable, the play is kept and the missing-state count is written to
`diagnostics.txt`; exclusion totals are also reported there by time band.

The default convergence requirement is `--tol 0.001`: offense/defense
alternation continues until the largest within-group deviation change is below
0.001 or `--max-passes` is reached. Bootstrap and split-half refits use the same
0.001 threshold.

Useful controls include `--folds`, `--max-passes`, `--tol`, `--cap`,
`--min-plays`, `--min-carries`, and `--seed`. Yardage is hard-clipped to the
range from -15 to +15 by default (`--cap 15`).

The output directory contains `rapm_rush.csv`, `rapm_pass.csv`,
`rapm_all_plays.csv`, `rapm_qb.csv`, and `diagnostics.txt`. With
`--export-expectations`, it also contains `play_expectations.csv`, with expected
yards for each side and offense/defense performance residuals for each of the
three fits. The pooled `rapm_all_plays.csv` retains its pooled RAPM and includes
separate `rush_rapm` and `pass_rapm` estimates, per-100-play values, and
play-count columns for each player's matching side and role; a blank component
means the player had no observations in that play-type fit. Leaderboards are
printed to the console.

With `--bootstrap`, `rapm_p10` and `rapm_p90` are the 10th and 90th percentiles
for raw `rapm`; `rapm_vs_group_avg_p10` and
`rapm_vs_group_avg_p90` are the corresponding percentiles for deviation from
the group target. Each game-cluster replicate refits the situational baseline and player
coefficients and reruns grouped CV to select lambda and the offense/defense
penalty ratio. The intervals therefore reflect game-sampling and penalty
selection variability, conditional on the chosen model specification and
candidate grids. Bootstrap runs are more computationally expensive because
each replicate repeats grouped CV.

## Aggregate existing seasons

To combine existing season exports without rerunning the play-level regression,
use `aggregate_rapm_years.py`. It reads `rapm_all_plays.csv` from each
`rapm_[YEAR]` folder and supports an inclusive year range:

```sh
python3 Code/nfl_RAPM/aggregate_rapm_years.py \
  --input-root . \
  --start-year 2018 \
  --end-year 2025 \
  --outdir rapm_multi_year
```

The aggregator emits one `final_RAPM` per player, using only that player's
primary side (the side with more observed plays across supplied seasons). It
never adds offensive and defensive ratings; any opposite-side records are
excluded from the final value and reported in diagnostic columns. For any
unqualified season (<200 plays), the observed RAPM is replaced by the
same-season, same-side, same-position-group 25th percentile among qualified
players. If that position group has no qualified players, it falls back to the
same-side seasonal 25th percentile; if that is unavailable too, it uses zero.
The unqualified season retains a capped low sample weight.

It prints the weighting formula and an auditable worked example before writing
files. Use `--preview-only` to inspect the formula without creating outputs.
Results include the player-level CSV, season-by-season calculations, examples,
input diagnostics, half-life and sample-weight sensitivity analyses, and a
summary report. Verify the calculation logic with:

```sh
python3 Code/nfl_RAPM/aggregate_rapm_years.py --selftest
```

## Limitations

One season of football RAPM is noisy. Players who are almost always on the field
together (such as a starting offensive line) cannot be separated reliably and
will be shrunk toward similar values. The response is yards only: incompletions
and interceptions appear as zero-yard plays, while turnovers, penalties, and
scoring are not modeled separately. Ratings adjust for teammates and opponents;
they are not proof of causal ability.
