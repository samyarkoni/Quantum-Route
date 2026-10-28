# NFL pre-game metric study (stats only: no spread, odds, power rankings)

Needs: python3, nflreadpy, polars, pandas, scikit-learn, scipy, pyarrow.
Run scripts from the repository root. Intermediate parquet files are stored in
`nfl_metric_study/data_cache/`.

## Run order
1. `build_team_game.py`  play-by-play 1999-2026 -> per-team-game offense metrics, QB lines (parquet)
2. `tune_ratings.py`, `tune_offseason.py`  choose rating half-life (20 weeks) and ridge strength using 2004-2014 only
3. `features.py`  leakage-free pre-game feature table (`model_table.parquet`)
4. `leak_test.py`  deletes everything from a week onward and checks that week's features are unchanged (must print 0 mismatches)
5. `reliability.py`  signal-vs-noise per metric, weight on last season, trailing-window length
6. `effects.py`  effect sizes in points (rest, travel, weather, slot, QB/coach), home-field advantage by era, offense vs defense persistence
7. `models.py`  rolling-origin comparison of feature groups (test seasons 2015-2025), bootstrap CIs
8. `final_eval.py`  simple rules, gradient-boosting check, margin model, 2026 holdout
9. `inj_eval.py`  injuries weighted by normal snap share
10. `verify.py`  independent recomputation of headline numbers from the raw schedule
11. `predictability_eval.py`  walk-forward score forecasts and pregame-adjusted live win probability

`common.py`, `featsets.py` are shared modules. Result tables: `results/*.csv`, full console log of step 7: `results/models_out.txt`.

`predictability_eval.py` deliberately does not load betting lines. It evaluates predicted
home/away points, game margin, total points, and in-game win probability against final
game outcomes. Team efficiency is estimated from nflreadpy's supplied play-level `epa`,
aggregated as offensive EPA per pass/rush play and adjusted for opponent, venue, recency,
and ridge shrinkage. Walk-forward score models test EPA efficiency on its own, jointly
with the points-based rating, and with starting-QB EPA; the points-only model remains the
control. Both team and QB EPA are read directly from `nflreadpy.load_pbp()`; the pipeline
does not recalculate EPA or fit an EPA model. QB EPA is attributed to `passer_player_id`
on regular-season QB dropbacks
(`qb_dropback == 1`), recency weighted, and shrunk toward zero using 200 prior dropbacks.
The walk-forward live model compares score/time/game-state features with the pregame
strength prior, then evaluates team-EPA and QB-EPA adjustments; each game is weighted
equally despite having many play-level live states.

Run it independently from the repository root with
`python3 nfl_metric_study/scripts/predictability_eval.py`. It uses nflreadpy schedules
and the local `GAMES/2018` through `GAMES/2025` play files. Metrics and game-level
pregame score predictions are written to `results/`; live play-level predictions are
evaluated in memory to avoid producing a large redundant file.

The score model keeps a simple points-based opponent-adjusted rating as its baseline and
compares it against versions using team EPA efficiency, QB EPA, or both. These are
walk-forward, points-calibrated score forecasts; they do not use betting lines. None of
these candidates adds injuries or weather. The league-average baseline establishes
whether the team-rating inputs carry predictive signal, not whether this is the strongest
available score model. The live-probability comparison is strictly forward by season,
covers regulation play states in 2020-2025, and has only six held-out seasons. Treat its
season-cluster intervals as preliminary, not as proof of a permanent lift.

## Protocol in one paragraph
Ratings for week W use only games before W's first kickoff (exponentially decayed, ridge-shrunk opponent adjustment).
Models train on seasons before each test season; regularization and rating hyperparameters were fixed using seasons <= 2014.
Targets are home win (ties dropped) and home margin; regular season only.

## Known limits
- Starting QB comes from the schedule file (the actual starter; known before kickoff in practice).
- Temperature/wind are missing for many recent games (filled with prior climatology), so the weather test is weak.
- Travel is approximated from home-city coordinates; short-week effects cannot be separated from Thursday games.
- EPA comes from nflfastR's expected-points model (built from play outcomes, not opinions).
