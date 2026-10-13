# Quantum Route

Quantum Route is an NFL prediction model built to improve betting efficiency and the accuracy of game predictions.

It works from the bottom up. Eight seasons of play-by-play data (2018–2025) feed an expected-points model. A drive simulator is built on top of that, and above it sit Bayesian team models fit with a hand-rolled MCMC sampler. The simulated games are then scored against the closing betting lines in a walk-forward backtest.

```
play-by-play CSVs ──► Expected Points (EP/EPA) ──► Team strength + coaching tendencies (MCMC)
        │                                                        │
        └──► Drive simulator (play calls, yards, FGs) ──────────►├──► Game simulator ──► Backtest vs. closing lines
```

## Project layout

```
quantum-route/
├── Code/
│   ├── First_Down_Distance/   Split plays by down, distance and run/pass
│   ├── Drive_Model/           Play loading, drive building, drive simulator
│   ├── Scoring_Model/         Expected points (EP) and EPA per play
│   ├── Team_Model/            MCMC team strength, play-calling tendencies, game sim, backtest
│   ├── Dashboard/             Streamlit dashboard over the backtest history
│   ├── db.py                  DuckDB (SQL) store for backtest history
│   └── Main.py                Scratch entry point (loads the schedule)
├── GAMES/<season>/<week>/     One CSV per game, every play (2018–2025)
├── Data/                      Generated outputs and caches (git-ignored)
├── writeups/                  Project plan (PDF) and research notes
└── nflreadpyTEST.py           nflreadpy demo + exporter that writes the GAMES/ CSVs
```

## The models

### 1. Down and distance: `Code/First_Down_Distance/`
`Distance_Stats.py` splits every run and pass play into buckets by down (1st to 4th) and yards to go (1–3, 4–7, 8–12, 13–20, 21+). It writes one CSV per bucket under `Data/First_Down_Distance/`. The later models use the same buckets.

### 2. Drive model: `Code/Drive_Model/`
- **`plays.py`**: loads every CSV in `GAMES/`, caches the result to `Data/plays.parquet`, and adds features such as drives, possession and game state.
- **`models.py`**: the building blocks of a play. They cover what gets called in each situation (`PlayCallModel`), how many yards the call gains (`EmpiricalYardsModel`, drawn from real plays), and whether a field goal is good (`FieldGoalModel`).
- **`simulate.py`**: `DriveSimulator` runs a drive one play at a time until it ends in a touchdown, field goal, punt, turnover or turnover on downs.
- **`run_drive_sim.py`**: gives the expected points and yards of a drive from every starting yard line, and checks the simulation against real drives.
- **`down_distance_ev.py`**: compares running and passing in each down-and-distance bucket. Both calls are simulated from the same sampled game states, so the difference reflects the call itself and not the situation it was made in.

### 3. Scoring model: `Code/Scoring_Model/`
`expected_points.py` fits a multinomial model of the **next score in the half**: touchdown, field goal, safety, the same three for the opponent, or no score. It uses yard line, down, distance and time left in the half. Expected points (EP) is the probability-weighted value of those outcomes, and **EPA** is the change in EP caused by a play. `run_ep_model.py` does three things:
- checks the model on the held-out 2025 season (log loss and calibration);
- writes an EP table by field position and down/distance;
- writes per-team offensive and defensive EPA profiles.

### 4. Team model: `Code/Team_Model/`
- **`mcmc.py`**: a hand-rolled Metropolis-within-Gibbs sampler. Its step sizes adapt during burn-in, it runs several chains, and it reports split R-hat and effective sample size.
- **`team_strength.py`**: a hierarchical Bayesian model of EPA per play. Each team gets an offense effect and a defense effect, estimated separately for runs and passes, plus a shared home-field term.
- **`play_calling.py`**: models coaching tendencies as offsets from league-average behavior. It measures **pass rate over expected** on 1st–3rd down and **4th-down go rate over expected**. Both account for down, distance, field position, score and clock.
- **`game_sim.py`**: `GamePredictor` fits everything on games played before a given week, weighting recent games more heavily. It then simulates full games as alternating drives, including kickoffs, the clock and overtime. Each batch of simulations uses a fresh posterior draw, so uncertainty about the parameters shows up in the predicted margin.
- **`run_backtest.py`**: a walk-forward backtest. Each week is predicted using only the games before it, and the predictions are scored against the closing spread, total and moneyline. The scores include mean absolute error (MAE), Brier score, and against-the-spread (ATS) record by edge size.
- **`coach_vs_player.py`**: a crossed random-effects model that splits offensive performance among the head coach, the starting QB and the rest of the roster.

### 5. Backtest history and dashboard: `Code/db.py`, `Code/Dashboard/`
Every backtest run is saved to a DuckDB database at `Data/quantum_route.duckdb`, so runs can be compared instead of overwriting `backtest_<season>.csv`. Each run records its season, start and finish time, git commit (and whether `Code/` had uncommitted changes) and an optional note. Predictions are saved week by week as the backtest goes, so a run that stops partway still shows up.

The scoring lives in SQL views that match `run_backtest.py report` with no slope adjustment:
- `run_summary`: one row per run with margin MAE, Brier score, totals MAE and ATS record, each next to the closing line's;
- `weekly_summary`: the same accuracy by week;
- `ats_by_edge`: cover rate by how far the model's margin is from the spread;
- `backtest_graded`: every game with its edge, cover and win flags, for your own queries.

The dashboard (`Code/Dashboard/app.py`) shows the run history, one run in detail (weekly MAE, ATS by edge, a filterable game table) and a side-by-side comparison of two runs.

## Research notes

[`writeups/coaching_vs_player_effects.txt`](writeups/coaching_vs_player_effects.txt) covers 2018–2025 (4,254 team-games):

| What is measured                  | Head coach | Starting QB | Rest of roster/staff |
|-----------------------------------|-----------:|------------:|---------------------:|
| Offensive efficiency (EPA/play)   | ~10%       | ~59%        | ~31%                 |
| Pass rate over expected (scheme)  | ~14%       | ~59%        | ~27%                 |
| 4th-down aggressiveness           | ~59%       | ~9%         | ~33%                 |

In short, offensive efficiency depends mostly on the quarterback, while 4th-down decisions depend mostly on the head coach. The note explains the method, the uncertainty intervals and how the results compare with published research.

[`writeups/Quantum_Route_Plans_Writeup_V1.pdf`](writeups/Quantum_Route_Plans_Writeup_V1.pdf) describes the overall project plan.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas pyarrow duckdb streamlit
```

- `duckdb` stores the backtest history and `streamlit` runs the dashboard.

- `pyarrow` is needed for the parquet caches in `Data/`.
- To re-export game data from nflverse, also install `nflreadpy` and `polars`.
- The team models download the nflverse schedule (`games.csv`, which has scores, closing lines, coaches and starting QBs) the first time they run.

## Running

Each script is run from its own folder. Outputs go to the matching folder under `Data/`.

```bash
# Down / distance splits
cd Code/First_Down_Distance && python Distance_Stats.py

# Drive simulator and run-vs-pass value by down and distance
cd Code/Drive_Model && python run_drive_sim.py
cd Code/Drive_Model && python down_distance_ev.py

# Expected points model
cd Code/Scoring_Model && python run_ep_model.py

# Team strength + coaching tendencies for a season (defaults to the latest)
cd Code/Team_Model && python run_team_model.py 2025

# Walk-forward backtest for a season, then score all saved seasons against the market.
# Each run is also saved to the backtest history; the optional note labels it.
cd Code/Team_Model && python run_backtest.py 2025 "tighter team priors"
cd Code/Team_Model && python run_backtest.py report

# Backtest history: load backtest CSVs from before the history existed, list runs,
# open the dashboard, or query the database directly (needs the duckdb CLI)
cd Code && python db.py import
cd Code && python db.py history
streamlit run Code/Dashboard/app.py
duckdb Data/quantum_route.duckdb "SELECT run_id, mae_model, mae_market FROM run_summary"

# Coach vs. QB vs. roster decomposition
cd Code/Team_Model && python coach_vs_player.py
```

The first run of the team model builds `Data/Team_Model/plays_epa_full.parquet`, and later runs reuse it. Delete that file to rebuild it after changing the EP model.

## Data

The play-by-play data comes from [nflverse](https://github.com/nflverse), accessed through `nflreadpy`. It is exported as one CSV per game at `GAMES/<season>/<week>/nfl_plays_<season>_<week>_<day>_<HOME>_<AWAY>.csv`. Each row is one play and includes:
- play type, yards gained, field position and the clock;
- down and distance, and the score;
- defensive personnel, coverage and pressure;
- the offensive and defensive players on the field.
