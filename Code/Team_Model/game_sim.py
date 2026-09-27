import numpy as np
import pandas as pd

from data import before, recency_weights
from mcmc import run_chains
from models import EmpiricalYardsModel, FieldGoalModel, PlayCallModel
from play_calling import ExpectedCallModel, TeamPlayCallModel, TeamTendencyModel
from plays import build_drives
from simulate import DriveSimulator
from team_strength import TeamStrengthModel

KICK_ENDINGS = ["punt", "field_goal", "missed_fg"]  # the kick is a snap that ends the drive
RELATIVE_ENDINGS = ["punt", "turnover", "downs", "missed_fg"]
PHYSICS_SEASONS = 3  # league-wide play outcomes come from the most recent seasons of history


class EpaYardsModel(EmpiricalYardsModel):
    """EmpiricalYardsModel that also keeps each pooled play's EPA, with pools capped in size so
    they can be reweighted quickly."""

    def __init__(self, plays, min_count=50, max_pool=2000, seed=0):
        super().__init__(plays, min_count)
        gains = plays[plays["is_scrimmage"] & plays["call"].isin(["run", "pass"])]
        self.epa = gains["epa"].fillna(0).to_numpy()
        rng = np.random.default_rng(seed)
        for level, pools in self.pools.items():
            for key, idx in pools.items():
                if len(idx) > max_pool:
                    pools[key] = rng.choice(idx, max_pool, replace=False)


class TiltedYardsModel:
    """One offense against one defense. Each situation's historical plays are reweighted by
    exp(lambda * epa), with lambda solved so the pool's mean EPA moves by the matchup's shift for
    that call. Good offenses then draw more explosive plays and fewer turnovers, in the proportions
    the league's own plays have."""

    def __init__(self, base, shift):
        self.base = base
        self.shift = shift  # {"run": epa/play, "pass": epa/play}
        self.cache = {}

    def _cdf(self, idx, delta, iterations=4):
        epa = self.base.epa[idx]
        centered = epa - epa.mean()
        lam = 0.0
        for _ in range(iterations):
            w = np.exp(np.clip(lam * centered, -50, 50))
            w /= w.sum()
            mean = w @ centered
            var = w @ (centered - mean) ** 2
            lam = float(np.clip(lam + (delta - mean) / max(var, 1e-6), -5, 5))
        return np.cumsum(w)

    def sample(self, call, down, yds_to_go, yardline, rng):
        idx = self.base.pool(call, down, yds_to_go, yardline)
        key = (id(idx), call)
        if key not in self.cache:
            self.cache[key] = self._cdf(idx, self.shift[call])
        cdf = self.cache[key]
        i = idx[min(np.searchsorted(cdf, rng.random() * cdf[-1]), len(idx) - 1)]
        return int(self.base.yards[i]), bool(self.base.turnover[i])


def possession_changes(plays):
    """One row per drive that handed the ball to the other team in the same half: how it ended,
    its last snap, and where the next drive started (yards from that team's goal line)."""
    drives = build_drives(plays)
    scrimmage = plays[plays["is_scrimmage"] & plays["drive_id"].notna()].reset_index()
    first = scrimmage.groupby("drive_id")
    drives = drives.join(pd.DataFrame({
        "posteam": first["posteam"].first(), "half": first["half"].first(),
        "last_snap": first["starting_yard"].last(), "order": first["index"].first(),
    }))
    drives.loc[(drives["outcome"] == "field_goal") & (drives["points"] <= 0), "outcome"] = "missed_fg"
    drives = drives.sort_values("order")
    by_half = drives.groupby(["game_id", "half"], sort=False)
    drives["next_start"] = by_half["start_yardline"].shift(-1)
    drives["next_team"] = by_half["posteam"].shift(-1)
    return drives[drives["next_start"].notna() & (drives["next_team"] != drives["posteam"])]


class FieldPositionModel:
    """Where the next possession starts, drawn from real possession changes: kickoffs and free kicks
    from the most recent season with enough of them (kickoff rules changed in 2024 and 2025), and
    punts / turnovers / downs / missed field goals as an offset from the spot of the last snap."""

    def __init__(self, plays, min_kicks=300):
        changes = possession_changes(plays)
        kicks = changes[changes["outcome"].isin(["touchdown", "field_goal"])]
        per_season = kicks.groupby("season").size()
        recent = per_season[per_season >= min_kicks].index.max()
        self.kickoffs = kicks.loc[kicks["season"] == recent, "next_start"].to_numpy()
        self.free_kicks = changes.loc[changes["outcome"] == "safety", "next_start"].to_numpy()

        # Conversion points after touchdowns (PAT and two-point tries)
        drives = build_drives(plays)
        tds = drives[(drives["outcome"] == "touchdown") & (drives["season"] == recent)]
        self.conversions = np.clip(tds["points"] - 6, 0, 2).round().astype(int).to_numpy()

        relative = changes[changes["outcome"].isin(RELATIVE_ENDINGS)]
        offset = relative["next_start"] - (100 - relative["last_snap"])
        bucket = (relative["last_snap"] // 10).astype(int)
        self.offsets = {key: grp.to_numpy() for key, grp in offset.groupby([relative["outcome"], bucket]) if len(grp) >= 30}
        self.offsets_any = {key: grp.to_numpy() for key, grp in offset.groupby(relative["outcome"])}

    def kickoff(self, rng):
        return int(self.kickoffs[rng.integers(len(self.kickoffs))])

    def free_kick(self, rng):
        return int(self.free_kicks[rng.integers(len(self.free_kicks))])

    def conversion(self, rng):
        return int(self.conversions[rng.integers(len(self.conversions))])

    def after(self, outcome, last_snap, rng):
        pool = self.offsets.get((outcome, int(last_snap) // 10), self.offsets_any[outcome])
        return int(np.clip(100 - last_snap + pool[rng.integers(len(pool))], 1, 99))


class GameSimulator:
    """Plays a game as alternating drives from DriveSimulator. Each play (and each kick that ends a
    drive) uses seconds_per_play of the 30-minute half, and a drive still going when the half runs
    out ends there. Ties go to one overtime period: a touchdown on the first possession wins, a
    field goal gives the other team a possession, then sudden death."""

    def __init__(self, field_positions, seconds_per_play, overtime_seconds=600):
        self.fp = field_positions
        self.seconds_per_play = seconds_per_play
        self.overtime_seconds = overtime_seconds

    def _drive(self, sims, offense, start, clock, score, rng):
        """Plays one drive and scores it. Returns (next offense, next start, clock left, ended)."""
        budget = max(1, int(np.ceil(clock / self.seconds_per_play)))
        r = sims[offense].simulate(start, rng, max_plays=budget)
        clock -= (r.plays + (r.outcome in KICK_ENDINGS)) * self.seconds_per_play
        other = 1 - offense
        if r.outcome == "touchdown":
            score[offense] += 6 + self.fp.conversion(rng)
            return other, self.fp.kickoff(rng), clock, False
        if r.outcome == "field_goal":
            score[offense] += 3
            return other, self.fp.kickoff(rng), clock, False
        if r.outcome == "defensive_td":
            score[other] += 6 + self.fp.conversion(rng)
            return offense, self.fp.kickoff(rng), clock, False  # the scoring team kicks off to them
        if r.outcome == "safety":
            score[other] += 2
            return other, self.fp.free_kick(rng), clock, False
        if r.outcome == "max_plays":
            return other, 0, 0, True
        return other, self.fp.after(r.outcome, r.last_snap, rng), clock, False

    def simulate(self, home, away, rng):
        """home, away: DriveSimulators for each offense in this matchup. Returns (home_pts, away_pts)."""
        sims, score = [home, away], [0, 0]
        first_receiver = int(rng.integers(2))
        for receiver in (first_receiver, 1 - first_receiver):
            offense, start, clock = receiver, self.fp.kickoff(rng), 1800.0
            while clock > 0:
                offense, start, clock, ended = self._drive(sims, offense, start, clock, score, rng)
                if ended:
                    break

        if score[0] == score[1]:
            offense, start, clock = int(rng.integers(2)), self.fp.kickoff(rng), float(self.overtime_seconds)
            possessions, first_was_fg = 0, False
            while clock > 0:
                before_drive = list(score)
                offense_before = offense
                offense, start, clock, ended = self._drive(sims, offense, start, clock, score, rng)
                possessions += 1
                if possessions == 1:
                    first_was_fg = score[offense_before] - before_drive[offense_before] == 3
                if ended or (score[0] != score[1] and not (possessions == 1 and first_was_fg)):
                    break
        return score[0], score[1]


class GamePredictor:
    """Everything fit on games before (season, week): league play physics, team strength and
    team tendencies (MCMC, recency weighted), then games simulated with a fresh posterior draw per
    batch of simulations so parameter uncertainty reaches the predicted margin."""

    def __init__(self, plays, season, week, n_chains=2, n_burn=500, n_draws=1000, seed=0, verbose=False):
        history = before(plays, season, week).reset_index(drop=True)
        physics = history[history["season"] >= history["season"].max() - PHYSICS_SEASONS + 1]

        self.base_calls = PlayCallModel(physics)
        self.yards = EpaYardsModel(physics)
        field_goals = FieldGoalModel(physics)
        turnovers = physics[physics["is_scrimmage"] & physics["is_turnover"]]
        self.defensive_td_rate = turnovers["defense_score"].mean()
        self.field_goals = field_goals

        field_positions = FieldPositionModel(physics)
        self.environment, seconds_per_play = self._calibrate(physics, field_positions, seed)
        self.game = GameSimulator(field_positions, seconds_per_play)

        weights = recency_weights(history, season, week)
        recent = history[weights > 0].reset_index(drop=True)
        w = weights[weights > 0]
        self.strength = TeamStrengthModel(recent, w)
        self.strength_draws, _ = run_chains(self.strength, n_chains, n_burn, n_draws, seed=seed)
        self.tendency = TeamTendencyModel(recent, ExpectedCallModel().fit(physics), w)
        self.tendency_draws, _ = run_chains(self.tendency, n_chains, n_burn, n_draws, seed=seed + 1)

        self.strength_index = {t: i for i, t in enumerate(self.strength.teams)}
        self.tendency_index = {t: i for i, t in enumerate(self.tendency.teams)}
        flat = lambda d: {k: v.reshape(-1, *v.shape[2:]) for k, v in d.items()}  # noqa: E731
        self.s_flat, self.t_flat = flat(self.strength_draws), flat(self.tendency_draws)
        if verbose:
            print(f"  fit through {season} week {week - 1}: {len(recent):,} weighted plays, "
                  f"environment {self.environment:+.3f} epa/play, {seconds_per_play:.1f} s/play")

    def _calibrate(self, physics, field_positions, seed, n_drives=4000, iterations=4):
        """Two league-wide constants the simulation can't get from play outcomes alone. Penalties
        aren't simulated, so drives come out slightly less efficient and longer than real ones:
        a small EPA shift for every offense is solved (secant method, common random numbers) so
        simulated points per drive match real drives from the same starting spots, then seconds
        per play are set so a game has as many drives as real games do."""
        drives = build_drives(physics)
        drives_per_game = drives.groupby("game_id").size().mean()
        complete = drives[drives["outcome"] != "end_of_half"]
        target = complete["points"].mean()
        starts = complete["start_yardline"].clip(1, 99).astype(int).to_numpy()
        conversion = field_positions.conversions.mean()
        value = {"touchdown": 6 + conversion, "field_goal": 3, "defensive_td": -6 - conversion, "safety": -2}

        def simulate(shift):
            rng = np.random.default_rng(seed)
            sim = DriveSimulator(self.base_calls, TiltedYardsModel(self.yards, {"run": shift, "pass": shift}),
                                 self.field_goals, td_points=6, defensive_td_rate=self.defensive_td_rate)
            results = [sim.simulate(starts[rng.integers(len(starts))], rng) for _ in range(n_drives)]
            points = np.mean([value.get(r.outcome, 0) for r in results])
            units = np.mean([r.plays + (r.outcome in KICK_ENDINGS) for r in results])
            return points, units

        shifts, errors = [0.0, 0.05], []
        for shift in shifts:
            errors.append(simulate(shift)[0] - target)
        for _ in range(iterations - 2):
            slope = (errors[-1] - errors[-2]) / (shifts[-1] - shifts[-2])
            shifts.append(shifts[-1] - errors[-1] / slope)
            errors.append(simulate(shifts[-1])[0] - target)
        environment = shifts[int(np.argmin(np.abs(errors)))]
        _, units = simulate(environment)
        return environment, 3600 / (drives_per_game * units)

    def _drive_sim(self, offense, defense, home_sign, draw):
        s, t = self.s_flat, self.t_flat
        o, d = self.strength_index[offense], self.strength_index[defense]
        shift = {call: self.environment + s["off"][draw, o, j] + s["def"][draw, d, j] + s["eta"][draw, 0] * home_sign
                 for j, call in enumerate(["run", "pass"])}
        k = self.tendency_index[offense]
        calls = TeamPlayCallModel(self.base_calls, t["pass_offset"][draw, k], t["go_offset"][draw, k])
        return DriveSimulator(calls, TiltedYardsModel(self.yards, shift), self.field_goals,
                              td_points=6, defensive_td_rate=self.defensive_td_rate)

    def simulate(self, home, away, neutral=False, n_draws=50, sims_per_draw=8, seed=0):
        """Array of (home_points, away_points), one row per simulated game."""
        rng = np.random.default_rng(seed)
        sign = 0 if neutral else 1
        draws = rng.integers(len(self.s_flat["mu"]), size=n_draws)
        results = []
        for draw in draws:
            home_sim = self._drive_sim(home, away, sign, draw)
            away_sim = self._drive_sim(away, home, -sign, draw)
            results += [self.game.simulate(home_sim, away_sim, rng) for _ in range(sims_per_draw)]
        return np.array(results)

    def predict(self, home, away, spread_line=None, neutral=False, **kwargs):
        scores = self.simulate(home, away, neutral, **kwargs)
        margin = scores[:, 0] - scores[:, 1]
        out = {
            "home_pts": scores[:, 0].mean(), "away_pts": scores[:, 1].mean(),
            "margin": margin.mean(), "margin_sd": margin.std(), "total": scores.sum(axis=1).mean(),
            "p_home_win": (margin > 0).mean() + 0.5 * (margin == 0).mean(),
        }
        if spread_line is not None:
            decided = margin != spread_line
            out["p_home_cover"] = (margin[decided] > spread_line).mean() if decided.any() else 0.5
        return out
