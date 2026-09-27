import numpy as np
import pandas as pd

from mcmc import StepTuner, log_half_normal, metropolis_update

TAU_SCALE = 0.5  # half-normal prior scale on the spread of team tendencies (logit units)
PROB_FLOOR = 0.005


def _logit(p):
    p = np.clip(p, PROB_FLOOR, 1 - PROB_FLOOR)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def call_features(plays):
    """Situation and game script: down, distance, field position, score, and clock. Trailing late
    forces passing and leading late forces running regardless of who is coaching, so tendencies are
    measured against this."""
    y = plays["starting_yard"].to_numpy(float)
    down = plays["down"].to_numpy(float)
    ytg = np.clip(plays["yds_to_go"].to_numpy(float), 1, None)
    diff = np.clip(plays["score_diff"].to_numpy(float), -28, 28) / 7
    secs = plays["half_seconds_left"].to_numpy(float)
    game_left = (secs + np.where(plays["half"] == 1, 1800, 0)) / 3600  # 1 at kickoff, 0 at the end
    late = np.clip(120 - secs, 0, None) / 120

    columns = [np.ones_like(y), y / 100] + [np.clip(y - k, 0, None) / 100 for k in (10, 20, 50, 80)]
    columns += [down == d for d in (2, 3, 4)]
    columns += [np.log(ytg), np.log(ytg) * (down == 2), np.log(ytg) * (down >= 3), ytg >= y]
    columns += [diff, np.clip(diff, 0, None), diff * (1 - game_left), np.clip(diff, 0, None) * (1 - game_left)]
    columns += [game_left, late, late * (plays["half"] == 2)]
    return np.column_stack(columns).astype(float)


def fit_logistic(X, y, ridge=1.0, iterations=25):
    """Newton-Raphson logistic regression (ridge on everything but the intercept)."""
    beta = np.zeros(X.shape[1])
    penalty = np.full(X.shape[1], ridge)
    penalty[0] = 0
    for _ in range(iterations):
        p = _sigmoid(X @ beta)
        gradient = X.T @ (y - p) - penalty * beta
        hessian = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def pass_mask(plays):
    """1st-3rd down run/pass calls in regulation (the pass-rate question)."""
    return (plays["call"].isin(["run", "pass"]) & plays["down"].between(1, 3) & (plays["half"] <= 2)
            & plays["posteam"].notna() & plays["score_diff"].notna()).to_numpy()


def fourth_mask(plays):
    """4th-down decisions in regulation: go (run/pass) vs kick (punt/field goal)."""
    return (plays["call"].isin(["run", "pass", "punt", "field_goal"]) & (plays["down"] == 4)
            & (plays["half"] <= 2) & plays["posteam"].notna() & plays["score_diff"].notna()).to_numpy()


class ExpectedCallModel:
    """League-average P(pass) on early downs and P(go) on 4th down given situation and game script."""

    def fit(self, plays):
        passes = plays[pass_mask(plays)]
        fourths = plays[fourth_mask(plays)]
        self.pass_beta = fit_logistic(call_features(passes), (passes["call"] == "pass").to_numpy(float))
        self.go_beta = fit_logistic(call_features(fourths), fourths["call"].isin(["run", "pass"]).to_numpy(float))
        return self

    def pass_logit(self, plays):
        return call_features(plays) @ self.pass_beta

    def go_logit(self, plays):
        return call_features(plays) @ self.go_beta


class TeamTendencyModel:
    """Coaching tendencies as offsets on the league-average logit:

        P(pass) = sigmoid(expected_pass_logit + pass_offset[team])       (1st-3rd down)
        P(go)   = sigmoid(expected_go_logit + go_offset[team])           (4th down)
        offsets ~ Normal(0, tau),  tau ~ HalfNormal(TAU_SCALE)

    pass_offset is pass rate over expected in logit units; given tau, each team's offset depends
    only on its own plays, so all teams are updated in one vectorized Metropolis step."""

    def __init__(self, plays, expected, weights=None):
        w = np.ones(len(plays)) if weights is None else np.asarray(weights, dtype=float)
        self.teams = np.array(sorted(plays["posteam"].dropna().unique()))
        index = {t: i for i, t in enumerate(self.teams)}
        self.data = {}
        for name, mask, logit_fn, target in (
            ("pass", pass_mask(plays), expected.pass_logit, lambda p: p["call"] == "pass"),
            ("go", fourth_mask(plays), expected.go_logit, lambda p: p["call"].isin(["run", "pass"])),
        ):
            keep = mask & (w > 0)
            rows = plays[keep]
            self.data[name] = {
                "team": rows["posteam"].map(index).to_numpy(),
                "base": logit_fn(rows),
                "y": target(rows).to_numpy(float),
                "w": w[keep],
            }

    def _team_loglik(self, name, offsets):
        d = self.data[name]
        eta = d["base"] + offsets[d["team"]]
        ll = d["w"] * (d["y"] * eta - np.logaddexp(0, eta))
        return np.bincount(d["team"], ll, minlength=len(self.teams))

    def initial_state(self, rng):
        n = len(self.teams)
        return {
            "pass_offset": rng.normal(0, 0.1, n), "go_offset": rng.normal(0, 0.1, n),
            "tau": rng.uniform(0.1, 0.5, 2),
            "tuners": {"pass_offset": StepTuner(n, 0.05), "go_offset": StepTuner(n, 0.2), "tau": StepTuner(2, 0.2)},
        }

    def sweep(self, s, rng, iteration, adapt):
        for j, name in enumerate(("pass", "go")):
            key = f"{name}_offset"
            tau = s["tau"][j]
            s[key] = metropolis_update(
                s[key], lambda v, name=name, tau=tau: self._team_loglik(name, v) - 0.5 * (v / tau) ** 2,
                s["tuners"][key], rng, iteration, adapt)

        effects = np.stack([s["pass_offset"], s["go_offset"]])
        n = effects.shape[1]

        def tau_density(log_tau):
            tau = np.exp(log_tau)
            return -n * log_tau - 0.5 * (effects ** 2).sum(axis=1) / tau ** 2 + log_half_normal(tau, TAU_SCALE) + log_tau
        s["tau"] = np.exp(metropolis_update(np.log(s["tau"]), tau_density, s["tuners"]["tau"], rng, iteration, adapt))

    def parameters(self, s):
        return {k: s[k] for k in ("pass_offset", "go_offset", "tau")}

    def summarize(self, draws):
        """Posterior offsets, plus pass rate over expected in percentage points on the team's own plays."""
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in draws.items()}
        d = self.data["pass"]
        base = _sigmoid(d["base"])
        shifted = _sigmoid(d["base"] + flat["pass_offset"].mean(axis=0)[d["team"]])
        n = np.bincount(d["team"], minlength=len(self.teams))
        proe = np.bincount(d["team"], shifted - base, minlength=len(self.teams)) / np.maximum(n, 1)
        return pd.DataFrame({
            "team": self.teams,
            "pass_offset": flat["pass_offset"].mean(axis=0), "pass_offset_sd": flat["pass_offset"].std(axis=0),
            "proe_pct": 100 * proe,
            "go_offset": flat["go_offset"].mean(axis=0), "go_offset_sd": flat["go_offset"].std(axis=0),
        }).sort_values("pass_offset", ascending=False, ignore_index=True)


class TeamPlayCallModel:
    """Drop-in replacement for PlayCallModel in the drive simulator: the league situational call
    mix, tilted by one team's pass and 4th-down-go offsets."""

    def __init__(self, base, pass_offset=0.0, go_offset=0.0):
        self.base = base
        self.pass_offset = pass_offset
        self.go_offset = go_offset
        self.cache = {}

    def _adjusted(self, down, yds_to_go, yardline):
        options, probs = self.base.probabilities(down, yds_to_go, yardline)
        probs = probs.astype(float).copy()
        go = np.isin(options, ["run", "pass"])
        if down < 4:
            is_pass = options == "pass"
            p_pass = _sigmoid(_logit(probs[is_pass].sum()) + self.pass_offset)
            probs = np.where(is_pass, p_pass, 1 - p_pass) if is_pass.any() and (~is_pass).any() else probs
        elif go.any() and (~go).any():
            p_go = probs[go].sum()
            new_go = _sigmoid(_logit(p_go) + self.go_offset)
            probs = np.where(go, probs * new_go / p_go, probs * (1 - new_go) / (1 - p_go))
        return options, np.cumsum(probs / probs.sum())

    def sample(self, down, yds_to_go, yardline, rng):
        key = (down, yds_to_go, yardline)
        if key not in self.cache:
            self.cache[key] = self._adjusted(down, yds_to_go, yardline)
        options, cdf = self.cache[key]
        return options[min(np.searchsorted(cdf, rng.random()), len(options) - 1)]
