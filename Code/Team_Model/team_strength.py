import numpy as np
import pandas as pd

from data import neutral_filter
from mcmc import StepTuner, log_half_normal, metropolis_update

CALLS = ["run", "pass"]
TAU_SCALE = 0.1   # half-normal prior scale on the spread of team effects (EPA/play)
ETA_SCALE = 0.05  # normal prior sd on home field (EPA/play for the home offense)


# Directions that leave every cell mean unchanged, as coefficients on (mu, off, def)
RIDGES = [(1, -1, 0), (1, 0, -1), (0, 1, -1)]


def strength_mask(plays):
    """Run and pass snaps (turnovers included, labeled by call) outside garbage time."""
    return (plays["call"].isin(CALLS) & plays["epa"].notna() & plays["posteam"].notna()
            & neutral_filter(plays)).to_numpy()


class TeamStrengthModel:
    """Hierarchical model of EPA per play:

        epa ~ Normal(mu[call] + eta * home + off[team, call] + def[opponent, call], sigma[call])
        off[., call] ~ Normal(0, tau_off[call]),  def[., call] ~ Normal(0, tau_def[call])
        tau ~ HalfNormal(TAU_SCALE),  eta ~ Normal(0, ETA_SCALE),  mu, log sigma flat

    home is +1 for the home offense, -1 for the away offense, 0 at neutral sites, so the home
    team's per-play edge is 2 * eta. Plays are collapsed to sufficient statistics per
    (offense, defense, call, home) cell, so a sweep costs O(cells), not O(plays). Optional play
    weights (aligned with `plays`, e.g. recency) enter the sufficient statistics as fractional counts.

    Team effects are identified only through their prior (mu + c, off - c fits the data equally
    well), so each sweep also proposes such shifts directly; they leave the likelihood unchanged
    and only the prior decides, which keeps the chain from crawling along that ridge."""

    def __init__(self, plays, weights=None):
        w = np.ones(len(plays)) if weights is None else np.asarray(weights, dtype=float)
        keep = strength_mask(plays) & (w > 0)
        plays, w = plays[keep], w[keep]

        self.teams = np.array(sorted(set(plays["posteam"]) | set(plays["defteam"])))
        team_index = {t: i for i, t in enumerate(self.teams)}
        home = np.where(plays["neutral"], 0, np.where(plays["posteam_is_home"] == 1, 1, -1))
        cells = pd.DataFrame({
            "off": plays["posteam"].map(team_index).to_numpy(),
            "def": plays["defteam"].map(team_index).to_numpy(),
            "call": plays["call"].map({c: i for i, c in enumerate(CALLS)}).to_numpy(),
            "home": home,
            "s0": w, "s1": w * plays["epa"].to_numpy(), "s2": w * plays["epa"].to_numpy() ** 2,
        }).groupby(["off", "def", "call", "home"], as_index=False).sum()
        self.off_idx, self.def_idx = cells["off"].to_numpy(), cells["def"].to_numpy()
        self.call_idx, self.home = cells["call"].to_numpy(), cells["home"].to_numpy()
        self.s0, self.s1, self.s2 = cells["s0"].to_numpy(), cells["s1"].to_numpy(), cells["s2"].to_numpy()
        self.n_teams = len(self.teams)
        self.team_call_off = self.off_idx * 2 + self.call_idx
        self.team_call_def = self.def_idx * 2 + self.call_idx
        self.plays_per_team_call = np.bincount(self.team_call_off, self.s0, minlength=2 * self.n_teams)

    # --- likelihood pieces -------------------------------------------------------------------
    def _means(self, s):
        return (s["mu"][self.call_idx] + s["eta"][0] * self.home
                + s["off"][self.off_idx, self.call_idx] + s["def"][self.def_idx, self.call_idx])

    def _cell_loglik(self, means, sigma):
        """Log likelihood of each cell given its mean (dropping terms constant in the mean)."""
        return (self.s1 * means - 0.5 * self.s0 * means ** 2) / sigma[self.call_idx] ** 2

    def _by_team_call(self, values, index):
        return np.bincount(index, values, minlength=2 * self.n_teams).reshape(self.n_teams, 2)

    # --- sampler interface -------------------------------------------------------------------
    def initial_state(self, rng):
        s0 = np.bincount(self.call_idx, self.s0, minlength=2)
        mean = np.bincount(self.call_idx, self.s1, minlength=2) / s0
        sd = np.sqrt(np.bincount(self.call_idx, self.s2, minlength=2) / s0 - mean ** 2)
        shape = (self.n_teams, 2)
        tuners = {name: StepTuner(size, initial) for name, size, initial in [
            ("mu", 2, 0.01), ("eta", 1, 0.01), ("off", shape, 0.03), ("def", shape, 0.03),
            ("ridge0", 2, 0.02), ("ridge1", 2, 0.02), ("ridge2", 2, 0.02),
            ("tau", (2, 2), 0.2), ("log_sigma", 2, 0.005)]}
        return {
            "mu": mean + rng.normal(0, 0.02, 2), "eta": rng.normal(0, 0.01, 1),
            "off": rng.normal(0, 0.05, shape), "def": rng.normal(0, 0.05, shape),
            "tau": rng.uniform(0.03, 0.15, (2, 2)), "sigma": sd, "tuners": tuners,
        }

    def sweep(self, s, rng, iteration, adapt):
        t = s["tuners"]
        means = self._means(s)

        # Team effects: given everything else, each (team, call) depends only on its own cells
        for side, index in (("off", self.team_call_off), ("def", self.team_call_def)):
            tau = s["tau"][0 if side == "off" else 1]
            current = s[side]

            def log_density(values, side=side, index=index, tau=tau, current=current):
                shifted = means + (values - current)[self.off_idx if side == "off" else self.def_idx, self.call_idx]
                return self._by_team_call(self._cell_loglik(shifted, s["sigma"]), index) - 0.5 * (values / tau) ** 2

            s[side] = metropolis_update(current, log_density, t[side], rng, iteration, adapt)
            means = self._means(s)

        # League means per call (flat prior) and home field
        def mu_density(values):
            shifted = means + (values - s["mu"])[self.call_idx]
            return np.bincount(self.call_idx, self._cell_loglik(shifted, s["sigma"]), minlength=2)
        s["mu"] = metropolis_update(s["mu"], mu_density, t["mu"], rng, iteration, adapt)
        means = self._means(s)

        def eta_density(values):
            shifted = means + (values[0] - s["eta"][0]) * self.home
            return np.array([self._cell_loglik(shifted, s["sigma"]).sum() - 0.5 * (values[0] / ETA_SCALE) ** 2])
        s["eta"] = metropolis_update(s["eta"], eta_density, t["eta"], rng, iteration, adapt)

        # Likelihood-neutral moves along the identifiability ridges (one per call), decided by the prior
        for r, (a_mu, a_off, a_def) in enumerate(RIDGES):
            def ridge_density(c, a_off=a_off, a_def=a_def):
                off, de = s["off"] + a_off * c, s["def"] + a_def * c
                return -0.5 * ((off / s["tau"][0]) ** 2).sum(axis=0) - 0.5 * ((de / s["tau"][1]) ** 2).sum(axis=0)

            c = metropolis_update(np.zeros(2), ridge_density, t[f"ridge{r}"], rng, iteration, adapt)
            s["mu"] = s["mu"] + a_mu * c
            s["off"] = s["off"] + a_off * c
            s["def"] = s["def"] + a_def * c

        # Spread of team effects, sampled on the log scale (Jacobian: + log tau)
        effects = np.stack([s["off"], s["def"]])  # (side, team, call)

        def tau_density(log_tau):
            tau = np.exp(log_tau)
            n = effects.shape[1]
            return (-n * log_tau - 0.5 * (effects ** 2).sum(axis=1) / tau ** 2
                    + log_half_normal(tau, TAU_SCALE) + log_tau)
        s["tau"] = np.exp(metropolis_update(np.log(s["tau"]), tau_density, t["tau"], rng, iteration, adapt))

        # Per-play noise per call (flat prior on log sigma)
        means = self._means(s)
        sse = np.bincount(self.call_idx, self.s2 - 2 * means * self.s1 + means ** 2 * self.s0, minlength=2)
        n = np.bincount(self.call_idx, self.s0, minlength=2)

        def sigma_density(log_sigma):
            return -n * log_sigma - 0.5 * sse / np.exp(2 * log_sigma)
        s["sigma"] = np.exp(metropolis_update(np.log(s["sigma"]), sigma_density, t["log_sigma"], rng, iteration, adapt))

    def parameters(self, s):
        return {k: s[k] for k in ("mu", "eta", "off", "def", "tau", "sigma")}


def summarize(model, draws, pass_rate=0.58):
    """Posterior mean and sd per team. net = offense minus defense at a typical call mix, in EPA/play."""
    mix = np.array([1 - pass_rate, pass_rate])
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in draws.items()}
    net = (flat["off"] - flat["def"]) @ mix
    table = pd.DataFrame({"team": model.teams})
    for side in ("off", "def"):
        for j, call in enumerate(CALLS):
            table[f"{side}_{call}"] = flat[side][:, :, j].mean(axis=0)
            table[f"{side}_{call}_sd"] = flat[side][:, :, j].std(axis=0)
    table["net"] = net.mean(axis=0)
    table["net_sd"] = net.std(axis=0)
    return table.sort_values("net", ascending=False, ignore_index=True)
