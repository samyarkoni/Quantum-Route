"""Hand-rolled Metropolis-within-Gibbs.

Every model here is built from groups of parameters that are conditionally independent given the
rest (e.g. given the defenses, each team's offense effect depends only on that team's own plays).
A group is updated with one random-walk proposal per component and each component is accepted or
rejected on its own log-density ratio, so it is exactly component-wise Metropolis, done for a
whole group in one vectorized step."""

import numpy as np


class StepTuner:
    """Per-component random-walk step sizes, adapted during burn-in (Robbins-Monro on log step)
    toward the ~0.44 acceptance rate that is efficient for one-dimensional updates."""

    def __init__(self, shape, initial=0.05, target=0.44):
        self.step = np.full(shape, initial, dtype=float)
        self.target = target
        self.accepted = np.zeros(shape)
        self.proposed = 0

    def adapt(self, accepted, iteration):
        self.step *= np.exp((accepted - self.target) * min(0.5, 5 / np.sqrt(iteration + 1)))

    def record(self, accepted):
        self.accepted += accepted
        self.proposed += 1

    @property
    def acceptance_rate(self):
        return self.accepted / max(self.proposed, 1)


def metropolis_update(values, log_density, tuner, rng, iteration, adapt):
    """One Metropolis step for each component of `values`. log_density(values) must return one log
    density per component (the component's full conditional, up to a constant)."""
    proposal = values + tuner.step * rng.standard_normal(values.shape)
    log_ratio = log_density(proposal) - log_density(values)
    accepted = np.log(rng.random(values.shape)) < log_ratio
    if adapt:
        tuner.adapt(accepted, iteration)
    else:
        tuner.record(accepted)
    return np.where(accepted, proposal, values)


def scale_update(tau, effects, log_likelihood, log_prior_tau, tuner, rng, iteration, adapt):
    """Joint move of a hierarchical scale and its effects: tau and every effect in its group are
    multiplied by the same exp(eps). The Jacobian exp(eps * (K + 1)) cancels the change in the
    effects' own Normal(0, tau) prior, leaving the likelihood ratio, the prior on tau and exp(eps).
    This moves along the funnel where one-at-a-time updates stall (small tau pins the effects
    near zero, and effects near zero pin tau small). log_likelihood takes the rescaled effects."""
    eps = tuner.step[0] * rng.standard_normal()
    factor = np.exp(eps)
    log_ratio = (log_likelihood(effects * factor) - log_likelihood(effects)
                 + log_prior_tau(tau * factor) - log_prior_tau(tau) + eps)
    accepted = np.array([np.log(rng.random()) < log_ratio])
    if adapt:
        tuner.adapt(accepted, iteration)
    else:
        tuner.record(accepted)
    return (tau * factor, effects * factor) if accepted[0] else (tau, effects)


def log_half_normal(x, scale):
    return np.where(x > 0, -0.5 * (x / scale) ** 2, -np.inf)


def run_chains(model, n_chains=4, n_burn=1000, n_draws=2000, thin=1, seed=0):
    """Runs independent chains from dispersed starts. model provides initial_state(rng),
    sweep(state, rng, iteration, adapt) and parameters(state) -> {name: array}.
    Returns ({name: array of shape (chains, draws, ...)}, {group: post-burn-in acceptance rate})."""
    rng = np.random.default_rng(seed)
    draws, acceptance = [], {}
    for _ in range(n_chains):
        state = model.initial_state(rng)
        chain = []
        for iteration in range(n_burn + n_draws):
            model.sweep(state, rng, iteration, adapt=iteration < n_burn)
            if iteration >= n_burn and (iteration - n_burn) % thin == 0:
                chain.append({k: np.copy(v) for k, v in model.parameters(state).items()})
        draws.append({k: np.stack([d[k] for d in chain]) for k in chain[0]})
        for name, tuner in state["tuners"].items():
            acceptance.setdefault(name, []).append(tuner.acceptance_rate.mean())
    stacked = {k: np.stack([d[k] for d in draws]) for k in draws[0]}
    return stacked, {name: float(np.mean(rates)) for name, rates in acceptance.items()}


def split_rhat(chains):
    """Split R-hat (Gelman-Rubin on half-chains) per component. chains: (chains, draws, ...)."""
    n = chains.shape[1] // 2
    halves = np.concatenate([chains[:, :n], chains[:, n:2 * n]], axis=0)
    within = halves.var(axis=1, ddof=1).mean(axis=0)
    between = n * halves.mean(axis=1).var(axis=0, ddof=1)
    var_plus = (n - 1) / n * within + between / n
    return np.sqrt(var_plus / np.where(within > 0, within, np.nan))


def effective_sample_size(chains):
    """ESS per component from the chain-averaged autocorrelation, summed over lags until the
    sum of a consecutive pair turns negative (Geyer's initial positive sequence)."""
    m, n = chains.shape[:2]
    flat = chains.reshape(m, n, -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=2 * n, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n] / n
    rho = acov.mean(axis=0) / np.where(acov.mean(axis=0)[0] > 0, acov.mean(axis=0)[0], np.nan)
    ess = np.empty(rho.shape[1])
    for j in range(rho.shape[1]):
        pairs = rho[:-1:2, j] + rho[1::2, j]
        stop = np.argmax(pairs < 0) if (pairs < 0).any() else len(pairs)
        ess[j] = m * n / max(-1 + 2 * pairs[:stop].sum(), 1e-9)
    return ess.reshape(chains.shape[2:])


def diagnostics(draws):
    """Worst R-hat and smallest ESS for each parameter group."""
    rows = []
    for name, chains in draws.items():
        rhat = split_rhat(chains)
        ess = effective_sample_size(chains)
        rows.append({"parameter": name, "size": int(np.prod(chains.shape[2:])),
                     "max_rhat": float(np.nanmax(rhat)), "min_ess": float(np.nanmin(ess))})
    return rows
