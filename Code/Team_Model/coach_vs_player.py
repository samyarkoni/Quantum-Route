"""How much of an offense is the head coach, the quarterback, and everyone else?

Each team-game outcome is split into crossed random effects:

    y[g] = mu + coach[c] + qb[q] + roster[team-season] + opponent[opp-season] + game noise
    effect ~ Normal(0, tau_group),  game noise ~ Normal(0, sigma_play^2 / n_plays[g] + omega^2)

Coaches and QBs are separable because they move independently: coaches change teams and QBs,
QBs change teams and coaches, and injured starters are replaced by backups mid-season under the
same coach and roster. The roster (team-season) term absorbs everyone else: the other 21 starters,
coordinators and staff, and anything else constant within a team-season. Fit with the same
Metropolis-within-Gibbs sampler as the team models.

Outcomes: offensive EPA/play (efficiency), pass rate over expected (scheme), and 4th-down go rate
over expected (decision-making), both "over expected" given down, distance, field position,
score and clock.
"""
import numpy as np
import pandas as pd

from data import OUTPUT_DIR, load_epa_plays, load_games, neutral_filter
from mcmc import StepTuner, diagnostics, log_half_normal, metropolis_update, run_chains, scale_update
from play_calling import ExpectedCallModel, _sigmoid, fourth_mask, pass_mask
from team_strength import strength_mask

GROUPS = ["coach", "qb", "roster", "opponent"]


def team_game_identities(games):
    """One row per team-game: the head coach, starting QB, team-season and opponent-season."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        sides.append(pd.DataFrame({
            "game_id": games["game_id"], "season": games["season"], "team": games[f"{side}_team"],
            "coach": games[f"{side}_coach"], "qb": games[f"{side}_qb_id"], "qb_name": games[f"{side}_qb_name"],
            "opp": games[f"{other}_team"],
        }))
    ids = pd.concat(sides, ignore_index=True)
    ids["roster"] = ids["team"] + "_" + ids["season"].astype(str)
    ids["opponent"] = ids["opp"] + "_" + ids["season"].astype(str)
    return ids


def outcome_table(plays, values, mask, ids):
    """Per team-game mean of `values` over masked plays, the play count, and the pooled
    within-game per-play variance."""
    rows = plays.loc[mask, ["game_id", "posteam"]].assign(v=np.asarray(values))
    g = rows.groupby(["game_id", "posteam"])["v"]
    table = pd.DataFrame({"y": g.mean(), "n": g.size()}).reset_index().rename(columns={"posteam": "team"})
    play_var = (rows["v"] - g.transform("mean")).pow(2).sum() / (len(rows) - len(table))
    table = table.merge(ids, on=["game_id", "team"]).dropna(subset=["coach", "qb"])
    return table, play_var


class CrossedEffectsModel:
    def __init__(self, table, play_var):
        self.y = table["y"].to_numpy(float)
        self.play_var = play_var / table["n"].to_numpy(float)
        self.levels, self.index = {}, {}
        for group in GROUPS:
            codes, levels = pd.factorize(table[group])
            self.index[group], self.levels[group] = codes, levels
        self.scale = self.y.std()

    def _residual(self, s):
        fitted = s["mu"][0] + sum(s[g][self.index[g]] for g in GROUPS)
        return self.y - fitted

    def _game_loglik(self, residual, omega):
        v = self.play_var + omega ** 2
        return -0.5 * (residual ** 2 / v + np.log(v))

    def initial_state(self, rng):
        state = {g: rng.normal(0, 0.1 * self.scale, len(self.levels[g])) for g in GROUPS}
        state.update({
            "mu": np.array([self.y.mean()]), "tau": rng.uniform(0.1, 0.5, len(GROUPS)) * self.scale,
            "omega": np.array([0.5 * self.scale]),
            "tuners": {name: StepTuner(size, 0.1 * self.scale) for name, size in
                       [(g, len(self.levels[g])) for g in GROUPS] + [("mu", 1), ("omega", 1)]}
                      | {"tau": StepTuner(len(GROUPS), 0.2)}
                      | {f"ridge_{g}": StepTuner(1, 0.05 * self.scale) for g in GROUPS}
                      | {f"scale_{g}": StepTuner(1, 0.2) for g in GROUPS},
        })
        return state

    def sweep(self, s, rng, iteration, adapt):
        t = s["tuners"]
        for j, group in enumerate(GROUPS):
            residual = self._residual(s)
            idx, current, tau = self.index[group], s[group], s["tau"][j]

            def density(values, idx=idx, current=current, residual=residual, tau=tau):
                shifted = residual - (values - current)[idx]
                ll = np.bincount(idx, self._game_loglik(shifted, s["omega"][0]), minlength=len(values))
                return ll - 0.5 * (values / tau) ** 2
            s[group] = metropolis_update(current, density, t[group], rng, iteration, adapt)

        residual = self._residual(s)
        s["mu"] = metropolis_update(
            s["mu"], lambda v: np.array([self._game_loglik(residual - (v[0] - s["mu"][0]), s["omega"][0]).sum()]),
            t["mu"], rng, iteration, adapt)

        # Prior-only moves along the (mu + c, group - c) ridges
        for j, group in enumerate(GROUPS):
            c = metropolis_update(
                np.zeros(1), lambda c, g=group, j=j: np.array([-0.5 * (((s[g] - c[0]) / s["tau"][j]) ** 2).sum()]),
                t[f"ridge_{group}"], rng, iteration, adapt)
            s["mu"] = s["mu"] + c
            s[group] = s[group] - c[0]

        def tau_density(log_tau):
            tau = np.exp(log_tau)
            sums = np.array([(s[g] ** 2).sum() for g in GROUPS])
            sizes = np.array([len(s[g]) for g in GROUPS])
            return -sizes * log_tau - 0.5 * sums / tau ** 2 + log_half_normal(tau, self.scale) + log_tau
        s["tau"] = np.exp(metropolis_update(np.log(s["tau"]), tau_density, t["tau"], rng, iteration, adapt))

        # Joint (tau, effects) rescaling per group, against the funnel
        for j, group in enumerate(GROUPS):
            base = self._residual(s) + s[group][self.index[group]]

            def log_likelihood(effects, base=base, group=group):
                return self._game_loglik(base - effects[self.index[group]], s["omega"][0]).sum()
            s["tau"][j], s[group] = scale_update(
                s["tau"][j], s[group], log_likelihood, lambda tau: float(log_half_normal(tau, self.scale)),
                t[f"scale_{group}"], rng, iteration, adapt)

        residual = self._residual(s)
        s["omega"] = np.exp(metropolis_update(
            np.log(s["omega"]),
            lambda lo: np.array([self._game_loglik(residual, np.exp(lo[0])).sum() + log_half_normal(np.exp(lo[0]), self.scale) + lo[0]]),
            t["omega"], rng, iteration, adapt))

    def parameters(self, s):
        return {k: s[k] for k in GROUPS + ["mu", "tau", "omega"]}


def variance_shares(draws):
    """Posterior variance components. Offense share = coach vs QB vs roster, of the three."""
    tau = draws["tau"].reshape(-1, len(GROUPS))
    var = pd.DataFrame(tau ** 2, columns=GROUPS)
    var["game"] = draws["omega"].ravel() ** 2
    offense = var[["coach", "qb", "roster"]].sum(axis=1)
    rows = []
    for g in GROUPS + ["game"]:
        sd = np.sqrt(var[g])
        rows.append({"component": g, "sd": sd.mean(), "sd_lo": sd.quantile(0.05), "sd_hi": sd.quantile(0.95)})
        if g in ("coach", "qb", "roster"):
            share = var[g] / offense
            rows[-1].update(offense_share=share.mean(), share_lo=share.quantile(0.05), share_hi=share.quantile(0.95))
    return pd.DataFrame(rows)


def top_effects(model, draws, group, names=None, k=6):
    flat = draws[group].reshape(-1, draws[group].shape[-1])
    table = pd.DataFrame({"level": model.levels[group], "effect": flat.mean(axis=0), "sd": flat.std(axis=0)})
    if names is not None:
        table["level"] = table["level"].map(names).fillna(table["level"])
    table = table.sort_values("effect", ascending=False)
    return pd.concat([table.head(k), table.tail(k)])


def persistence(table):
    """Year-over-year correlation of team-season averages, split by whether the head coach and the
    primary QB (most starts) stayed the same."""
    season = table.groupby(["team", "season"]).apply(lambda d: pd.Series({
        "y": np.average(d["y"], weights=d["n"]),
        "coach": d["coach"].mode().iloc[0], "qb": d["qb"].mode().iloc[0],
    })).reset_index()
    nxt = season.assign(season=season["season"] - 1)
    pairs = season.merge(nxt, on=["team", "season"], suffixes=("", "_next"))
    pairs["case"] = np.select(
        [(pairs["coach"] == pairs["coach_next"]) & (pairs["qb"] == pairs["qb_next"]),
         pairs["coach"] == pairs["coach_next"], pairs["qb"] == pairs["qb_next"]],
        ["same coach, same QB", "same coach, new QB", "new coach, same QB"], default="new coach, new QB")
    return pairs.groupby("case").apply(lambda d: pd.Series({"pairs": len(d), "corr": d["y"].corr(d["y_next"])}))


def scramble_check(regular, ids, seasons=(2023, 2024, 2025)):
    """Scrambles are recorded as runs, so a scrambling QB lowers 'pass rate' even when the call was
    a pass. Coverage is charted on dropbacks, and from 2023 it is also charted on the runs that
    were scrambles, so on those seasons pass rate can be redone as dropback rate."""
    recent = regular[regular["season"].isin(seasons)]
    scramble = (recent["play_type"] == "run") & recent["defense_coverage_type"].notna()
    lines = []
    for label, data in (("as labeled", recent),
                        ("scrambles counted as dropbacks", recent.assign(call=recent["call"].mask(scramble, "pass")))):
        mask = pass_mask(data) & neutral_filter(data).to_numpy()
        expected = ExpectedCallModel().fit(data)
        values = 100 * ((data.loc[mask, "call"] == "pass") - _sigmoid(expected.pass_logit(data[mask])))
        table, play_var = outcome_table(data, values, mask, ids)
        draws, _ = run_chains(CrossedEffectsModel(table, play_var), n_chains=4, n_burn=2000, n_draws=10000, thin=4, seed=7)
        shares = variance_shares(draws)
        diag = pd.DataFrame(diagnostics(draws))
        block = [f"--- Scramble check, pass rate over expected {seasons[0]}-{seasons[-1]}, {label} "
                 f"({scramble.sum():,} scrambles; max R-hat {diag['max_rhat'].max():.3f}) ---",
                 shares.round(3).to_string(index=False)]
        print("\n".join(block) + "\n", flush=True)
        lines += block + [""]
    return lines


def main():
    plays = load_epa_plays()
    games = load_games()
    games = games[games["game_type"] == "REG"]
    ids = team_game_identities(games)
    expected = ExpectedCallModel().fit(plays)
    regular = plays[plays["game_id"].isin(games["game_id"])]

    epa_mask = strength_mask(regular)
    pm = pass_mask(regular) & neutral_filter(regular).to_numpy()
    fm = fourth_mask(regular)
    outcomes = {
        "Offensive EPA/play": outcome_table(regular, regular.loc[epa_mask, "epa"], epa_mask, ids),
        "Pass rate over expected (pct pts)": outcome_table(
            regular, 100 * ((regular.loc[pm, "call"] == "pass") - _sigmoid(expected.pass_logit(regular[pm]))), pm, ids),
        "4th-down go rate over expected (pct pts)": outcome_table(
            regular, 100 * (regular.loc[fm, "call"].isin(["run", "pass"]) - _sigmoid(expected.go_logit(regular[fm]))), fm, ids),
    }

    names = ids.drop_duplicates("qb").set_index("qb")["qb_name"]
    lines = []
    pd.set_option("display.width", 200)
    for title, (table, play_var) in outcomes.items():
        model = CrossedEffectsModel(table, play_var)
        draws, _ = run_chains(model, n_chains=4, n_burn=2000, n_draws=10000, thin=4, seed=7)
        diag = pd.DataFrame(diagnostics(draws))
        shares = variance_shares(draws)
        block = [f"=== {title}: {len(table):,} team-games, {len(model.levels['coach'])} head coaches, "
                 f"{len(model.levels['qb'])} starting QBs, {len(model.levels['roster'])} team-seasons ===",
                 shares.round(3).to_string(index=False),
                 f"(max R-hat {diag['max_rhat'].max():.3f}, min ESS {diag['min_ess'].min():.0f})",
                 "Year-over-year persistence of team-season averages:",
                 persistence(table).round(2).to_string(),
                 "Largest coach effects:", top_effects(model, draws, "coach").round(3).to_string(index=False),
                 "Largest QB effects:", top_effects(model, draws, "qb", names).round(3).to_string(index=False)]
        print("\n".join(block) + "\n", flush=True)
        lines += block + [""]

    lines += scramble_check(regular, ids)

    # How the design identifies coaches vs QBs
    reg = ids.dropna(subset=["coach", "qb"])
    qbs_per_coach = reg.groupby("coach")["qb"].nunique()
    coaches_per_qb = reg.groupby("qb")["coach"].nunique()
    teams_per_coach = reg.groupby("coach")["team"].nunique()
    ident = (f"Identification: {(teams_per_coach > 1).sum()} head coaches led 2+ teams; "
             f"{(qbs_per_coach > 1).sum()} of {len(qbs_per_coach)} coaches started 2+ QBs; "
             f"{(coaches_per_qb > 1).sum()} of {len(coaches_per_qb)} QBs started under 2+ head coaches.")
    print(ident)
    lines.append(ident)
    (OUTPUT_DIR / "coach_vs_player_results.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT_DIR / 'coach_vs_player_results.txt'}")


if __name__ == "__main__":
    main()
