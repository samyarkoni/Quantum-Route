"""Full-season team profiles: strength (EPA effects) and coaching tendencies, with MCMC diagnostics.

    python run_team_model.py            # latest season
    python run_team_model.py 2023
"""
import sys

import pandas as pd

from data import OUTPUT_DIR, SEASONS, load_epa_plays
from mcmc import diagnostics, run_chains
from play_calling import ExpectedCallModel, TeamTendencyModel
from team_strength import TeamStrengthModel, summarize


def main(season):
    plays = load_epa_plays()
    this_season = plays[plays["season"] == season].reset_index(drop=True)
    pd.set_option("display.width", 200)

    strength = TeamStrengthModel(this_season)
    strength_draws, acceptance = run_chains(strength, n_chains=4, n_burn=1500, n_draws=3000)
    expected = ExpectedCallModel().fit(plays[plays["season"] != season])
    tendency = TeamTendencyModel(this_season, expected)
    tendency_draws, tendency_acceptance = run_chains(tendency, n_chains=4, n_burn=1000, n_draws=2000, seed=1)

    print(f"{season} MCMC diagnostics (target: R-hat < 1.05, ESS in the hundreds, acceptance ~0.44)")
    print(pd.DataFrame(diagnostics(strength_draws) + diagnostics(tendency_draws)).round(3).to_string(index=False))
    print("acceptance:", {k: round(v, 2) for k, v in {**acceptance, **tendency_acceptance}.items()})

    eta = strength_draws["eta"].ravel()
    print(f"\nHome field: {2 * eta.mean():+.3f} EPA/play for the home team (90% interval "
          f"{2 * pd.Series(eta).quantile(0.05):+.3f} to {2 * pd.Series(eta).quantile(0.95):+.3f})")
    tau = strength_draws["tau"].reshape(-1, 2, 2).mean(axis=0)
    print(f"Team spread (tau, EPA/play): offense run {tau[0, 0]:.3f} pass {tau[0, 1]:.3f}, "
          f"defense run {tau[1, 0]:.3f} pass {tau[1, 1]:.3f}")

    table = summarize(strength, strength_draws).merge(tendency.summarize(tendency_draws), on="team")
    table.round(4).to_csv(OUTPUT_DIR / f"team_profiles_{season}.csv", index=False)
    cols = ["team", "net", "net_sd", "off_pass", "off_run", "def_pass", "def_run", "proe_pct", "go_offset"]
    print(f"\n{season} teams by net EPA/play (offense minus defense), with coaching tendencies:")
    print(table[cols].round(3).to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR / f'team_profiles_{season}.csv'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else max(SEASONS))
