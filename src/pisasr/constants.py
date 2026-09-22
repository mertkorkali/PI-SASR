"""Constants shared by the unit-commitment model, the evaluator and PI-SASR.

The values are those used for every result in the paper. Changing them changes
the numbers in ``results/``.
"""

from __future__ import annotations

N_HOURS = 24
"""Length of the scheduling horizon (hours)."""

HOUR_COLS = [f"hour_{h:02d}" for h in range(N_HOURS)]
"""Column names of the 24-hour profiles in every scenario and commitment file."""

VOLL = 10000.0
"""Value of lost load ($/MWh), the price of unserved energy in the recourse."""

SPILL_PENALTY = 1.0
"""Penalty ($/MWh) for spilling surplus output when the committed minimum
outputs exceed the net load of an hour."""

SOLVER_NAME = "gurobi"
"""Pyomo solver used for every MILP and LP."""

GUROBI_OPTIONS = {
    "Method": 2,          # barrier for the root relaxation
    "Crossover": 0,       # passed as in the runs of the paper; Gurobi 13.0.2 ignores it
                          # for a MILP (see docs/MODEL.md, Section 2)
    "MIPGap": 1e-3,       # 0.1% relative MILP tolerance
    "NodefileStart": 8,   # GB of node memory before node files are written
}
"""Gurobi parameters of the MILPs. No thread count, time limit or random
seed is set, so Gurobi uses its defaults (all available cores)."""

SCENARIO_SUBSET_SEED = 42
"""Random seed of the single permutation of the training set that gives nested
scenario subsets of size S."""

SCENARIO_COUNTS = (100, 200, 500, 1000, 2500)
"""Scenario counts S of the scaling study."""

EVALUATION_BATCH_SIZE = 50
"""Number of evaluation scenarios in one LP of the exact evaluation."""

PISASR_DEFAULTS = {
    "k_critical": 90,      # K_c, critical scenarios
    "k_cover": 60,         # K_r, coverage scenarios
    "design_size": 140,    # size of the design set
    "cg_rounds": 1,        # enrichment budget (rounds)
    "cg_add": 40,          # scenarios added in one enrichment round
    "svd_rank": 5,         # rank r of the low-rank embedding
    "random_seed": 7,      # random seed of the design set and of its split
    "cg_tol": 0.002,       # enrichment tolerance epsilon (0.2%)
}
"""PI-SASR settings used on both systems (no system-specific tuning)."""
