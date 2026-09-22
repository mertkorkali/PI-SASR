"""Unit-commitment model: data loading, model construction, solution and output.

The same model builder serves the deterministic (expected-value) UC, the full
(extensive-form) two-stage stochastic UC over S scenarios and the reduced UC of
PI-SASR over the active set. First-stage decisions are the commitment, startup
and shutdown binaries; the second stage is an economic dispatch for each
scenario with unserved energy priced at the value of lost load and spilled
surplus priced at a small penalty. ``docs/MODEL.md`` states the formulation as
implemented here.

Module-level settings
---------------------
``SCENARIO_FOLDER``, ``GENERATOR_FILE`` and ``OUTPUT_ROOT`` select the test system
and the output folder. They default to TX-123BT and are set for a given system by
:func:`pisasr.systems.activate`, which every script calls first.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .constants import (
    GUROBI_OPTIONS,
    HOUR_COLS,
    SCENARIO_SUBSET_SEED,
    SOLVER_NAME,
    SPILL_PENALTY,
    VOLL,
)
from .paths import DATA_DIR, OUTPUT_DIR

# ============================================================
# Settings (set for a given system by pisasr.systems.activate)
# ============================================================

SCENARIO_FOLDER = DATA_DIR / "tx123bt"
"""Folder that holds ``net_load_scenarios_train.csv`` of the active system."""

GENERATOR_FILE = SCENARIO_FOLDER / "generator_portfolio.csv"
"""Generator portfolio of the active system."""

OUTPUT_ROOT = OUTPUT_DIR / "tx123bt"
"""Output folder of the active system; runs are written to ``OUTPUT_ROOT/runs``."""

RANDOM_SEED = SCENARIO_SUBSET_SEED
"""Seed of the permutation that defines the nested scenario subsets."""


# ============================================================
# Data loading
# ============================================================

def read_scenario_file(path: Path) -> pd.DataFrame:
    """Read a scenario file and return its ``hour_00`` ... ``hour_23`` columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing scenario file: {path}")

    df = pd.read_csv(path, index_col=0)
    expected_cols = list(HOUR_COLS)

    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    return df[expected_cols].copy()


def load_net_load_scenarios() -> pd.DataFrame:
    """Return the training set of the active system (2,500 scenarios).

    The evaluation set is kept separate and is read only by
    :mod:`pisasr.evaluation`.
    """
    return read_scenario_file(SCENARIO_FOLDER / "net_load_scenarios_train.csv")


def load_generators() -> pd.DataFrame:
    """Return the generator portfolio of the active system (file row order)."""
    if not Path(GENERATOR_FILE).exists():
        raise FileNotFoundError(f"Missing generator file: {GENERATOR_FILE}")

    gen = pd.read_csv(GENERATOR_FILE)

    required_cols = {
        "unit",
        "p_min_mw",
        "p_max_mw",
        "variable_cost",
        "startup_cost",
        "shutdown_cost",
        "no_load_cost",
        "ramp_up_mw",
        "ramp_down_mw",
        "min_up_hours",
        "min_down_hours",
        "initial_status",
    }

    missing = required_cols - set(gen.columns)
    if missing:
        raise ValueError(f"Generator file is missing columns: {sorted(missing)}")

    gen["unit"] = gen["unit"].astype(str)
    return gen


def select_stochastic_scenarios(
    net_load_df: pd.DataFrame,
    n_scenarios: int,
) -> pd.DataFrame:
    """Return the first ``n_scenarios`` rows of a fixed random permutation.

    One permutation of the training set (random seed 42) is used for every S,
    so the subsets are nested: the 100 scenarios are contained in the 200,
    which are contained in the 500, and so on. The subsets are random rather
    than peak-biased, so the full model at each S is a sample-average
    approximation of the expected cost.
    """
    if n_scenarios > len(net_load_df):
        raise ValueError(f"Requested {n_scenarios}, available {len(net_load_df)}")

    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(net_load_df))
    selected = net_load_df.index[order[:n_scenarios]]

    return net_load_df.loc[selected].copy()


def expected_net_load(net_load_df: pd.DataFrame) -> np.ndarray:
    """Mean 24-hour net-load profile of the training set (deterministic UC input)."""
    return net_load_df[HOUR_COLS].mean(axis=0).to_numpy(dtype=float)


# ============================================================
# Model builder
# ============================================================

def build_uc_master_model(
    gen: pd.DataFrame,
    demand_scenarios: dict[str, np.ndarray],
    probabilities: dict[str, float] | None,
    model_type: str,
    availability: dict[tuple[str, str], float] | None = None,
) -> pyo.ConcreteModel:
    """Build the deterministic or two-stage stochastic UC model.

    Parameters
    ----------
    gen
        Generator portfolio; its row order fixes the order of the unit set.
    demand_scenarios
        ``{scenario name: 24-hour net-load profile}``.
    probabilities
        ``{scenario name: probability}``; uniform when ``None``.
    model_type
        ``"deterministic"`` (one scenario, the expected net load) or
        ``"stochastic"`` (expected recourse cost over all scenarios).
    availability
        Optional ``{(scenario name, unit): factor}``: 0.0 for an outaged unit,
        1.0 for an available unit, and a fraction for an expected derate.
        Missing pairs default to 1.0.
    """
    if model_type not in {"deterministic", "stochastic"}:
        raise ValueError(f"Invalid model_type: {model_type}")

    m = pyo.ConcreteModel(name=f"{model_type}_UC")

    units = list(gen["unit"])
    hours = list(range(24))
    scenario_names = list(demand_scenarios.keys())

    gen_by_unit = gen.set_index("unit")

    m.G = pyo.Set(initialize=units)
    m.T = pyo.Set(initialize=hours, ordered=True)
    m.S = pyo.Set(initialize=scenario_names)

    pmin = gen_by_unit["p_min_mw"].to_dict()
    pmax = gen_by_unit["p_max_mw"].to_dict()
    variable_cost = gen_by_unit["variable_cost"].to_dict()
    startup_cost = gen_by_unit["startup_cost"].to_dict()
    shutdown_cost = gen_by_unit["shutdown_cost"].to_dict()
    no_load_cost = gen_by_unit["no_load_cost"].to_dict()
    ramp_up = gen_by_unit["ramp_up_mw"].to_dict()
    ramp_down = gen_by_unit["ramp_down_mw"].to_dict()
    min_up = gen_by_unit["min_up_hours"].astype(int).to_dict()
    min_down = gen_by_unit["min_down_hours"].astype(int).to_dict()
    initial_status = gen_by_unit["initial_status"].astype(int).to_dict()

    demand = {
        (s, t): float(demand_scenarios[s][t])
        for s in scenario_names
        for t in hours
    }

    if probabilities is None:
        probabilities = {s: 1.0 / len(scenario_names) for s in scenario_names}

    if availability is None:
        availability = {}

    def avail(s, g):
        return availability.get((s, g), 1.0)

    # First stage: commitment, startup and shutdown.
    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)
    m.startup = pyo.Var(m.G, m.T, within=pyo.Binary)
    m.shutdown = pyo.Var(m.G, m.T, within=pyo.Binary)

    # Second stage: dispatch, unserved energy and spilled surplus for each
    # scenario. Spill is needed when the committed minimum outputs exceed a
    # low net-load hour.
    m.p = pyo.Var(m.G, m.T, m.S, within=pyo.NonNegativeReals)
    m.ens = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)
    m.spill = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)

    # -------------------------
    # Commitment logic
    # -------------------------

    def commitment_logic_rule(m, g, t):
        if t == 0:
            return (
                m.u[g, t] - initial_status[g]
                == m.startup[g, t] - m.shutdown[g, t]
            )
        return (
            m.u[g, t] - m.u[g, t - 1]
            == m.startup[g, t] - m.shutdown[g, t]
        )

    m.commitment_logic = pyo.Constraint(m.G, m.T, rule=commitment_logic_rule)

    # -------------------------
    # Generation limits
    # -------------------------

    def gen_min_rule(m, g, t, s):
        return m.p[g, t, s] >= pmin[g] * m.u[g, t] * avail(s, g)

    def gen_max_rule(m, g, t, s):
        return m.p[g, t, s] <= pmax[g] * m.u[g, t] * avail(s, g)

    m.gen_min = pyo.Constraint(m.G, m.T, m.S, rule=gen_min_rule)
    m.gen_max = pyo.Constraint(m.G, m.T, m.S, rule=gen_max_rule)

    # -------------------------
    # Power balance on net load
    # -------------------------

    def balance_rule(m, t, s):
        return (
            sum(m.p[g, t, s] for g in m.G) + m.ens[t, s] - m.spill[t, s]
            == demand[s, t]
        )

    m.balance = pyo.Constraint(m.T, m.S, rule=balance_rule)

    def ens_limit_rule(m, t, s):
        return m.ens[t, s] <= max(demand[s, t], 0.0)

    m.ens_limit = pyo.Constraint(m.T, m.S, rule=ens_limit_rule)

    # -------------------------
    # Ramping (with startup and shutdown allowances)
    # -------------------------

    def ramp_up_rule(m, g, t, s):
        # An outaged unit drops to zero at once; its ramp limits do not apply.
        if avail(s, g) == 0.0:
            return pyo.Constraint.Skip
        if t == 0:
            previous_p = pmin[g] * initial_status[g]
            return (
                m.p[g, t, s] - previous_p
                <= ramp_up[g] + pmax[g] * m.startup[g, t]
            )

        return (
            m.p[g, t, s] - m.p[g, t - 1, s]
            <= ramp_up[g] + pmax[g] * m.startup[g, t]
        )

    def ramp_down_rule(m, g, t, s):
        if avail(s, g) == 0.0:
            return pyo.Constraint.Skip
        if t == 0:
            previous_p = pmin[g] * initial_status[g]
            return (
                previous_p - m.p[g, t, s]
                <= ramp_down[g] + pmax[g] * m.shutdown[g, t]
            )

        return (
            m.p[g, t - 1, s] - m.p[g, t, s]
            <= ramp_down[g] + pmax[g] * m.shutdown[g, t]
        )

    m.ramp_up = pyo.Constraint(m.G, m.T, m.S, rule=ramp_up_rule)
    m.ramp_down = pyo.Constraint(m.G, m.T, m.S, rule=ramp_down_rule)

    # -------------------------
    # Minimum up and down times (truncated at the end of the horizon)
    # -------------------------

    def min_up_rule(m, g, t):
        if min_up[g] <= 1:
            return pyo.Constraint.Skip

        end = min(t + min_up[g], 24)
        return sum(m.u[g, tau] for tau in range(t, end)) >= (end - t) * m.startup[g, t]

    def min_down_rule(m, g, t):
        if min_down[g] <= 1:
            return pyo.Constraint.Skip

        end = min(t + min_down[g], 24)
        return sum(1 - m.u[g, tau] for tau in range(t, end)) >= (end - t) * m.shutdown[g, t]

    m.min_up = pyo.Constraint(m.G, m.T, rule=min_up_rule)
    m.min_down = pyo.Constraint(m.G, m.T, rule=min_down_rule)

    # -------------------------
    # Objective: first-stage cost plus expected recourse cost
    # -------------------------

    fixed_cost = sum(
        no_load_cost[g] * m.u[g, t]
        + startup_cost[g] * m.startup[g, t]
        + shutdown_cost[g] * m.shutdown[g, t]
        for g in m.G
        for t in m.T
    )

    def recourse_cost(m, s):
        return sum(
            variable_cost[g] * m.p[g, t, s]
            for g in m.G
            for t in m.T
        ) + sum(
            VOLL * m.ens[t, s] + SPILL_PENALTY * m.spill[t, s]
            for t in m.T
        )

    expected_cost = sum(probabilities[s] * recourse_cost(m, s) for s in m.S)

    m.objective = pyo.Objective(
        expr=fixed_cost + expected_cost,
        sense=pyo.minimize,
    )

    return m


# ============================================================
# Solution and output
# ============================================================

def solve_pyomo_model(model: pyo.ConcreteModel) -> dict:
    """Solve a model with Gurobi and the options in ``GUROBI_OPTIONS``.

    The returned ``solve_time_seconds`` covers the solver call only (Pyomo
    writes the model file and Gurobi solves it); model construction is not
    included.
    """
    solver = pyo.SolverFactory(SOLVER_NAME)

    if SOLVER_NAME == "gurobi":
        for key, value in GUROBI_OPTIONS.items():
            solver.options[key] = value

    start = time.time()
    result = solver.solve(model, tee=True)
    elapsed = time.time() - start

    return {
        "status": str(result.solver.status),
        "termination": str(result.solver.termination_condition),
        "objective": pyo.value(model.objective),
        "solve_time_seconds": elapsed,
    }


def get_commitment_from_model(model: pyo.ConcreteModel) -> pd.DataFrame:
    """Return the commitment ``u`` as a table with one row for each unit."""
    rows = []

    for g in model.G:
        row = {"unit": str(g)}
        for t in model.T:
            row[f"hour_{t:02d}"] = int(round(pyo.value(model.u[g, t])))
        rows.append(row)

    return pd.DataFrame(rows)


def save_model_outputs(
    model: pyo.ConcreteModel,
    output_folder: Path,
    metrics: dict,
) -> None:
    """Write ``commitment.csv``, ``startup.csv``, ``shutdown.csv``,
    ``scenario_operating_summary.csv`` and ``solve_metrics.csv``."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    commitment = get_commitment_from_model(model)
    commitment.to_csv(output_folder / "commitment.csv", index=False)

    startup_rows = []
    shutdown_rows = []

    for g in model.G:
        start_row = {"unit": str(g)}
        shut_row = {"unit": str(g)}

        for t in model.T:
            start_row[f"hour_{t:02d}"] = int(round(pyo.value(model.startup[g, t])))
            shut_row[f"hour_{t:02d}"] = int(round(pyo.value(model.shutdown[g, t])))

        startup_rows.append(start_row)
        shutdown_rows.append(shut_row)

    pd.DataFrame(startup_rows).to_csv(output_folder / "startup.csv", index=False)
    pd.DataFrame(shutdown_rows).to_csv(output_folder / "shutdown.csv", index=False)

    summary_rows = []

    for s in model.S:
        total_dispatch = 0.0
        total_ens = 0.0

        for t in model.T:
            total_dispatch += sum(pyo.value(model.p[g, t, s]) for g in model.G)
            total_ens += pyo.value(model.ens[t, s])

        summary_rows.append(
            {
                "scenario": str(s),
                "total_dispatch_mwh": total_dispatch,
                "total_ens_mwh": total_ens,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        output_folder / "scenario_operating_summary.csv",
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(output_folder / "solve_metrics.csv", index=False)


# ============================================================
# Single-scenario dispatch LP (merit-order validation)
# ============================================================

def evaluate_fixed_commitment_dispatch_cost(
    gen: pd.DataFrame,
    commitment: pd.DataFrame,
    demand: np.ndarray,
    outaged_unit: str | None = None,
) -> float:
    """Minimum dispatch cost of one 24-hour net-load profile under a fixed commitment.

    Used by :func:`pisasr.merit_order.validate_against_lp` to measure the error
    of the merit-order recourse. ``outaged_unit``, if given, is unavailable in
    every hour. Note that the ramp limits of this LP have no startup or
    shutdown allowance, unlike the model above and the exact evaluator in
    :mod:`pisasr.evaluation`. Returns 1e15 if the LP is not solved to
    optimality.
    """
    units = list(gen["unit"])
    hours = list(range(24))
    gen_by_unit = gen.set_index("unit")

    pmin = gen_by_unit["p_min_mw"].to_dict()
    pmax = gen_by_unit["p_max_mw"].to_dict()
    variable_cost = gen_by_unit["variable_cost"].to_dict()
    ramp_up = gen_by_unit["ramp_up_mw"].to_dict()
    ramp_down = gen_by_unit["ramp_down_mw"].to_dict()
    initial_status = gen_by_unit["initial_status"].astype(int).to_dict()

    commitment_map = commitment.set_index("unit").to_dict(orient="index")

    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=units)
    m.T = pyo.Set(initialize=hours, ordered=True)

    m.p = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)
    m.ens = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.spill = pyo.Var(m.T, within=pyo.NonNegativeReals)

    def balance_rule(m, t):
        return (
            sum(m.p[g, t] for g in m.G) + m.ens[t] - m.spill[t]
            == float(demand[t])
        )

    m.balance = pyo.Constraint(m.T, rule=balance_rule)

    def unit_avail(g):
        return 0.0 if (outaged_unit is not None and g == outaged_unit) else 1.0

    def gen_min_rule(m, g, t):
        u = commitment_map[g][f"hour_{t:02d}"]
        return m.p[g, t] >= pmin[g] * u * unit_avail(g)

    def gen_max_rule(m, g, t):
        u = commitment_map[g][f"hour_{t:02d}"]
        return m.p[g, t] <= pmax[g] * u * unit_avail(g)

    m.gen_min = pyo.Constraint(m.G, m.T, rule=gen_min_rule)
    m.gen_max = pyo.Constraint(m.G, m.T, rule=gen_max_rule)

    def ramp_up_rule(m, g, t):
        if unit_avail(g) == 0.0:
            return pyo.Constraint.Skip
        if t == 0:
            previous_p = pmin[g] * initial_status[g]
            return m.p[g, t] - previous_p <= ramp_up[g]
        return m.p[g, t] - m.p[g, t - 1] <= ramp_up[g]

    def ramp_down_rule(m, g, t):
        if unit_avail(g) == 0.0:
            return pyo.Constraint.Skip
        if t == 0:
            previous_p = pmin[g] * initial_status[g]
            return previous_p - m.p[g, t] <= ramp_down[g]
        return m.p[g, t - 1] - m.p[g, t] <= ramp_down[g]

    m.ramp_up = pyo.Constraint(m.G, m.T, rule=ramp_up_rule)
    m.ramp_down = pyo.Constraint(m.G, m.T, rule=ramp_down_rule)

    m.objective = pyo.Objective(
        expr=sum(variable_cost[g] * m.p[g, t] for g in m.G for t in m.T)
        + sum(VOLL * m.ens[t] + SPILL_PENALTY * m.spill[t] for t in m.T),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(SOLVER_NAME)
    result = solver.solve(m, tee=False)

    termination = str(result.solver.termination_condition)
    if termination.lower() not in {"optimal", "feasible"}:
        return 1e15

    return float(pyo.value(m.objective))
