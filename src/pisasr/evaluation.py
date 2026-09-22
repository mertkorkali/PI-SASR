"""Exact evaluation of a fixed commitment on the evaluation set.

For a given commitment the recourse of each evaluation scenario is the
ramp-coupled economic-dispatch LP of the unit-commitment model (startup and
shutdown allowances included), under the same N-1 outage draw as in training.
Scenarios are solved in batches of 50 in one LP; the LP separates across
scenarios, so the batch size does not change the cost of any scenario.

The evaluation set is the first ``EVALUATION_SCENARIOS`` (2,000) rows of the
system's ``net_load_scenarios_test.csv``, which is disjoint from the training
set.

Module-level settings
---------------------
``SCENARIO_FILE``, ``GENERATOR_FILE``, ``RESULTS_ROOT`` and
``EVALUATION_SCENARIOS`` default to TX-123BT and are set for a given system by
:func:`pisasr.systems.activate`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .constants import EVALUATION_BATCH_SIZE, HOUR_COLS, SOLVER_NAME, SPILL_PENALTY, VOLL
from .outages import draw_scenario_availability
from .paths import DATA_DIR, OUTPUT_DIR
from .systems import parse_run_name

# ============================================================
# Settings (set for a given system by pisasr.systems.activate)
# ============================================================

SCENARIO_FILE = DATA_DIR / "tx123bt" / "net_load_scenarios_test.csv"
"""Scenario file whose first rows form the evaluation set."""

GENERATOR_FILE = DATA_DIR / "tx123bt" / "generator_portfolio.csv"
"""Generator portfolio of the active system."""

RESULTS_ROOT = OUTPUT_DIR / "tx123bt"
"""Output folder of the active system."""

EVALUATION_SCENARIOS = 2000
"""Size of the evaluation set (first rows of ``SCENARIO_FILE``)."""

BATCH_SIZE = EVALUATION_BATCH_SIZE
"""Number of scenarios in one evaluation LP."""


# ============================================================
# Data loading
# ============================================================

def load_generators() -> pd.DataFrame:
    """Return the generator portfolio of the active system."""
    gen = pd.read_csv(GENERATOR_FILE)
    gen["unit"] = gen["unit"].astype(str)
    return gen


def load_net_load_scenarios() -> pd.DataFrame:
    """Return the evaluation set: the first ``EVALUATION_SCENARIOS`` rows."""
    df = pd.read_csv(SCENARIO_FILE, index_col=0)
    df = df[list(HOUR_COLS)].copy()

    if EVALUATION_SCENARIOS is not None:
        df = df.iloc[:EVALUATION_SCENARIOS].copy()

    return df


# ============================================================
# Run-folder names
# ============================================================

def parse_model_case(folder_name: str) -> dict:
    """Return the model label and scenario count encoded in a run-folder name.

    The labels are those of ``summary/evaluation_metrics.csv``
    (:func:`pisasr.systems.parse_run_name`): ``deterministic`` -> model
    ``deterministic`` with S = 1; ``full_S<n>`` -> ``full`` with S = n;
    ``pisasr_S<n>`` (or ``pisasr_S<n>_<tag>``) -> ``pisasr`` with S = n. Any
    other name is returned as the model with no scenario count.
    """
    model, n_scenarios = parse_run_name(folder_name)
    return {"model": model, "S": np.nan if n_scenarios is None else n_scenarios}


# ============================================================
# First-stage cost of a fixed commitment
# ============================================================

def compute_startup_shutdown_from_commitment(
    commitment: pd.DataFrame,
    gen: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Startup and shutdown indicators implied by a commitment and the initial status."""
    gen_by_unit = gen.set_index("unit")
    commitment = commitment.copy()

    startup_rows = []
    shutdown_rows = []

    for _, row in commitment.iterrows():
        unit = str(row["unit"])
        initial_status = int(gen_by_unit.loc[unit, "initial_status"])

        startup_row = {"unit": unit}
        shutdown_row = {"unit": unit}

        previous = initial_status

        for h in range(24):
            current = int(round(row[f"hour_{h:02d}"]))

            startup_row[f"hour_{h:02d}"] = max(current - previous, 0)
            shutdown_row[f"hour_{h:02d}"] = max(previous - current, 0)

            previous = current

        startup_rows.append(startup_row)
        shutdown_rows.append(shutdown_row)

    return pd.DataFrame(startup_rows), pd.DataFrame(shutdown_rows)


def compute_fixed_commitment_cost(
    commitment: pd.DataFrame,
    gen: pd.DataFrame,
) -> dict:
    """First-stage cost F(u): no-load, startup and shutdown costs of a commitment."""
    gen_by_unit = gen.set_index("unit")
    hour_cols = list(HOUR_COLS)

    startup, shutdown = compute_startup_shutdown_from_commitment(commitment, gen)

    commitment = commitment.set_index("unit")
    startup = startup.set_index("unit")
    shutdown = shutdown.set_index("unit")

    no_load_cost = 0.0
    startup_cost = 0.0
    shutdown_cost = 0.0

    for unit in commitment.index:
        for h in hour_cols:
            no_load_cost += (
                float(gen_by_unit.loc[unit, "no_load_cost"])
                * float(commitment.loc[unit, h])
            )

            startup_cost += (
                float(gen_by_unit.loc[unit, "startup_cost"])
                * float(startup.loc[unit, h])
            )

            shutdown_cost += (
                float(gen_by_unit.loc[unit, "shutdown_cost"])
                * float(shutdown.loc[unit, h])
            )

    return {
        "no_load_cost": no_load_cost,
        "startup_cost": startup_cost,
        "shutdown_cost": shutdown_cost,
        "fixed_commitment_cost": no_load_cost + startup_cost + shutdown_cost,
        "startup_count": float(startup[hour_cols].sum().sum()),
        "committed_unit_hours": float(commitment[hour_cols].sum().sum()),
    }


# ============================================================
# Exact recourse of a fixed commitment
# ============================================================

def evaluate_commitment_batch(
    commitment: pd.DataFrame,
    gen: pd.DataFrame,
    scenario_batch: pd.DataFrame,
) -> pd.DataFrame:
    """Solve the ramp-coupled dispatch LP of a fixed commitment for a batch of scenarios.

    Returns one row for each scenario with the dispatch (recourse) cost, the
    unserved energy, the spilled energy, the number of hours with unserved
    energy and the total net load. Raises ``RuntimeError`` if the LP is not
    solved to optimality.
    """
    units = list(gen["unit"])
    hours = list(range(24))
    scenarios = list(scenario_batch.index)

    gen_by_unit = gen.set_index("unit")
    commitment_map = commitment.set_index("unit").to_dict(orient="index")

    pmin = gen_by_unit["p_min_mw"].to_dict()
    pmax = gen_by_unit["p_max_mw"].to_dict()
    var_cost = gen_by_unit["variable_cost"].to_dict()
    ramp_up = gen_by_unit["ramp_up_mw"].to_dict()
    ramp_down = gen_by_unit["ramp_down_mw"].to_dict()
    initial_status = gen_by_unit["initial_status"].astype(int).to_dict()

    startup, shutdown = compute_startup_shutdown_from_commitment(commitment, gen)
    startup_map = startup.set_index("unit").to_dict(orient="index")
    shutdown_map = shutdown.set_index("unit").to_dict(orient="index")

    # Outage draw of each evaluation scenario, seeded by the scenario
    # identifier only, so every commitment faces identical outages.
    availability = {}
    for s in scenarios:
        scen_avail = draw_scenario_availability(units, s)
        for g in units:
            availability[(g, s)] = scen_avail[g]

    m = pyo.ConcreteModel()

    m.G = pyo.Set(initialize=units)
    m.T = pyo.Set(initialize=hours, ordered=True)
    m.S = pyo.Set(initialize=scenarios)

    m.p = pyo.Var(m.G, m.T, m.S, within=pyo.NonNegativeReals)
    m.ens = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)
    m.spill = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)

    def balance_rule(m, t, s):
        demand = float(scenario_batch.loc[s, f"hour_{t:02d}"])
        return sum(m.p[g, t, s] for g in m.G) + m.ens[t, s] - m.spill[t, s] == demand

    m.balance = pyo.Constraint(m.T, m.S, rule=balance_rule)

    def gen_min_rule(m, g, t, s):
        u = int(round(commitment_map[g][f"hour_{t:02d}"]))
        return m.p[g, t, s] >= pmin[g] * u * availability[(g, s)]

    def gen_max_rule(m, g, t, s):
        u = int(round(commitment_map[g][f"hour_{t:02d}"]))
        return m.p[g, t, s] <= pmax[g] * u * availability[(g, s)]

    m.gen_min = pyo.Constraint(m.G, m.T, m.S, rule=gen_min_rule)
    m.gen_max = pyo.Constraint(m.G, m.T, m.S, rule=gen_max_rule)

    def ramp_up_rule(m, g, t, s):
        # An outaged unit drops to zero at once; its ramp limits do not apply.
        if availability[(g, s)] == 0.0:
            return pyo.Constraint.Skip
        y = int(round(startup_map[g][f"hour_{t:02d}"]))

        if t == 0:
            previous_p = pmin[g] * initial_status[g]
            return m.p[g, t, s] - previous_p <= ramp_up[g] + pmax[g] * y

        return m.p[g, t, s] - m.p[g, t - 1, s] <= ramp_up[g] + pmax[g] * y

    def ramp_down_rule(m, g, t, s):
        if availability[(g, s)] == 0.0:
            return pyo.Constraint.Skip
        z = int(round(shutdown_map[g][f"hour_{t:02d}"]))

        if t == 0:
            previous_p = pmin[g] * initial_status[g]
            return previous_p - m.p[g, t, s] <= ramp_down[g] + pmax[g] * z

        return m.p[g, t - 1, s] - m.p[g, t, s] <= ramp_down[g] + pmax[g] * z

    m.ramp_up = pyo.Constraint(m.G, m.T, m.S, rule=ramp_up_rule)
    m.ramp_down = pyo.Constraint(m.G, m.T, m.S, rule=ramp_down_rule)

    m.objective = pyo.Objective(
        expr=sum(var_cost[g] * m.p[g, t, s] for g in m.G for t in m.T for s in m.S)
        + sum(VOLL * m.ens[t, s] + SPILL_PENALTY * m.spill[t, s] for t in m.T for s in m.S),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(SOLVER_NAME)
    result = solver.solve(m, tee=False)

    termination = str(result.solver.termination_condition)
    if termination.lower() not in {"optimal", "feasible"}:
        raise RuntimeError(f"Batch dispatch failed with termination: {termination}")

    rows = []

    for s in scenarios:
        dispatch_cost = 0.0
        ens_mwh = 0.0
        spill_mwh = 0.0
        ens_hours = 0

        for t in hours:
            for g in units:
                dispatch_cost += var_cost[g] * pyo.value(m.p[g, t, s])

            ens = pyo.value(m.ens[t, s])
            spill = pyo.value(m.spill[t, s])

            dispatch_cost += VOLL * ens + SPILL_PENALTY * spill
            ens_mwh += ens
            spill_mwh += spill

            if ens > 1e-4:
                ens_hours += 1

        total_net_load = scenario_batch.loc[s].sum()

        rows.append(
            {
                "scenario": s,
                "dispatch_cost": dispatch_cost,
                "ens_mwh": ens_mwh,
                "spill_mwh": spill_mwh,
                "ens_hours": ens_hours,
                "total_net_load_mwh": total_net_load,
            }
        )

    return pd.DataFrame(rows)


def evaluate_scenarios(
    commitment: pd.DataFrame,
    gen: pd.DataFrame,
    net_load_scenarios: pd.DataFrame,
    batch_size: int | None = None,
    label: str = "",
) -> pd.DataFrame:
    """Evaluate a commitment on every scenario of ``net_load_scenarios``.

    Returns the table of :func:`evaluate_commitment_batch` (one row for each scenario) with the
    column ``total_cost_with_fixed`` (first-stage cost plus dispatch cost).
    """
    batch_size = BATCH_SIZE if batch_size is None else int(batch_size)
    commitment = commitment.copy()
    commitment["unit"] = commitment["unit"].astype(str)
    fixed = compute_fixed_commitment_cost(commitment, gen)["fixed_commitment_cost"]

    scenario_indices = list(net_load_scenarios.index)
    parts = []
    for start in range(0, len(scenario_indices), batch_size):
        end = min(start + batch_size, len(scenario_indices))
        if label:
            print(f"Evaluating {label}: scenarios {start + 1}-{end}")
        batch_df = net_load_scenarios.loc[scenario_indices[start:end]].copy()
        parts.append(evaluate_commitment_batch(commitment, gen, batch_df))

    scenario_results = pd.concat(parts, ignore_index=True)
    scenario_results["total_cost_with_fixed"] = fixed + scenario_results["dispatch_cost"]
    return scenario_results


def summarize_evaluation(scenario_results: pd.DataFrame) -> dict:
    """Summary statistics of an evaluation table (one row for each scenario).

    ``scenario_results`` must hold the columns written to ``evaluation.csv``
    (``dispatch_cost``, ``total_cost_with_fixed``, ``ens_mwh``, ``ens_hours``,
    ``spill_mwh``, ``total_net_load_mwh``). The unserved-energy rate is the total
    unserved energy divided by the total net load over the evaluation set.
    """
    return {
        "n_evaluation_scenarios": len(scenario_results),
        "expected_dispatch_cost": scenario_results["dispatch_cost"].mean(),
        "expected_total_cost": scenario_results["total_cost_with_fixed"].mean(),
        "p95_total_cost": scenario_results["total_cost_with_fixed"].quantile(0.95),
        "worst_total_cost": scenario_results["total_cost_with_fixed"].max(),
        "expected_ens_mwh": scenario_results["ens_mwh"].mean(),
        "total_ens_mwh": scenario_results["ens_mwh"].sum(),
        "ens_probability": (scenario_results["ens_mwh"] > 1e-4).mean(),
        "expected_ens_hours": scenario_results["ens_hours"].mean(),
        "total_spill_mwh": scenario_results["spill_mwh"].sum(),
        "unserved_energy_rate": (
            scenario_results["ens_mwh"].sum()
            / max(scenario_results["total_net_load_mwh"].sum(), 1e-6)
        ),
    }


RUN_SUMMARY_COLUMNS = [
    "run", "model", "S",
    "fixed_commitment_cost", "expected_dispatch_cost", "expected_total_cost",
    "p95_total_cost", "worst_total_cost",
    "expected_ens_mwh", "total_ens_mwh", "ens_probability", "expected_ens_hours",
    "total_spill_mwh", "unserved_energy_rate", "n_evaluation_scenarios",
    "no_load_cost", "startup_cost", "shutdown_cost", "startup_count", "committed_unit_hours",
    "evaluation_time_seconds",
]
"""Columns of ``evaluation_summary.csv``, the one-row file that ``scripts/evaluate.py``
writes in each run folder: the columns of ``summary/evaluation_metrics.csv`` (without
the two normalized costs) followed by the wall-clock time of the evaluation."""


def evaluate_one_commitment(
    commitment_file: Path,
    gen: pd.DataFrame,
    net_load_scenarios: pd.DataFrame,
    scenario_output: Path | None = None,
    batch_size: int | None = None,
) -> dict:
    """Evaluate one ``commitment.csv`` on the evaluation set.

    Parameters
    ----------
    commitment_file
        Path to a ``commitment.csv``; its folder name labels the run.
    gen, net_load_scenarios
        Generator portfolio and evaluation set.
    scenario_output
        If given, the table with one row for each scenario is written there
        (``evaluation.csv``).
    batch_size
        Scenarios in one LP (default ``BATCH_SIZE``).

    Returns
    -------
    dict
        Model label and scenario count (:func:`parse_model_case`), first-stage
        cost terms, the summary statistics of :func:`summarize_evaluation` and
        the evaluation time; the keys of :data:`RUN_SUMMARY_COLUMNS` except
        ``run``.
    """
    commitment_file = Path(commitment_file)
    folder = commitment_file.parent
    case_info = parse_model_case(folder.name)

    commitment = pd.read_csv(commitment_file)
    commitment["unit"] = commitment["unit"].astype(str)

    fixed_cost_info = compute_fixed_commitment_cost(commitment, gen)

    start_time = time.time()
    scenario_results = evaluate_scenarios(
        commitment, gen, net_load_scenarios, batch_size=batch_size, label=folder.name
    )
    elapsed = time.time() - start_time

    if scenario_output is not None:
        scenario_output = Path(scenario_output)
        scenario_output.parent.mkdir(parents=True, exist_ok=True)
        scenario_results.to_csv(scenario_output, index=False)

    summary = summarize_evaluation(scenario_results)
    fixed_cost_info["startup_count"] = int(round(fixed_cost_info["startup_count"]))
    fixed_cost_info["committed_unit_hours"] = int(round(fixed_cost_info["committed_unit_hours"]))
    row = {**case_info, **fixed_cost_info, **summary, "evaluation_time_seconds": elapsed}
    return {k: row[k] for k in RUN_SUMMARY_COLUMNS if k in row}
