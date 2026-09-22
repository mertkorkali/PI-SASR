"""Deterministic (expected-value) UC and the full (extensive-form) model.

* Deterministic UC: one scenario, the mean net-load profile of the training set,
  with every unit's minimum and maximum output derated by (1 - forced-outage
  rate) to account for outages in expectation.
* Full model: the two-stage stochastic UC over the first S scenarios of the
  fixed permutation of the training set, each with its own N-1 outage draw.

Timing. Each ``solve_metrics.csv`` records ``build_time_seconds`` (Pyomo model
construction) and ``solver_time_seconds`` (the solver call, which includes
Pyomo's model-file transfer). ``solve_time_seconds`` follows the definition used
in the paper for each system (``System.full_timing_basis``): the solver call only
on TX-123BT, and model construction plus solver call on Texas2k. Texas2k at
S = 2,500 is solved from an LP file by :func:`run_full_file_based`, and its
``solve_time_seconds`` covers construction, model-file writing and solution.
"""

from __future__ import annotations

import gc
import os
import re
import tempfile
import time
from pathlib import Path

import pandas as pd

from . import uc_model as r
from .constants import GUROBI_OPTIONS, HOUR_COLS
from .outages import GEN_FORCED_OUTAGE_RATE, draw_scenario_availability
from .paths import display_path
from .systems import System, get_system

__all__ = [
    "scenario_availability",
    "run_deterministic",
    "run_full",
    "run_full_file_based",
    "use_file_based_path",
]


def scenario_availability(units, scenario_ids) -> dict[tuple[str, str], float]:
    """Outage draw ``{(scenario, unit): availability}`` for a set of scenarios."""
    availability = {}
    for s in scenario_ids:
        scen_avail = draw_scenario_availability(units, s)
        for g in units:
            availability[(str(s), g)] = scen_avail[g]
    return availability


def _full_model_inputs(gen: pd.DataFrame, net: pd.DataFrame, n_scenarios: int):
    """Scenario profiles, uniform probabilities and outage draws of the full model."""
    units = [str(u) for u in gen["unit"]]
    selected = r.select_stochastic_scenarios(net, n_scenarios)
    demand_scenarios = {str(s): selected.loc[s, HOUR_COLS].to_numpy(float) for s in selected.index}
    probabilities = {s: 1.0 / n_scenarios for s in demand_scenarios}
    availability = scenario_availability(units, selected.index)
    return demand_scenarios, probabilities, availability


def _paper_time(system: System, build_s: float, solver_s: float, build_and_solve_s: float) -> float:
    """``solve_time_seconds`` with the definition of the paper for ``system``."""
    if system.full_timing_basis == "solver":
        return solver_s
    if system.full_timing_basis == "build+solver":
        return build_and_solve_s
    raise ValueError(f"Unknown timing basis {system.full_timing_basis!r}")


def _skip(run_folder: Path, force: bool) -> bool:
    if (run_folder / "solve_metrics.csv").exists() and not force:
        print(f"{display_path(run_folder)} already holds solve_metrics.csv; "
              "skipping (use --force to solve again)")
        return True
    return False


def run_deterministic(
    system: str | System,
    gen: pd.DataFrame,
    net: pd.DataFrame,
    run_folder: str | Path,
    force: bool = False,
) -> dict | None:
    """Solve the deterministic (expected-value) UC and write its run folder.

    Returns the metrics written to ``solve_metrics.csv``, or ``None`` when the
    run folder already holds a result and ``force`` is false.
    """
    system = get_system(system)
    run_folder = Path(run_folder)
    if _skip(run_folder, force):
        return None

    print(f"\n=== {system.display_name}: deterministic (expected-value) UC ===")
    expected = r.expected_net_load(net)
    units = [str(u) for u in gen["unit"]]
    # Expected derate: the single scenario accounts for outages by scaling each
    # unit's output limits by (1 - forced-outage rate).
    availability = {("expected", g): 1.0 - GEN_FORCED_OUTAGE_RATE for g in units}

    t_start = time.time()
    model = r.build_uc_master_model(
        gen=gen,
        demand_scenarios={"expected": expected},
        probabilities={"expected": 1.0},
        model_type="deterministic",
        availability=availability,
    )
    build_s = time.time() - t_start
    metrics = r.solve_pyomo_model(model)
    build_and_solve_s = time.time() - t_start
    solver_s = metrics["solve_time_seconds"]

    metrics["solve_time_seconds"] = _paper_time(system, build_s, solver_s, build_and_solve_s)
    metrics.update({
        "model": "deterministic",
        "n_scenarios": 1,
        "build_time_seconds": build_s,
        "solver_time_seconds": solver_s,
        "timing_basis": system.full_timing_basis,
    })
    r.save_model_outputs(model, run_folder, metrics)
    print(f"deterministic: objective {metrics['objective']:,.2f}, "
          f"time {metrics['solve_time_seconds']:.1f} s -> {display_path(run_folder)}")
    return metrics


def run_full(
    system: str | System,
    gen: pd.DataFrame,
    net: pd.DataFrame,
    n_scenarios: int,
    run_folder: str | Path,
    force: bool = False,
) -> dict | None:
    """Solve the full (extensive-form) model over ``n_scenarios`` scenarios in memory."""
    system = get_system(system)
    run_folder = Path(run_folder)
    if _skip(run_folder, force):
        return None

    print(f"\n=== {system.display_name}: full two-stage stochastic UC, S={n_scenarios} ===")
    demand_scenarios, probabilities, availability = _full_model_inputs(gen, net, n_scenarios)

    t_start = time.time()
    model = r.build_uc_master_model(
        gen, demand_scenarios, probabilities, "stochastic", availability=availability
    )
    build_s = time.time() - t_start
    metrics = r.solve_pyomo_model(model)
    build_and_solve_s = time.time() - t_start
    solver_s = metrics["solve_time_seconds"]

    metrics["solve_time_seconds"] = _paper_time(system, build_s, solver_s, build_and_solve_s)
    metrics.update({
        "model": "stochastic",
        "n_scenarios": n_scenarios,
        "build_time_seconds": build_s,
        "solver_time_seconds": solver_s,
        "timing_basis": system.full_timing_basis,
    })
    r.save_model_outputs(model, run_folder, metrics)
    print(f"full S={n_scenarios}: objective {metrics['objective']:,.2f}, "
          f"time {metrics['solve_time_seconds']:.1f} s -> {display_path(run_folder)}")
    return metrics


def run_full_file_based(
    system: str | System,
    gen: pd.DataFrame,
    net: pd.DataFrame,
    n_scenarios: int,
    run_folder: str | Path,
    model_file_dir: str | Path | None = None,
    force: bool = False,
) -> dict | None:
    """Solve the full model through an LP file to limit memory use.

    The in-memory Pyomo model of Texas2k at S = 2,500 (about 34 million
    variables) did not fit, together with Gurobi's copy and the barrier
    factorization, in 192 GB. This function builds the model, writes an LP file
    with symbolic names, releases the Pyomo model and solves the file with
    gurobipy using the options in ``GUROBI_OPTIONS``. The commitment is
    recovered from the variable names ``u(<unit>_<hour>)``.

    Parameters
    ----------
    model_file_dir
        Folder for the (multi-gigabyte) LP file; the system temporary folder by
        default. The file is deleted after the solution.
    """
    import gurobipy as gp

    system = get_system(system)
    run_folder = Path(run_folder)
    if _skip(run_folder, force):
        return None

    model_file_dir = Path(tempfile.gettempdir() if model_file_dir is None else model_file_dir)
    model_file_dir.mkdir(parents=True, exist_ok=True)
    lp_file = model_file_dir / f"pisasr_{system.key}_full_S{n_scenarios}.lp"

    print(f"\n=== {system.display_name}: full two-stage stochastic UC, S={n_scenarios} "
          "(solved from an LP file) ===")
    units = [str(u) for u in gen["unit"]]
    demand_scenarios, probabilities, availability = _full_model_inputs(gen, net, n_scenarios)

    print("building the Pyomo model ...")
    t0 = time.time()
    model = r.build_uc_master_model(gen, demand_scenarios, probabilities,
                                    "stochastic", availability=availability)
    build_s = time.time() - t0
    print(f"built in {build_s:.0f} s; writing {lp_file} with symbolic names ...")
    t_write = time.time()
    model.write(str(lp_file), io_options={"symbolic_solver_labels": True})
    write_s = time.time() - t_write
    print(f"model file written ({os.path.getsize(lp_file) / 1e9:.1f} GB); "
          "releasing the Pyomo model")
    del model, demand_scenarios, availability
    gc.collect()

    print("solving with gurobipy ...")
    t1 = time.time()
    m = gp.read(str(lp_file))
    for key, value in GUROBI_OPTIONS.items():
        m.setParam(key, value)
    m.optimize()
    solver_s = time.time() - t1
    if m.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi returned status {m.Status}")
    obj = m.ObjVal
    print(f"solved: objective {obj:,.2f}, solver time {solver_s:.0f} s")

    # First-stage schedule from the symbolic names u(g_t), startup(g_t), shutdown(g_t).
    pattern = re.compile(r"^(u|startup|shutdown)\((.+)_([0-9]+)\)$")
    schedule = {"u": {}, "startup": {}, "shutdown": {}}
    for v in m.getVars():
        match = pattern.match(v.VarName)
        if match:
            kind, g_name, t = match.group(1), match.group(2), int(match.group(3))
            schedule[kind][(g_name, t)] = int(round(v.X))
    n_u = len(schedule["u"])
    if n_u != len(units) * 24:
        raise RuntimeError(f"expected {len(units) * 24} commitment binaries, found {n_u}")

    run_folder.mkdir(parents=True, exist_ok=True)
    for kind, fname in [("u", "commitment.csv"), ("startup", "startup.csv"),
                        ("shutdown", "shutdown.csv")]:
        rows = []
        for g_name in units:
            row = {"unit": g_name}
            for t in range(24):
                row[f"hour_{t:02d}"] = schedule[kind].get((g_name, t), 0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(run_folder / fname, index=False)

    metrics = {
        "status": "ok",
        "termination": "optimal",
        "objective": obj,
        "solve_time_seconds": time.time() - t0,   # construction + file writing + solution
        "model": "stochastic",
        "n_scenarios": n_scenarios,
        "lean_solver_only_seconds": solver_s,
        "build_time_seconds": build_s,
        "write_time_seconds": write_s,
        "solver_time_seconds": solver_s,
        "timing_basis": "build+write+solver",
    }
    pd.DataFrame([metrics]).to_csv(run_folder / "solve_metrics.csv", index=False)
    del m
    os.remove(lp_file)
    print(f"full S={n_scenarios}: objective {obj:,.2f}, time {metrics['solve_time_seconds']:.0f} s "
          f"-> {display_path(run_folder)}")
    return metrics


def use_file_based_path(system: str | System, n_scenarios: int, mode: str = "auto") -> bool:
    """Whether the full model is solved through an LP file.

    ``mode="auto"`` reproduces the paper: only Texas2k at S >= 2,500.
    """
    if mode == "always":
        return True
    if mode == "never":
        return False
    return get_system(system).key == "texas2k" and n_scenarios >= 2500
