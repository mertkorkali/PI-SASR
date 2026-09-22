"""The exact LP evaluation of the stored commitments reproduces results/<system>/runs/*/evaluation.csv.

Each stored commitment is evaluated on the first 20 evaluation scenarios and
the cost of each scenario is compared with the reference files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pisasr import evaluation as ev
from pisasr import paths, systems
from pisasr.constants import SCENARIO_COUNTS

pytestmark = pytest.mark.usefixtures("restore_active_system")

N_CHECK = 20
RUNS = ["deterministic"] + [f"full_S{S}" for S in SCENARIO_COUNTS] + [f"pisasr_S{S}" for S in SCENARIO_COUNTS]
# The Texas2k LP is about four times larger; the S=1000 runs of Table 2 and the
# deterministic run are checked by default and the others under the "slow" marker.
TEXAS2K_DEFAULT = {"deterministic", "full_S1000", "pisasr_S1000"}

CASES = [pytest.param("tx123bt", run, marks=pytest.mark.gurobi, id=f"tx123bt-{run}") for run in RUNS]
CASES += [
    pytest.param("texas2k", run,
                 marks=[pytest.mark.gurobi] + ([] if run in TEXAS2K_DEFAULT else [pytest.mark.slow]),
                 id=f"texas2k-{run}")
    for run in RUNS
]


def test_reference_files_complete():
    for key in systems.SYSTEMS:
        for run in RUNS:
            folder = systems.reference_run_dir(run, key)
            assert (folder / "commitment.csv").exists(), folder
            df = pd.read_csv(folder / "evaluation.csv")
            assert len(df) == 2000, folder
            assert "total_cost_with_fixed" in df.columns


def test_fixed_cost_consistent_with_reference():
    """total_cost_with_fixed - dispatch_cost equals the first-stage cost of the commitment."""
    for key in systems.SYSTEMS:
        systems.activate(key)
        gen = ev.load_generators()
        for run in ("deterministic", "full_S1000", "pisasr_S1000"):
            folder = systems.reference_run_dir(run, key)
            commitment = pd.read_csv(folder / "commitment.csv")
            commitment["unit"] = commitment["unit"].astype(str)
            fixed = ev.compute_fixed_commitment_cost(commitment, gen)["fixed_commitment_cost"]
            ref = pd.read_csv(folder / "evaluation.csv")
            np.testing.assert_allclose(ref["total_cost_with_fixed"] - ref["dispatch_cost"], fixed, rtol=1e-9)


def test_summary_statistics_texas2k():
    """summarize_evaluation reproduces results/texas2k/summary/evaluation_metrics.csv."""
    summary = pd.read_csv(paths.REFERENCE_DIR / "texas2k" / "summary" / "evaluation_metrics.csv")
    names = {"deterministic": "deterministic", "full_S1000": "stochastic_1000_scenarios",
             "pisasr_S1000": "novel_pisasr_S1000"}
    for run, case in names.items():
        ref = pd.read_csv(systems.reference_run_dir(run, "texas2k") / "evaluation.csv")
        stats = ev.summarize_evaluation(ref)
        if "run" in summary.columns and (summary["run"] == run).any():
            row = summary[summary["run"] == run].iloc[0]
        elif "case" in summary.columns:
            row = summary[summary["case"] == case].iloc[0]
        else:
            pytest.skip("unrecognized layout of evaluation_metrics.csv")
        for col in ("expected_total_cost", "unserved_energy_rate", "total_ens_mwh", "p95_total_cost"):
            assert stats[col] == pytest.approx(float(row[col]), rel=1e-9), (run, col)


@pytest.mark.parametrize("key, run", CASES)
def test_exact_lp_matches_reference(key, run):
    systems.activate(key)
    gen = ev.load_generators()
    test = ev.load_net_load_scenarios().iloc[:N_CHECK]
    folder = systems.reference_run_dir(run, key)
    commitment = pd.read_csv(folder / "commitment.csv")
    ref = pd.read_csv(folder / "evaluation.csv").iloc[:N_CHECK]

    new = ev.evaluate_scenarios(commitment, gen, test)

    assert list(new["scenario"]) == list(ref["scenario"])
    np.testing.assert_allclose(new["dispatch_cost"], ref["dispatch_cost"], rtol=1e-6)
    np.testing.assert_allclose(new["total_cost_with_fixed"], ref["total_cost_with_fixed"], rtol=1e-6)
    np.testing.assert_allclose(new["ens_mwh"], ref["ens_mwh"], rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(new["total_net_load_mwh"], ref["total_net_load_mwh"], rtol=1e-12)
