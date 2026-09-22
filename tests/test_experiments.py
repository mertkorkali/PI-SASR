"""Checks of the additional experiments against the reference results in ``results/``.

The fast tests recompute, without a solver, the certified gaps of Section 6.1 from
the stored lower-bound batches, the spectrum of the net-load matrix of Section 4.1,
one row of the GP kernel comparison of Section 4.2 and the merit-order statistics
of Section 6.5. The tests marked ``gurobi`` solve the dispatch LP of the
merit-order check for three scenarios and one reduced UC of the budget-sensitivity
study.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pisasr import experiments as ex
from pisasr import paths, systems
from pisasr import uc_model as r
from pisasr.statistics import batch_lower_bound, certified_gap, evaluation_upper_bound

pytestmark = pytest.mark.usefixtures("restore_active_system")

SYSTEM_KEYS = ["tx123bt", "texas2k"]


def _experiments(system: str):
    return paths.REFERENCE_DIR / system / "experiments"


def _read_exact(path) -> pd.DataFrame:
    """Read a CSV file without the rounding of the default float parser."""
    return pd.read_csv(path, float_precision="round_trip")


# ---------------------------------------------------------------------------
# Lower bound and certified gap (Sections 4.4 and 6.1)
# ---------------------------------------------------------------------------

def test_batch_lower_bound_formula():
    z = np.array([100.0, 104.0, 98.0, 101.0, 97.0])
    lb = batch_lower_bound(z, mip_gap=1e-3, confidence=0.95)
    zd = z * (1.0 - 1e-3)
    q = stats.t.ppf(0.975, len(z) - 1)
    assert lb["M"] == 5
    assert lb["lb_mean"] == pytest.approx(zd.mean(), rel=1e-15)
    assert lb["lb_sd"] == pytest.approx(zd.std(ddof=1), rel=1e-15)
    assert lb["lb_95"] == pytest.approx(zd.mean() - q * zd.std(ddof=1) / np.sqrt(5), rel=1e-15)


def test_certified_gap_formula():
    rng = np.random.default_rng(0)
    costs = rng.normal(110.0, 5.0, size=200)
    batches = np.array([100.0, 101.0, 99.0])
    res = certified_gap(batches, costs, insample_objective=108.0)
    ub = evaluation_upper_bound(costs)
    lb = batch_lower_bound(batches)
    assert res["ub_95"] == ub["ub_95"] and res["lb_95"] == lb["lb_95"]
    assert res["gap_cert95_pct"] == pytest.approx(100 * (ub["ub_95"] - lb["lb_95"]) / lb["lb_95"], rel=1e-15)
    assert res["gap_point_pct"] == pytest.approx(100 * (ub["ub_mean"] - lb["lb_mean"]) / lb["lb_mean"], rel=1e-15)
    assert res["gap_insample_vs_lb95_pct"] == pytest.approx(100 * (108.0 - lb["lb_95"]) / lb["lb_95"], rel=1e-15)
    with pytest.raises(ValueError):
        batch_lower_bound([1.0])


@pytest.mark.parametrize("system", SYSTEM_KEYS)
def test_certified_gap_reproduces_reference(system, tmp_path):
    """certified_gap.csv is recomputed byte for byte from the stored batches."""
    systems.activate(system, tmp_path / system)   # no reruns there: commitments come from results/
    batches = pd.read_csv(_experiments(system) / "lower_bound_batches.csv")
    table = ex.certified_gap_table(batches)
    ref_file = _experiments(system) / "certified_gap.csv"
    assert table.to_csv(index=False) == ref_file.read_text()


def test_certified_gap_values_of_the_paper():
    tx = _read_exact(_experiments("tx123bt") / "certified_gap.csv").set_index(["case", "batch_size"])
    tk = _read_exact(_experiments("texas2k") / "certified_gap.csv").set_index(["case", "batch_size"])
    assert round(tx.loc[("pisasr", 200), "gap_cert95_pct"], 2) == 2.79
    assert round(tx.loc[("full", 200), "gap_cert95_pct"], 2) == 1.68
    assert round(tk.loc[("pisasr", 200), "gap_cert95_pct"], 2) == 3.76
    assert round(tk.loc[("full", 200), "gap_cert95_pct"], 2) == 2.75


# ---------------------------------------------------------------------------
# Embedding spectrum (Section 4.1) and GP kernels (Section 4.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system", SYSTEM_KEYS)
def test_embedding_spectrum_reproduces_reference(system, tmp_path):
    systems.activate(system, tmp_path / system)
    target = r.select_stochastic_scenarios(r.load_net_load_scenarios(), 1000)
    spec = ex.embedding_spectrum(target)
    ref_file = _experiments(system) / "embedding_spectrum_S1000.csv"
    assert spec.to_csv(index=False) == ref_file.read_text()


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gp_kernel_row_reproduces_reference(tmp_path):
    """One row of gp_kernels_S1000.csv (TX-123BT, Matérn-5/2 ARD, random seed 7)."""
    systems.activate("tx123bt", tmp_path / "tx123bt")
    path = ex.stage_gp_kernels(tmp_path / "experiments", 1000, kernels=["matern52_ard"], seeds=(7,))
    got = pd.read_csv(path, float_precision="round_trip").iloc[0]
    ref = _read_exact(_experiments("tx123bt") / "gp_kernels_S1000.csv")
    ref = ref[(ref["kernel"] == "matern52_ard") & (ref["seed"] == 7)].iloc[0]
    for col in ("r2_holdout", "spearman_allS", "top90_overlap", "top148_overlap"):
        assert got[col] == pytest.approx(ref[col], rel=1e-9), col


# ---------------------------------------------------------------------------
# Merit-order check (Section 6.5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system, expected", [
    ("tx123bt", (0.665, 5.008, 0.99962)),
    ("texas2k", (0.572, 8.226, 0.99995)),
])
def test_merit_order_statistics(system, expected):
    table = pd.read_csv(paths.REFERENCE_DIR / system / "summary" / ex.MERIT_ORDER_FILE)
    assert list(table.columns) == ["scenario", "merit_order", "lp", "rel_err_pct"]
    s = ex.merit_order_summary(table)
    assert s["n"] == ex.MERIT_ORDER_SCENARIOS == 70
    assert (round(s["mean_abs_err_pct"], 3), round(s["max_abs_err_pct"], 3), round(s["r2"], 5)) == expected
    assert s["n_not_above_lp"] == 70          # the merit order never exceeds the LP


def test_merit_order_scenarios_are_the_first_training_rows():
    for system in SYSTEM_KEYS:
        systems.activate(system)
        table = pd.read_csv(paths.REFERENCE_DIR / system / "summary" / ex.MERIT_ORDER_FILE)
        first = list(r.load_net_load_scenarios().index[:70])
        assert table["scenario"].tolist() == first


@pytest.mark.gurobi
@pytest.mark.parametrize("system", SYSTEM_KEYS)
def test_merit_order_rows_reproduce_reference(system, tmp_path):
    systems.activate(system, tmp_path / system)
    got = ex.merit_order_validation(3)
    ref = _read_exact(paths.REFERENCE_DIR / system / "summary" / ex.MERIT_ORDER_FILE).iloc[:3]
    assert got["scenario"].tolist() == ref["scenario"].tolist()
    for col in ("merit_order", "lp", "rel_err_pct"):
        np.testing.assert_allclose(got[col].to_numpy(float), ref[col].to_numpy(float), rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# One reduced UC of the budget-sensitivity study (Figure 3(b))
# ---------------------------------------------------------------------------

@pytest.mark.gurobi
def test_budget_sensitivity_point(tmp_path):
    """TX-123BT, S = 200, (K_c, K_r) = (15, 10): same active set, objective and gap (about 1 min)."""
    systems.activate("tx123bt", tmp_path / "tx123bt")
    path = ex.stage_budget_sensitivity(tmp_path / "experiments", [200], budgets=[(15, 10)])
    got = pd.read_csv(path, float_precision="round_trip").iloc[0]
    ref = _read_exact(_experiments("tx123bt") / "budget_sensitivity.csv")
    ref = ref[(ref["S"] == 200) & (ref["k_critical"] == 15) & (ref["k_cover"] == 10)].iloc[0]
    assert int(got["K"]) == int(ref["K"]) == 25
    for col in ("objective", "gap_pct", "under_rep_pct"):
        assert got[col] == pytest.approx(ref[col], rel=1e-9), col


# ---------------------------------------------------------------------------
# Result files and scripts
# ---------------------------------------------------------------------------

def test_upsert_row_appends_skips_and_replaces(tmp_path):
    path = tmp_path / "rows.csv"
    ex.upsert_row(path, {"S": 200, "k": 1, "value": 1.0}, ["S", "k"])
    ex.upsert_row(path, {"S": 200, "k": 2, "value": 2.0}, ["S", "k"])
    ex.upsert_row(path, {"S": 500, "k": 1, "value": 3.0}, ["S", "k"])
    assert ex.done_keys(path, ["S", "k"]) == {("200", "1"), ("200", "2"), ("500", "1")}
    ex.upsert_row(path, {"S": 200, "k": 2, "value": 9.0}, ["S", "k"])      # replaced in place
    df = pd.read_csv(path)
    assert df["value"].tolist() == [1.0, 9.0, 3.0]


def test_stage_defaults_follow_the_paper():
    assert ex.BUDGETS["tx123bt"] == [(15, 10), (30, 20), (45, 30), (60, 40), (90, 60), (120, 80)]
    assert ex.BUDGETS["texas2k"] == [(30, 20), (45, 30), (60, 40), (90, 60), (120, 80)]
    assert ex.REPLICATION_SEEDS == (7, 17, 27, 37, 47)
    assert ex.KERNEL_SEEDS == (7, 17, 27)
    for system in SYSTEM_KEYS:
        bs = pd.read_csv(_experiments(system) / "budget_sensitivity.csv")
        assert sorted(bs["S"].unique().tolist()) == ex.DEFAULT_S["budget-sensitivity"][system]
        assert sorted(zip(bs["k_critical"], bs["k_cover"])) == sorted(ex.BUDGETS[system] * bs["S"].nunique())
        lb = pd.read_csv(_experiments(system) / "lower_bound_batches.csv")
        assert sorted(lb["batch_size"].unique().tolist()) == ex.DEFAULT_S["lower-bound"][system]
        assert (lb.groupby("batch_size").size() == ex.LOWER_BOUND_BATCHES).all()


@pytest.mark.parametrize("script", ["run_experiments.py", "run_replication.py", "validate_merit_order.py"])
def test_help(script, script_env):
    proc = subprocess.run([sys.executable, str(paths.REPO_ROOT / "scripts" / script), "--help"],
                          capture_output=True, text=True, env=script_env)
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


def test_unknown_names_fail_before_any_computation(tmp_path):
    systems.activate("tx123bt", tmp_path / "tx123bt")
    with pytest.raises(ValueError):
        ex.stage_component_timing(tmp_path, 1000, variants=["no_such_variant"])
    with pytest.raises(ValueError):
        ex.stage_criticality_only(tmp_path, 1000, variants=["no_such_variant"])
    with pytest.raises(ValueError):
        ex.stage_gp_kernels(tmp_path, 1000, kernels=["no_such_kernel"])
    assert not list(tmp_path.glob("*.csv"))
