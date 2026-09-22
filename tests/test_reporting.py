"""Fast checks of the reporting code against the reference results (no solver needed).

The tests recompute the paired confidence intervals of Section 6.3, a selection of
table values, the evaluation summaries and the register of paper numbers from the
CSV files in ``results/`` and ``data/``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pisasr import reporting as rp
from pisasr.constants import HOUR_COLS
from pisasr.paths import DATA_DIR, REFERENCE_DIR
from pisasr.statistics import paired_difference_interval

SYSTEMS = rp.SYSTEM_KEYS


@pytest.fixture(scope="module")
def src():
    return rp.Sources(REFERENCE_DIR, DATA_DIR)


# ---------------------------------------------------------------------------
# Paired confidence intervals (Section 6.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system, quantity, expected", [
    ("tx123bt", "total_cost", (0.915, 0.650, 1.179)),
    ("texas2k", "total_cost", (1.105, 0.896, 1.314)),
    ("texas2k", "dispatch_cost", (0.007, -0.245, 0.259)),
])
def test_paired_interval_reproduces_paper(system, quantity, expected):
    table = rp.paired_cost_difference(REFERENCE_DIR, system, S=1000).set_index("quantity")
    row = table.loc[quantity]
    got = (row["mean_difference_pct"], row["ci_low_pct"], row["ci_high_pct"])
    assert [round(x, 3) for x in got] == list(expected)
    assert row["n_pairs"] == 2000
    assert row["distribution"] == "t"


def test_normal_interval_does_not_reproduce_texas2k():
    """The printed intervals use the Student-t quantile; the normal one gives 0.897 for Texas2k."""
    a = pd.read_csv(REFERENCE_DIR / "texas2k/runs/pisasr_S1000/evaluation.csv")
    b = pd.read_csv(REFERENCE_DIR / "texas2k/runs/full_S1000/evaluation.csv")
    t = paired_difference_interval(a["total_cost_with_fixed"], b["total_cost_with_fixed"], distribution="t")
    z = paired_difference_interval(a["total_cost_with_fixed"], b["total_cost_with_fixed"], distribution="normal")
    assert round(t.ci_low_pct, 3) == 0.896
    assert round(z.ci_low_pct, 3) == 0.897


def test_paired_interval_on_synthetic_data():
    rng = np.random.default_rng(0)
    b = rng.normal(100.0, 5.0, size=400)
    a = b + 2.0 + rng.normal(0.0, 1.0, size=400)
    r = paired_difference_interval(a, b, confidence=0.95)
    d = a - b
    assert math.isclose(r.mean_difference, d.mean())
    assert math.isclose(r.ci_high - r.mean_difference, r.mean_difference - r.ci_low)
    assert r.ci_low < 2.0 < r.ci_high
    assert math.isclose(r.mean_difference_pct, 100 * d.mean() / b.mean())
    with pytest.raises(ValueError):
        paired_difference_interval(a, b[:-1])


@pytest.mark.parametrize("system", SYSTEMS)
def test_stored_paired_table_is_current(system):
    stored = REFERENCE_DIR / system / "summary" / "paired_cost_difference_S1000.csv"
    if not stored.is_file():
        pytest.skip("paired_cost_difference_S1000.csv not generated yet")
    new = rp.paired_cost_difference(REFERENCE_DIR, system, S=1000)
    assert rp.compare_frames(new, pd.read_csv(stored), key="quantity") == []


# ---------------------------------------------------------------------------
# Evaluation summaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system", SYSTEMS)
def test_evaluation_summary_matches_stored_files(system):
    summary = rp.summarize_system(REFERENCE_DIR, system, DATA_DIR)
    assert len(summary) == 11
    assert (summary["n_evaluation_scenarios"] == 2000).all()
    stored = REFERENCE_DIR / system / "summary" / "evaluation_metrics.csv"
    if stored.is_file():
        assert rp.compare_frames(summary, pd.read_csv(stored)) == []
    scaling = pd.read_csv(REFERENCE_DIR / system / "summary" / "scaling.csv")
    assert rp.check_scaling_evaluation(summary, scaling) == []


@pytest.mark.parametrize("system", SYSTEMS)
def test_evaluation_set_is_first_2000_test_scenarios(system):
    test_ids = pd.read_csv(DATA_DIR / system / "net_load_scenarios_test.csv", usecols=[0]).iloc[:2000, 0]
    train_ids = set(pd.read_csv(DATA_DIR / system / "net_load_scenarios_train.csv", usecols=[0]).iloc[:, 0])
    for run in ("deterministic", "full_S1000", "pisasr_S1000"):
        ev = pd.read_csv(REFERENCE_DIR / system / "runs" / run / "evaluation.csv")
        assert list(ev["scenario"]) == list(test_ids)
    assert train_ids.isdisjoint(test_ids)


def test_fixed_cost_of_tx123bt_commitments():
    gen = rp.load_portfolio("tx123bt")
    full = rp.commitment_costs(pd.read_csv(REFERENCE_DIR / "tx123bt/runs/full_S1000/commitment.csv"), gen)
    pis = rp.commitment_costs(pd.read_csv(REFERENCE_DIR / "tx123bt/runs/pisasr_S1000/commitment.csv"), gen)
    assert full["fixed_commitment_cost"] == pytest.approx(2_659_639.0, rel=1e-12)
    assert pis["fixed_commitment_cost"] == pytest.approx(2_645_256.0, rel=1e-12)
    assert full["committed_unit_hours"] == 2493


# ---------------------------------------------------------------------------
# Table values
# ---------------------------------------------------------------------------
def test_table1_values(src):
    for system, units, cap, peak in (("tx123bt", 138, "76.4", "77.2"), ("texas2k", 568, "77.8", "76.8")):
        gen = src.data(system, "generator_portfolio.csv")
        assert len(gen) == units
        assert rp.fmt(gen["p_max_mw"].sum() / 1000, 1) == cap
        pool = src.scenario_pool(system)
        assert len(pool) == 7500
        assert rp.fmt(pool[HOUR_COLS].max(axis=1).max() / 1000, 1) == peak


def test_table2_values(src):
    tx = src.solve("tx123bt", "pisasr_S1000")
    assert rp.fmt(tx["objective"] / 1e6, 2) == "38.37"
    assert rp.fmt(tx["opt_gap_pct_vs_full"], 2) == "0.69"
    assert int(tx["n_scenarios_used"]) == 148
    assert rp.fmt(src.ens_pct("tx123bt", "deterministic"), 3) == "1.755"
    assert rp.fmt(src.ens_pct("texas2k", "full_S1000"), 4) == "0.0019"
    ratio = src.eval_cost("texas2k", "deterministic") / src.eval_cost("texas2k", "full_S1000")
    assert round(ratio) == 35


def test_table3_values(src):
    tx = src.scaling("tx123bt").set_index("S")
    tk = src.scaling("texas2k").set_index("S")
    assert rp.fmt_speedup(tx.at[2500, "speedup_x"]) == "21.3"
    assert rp.fmt_speedup(tk.at[2500, "speedup_x"]) == "24.0"
    assert rp.fmt_gap(tx.at[100, "pisasr_gap_pct"]) == "0.004"
    assert rp.fmt(src.under_rep_pct("texas2k", 2500), 2) == "-15.87"
    stored = pd.read_csv(REFERENCE_DIR / "tx123bt/summary/scaling.csv").set_index("S")
    for col in ("full_obj", "full_time_s", "pisasr_obj", "pisasr_gap_pct", "pisasr_time_s", "speedup_x", "K_used"):
        np.testing.assert_allclose(tx[col].to_numpy(float), stored[col].to_numpy(float), rtol=1e-12)


def test_table4_and_kmin(src):
    d = src.experiment("tx123bt", "component_timing_S1000.csv").set_index("variant")
    assert rp.fmt(d.at["pisasr", "t_online_s"], 0) == "77"
    assert rp.fmt(d.at["ffs_K148", "gap_pct"], 2, sign=True) == "+4.99"
    km_tx = rp.kmin_from_budget(src.experiment("tx123bt", "budget_sensitivity.csv"), 1.0)
    assert {S: v[0] for S, v in km_tx.items()} == {200: 90, 500: 143, 1000: 75, 2500: 100}
    km_tk = rp.kmin_from_budget(src.experiment("texas2k", "budget_sensitivity.csv"), 1.0)
    assert km_tk[2500][0] is None and km_tk[2500][1] == 198


def test_rendered_tables_contain_paper_cells(src):
    t2 = rp.table_primary(src)
    assert "\\textbf{74}" in t2.tex and "12.13" in t2.tex and "\\rev" not in t2.tex
    t3 = rp.table_scaling(src)
    assert "14689\\tnote{c}" in t3.tex and "| 1,000 | 28.59 | 4324 |" in t3.md
    assert "solver call only" in t3.tex


# ---------------------------------------------------------------------------
# Register of paper numbers
# ---------------------------------------------------------------------------
def test_register_has_no_differences(src):
    reg = rp.build_register(src)
    statuses = {c.status for c in reg.claims}
    assert statuses <= {"yes", "not yet generated"}
    missing = [c for c in reg.claims if c.status == "not yet generated"]
    assert all("merit_order_validation" in c.computed for c in missing)
    assert sum(c.status == "yes" for c in reg.claims) > 400
