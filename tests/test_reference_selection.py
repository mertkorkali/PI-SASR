"""The selection stage and the merit-order verification reproduce the reference results.

These tests need no solver. With the stored reference commitment u_ref, the
active set, the GP accuracy and the merit-order verified objective match the
values in results/<system>/runs/pisasr_S1000/ exactly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pisasr import systems
from pisasr import uc_model as r
from pisasr.evaluation import compute_fixed_commitment_cost
from pisasr.merit_order import MeritOrderDispatch
from pisasr.pisasr import _avail_vec, read_commitment, select_active_set

pytestmark = pytest.mark.usefixtures("restore_active_system")

TX_R2 = 0.5112988860331589
TX_VERIFIED_OBJECTIVE = 38_369_743.7514146


def _stored_metrics(system, run):
    return pd.read_csv(systems.reference_run_dir(run, system) / "solve_metrics.csv").iloc[0]


def _selection(key, S=1000):
    system = systems.activate(key)
    gen = r.load_generators()
    net = r.load_net_load_scenarios()
    ref = systems.resolve_reference_commitment("results")
    return system, gen, net, select_active_set(gen, net, ref, n_target=S)


def test_tx123bt_selection_S1000():
    system, _, _, sel = _selection("tx123bt")
    stored = _stored_metrics(system, "pisasr_S1000")
    assert len(sel.active_set) == 148 == int(stored["n_scenarios_used"])
    assert sel.surrogate_accuracy["r2"] == TX_R2
    assert sel.surrogate_accuracy["r2"] == float(stored["surrogate_r2"])
    assert sel.surrogate_accuracy["mae"] == pytest.approx(float(stored["surrogate_mae"]), rel=1e-12)
    assert len(sel.baseload) == 64 == int(stored["n_fixed_baseload_units"])
    assert len(sel.critical) == 90 and len(sel.cover) == 60
    assert sel.active_set[:90] == sel.critical


def test_texas2k_selection_S1000():
    system, _, _, sel = _selection("texas2k")
    stored = _stored_metrics(system, "pisasr_S1000")
    assert len(sel.active_set) == 146 == int(stored["n_scenarios_used"])
    assert sel.surrogate_accuracy["r2"] == float(stored["surrogate_r2"])
    assert len(sel.baseload) == 58 == int(stored["n_fixed_baseload_units"])


def test_tx123bt_merit_order_verification_S1000():
    system = systems.activate("tx123bt")
    gen = r.load_generators()
    net = r.load_net_load_scenarios()
    units = [str(u) for u in gen["unit"]]
    target_ids = list(r.select_stochastic_scenarios(net, 1000).index)
    commitment = read_commitment(systems.reference_run_dir("pisasr_S1000") / "commitment.csv")
    recourse = MeritOrderDispatch(gen, commitment).recourse_over(
        net, target_ids, avail_fn=lambda s: _avail_vec(units, s))
    verified = compute_fixed_commitment_cost(commitment, gen)["fixed_commitment_cost"] + float(recourse.mean())
    assert verified == pytest.approx(TX_VERIFIED_OBJECTIVE, rel=1e-6)
    assert verified == pytest.approx(float(_stored_metrics(system, "pisasr_S1000")["approx_objective"]), rel=1e-12)


def test_tx123bt_benchmark_gap_S1000():
    """Benchmark gap of the stored PI-SASR result against the stored full model."""
    system = systems.get_system("tx123bt")
    pis = _stored_metrics(system, "pisasr_S1000")
    full = _stored_metrics(system, "full_S1000")
    gap = 100.0 * (float(pis["objective"]) - float(full["objective"])) / float(full["objective"])
    assert gap == pytest.approx(0.68715, abs=1e-5)
