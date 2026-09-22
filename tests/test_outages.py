"""The N-1 outage draw is deterministic and matches the draws of the reference results."""

from __future__ import annotations

import zlib

import pandas as pd
import pytest

from pisasr.outages import (
    GEN_FORCED_OUTAGE_RATE,
    MAX_OUTAGES_PER_SCENARIO,
    OUTAGE_BASE_SEED,
    _stable_seed,
    draw_scenario_availability,
)

# Outaged unit of the first eight training and first four evaluation scenarios,
# recorded with the code that produced the reference results.
EXPECTED_OUTAGES = {
    "tx123bt": {
        "scenario_1018": "G257", "scenario_6247": "G129", "scenario_5834": "G130",
        "scenario_5043": "G129", "scenario_2640": "G262", "scenario_0484": "G178",
        "scenario_0513": "G179", "scenario_6727": "G179", "scenario_6782": "G122",
        "scenario_1354": "G76", "scenario_3257": "G212", "scenario_1885": "G51",
    },
    "texas2k": {
        "scenario_1018": "T467", "scenario_6247": "T621", "scenario_5834": "T215",
        "scenario_5043": "T104", "scenario_2640": "T466", "scenario_0484": "T608",
        "scenario_0513": "T146", "scenario_6727": "T268", "scenario_6782": "T197",
        "scenario_1354": "T377", "scenario_3257": "T509", "scenario_1885": "T167",
    },
}


def _units(system):
    gen = pd.read_csv(system.generator_file)
    return [str(u) for u in gen["unit"]]


def test_parameters():
    assert GEN_FORCED_OUTAGE_RATE == 0.05
    assert MAX_OUTAGES_PER_SCENARIO == 1
    assert OUTAGE_BASE_SEED == 12345


def test_stable_seed_formula():
    sid = "scenario_1018"
    assert _stable_seed(sid, OUTAGE_BASE_SEED) == (zlib.crc32(sid.encode()) ^ 12345) & 0xFFFFFFFF


def test_draw_is_deterministic(tx123bt):
    units = _units(tx123bt)
    for sid in ["scenario_0001", "scenario_1018", "scenario_7500"]:
        assert draw_scenario_availability(units, sid) == draw_scenario_availability(units, sid)


@pytest.mark.parametrize("key", ["tx123bt", "texas2k"])
def test_reference_draws(key, tx123bt, texas2k):
    system = tx123bt if key == "tx123bt" else texas2k
    units = _units(system)
    for sid, unit in EXPECTED_OUTAGES[key].items():
        avail = draw_scenario_availability(units, sid)
        outaged = [u for u, a in avail.items() if a == 0.0]
        assert outaged == [unit], sid


def test_at_most_one_outage(tx123bt):
    units = _units(tx123bt)
    train = pd.read_csv(tx123bt.train_file, index_col=0)
    for sid in train.index[:300]:
        avail = draw_scenario_availability(units, sid)
        assert set(avail.values()) <= {0.0, 1.0}
        assert sum(1 for a in avail.values() if a == 0.0) <= 1
