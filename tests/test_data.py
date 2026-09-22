"""The processed inputs match data/SHA256SUMS and have the sizes stated in the paper."""

from __future__ import annotations

import pandas as pd
import pytest

from pisasr import paths, systems
from pisasr import uc_model as r
from pisasr.constants import HOUR_COLS

pytestmark = pytest.mark.usefixtures("restore_active_system")


def test_checksums():
    assert paths.verify_checksums(paths.DATA_DIR, verbose=False)


def test_checksum_file_lists_every_data_file():
    listed = set(paths.read_checksums())
    on_disk = {
        p.relative_to(paths.DATA_DIR).as_posix()
        for p in paths.DATA_DIR.rglob("*")
        if p.is_file() and p.suffix == ".csv" and not p.relative_to(paths.DATA_DIR).as_posix().startswith("raw/")
    }
    assert on_disk <= listed


@pytest.mark.parametrize("key, n_units, capacity_gw", [("tx123bt", 138, 76.4), ("texas2k", 568, 77.8)])
def test_generator_portfolio(key, n_units, capacity_gw):
    gen = pd.read_csv(systems.get_system(key).generator_file)
    assert len(gen) == n_units
    assert round(gen["p_max_mw"].sum() / 1000, 1) == capacity_gw


@pytest.mark.parametrize("key", ["tx123bt", "texas2k"])
def test_scenario_sets(key):
    system = systems.get_system(key)
    train = pd.read_csv(system.train_file, index_col=0)
    test = pd.read_csv(system.test_file, index_col=0)
    assert train.shape == (2500, 24) and test.shape == (5000, 24)
    assert list(train.columns) == HOUR_COLS
    assert not set(train.index) & set(test.index)
    assert system.n_eval == 2000


def test_nested_subsets(tx123bt):
    systems.activate(tx123bt)
    net = r.load_net_load_scenarios()
    previous = set()
    for S in (100, 200, 500, 1000, 2500):
        ids = list(r.select_stochastic_scenarios(net, S).index)
        assert len(ids) == S
        assert previous <= set(ids)
        previous = set(ids)
    assert list(r.select_stochastic_scenarios(net, 5).index) == list(
        r.select_stochastic_scenarios(net, 1000).index[:5])


def test_tx123bt_year_parsing_ignores_parent_folders():
    """The year comes from the profile folder, not from folders higher up the path."""
    import importlib.util
    from pathlib import Path

    script = paths.REPO_ROOT / "scripts" / "data" / "prepare_tx123bt.py"
    spec = importlib.util.spec_from_file_location("prepare_tx123bt", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = Path("projects") / "HICSS2026" / "Load_5y" / "Load_annual_2017" / "load_annual_D152.txt"
    assert module.extract_year_and_day(path) == (2017, 152)
    path = Path("archive_2030") / "Wind_5y" / "wind_2021" / "wind_annual_D242.txt"
    assert module.extract_year_and_day(path) == (2021, 242)
