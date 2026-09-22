"""Shared fixtures and markers.

Markers (registered in pyproject.toml):

* ``gurobi``: the test solves an LP or MILP and needs Gurobi with a valid
  license; such tests are skipped automatically when Gurobi is not usable,
  including under the size-limited license installed with ``gurobipy``.
* ``slow``: the test takes several minutes.

``make test-fast`` runs ``pytest -m "not gurobi and not slow"``.
"""

from __future__ import annotations

import functools
import os

import pytest

from pisasr import paths, systems


SIZE_LIMIT_CHECK_VARIABLES = 2001
"""Variables of the model solved by :func:`gurobi_available`. The size-limited
license that comes with the ``gurobipy`` package accepts at most 2,000 variables,
fewer than any model of the tests."""


@functools.lru_cache(maxsize=1)
def gurobi_available() -> bool:
    """True when gurobipy imports and its license can solve a model above the size limit.

    Creating an empty model succeeds even under the size-limited license, so a
    model with 2,001 variables is solved; a ``GurobiError`` (no license, or the
    error "Model too large for size-limited license") means that Gurobi is not usable.
    """
    try:
        import gurobipy as gp
    except ImportError:
        return False
    env = model = None
    try:
        env = gp.Env(params={"OutputFlag": 0})
        model = gp.Model(env=env)
        model.addVars(SIZE_LIMIT_CHECK_VARIABLES)
        model.optimize()
        return model.Status == gp.GRB.OPTIMAL
    except gp.GurobiError:
        return False
    finally:
        if model is not None:
            model.dispose()
        if env is not None:
            env.dispose()


def pytest_collection_modifyitems(config, items):
    if gurobi_available():
        return
    skip = pytest.mark.skip(reason="Gurobi with a valid license is not available")
    for item in items:
        if "gurobi" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def restore_active_system():
    """Reset the active system to TX-123BT (with outputs/ as root) after the test.

    Use it in modules that call :func:`pisasr.systems.activate`, with
    ``pytestmark = pytest.mark.usefixtures("restore_active_system")``.
    """
    yield
    systems.activate("tx123bt")


@pytest.fixture(scope="session")
def script_env():
    """Environment for running a script in a subprocess.

    The ``src/`` folder of the tree under test is put first on ``PYTHONPATH``, so
    that the script imports the same package as the test process, whatever the
    working directory of the subprocess and whichever copy is installed.
    """
    env = dict(os.environ)
    parts = [str(paths.REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


@pytest.fixture(scope="session")
def repo_root():
    return paths.REPO_ROOT


@pytest.fixture(scope="session")
def reference_dir():
    return paths.REFERENCE_DIR


@pytest.fixture(scope="session")
def tx123bt():
    return systems.get_system("tx123bt")


@pytest.fixture(scope="session")
def texas2k():
    return systems.get_system("texas2k")
