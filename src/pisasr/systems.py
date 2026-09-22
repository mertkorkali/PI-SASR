"""The two test systems and the selection of the active system.

``SYSTEMS`` describes TX-123BT and Texas2k Series25: input files, reference
results and the default output folder. :func:`activate` points the module-level
settings of :mod:`pisasr.uc_model` and :mod:`pisasr.evaluation` at one system;
every script calls it in its ``main`` function before any computation.

Folder layout (identical for the reference results and for reruns)::

    <root>/runs/deterministic/          deterministic (expected-value) UC
    <root>/runs/full_S<S>/              full (extensive-form) model over S scenarios
    <root>/runs/pisasr_S<S>/            PI-SASR at scenario count S

where ``<root>`` is ``results/<system>`` for the reference results and
``outputs/<system>`` (or ``<--output-dir>/<system>``) for reruns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import paths


@dataclass(frozen=True)
class System:
    """One test system.

    Attributes
    ----------
    key
        Short name used in folder names and on the command line.
    display_name
        Name used in the paper.
    data_dir
        Folder of the processed inputs (``data/<key>``).
    n_eval
        Size of the evaluation set (first rows of the test file).
    full_timing_basis
        Definition of ``solve_time_seconds`` for the deterministic UC and the
        full model, as in the paper: ``"solver"`` (solver call only) for
        TX-123BT and ``"build+solver"`` (model construction and solver call)
        for Texas2k.
    """

    key: str
    display_name: str
    data_dir: Path
    n_eval: int = 2000
    full_timing_basis: str = "solver"

    @property
    def generator_file(self) -> Path:
        return self.data_dir / "generator_portfolio.csv"

    @property
    def train_file(self) -> Path:
        return self.data_dir / "net_load_scenarios_train.csv"

    @property
    def test_file(self) -> Path:
        return self.data_dir / "net_load_scenarios_test.csv"

    @property
    def reference_dir(self) -> Path:
        """Reference results of the paper (``results/<key>``)."""
        return paths.REFERENCE_DIR / self.key

    @property
    def output_dir(self) -> Path:
        """Default output folder of reruns (``outputs/<key>``)."""
        return paths.OUTPUT_DIR / self.key

    def output_root(self, output_dir: str | Path | None = None) -> Path:
        """System folder below a top-level output folder (default ``outputs/``).

        The result is always absolute: a relative ``output_dir`` is taken
        relative to the working directory.
        """
        base = paths.OUTPUT_DIR if output_dir is None else paths.absolute_path(output_dir)
        return base / self.key


SYSTEMS: dict[str, System] = {
    "tx123bt": System(
        key="tx123bt",
        display_name="TX-123BT",
        data_dir=paths.DATA_DIR / "tx123bt",
        full_timing_basis="solver",
    ),
    "texas2k": System(
        key="texas2k",
        display_name="Texas2k Series25",
        data_dir=paths.DATA_DIR / "texas2k",
        full_timing_basis="build+solver",
    ),
}

_ACTIVE: System = SYSTEMS["tx123bt"]
_ACTIVE_ROOT: Path = SYSTEMS["tx123bt"].output_dir


_RUN_NAME = re.compile(r"(full|pisasr)_S(\d+)(?:_.+)?")
_LEGACY_FULL_NAME = re.compile(r"stochastic_(\d+)_scenarios")


def parse_run_name(run: str) -> tuple[str, int | None]:
    """Return ``(model, S)`` for a run-folder name.

    ``deterministic`` gives ``("deterministic", 1)`` (a single net-load
    profile); ``full_S<S>`` gives ``("full", S)``; ``pisasr_S<S>``, or
    ``pisasr_S<S>_<tag>`` for a run with other settings, gives ``("pisasr", S)``.
    ``stochastic_<S>_scenarios``, the name of a full-model folder in the
    original code, also gives ``("full", S)``. Any other name is returned as the
    model with ``S = None``.
    """
    if run == "deterministic":
        return "deterministic", 1
    m = _RUN_NAME.fullmatch(run)
    if m:
        return m.group(1), int(m.group(2))
    m = _LEGACY_FULL_NAME.fullmatch(run)
    if m:
        return "full", int(m.group(1))
    return run, None


def get_system(system: str | System) -> System:
    """Return the :class:`System` for a key such as ``"tx123bt"``."""
    if isinstance(system, System):
        return system
    try:
        return SYSTEMS[str(system).lower()]
    except KeyError:
        raise ValueError(f"Unknown system {system!r}; choose from {sorted(SYSTEMS)}") from None


def activate(system: str | System, results_dir: str | Path | None = None) -> System:
    """Point :mod:`pisasr.uc_model` and :mod:`pisasr.evaluation` at one system.

    Parameters
    ----------
    system
        ``"tx123bt"``, ``"texas2k"`` or a :class:`System`.
    results_dir
        System output folder (the ``<root>`` of the module docstring). Defaults
        to ``outputs/<key>``; a relative path is taken relative to the working
        directory.

    Returns
    -------
    System
        The activated system.
    """
    from . import evaluation, uc_model

    global _ACTIVE, _ACTIVE_ROOT
    paths.check_layout()
    sys_ = get_system(system)
    root = sys_.output_dir if results_dir is None else paths.absolute_path(results_dir)

    uc_model.SCENARIO_FOLDER = sys_.data_dir
    uc_model.GENERATOR_FILE = sys_.generator_file
    uc_model.OUTPUT_ROOT = root

    evaluation.SCENARIO_FILE = sys_.test_file
    evaluation.GENERATOR_FILE = sys_.generator_file
    evaluation.RESULTS_ROOT = root
    evaluation.EVALUATION_SCENARIOS = sys_.n_eval

    _ACTIVE = sys_
    _ACTIVE_ROOT = root
    return sys_


def active_system() -> System:
    """The system selected by the last call of :func:`activate` (TX-123BT by default)."""
    return _ACTIVE


def active_root() -> Path:
    """The output folder selected by the last call of :func:`activate`."""
    return _ACTIVE_ROOT


def run_dir(name: str, root: str | Path | None = None) -> Path:
    """Run folder ``<root>/runs/<name>`` (``root`` defaults to the active output folder)."""
    base = active_root() if root is None else Path(root)
    return base / "runs" / name


def reference_run_dir(name: str, system: str | System | None = None) -> Path:
    """Run folder of the reference results, ``results/<system>/runs/<name>``."""
    sys_ = active_system() if system is None else get_system(system)
    return sys_.reference_dir / "runs" / name


def resolve_reference_commitment(
    source: str = "results",
    system: str | System | None = None,
    root: str | Path | None = None,
) -> Path:
    """Location of the reference commitment u_ref (the deterministic UC schedule).

    ``source="results"`` (default) selects the stored schedule of the paper,
    ``results/<system>/runs/deterministic/commitment.csv``. The deterministic
    UC is a MILP solved to a 0.1% tolerance, so solving it again can return a
    different schedule of equal quality (on TX-123BT a new solution differs in 59
    of 3,312 unit-hours and has 65 instead of 64 baseload units); the stored
    schedule is therefore the one that reproduces the paper.
    ``source="outputs"`` selects ``<root>/runs/deterministic/commitment.csv``
    written by ``scripts/run_full_model.py --deterministic``.
    """
    if source == "results":
        path = reference_run_dir("deterministic", system) / "commitment.csv"
    elif source == "outputs":
        path = run_dir("deterministic", root) / "commitment.csv"
    else:
        raise ValueError("source must be 'results' or 'outputs'")
    if not path.exists():
        hint = (" Run scripts/run_full_model.py --deterministic first."
                if source == "outputs" else "")
        raise FileNotFoundError(f"Reference commitment not found: {path}.{hint}")
    return path


def resolve_full_objective(
    n_scenarios: int,
    system: str | System | None = None,
    root: str | Path | None = None,
    verbose: bool = True,
) -> tuple[float, str]:
    """Objective of the full model at S scenarios, used for the benchmark gap.

    The rerun in ``<root>/runs/full_S<S>`` is used when present, otherwise the
    reference result in ``results/<system>/runs/full_S<S>``.

    Returns
    -------
    (float, str)
        The objective (NaN when neither file exists) and its source
        (``"outputs"``, ``"results"`` or ``"none"``).
    """
    name = f"full_S{int(n_scenarios)}"
    candidates = [
        ("outputs", run_dir(name, root) / "solve_metrics.csv"),
        ("results", reference_run_dir(name, system) / "solve_metrics.csv"),
    ]
    for label, path in candidates:
        if path.exists():
            value = float(pd.read_csv(path)["objective"].iloc[0])
            if verbose:
                print(f"Full-model objective for the benchmark gap ({label}): "
                      f"{value:,.2f} from {paths.display_path(path)}")
            return value, label
    if verbose:
        print(f"No full-model objective found for S={n_scenarios}; "
              "the benchmark gap is set to NaN.")
    return float("nan"), "none"
