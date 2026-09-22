"""Every script shows its help text, and the module-level settings follow the active system."""

from __future__ import annotations

import subprocess
import sys

import pytest

from pisasr import evaluation, paths, systems, uc_model

pytestmark = pytest.mark.usefixtures("restore_active_system")

SCRIPTS = [
    "run_full_model.py",
    "run_pisasr.py",
    "evaluate.py",
    "data/download_tx123bt.py",
    "data/prepare_tx123bt.py",
    "data/prepare_texas2k.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help(script, script_env):
    proc = subprocess.run([sys.executable, str(paths.REPO_ROOT / "scripts" / script), "--help"],
                          capture_output=True, text=True, env=script_env)
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout
    assert str(paths.REPO_ROOT) not in proc.stdout   # no machine-specific default paths


def test_run_full_model_needs_a_task(script_env):
    proc = subprocess.run([sys.executable, str(paths.REPO_ROOT / "scripts" / "run_full_model.py"),
                           "--system", "tx123bt"], capture_output=True, text=True, env=script_env)
    assert proc.returncode != 0
    assert "nothing to do" in proc.stderr


def test_relative_output_dir_is_made_absolute(tmp_path, monkeypatch):
    """A relative --output-dir is taken relative to the working directory, once."""
    from pisasr.pisasr import _resolve_output_folder

    monkeypatch.chdir(tmp_path)
    system = systems.get_system("tx123bt")
    root = system.output_root("rel")
    assert root.is_absolute() and root == tmp_path.resolve() / "rel" / "tx123bt"
    systems.activate(system, root)
    run_folder = systems.run_dir("pisasr_S20", root)
    assert _resolve_output_folder(run_folder) == run_folder
    assert _resolve_output_folder("pisasr_S20") == run_folder
    systems.activate(system, "rel/tx123bt")
    assert uc_model.OUTPUT_ROOT == root and evaluation.RESULTS_ROOT == root


def test_evaluate_unknown_run_fails_cleanly(tmp_path, script_env):
    proc = subprocess.run([sys.executable, str(paths.REPO_ROOT / "scripts" / "evaluate.py"),
                           "--system", "tx123bt", "--runs", "pisasr_S999", "--output-dir", str(tmp_path)],
                          capture_output=True, text=True, env=script_env, cwd=tmp_path)
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "pisasr_S999" in proc.stderr and "pisasr_S1000" in proc.stderr
    assert not list(tmp_path.rglob("evaluation*.csv"))


def test_download_missing_archive_fails_cleanly(tmp_path, script_env):
    proc = subprocess.run([sys.executable, str(paths.REPO_ROOT / "scripts" / "data" / "download_tx123bt.py"),
                           "--zip", str(tmp_path / "missing.zip"), "--dest", str(tmp_path / "raw")],
                          capture_output=True, text=True, env=script_env)
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr and "archive not found" in proc.stderr


def test_run_names_share_the_labels_of_the_summaries():
    assert evaluation.parse_model_case("deterministic") == {"model": "deterministic", "S": 1}
    assert evaluation.parse_model_case("full_S1000") == {"model": "full", "S": 1000}
    assert evaluation.parse_model_case("pisasr_S1000") == {"model": "pisasr", "S": 1000}
    assert evaluation.parse_model_case("pisasr_S1000_k120") == {"model": "pisasr", "S": 1000}
    assert systems.parse_run_name("full_S2500") == ("full", 2500)


def test_activate_sets_module_settings(tmp_path):
    system = systems.activate("texas2k", tmp_path / "texas2k")
    assert uc_model.SCENARIO_FOLDER == paths.DATA_DIR / "texas2k"
    assert uc_model.GENERATOR_FILE == paths.DATA_DIR / "texas2k" / "generator_portfolio.csv"
    assert uc_model.OUTPUT_ROOT == tmp_path / "texas2k"
    assert evaluation.SCENARIO_FILE == paths.DATA_DIR / "texas2k" / "net_load_scenarios_test.csv"
    assert evaluation.GENERATOR_FILE == uc_model.GENERATOR_FILE
    assert evaluation.RESULTS_ROOT == tmp_path / "texas2k"
    assert evaluation.EVALUATION_SCENARIOS == 2000
    assert systems.active_system() is system
    assert systems.run_dir("pisasr_S100") == tmp_path / "texas2k" / "runs" / "pisasr_S100"
    assert len(evaluation.load_net_load_scenarios()) == 2000


def test_reference_resolution(tmp_path):
    systems.activate("tx123bt", tmp_path / "tx123bt")
    ref = systems.resolve_reference_commitment("results")
    assert ref == paths.REFERENCE_DIR / "tx123bt" / "runs" / "deterministic" / "commitment.csv"
    with pytest.raises(FileNotFoundError):
        systems.resolve_reference_commitment("outputs")
    value, source = systems.resolve_full_objective(1000, verbose=False)
    assert source == "results" and value == pytest.approx(38108031.006481566, rel=1e-12)
    value, source = systems.resolve_full_objective(20, verbose=False)
    assert source == "none" and value != value
