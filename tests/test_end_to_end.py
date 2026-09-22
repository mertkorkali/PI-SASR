"""Short end-to-end check on TX-123BT at S = 20 (full model and PI-SASR, about 1 min with Gurobi).

The scripts are run as a user would run them, from a temporary working folder
and with a relative ``--output-dir``. At S = 20 the active set contains all 20
scenarios (K = 20), and the benchmark gap is about -0.04%: PI-SASR's baseload
pre-commitment and the 0.1% MILP tolerance of both solutions allow either sign.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from pisasr import paths

SCRIPTS = paths.REPO_ROOT / "scripts"
OUT = "out"   # relative to the working folder of the scripts


def _run(script, *args, cwd, env):
    """Run a script in ``cwd`` (a temporary folder, which also receives any solver files)."""
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


@pytest.mark.gurobi
def test_end_to_end_tx123bt_S20(tmp_path, script_env):
    out = _run("run_full_model.py", "--system", "tx123bt", "--S", 20,
               "--output-dir", OUT, cwd=tmp_path, env=script_env)
    assert "full S=20" in out
    full = pd.read_csv(tmp_path / OUT / "tx123bt" / "runs" / "full_S20" / "solve_metrics.csv").iloc[0]
    assert full["termination"] == "optimal"
    for col in ("build_time_seconds", "solver_time_seconds", "solve_time_seconds"):
        assert full[col] > 0
    assert full["solve_time_seconds"] == full["solver_time_seconds"]   # TX-123BT timing basis

    out = _run("run_pisasr.py", "--system", "tx123bt", "--S", 20,
               "--output-dir", OUT, cwd=tmp_path, env=script_env)
    assert "(outputs)" in out   # the benchmark gap uses the full model just solved
    run = tmp_path / OUT / "tx123bt" / "runs" / "pisasr_S20"
    res = pd.read_csv(run / "solve_metrics.csv").iloc[0]
    assert int(res["n_scenarios_used"]) == 20
    assert int(res["n_fixed_baseload_units"]) == 64
    assert res["full_objective_source"] == "outputs"
    assert res["reference_source"] == "results"
    assert res["full_obj"] == pytest.approx(float(full["objective"]), rel=1e-12)
    assert abs(float(res["opt_gap_pct_vs_full"])) < 0.5
    assert (run / "commitment.csv").exists() and (run / "pisasr_log.csv").exists()
    # Both runs lie directly in the folder named on the command line (not below a repeated path).
    assert sorted(p.name for p in (tmp_path / OUT / "tx123bt" / "runs").iterdir()) == ["full_S20", "pisasr_S20"]

    # A second call skips the existing run.
    out = _run("run_pisasr.py", "--system", "tx123bt", "--S", 20,
               "--output-dir", OUT, cwd=tmp_path, env=script_env)
    assert "skipping" in out
