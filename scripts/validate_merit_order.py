#!/usr/bin/env python
"""Merit-order verification against the dispatch LP (Section 6.5 of the paper).

Examples
--------
The check of the paper for TX-123BT, written to ``outputs/`` and compared with
the reference file in ``results/``::

    python scripts/validate_merit_order.py --system tx123bt --check

Write the reference file itself (``results/tx123bt/summary/``)::

    python scripts/validate_merit_order.py --system tx123bt --output-dir results

The recourse cost of the first 70 training scenarios is computed under the
reference commitment u_ref (the deterministic UC schedule, by default the stored
``results/<system>/runs/deterministic/commitment.csv``) twice: with the
merit-order dispatch used by PI-SASR, and with a single-scenario dispatch LP that
keeps the ramp limits. Each scenario has its own unit outage in both. The table
``<output-dir>/<system>/summary/merit_order_validation.csv`` has the columns
``scenario``, ``merit_order``, ``lp`` and ``rel_err_pct`` (100 (merit order - LP) /
LP), and the script prints the mean and largest absolute error, the squared
Pearson correlation R^2 of the two costs (the value that the paper calls a
coefficient of determination) and the number of scenarios in which the merit
order does not exceed the LP. Gurobi solves one LP for each scenario (about 12 s
on TX-123BT and 46 s on Texas2k).

An existing file is not recomputed unless ``--force`` is given; its summary is
printed instead. With ``--check`` the table is also compared with
``results/<system>/summary/merit_order_validation.csv`` (relative tolerance 1e-9),
and the exit status is 1 when they differ.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pisasr import experiments as ex
from pisasr import paths, systems
from pisasr import uc_model as r

RTOL = 1e-9


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the merit-order recourse with the dispatch LP on the first training scenarios.",
    )
    parser.add_argument("--system", required=True, choices=sorted(systems.SYSTEMS), help="test system")
    parser.add_argument("--n-scenarios", type=int, default=ex.MERIT_ORDER_SCENARIOS,
                        help="first rows of the training set to compare (default: %(default)s)")
    parser.add_argument("--reference", choices=["results", "outputs"], default="results",
                        help="source of the reference commitment u_ref (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=paths.OUTPUT_DIR,
                        help="top-level output folder; use 'results' to write the reference file "
                             "(default: outputs/ in the repository, or $PISASR_OUTPUT_DIR)")
    parser.add_argument("--force", action="store_true", help="recompute an existing file")
    parser.add_argument("--check", action="store_true",
                        help="compare with results/<system>/summary/merit_order_validation.csv; "
                             "exit status 1 when they differ")
    return parser.parse_args(argv)


def compare_tables(new: pd.DataFrame, ref: pd.DataFrame, rtol: float = RTOL) -> list[str]:
    """Differences between two merit-order tables (empty when they agree)."""
    problems = []
    if list(new.columns) != list(ref.columns):
        return [f"columns differ: {list(new.columns)} vs {list(ref.columns)}"]
    if len(new) != len(ref):
        return [f"row counts differ: {len(new)} vs {len(ref)}"]
    if not (new["scenario"].astype(str).to_numpy() == ref["scenario"].astype(str).to_numpy()).all():
        problems.append("scenario order differs")
    for col in ("merit_order", "lp", "rel_err_pct"):
        a, b = new[col].to_numpy(float), ref[col].to_numpy(float)
        if not np.allclose(a, b, rtol=rtol, atol=0.0):
            worst = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
            problems.append(f"{col}: largest relative difference {worst:.3e}")
    return problems


def main(argv=None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore")
    system = systems.get_system(args.system)
    root = system.output_root(args.output_dir)
    systems.activate(system, root)
    out = root / "summary" / ex.MERIT_ORDER_FILE
    reference = systems.resolve_reference_commitment(args.reference, system, root)

    print(f"System: {system.display_name}; first {args.n_scenarios} training scenarios")
    print(f"Reading generators: {paths.display_path(r.GENERATOR_FILE)}")
    print(f"Reading training set: {paths.display_path(system.train_file)}")
    print(f"Reading reference commitment u_ref ({args.reference}): {paths.display_path(reference)}")

    if out.exists() and not args.force:
        print(f"{paths.display_path(out)} exists; not recomputed (use --force to recompute)")
        table = pd.read_csv(out)
    else:
        t0 = time.time()
        table = ex.merit_order_validation(args.n_scenarios, reference_commitment=reference)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"Wrote {paths.display_path(out)} ({len(table)} scenarios, {time.time() - t0:,.1f} s)")

    s = ex.merit_order_summary(table)
    print(f"Mean |error| {s['mean_abs_err_pct']:.3f}%, largest |error| {s['max_abs_err_pct']:.3f}%, "
          f"R^2 {s['r2']:.5f}; merit order at or below the LP in {s['n_not_above_lp']} of {s['n']} scenarios")

    if not args.check:
        return 0
    ref_file = system.reference_dir / "summary" / ex.MERIT_ORDER_FILE
    if not ref_file.exists():
        print(f"Check: {paths.display_path(ref_file)} does not exist")
        return 1
    problems = compare_tables(table, pd.read_csv(ref_file))
    print(f"Check against {paths.display_path(ref_file)}: "
          + ("identical within a relative tolerance of 1e-9" if not problems else f"{len(problems)} differences"))
    for line in problems:
        print(f"    {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(paths.run_script(main))
