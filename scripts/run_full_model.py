#!/usr/bin/env python
"""Solve the deterministic (expected-value) UC and the full (extensive-form) model.

Examples
--------
Deterministic UC and the full model at every scenario count of the paper::

    python scripts/run_full_model.py --system tx123bt --deterministic --S 100 200 500 1000 2500

Results are written to ``<output-dir>/<system>/runs/{deterministic,full_S<S>}/``
(default ``outputs/``). A run whose ``solve_metrics.csv`` exists is skipped
unless ``--force`` is given. Every model needs Gurobi; the Texas2k full models
take hours (see docs/REPRODUCIBILITY.md).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from pisasr import baselines, paths, systems
from pisasr import uc_model as r


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the deterministic (expected-value) UC, the full two-stage "
            "stochastic UC over S training scenarios, or both, for one test system."
        ),
        epilog=(
            "Timing: solve_time_seconds follows the paper (TX-123BT: solver call "
            "only; Texas2k: model construction and solver call; Texas2k at S=2,500: "
            "construction, model-file writing and solver call). build_time_seconds "
            "and solver_time_seconds are recorded separately."
        ),
    )
    parser.add_argument("--system", required=True, choices=sorted(systems.SYSTEMS),
                        help="test system")
    parser.add_argument("--S", type=int, nargs="+", default=[], metavar="S",
                        help="scenario counts of the full model (for example 100 200 500 1000 2500)")
    parser.add_argument("--deterministic", action="store_true",
                        help="solve the deterministic (expected-value) UC")
    parser.add_argument("--output-dir", type=Path, default=paths.OUTPUT_DIR,
                        help="top-level output folder (default: outputs/ in the repository, or $PISASR_OUTPUT_DIR); runs go to "
                             "<output-dir>/<system>/runs/")
    parser.add_argument("--force", action="store_true",
                        help="solve again even if the run folder already holds a result")
    parser.add_argument("--file-based", choices=["auto", "always", "never"], default="auto",
                        help="solve the full model through an LP file to limit memory use; "
                             "'auto' (default) does so only for Texas2k at S>=2500, as in the paper")
    parser.add_argument("--model-file-dir", type=Path, default=None,
                        help="folder for the temporary LP file of the solution from a file "
                             "(default: the system temporary folder)")
    args = parser.parse_args(argv)
    if not args.deterministic and not args.S:
        parser.error("nothing to do: give --deterministic, --S or both")
    return args


def _compare_with_reference(system, run_name: str, metrics: dict | None) -> None:
    ref = systems.reference_run_dir(run_name, system) / "solve_metrics.csv"
    if metrics is None or not ref.exists():
        return
    ref_obj = float(pd.read_csv(ref)["objective"].iloc[0])
    rel = 100.0 * (metrics["objective"] - ref_obj) / ref_obj
    print(f"  reference objective {ref_obj:,.2f} ({paths.display_path(ref)}); "
          f"difference {rel:+.4f}% (MILP tolerance 0.1%)")


def main(argv=None) -> int:
    args = parse_args(argv)
    system = systems.get_system(args.system)
    root = system.output_root(args.output_dir)
    systems.activate(system, root)

    print(f"System: {system.display_name}")
    print(f"Reading generators: {paths.display_path(r.GENERATOR_FILE)}")
    print(f"Reading training set: {paths.display_path(system.train_file)}")
    print(f"Writing runs to: {paths.display_path(root / 'runs')}")
    gen = r.load_generators()
    net = r.load_net_load_scenarios()
    print(f"{len(gen)} committable units, {gen['p_max_mw'].sum() / 1000:.1f} GW; "
          f"{len(net)} training scenarios")

    t_all = time.time()
    if args.deterministic:
        run_folder = systems.run_dir("deterministic", root)
        metrics = baselines.run_deterministic(system, gen, net, run_folder, force=args.force)
        _compare_with_reference(system, "deterministic", metrics)

    for S in args.S:
        run_folder = systems.run_dir(f"full_S{S}", root)
        if baselines.use_file_based_path(system, S, args.file_based):
            metrics = baselines.run_full_file_based(
                system, gen, net, S, run_folder,
                model_file_dir=args.model_file_dir, force=args.force)
        else:
            metrics = baselines.run_full(system, gen, net, S, run_folder, force=args.force)
        _compare_with_reference(system, f"full_S{S}", metrics)

    print(f"\nDone in {time.time() - t_all:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(paths.run_script(main))
