#!/usr/bin/env python
"""Run PI-SASR at one or more scenario counts.

Examples
--------
PI-SASR at S = 1,000 on TX-123BT with the settings of the paper::

    python scripts/run_pisasr.py --system tx123bt --S 1000

Each run writes ``commitment.csv``, ``solve_metrics.csv`` and ``pisasr_log.csv`` to
``<output-dir>/<system>/runs/pisasr_S<S>/`` (default ``outputs/``); an existing
run is skipped unless ``--force`` is given.

Reference commitment. PI-SASR learns scenario criticality under the reference
commitment u_ref, the deterministic UC schedule. By default the stored schedule
``results/<system>/runs/deterministic/commitment.csv`` is used, which reproduces
the paper exactly; ``--reference outputs`` uses the schedule solved by
``scripts/run_full_model.py --deterministic`` instead (a schedule obtained by
solving the MILP again agrees with the stored one only within the 0.1% MILP
tolerance).

Benchmark gap. The full-model objective is read from
``<output-dir>/<system>/runs/full_S<S>/solve_metrics.csv`` if present, else from
``results/<system>/runs/full_S<S>/``; the script prints which one is used.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from pisasr import paths, systems
from pisasr import uc_model as r
from pisasr.constants import PISASR_DEFAULTS
from pisasr.pisasr import run_pisasr


def parse_args(argv=None) -> argparse.Namespace:
    d = PISASR_DEFAULTS
    parser = argparse.ArgumentParser(
        description="Run PI-SASR (physics-informed surrogate-assisted scenario reduction).",
    )
    parser.add_argument("--system", required=True, choices=sorted(systems.SYSTEMS),
                        help="test system")
    parser.add_argument("--S", type=int, nargs="+", required=True, metavar="S",
                        help="scenario counts (for example 100 200 500 1000 2500)")
    parser.add_argument("--seed", type=int, default=d["random_seed"],
                        help="random seed of the design set, of its split, of the GP restarts "
                             "and of k-means (default: %(default)s)")
    parser.add_argument("--reference", choices=["results", "outputs"], default="results",
                        help="source of the reference commitment u_ref (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=paths.OUTPUT_DIR,
                        help="top-level output folder (default: outputs/ in the repository, or $PISASR_OUTPUT_DIR)")
    parser.add_argument("--force", action="store_true",
                        help="run again even if the run folder already holds a result")
    parser.add_argument("--tag", default="",
                        help="suffix of the run-folder name (pisasr_S<S>_<tag>), for runs "
                             "with settings other than those of the paper")
    parser.add_argument("--k-critical", type=int, default=d["k_critical"],
                        help="number of critical scenarios K_c (default: %(default)s)")
    parser.add_argument("--k-cover", type=int, default=d["k_cover"],
                        help="number of coverage scenarios K_r (default: %(default)s)")
    parser.add_argument("--design-size", type=int, default=d["design_size"],
                        help="size of the design set (default: %(default)s)")
    parser.add_argument("--svd-rank", type=int, default=d["svd_rank"],
                        help="rank r of the low-rank embedding (default: %(default)s)")
    parser.add_argument("--enrichment-rounds", type=int, default=d["cg_rounds"],
                        help="enrichment budget in rounds (default: %(default)s)")
    parser.add_argument("--enrichment-add", type=int, default=d["cg_add"],
                        help="scenarios added in one enrichment round (default: %(default)s)")
    parser.add_argument("--tolerance", type=float, default=d["cg_tol"],
                        help="enrichment tolerance epsilon as a fraction (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    system = systems.get_system(args.system)
    root = system.output_root(args.output_dir)
    systems.activate(system, root)
    reference = systems.resolve_reference_commitment(args.reference, system, root)

    print(f"System: {system.display_name}")
    print(f"Reading generators: {paths.display_path(r.GENERATOR_FILE)}")
    print(f"Reading training set: {paths.display_path(system.train_file)}")
    print(f"Reading reference commitment u_ref ({args.reference}): {paths.display_path(reference)}")

    rows = []
    for S in args.S:
        name = f"pisasr_S{S}" + (f"_{args.tag}" if args.tag else "")
        run_folder = systems.run_dir(name, root)
        if (run_folder / "solve_metrics.csv").exists() and not args.force:
            print(f"\n{paths.display_path(run_folder)} already holds solve_metrics.csv; "
                  "skipping (use --force to run again)")
            continue
        print(f"\n=== {system.display_name}: PI-SASR, S={S} ===")
        print(f"Writing: {paths.display_path(run_folder)}")
        t = time.time()
        res = run_pisasr(
            n_target=S,
            k_critical=args.k_critical,
            k_cover=args.k_cover,
            design_size=args.design_size,
            cg_rounds=args.enrichment_rounds,
            cg_add=args.enrichment_add,
            svd_rank=args.svd_rank,
            random_seed=args.seed,
            cg_tol=args.tolerance,
            output_folder=run_folder,
            reference_commitment=reference,
        )
        print(f"PI-SASR S={S}: K={res['n_scenarios_used']}, "
              f"baseload units={res['n_fixed_baseload_units']}, "
              f"GP R^2={res['surrogate_r2']:.4f}, "
              f"objective={res['objective']:,.2f}, "
              f"benchmark gap={res['opt_gap_pct_vs_full']:+.5f}%, "
              f"solution time={res['solve_time_seconds']:.1f} s "
              f"(exact recomputation {res['exact_validation_s']:.1f} s; "
              f"wall {time.time() - t:.1f} s)")
        rows.append({"S": S, **res})

    if rows:
        cols = ["S", "n_scenarios_used", "objective", "opt_gap_pct_vs_full",
                "solve_time_seconds", "surrogate_r2"]
        print("\n" + pd.DataFrame(rows)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(paths.run_script(main))
