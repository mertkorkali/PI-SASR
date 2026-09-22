#!/usr/bin/env python
"""Five random draws of the design set, and repeated timings of the full model (Section 6.2).

Example
-------
The study of the paper (TX-123BT, S = 1,000; about 50 min)::

    python scripts/run_replication.py

PI-SASR is run at S with the settings of the paper and the random seeds 7, 17, 27,
37 and 47 of the design set (and of the split of the design set, the GP restarts
and k-means); seed 7 is the run used everywhere else in the paper. The full model at S is
then solved three times to measure the spread of its solution time (model
construction and solution). The rows are written to
``<output-dir>/<system>/experiments/design_set_replication_S<S>.csv`` (default
``outputs/``) with the columns ``kind`` (``pisasr`` or ``full``), ``seed`` (the
repetition number for ``full``), ``gap_pct``, ``time_s`` and ``K``. Rows already in
the file are skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from pisasr import experiments as ex
from pisasr import paths, systems
from pisasr import uc_model as r


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PI-SASR with several random draws of the design set, and repeated "
                    "full-model timings.",
    )
    parser.add_argument("--system", default="tx123bt", choices=sorted(systems.SYSTEMS),
                        help="test system (default: %(default)s, as in the paper)")
    parser.add_argument("--S", type=int, default=1000, help="scenario count (default: %(default)s)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ex.REPLICATION_SEEDS),
                        metavar="SEED", help="random seeds of the design set (default: 7 17 27 37 47)")
    parser.add_argument("--full-repeats", type=int, default=ex.FULL_TIMING_REPEATS,
                        help="number of full-model solutions to time (default: %(default)s; 0 to skip)")
    parser.add_argument("--reference", choices=["results", "outputs"], default="results",
                        help="source of the reference commitment u_ref (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=paths.OUTPUT_DIR,
                        help="top-level output folder (default: outputs/ in the repository, or $PISASR_OUTPUT_DIR)")
    parser.add_argument("--force", action="store_true",
                        help="recompute rows that are already in the output file")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    system = systems.get_system(args.system)
    root = system.output_root(args.output_dir)
    systems.activate(system, root)
    out = root / "experiments"
    reference = systems.resolve_reference_commitment(args.reference, system, root)

    print(f"System: {system.display_name}; S={args.S}")
    print(f"Reading generators: {paths.display_path(r.GENERATOR_FILE)}")
    print(f"Reading training set: {paths.display_path(system.train_file)}")
    print(f"Reading reference commitment u_ref ({args.reference}): {paths.display_path(reference)}")
    print(f"Writing to: {paths.display_path(out)}")
    if system.key != "tx123bt":
        print("Note: the paper reports the design-set draws for TX-123BT only.")

    t0 = time.time()
    path = ex.design_set_replication(out, S=args.S, seeds=args.seeds, full_repeats=args.full_repeats,
                                     reference_commitment=reference, force=args.force)
    print(f"\nDone in {time.time() - t0:,.1f} s: {paths.display_path(path)}")
    if path.exists():
        df = pd.read_csv(path)
        summary = df.groupby("kind").agg(n=("time_s", "size"), gap_mean=("gap_pct", "mean"),
                                         gap_sd=("gap_pct", "std"), time_mean=("time_s", "mean"),
                                         time_sd=("time_s", "std"))
        print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(paths.run_script(main))
