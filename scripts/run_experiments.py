#!/usr/bin/env python
"""Run the additional experiments of the paper (Sections 4, 6 and 7.3, Table 4).

Examples
--------
Component analysis of Table 4 (TX-123BT, S = 1,000)::

    python scripts/run_experiments.py --system tx123bt --stage component-timing

Budget sensitivity of Figure 3(b) at the scenario counts of the paper::

    python scripts/run_experiments.py --system texas2k --stage budget-sensitivity

Certified gaps from the stored lower-bound batches (no MILP)::

    python scripts/run_experiments.py --system tx123bt --stage lower-bound \\
        --batches-file results/tx123bt/experiments/lower_bound_batches.csv

Experiments and files (written to ``<output-dir>/<system>/experiments/``, default
``outputs/``):

========================  =========================================  ================
``--stage``               file                                       solver
========================  =========================================  ================
``component-timing``      ``component_timing_S<S>.csv``              MILP, LP
``criticality-only``      ``criticality_only_S<S>.csv``              MILP, LP
``gp-kernels``            ``gp_kernels_S<S>.csv``                    none
``embedding``             ``embedding_spectrum_S<S>.csv`` and        none; MILP, LP
                          ``embedding_space_S<S>.csv``
``scenario-weights``      ``scenario_weights_S<S>.csv``              MILP, LP
``weights-s2500``         ``scenario_weights_S2500.csv``             MILP, LP
``lower-bound``           ``lower_bound_batches.csv`` and            MILP (LP)
                          ``certified_gap.csv``
``budget-sensitivity``    ``budget_sensitivity.csv``                 MILP, LP
========================  =========================================  ================

Without ``--S`` each experiment uses the scenario counts of the paper: S = 1,000
(2,500 for ``weights-s2500``); batch sizes S' = 200 and 500 on TX-123BT and 200 on
Texas2k for ``lower-bound``; S = 200, 500, 1,000, 2,500 on TX-123BT and 500, 1,000,
2,500 on Texas2k for ``budget-sensitivity``.

Rows already present in a file are skipped, so an experiment can be resumed after an
interruption; ``--force`` computes them again and replaces them. The certified-gap
file is always recomputed from the batch file. PI-SASR uses the stored reference
commitment ``results/<system>/runs/deterministic/commitment.csv`` unless
``--reference outputs`` is given, and the benchmark gap uses the full-model objective
of ``<output-dir>/<system>/runs/full_S<S>`` if present, else of ``results/``.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from pisasr import experiments as ex
from pisasr import paths, systems
from pisasr import uc_model as r


def _budget(text: str) -> tuple[int, int]:
    try:
        kc, kr = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"budget {text!r} is not of the form K_c,K_r (for example 90,60)")
    return kc, kr


VARIANT_CHOICES = {
    "component-timing": ex.COMPONENT_VARIANTS,
    "criticality-only": ex.CRITICALITY_ONLY_CHOICES,
    "gp-kernels": ex.KERNELS,
    "embedding": ["svd5", "raw24", "svd2", "svd10"],
    "scenario-weights": ex.WEIGHT_SCHEMES,
    "weights-s2500": ex.WEIGHT_SCHEMES,
}
"""Names accepted by --variants for each value of --stage (the other values take none)."""

TX123BT_ONLY_STAGES = ("component-timing",)
"""Experiments (values of --stage) that the paper reports for TX-123BT only."""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one of the additional experiments of the paper (chosen with --stage).",
        epilog="See the module docstring (python -m pydoc scripts/run_experiments.py) or "
               "docs/RESULTS_GUIDE.md for the files and columns.",
    )
    parser.add_argument("--system", required=True, choices=sorted(systems.SYSTEMS), help="test system")
    parser.add_argument("--stage", required=True, choices=ex.STAGES, help="experiment to run")
    parser.add_argument("--S", type=int, nargs="+", metavar="S",
                        help="scenario counts (batch sizes S' for lower-bound); "
                             "default: the values of the paper")
    parser.add_argument("--variants", nargs="+", metavar="NAME",
                        help="subset of variants (component-timing: " + ", ".join(ex.COMPONENT_VARIANTS)
                             + "; criticality-only: " + ", ".join(ex.CRITICALITY_ONLY_CHOICES)
                             + "; gp-kernels: " + ", ".join(ex.KERNELS)
                             + "; embedding: svd5, raw24, svd2, svd10; scenario-weights: "
                             + ", ".join(ex.WEIGHT_SCHEMES) + ")")
    parser.add_argument("--budgets", type=_budget, nargs="+", metavar="KC,KR",
                        help="budgets (K_c, K_r) of budget-sensitivity (default: those of the paper)")
    parser.add_argument("--kernel-seeds", type=int, nargs="+", default=list(ex.KERNEL_SEEDS),
                        metavar="SEED", help="design-set seeds of gp-kernels (default: 7 17 27)")
    parser.add_argument("--spectrum-only", action="store_true",
                        help="embedding: write the spectrum only (no MILP)")
    parser.add_argument("--batches", type=int, default=ex.LOWER_BOUND_BATCHES, metavar="M",
                        help="lower-bound: number of batches of each size (default: %(default)s)")
    parser.add_argument("--S-ref", type=int, default=ex.LOWER_BOUND_S_REF,
                        help="lower-bound: scenario count of the certified commitments (default: %(default)s)")
    parser.add_argument("--batches-file", type=Path,
                        help="lower-bound: compute the certified gaps from this batch file without "
                             "solving (for example results/<system>/experiments/lower_bound_batches.csv)")
    parser.add_argument("--seed", type=int, default=ex.SEED,
                        help="random seed of PI-SASR (default: %(default)s)")
    parser.add_argument("--reference", choices=["results", "outputs"], default="results",
                        help="source of the reference commitment u_ref (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=paths.OUTPUT_DIR,
                        help="top-level output folder (default: outputs/ in the repository, or $PISASR_OUTPUT_DIR)")
    parser.add_argument("--force", action="store_true",
                        help="recompute rows that are already in the output file")
    args = parser.parse_args(argv)
    if args.variants:
        allowed = VARIANT_CHOICES.get(args.stage)
        if allowed is None:
            parser.error(f"--variants is not used by --stage {args.stage}")
        unknown = [name for name in args.variants if name not in allowed]
        if unknown:
            parser.error(f"unknown name(s) {', '.join(unknown)} for --stage {args.stage}; "
                         f"choose from {', '.join(allowed)}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    system = systems.get_system(args.system)
    root = system.output_root(args.output_dir)
    systems.activate(system, root)
    out = root / "experiments"
    reference = systems.resolve_reference_commitment(args.reference, system, root)

    print(f"System: {system.display_name}; stage: {args.stage}")
    print(f"Reading generators: {paths.display_path(r.GENERATOR_FILE)}")
    print(f"Reading training set: {paths.display_path(system.train_file)}")
    print(f"Reading reference commitment u_ref ({args.reference}): {paths.display_path(reference)}")
    print(f"Writing to: {paths.display_path(out)}")
    if args.stage in TX123BT_ONLY_STAGES and system.key != "tx123bt":
        print(f"Note: the paper reports the {args.stage} experiment for TX-123BT only.")

    t0 = time.time()
    written = ex.run_stage(
        args.stage, out,
        S_list=args.S,
        variants=args.variants,
        seed=args.seed,
        reference_commitment=reference,
        force=args.force,
        budgets=args.budgets,
        kernel_seeds=args.kernel_seeds,
        spectrum_only=args.spectrum_only,
        n_batches=args.batches,
        S_ref=args.S_ref,
        batches_file=args.batches_file,
    )
    print(f"\n{system.display_name} {args.stage}: done in {time.time() - t0:,.1f} s")
    for path in written:
        print(f"  {paths.display_path(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(paths.run_script(main))
