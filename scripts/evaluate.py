#!/usr/bin/env python
"""Evaluate commitments on the evaluation set with the exact ramp-coupled LP.

Examples
--------
Evaluate the three commitments of Table 2 on TX-123BT::

    python scripts/evaluate.py --system tx123bt --runs deterministic full_S1000 pisasr_S1000

For each run the commitment is read from ``<output-dir>/<system>/runs/<run>/`` if
it exists there, otherwise from the reference results ``results/<system>/runs/<run>/``
(``--source`` fixes the choice). The table with one row for each scenario is written to
``<output-dir>/<system>/runs/<run>/evaluation.csv``, and a one-row
``evaluation_summary.csv`` in the same folder holds its summary statistics (the
columns of ``summary/evaluation_metrics.csv``) and the wall-clock time of the
evaluation. An existing ``evaluation.csv`` is kept unless ``--force`` is given.

The evaluation set is the first 2,000 scenarios of
``data/<system>/net_load_scenarios_test.csv``, which is disjoint from the training
set. One commitment takes about 6 min on TX-123BT and 25 min on Texas2k.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pisasr import evaluation as ev
from pisasr import paths, systems
from pisasr.constants import EVALUATION_BATCH_SIZE, SCENARIO_COUNTS

STANDARD_RUNS = (["deterministic"]
                 + [f"full_S{S}" for S in SCENARIO_COUNTS]
                 + [f"pisasr_S{S}" for S in SCENARIO_COUNTS])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate commitments on the evaluation set (exact ramp-coupled LP "
                    "under the same N-1 outage draws for every commitment).",
    )
    parser.add_argument("--system", required=True, choices=sorted(systems.SYSTEMS),
                        help="test system")
    parser.add_argument("--runs", nargs="+", required=True, metavar="RUN",
                        help="run-folder names (deterministic, full_S1000, pisasr_S1000, ...); "
                             "'all' selects the 11 runs of the paper")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="number of evaluation scenarios, taken from the start of the "
                             "test file (default: 2000, as in the paper)")
    parser.add_argument("--source", choices=["auto", "outputs", "results"], default="auto",
                        help="where to read the commitments: 'auto' (default) reads the output "
                             "folder if it holds the run, otherwise the reference results")
    parser.add_argument("--output-dir", type=Path, default=paths.OUTPUT_DIR,
                        help="top-level output folder (default: outputs/ in the repository, "
                             "or $PISASR_OUTPUT_DIR)")
    parser.add_argument("--batch-size", type=int, default=EVALUATION_BATCH_SIZE,
                        help="scenarios in one LP (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="evaluate again even if evaluation.csv exists")
    return parser.parse_args(argv)


def _commitment_file(run: str, source: str, root: Path, system) -> Path:
    candidates = {
        "outputs": systems.run_dir(run, root) / "commitment.csv",
        "results": systems.reference_run_dir(run, system) / "commitment.csv",
    }
    order = ["outputs", "results"] if source == "auto" else [source]
    for label in order:
        if candidates[label].exists():
            return candidates[label]
    raise FileNotFoundError(
        f"No commitment.csv for run {run!r} in "
        + " or ".join(paths.display_path(candidates[k]) for k in order)
    )


def _available_runs(source: str, root: Path, system) -> list[str]:
    """Run folders that hold a commitment.csv in the folders searched by ``source``."""
    folders = []
    if source in ("auto", "outputs"):
        folders.append(root / "runs")
    if source in ("auto", "results"):
        folders.append(system.reference_dir / "runs")
    names = {p.parent.name for f in folders if f.is_dir() for p in f.glob("*/commitment.csv")}
    return sorted(names, key=lambda n: (systems.parse_run_name(n)[0], systems.parse_run_name(n)[1] or 0, n))


def main(argv=None) -> int:
    args = parse_args(argv)
    system = systems.get_system(args.system)
    root = system.output_root(args.output_dir)
    systems.activate(system, root)
    if args.n_eval is not None:
        ev.EVALUATION_SCENARIOS = args.n_eval

    runs = STANDARD_RUNS if args.runs == ["all"] else args.runs
    print(f"System: {system.display_name}")
    print(f"Reading generators: {paths.display_path(ev.GENERATOR_FILE)}")
    print(f"Reading evaluation set: first {ev.EVALUATION_SCENARIOS} rows of "
          f"{paths.display_path(ev.SCENARIO_FILE)}")
    gen = ev.load_generators()
    test = ev.load_net_load_scenarios()

    # Find every commitment before the first (long) evaluation starts.
    todo = []
    for run in runs:
        out_folder = systems.run_dir(run, root)
        out_file = out_folder / "evaluation.csv"
        if out_file.exists() and not args.force:
            print(f"\n{paths.display_path(out_file)} exists; skipping (use --force)")
            continue
        try:
            todo.append((run, out_folder, _commitment_file(run, args.source, root, system)))
        except FileNotFoundError as exc:
            available = _available_runs(args.source, root, system)
            sys.stdout.flush()
            print(f"\nError: {exc}.", file=sys.stderr)
            print("Runs with a commitment: " + (", ".join(available) if available else "none"),
                  file=sys.stderr)
            return 1

    rows = []
    for run, out_folder, commitment_file in todo:
        out_file = out_folder / "evaluation.csv"
        print(f"\n=== {system.display_name}: evaluating {run} ===")
        print(f"Reading commitment: {paths.display_path(commitment_file)}")
        metrics = ev.evaluate_one_commitment(
            commitment_file, gen, test, scenario_output=out_file, batch_size=args.batch_size)
        metrics = {"run": run, **metrics}
        pd.DataFrame([metrics]).reindex(columns=ev.RUN_SUMMARY_COLUMNS).to_csv(
            out_folder / "evaluation_summary.csv", index=False)
        print(f"Wrote: {paths.display_path(out_file)} and evaluation_summary.csv")
        print(f"{run}: expected total cost {metrics['expected_total_cost']:,.2f}, "
              f"unserved energy {100 * metrics['unserved_energy_rate']:.4f}% of net load, "
              f"{metrics['evaluation_time_seconds']:.0f} s")

        ref = systems.reference_run_dir(run, system) / "evaluation.csv"
        if ref.exists():
            ref_df = pd.read_csv(ref).iloc[:metrics["n_evaluation_scenarios"]]
            new_df = pd.read_csv(out_file)
            if list(ref_df["scenario"]) == list(new_df["scenario"]):
                rel = ((new_df["total_cost_with_fixed"] - ref_df["total_cost_with_fixed"]).abs()
                       / ref_df["total_cost_with_fixed"].abs()).max()
                print(f"  largest relative difference from {paths.display_path(ref)}: {rel:.2e}")
        rows.append(metrics)

    if rows:
        cols = ["run", "expected_total_cost", "unserved_energy_rate", "evaluation_time_seconds"]
        print("\n" + pd.DataFrame(rows)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(paths.run_script(main))
