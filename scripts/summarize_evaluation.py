#!/usr/bin/env python
"""Summarize the evaluation-set results of every run of a system.

Examples
--------
Rebuild the two summaries of the reference results::

    python scripts/summarize_evaluation.py

Summarize a rerun in ``outputs/`` and check a reference summary without writing::

    python scripts/summarize_evaluation.py --results-dir outputs
    python scripts/summarize_evaluation.py --system texas2k --check

For each run folder ``<results-dir>/<system>/runs/<run>/`` that contains an
``evaluation.csv`` (one row for each of the 2,000 evaluation scenarios), one row
of ``<results-dir>/<system>/summary/evaluation_metrics.csv`` is written: expected,
95th-percentile and worst total cost, expected dispatch cost, the fixed
commitment cost, unserved energy and spilled energy, and the unserved-energy rate
(a fraction). The commitment columns (no-load, startup and shutdown costs,
startups and committed unit-hours) are recomputed from ``commitment.csv`` and
``data/<system>/generator_portfolio.csv``. The run also compares the evaluation
columns of ``summary/scaling.csv``, where present, with the recomputed values.
No solver is needed.

A system without any ``evaluation.csv`` is skipped. The exit status is 1 when no
system could be summarized, when ``summary/scaling.csv`` disagrees with the
recomputed values, or, with ``--check``, when a stored summary differs or is
missing; otherwise it is 0.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from pisasr import reporting as rp
from pisasr.paths import DATA_DIR, REFERENCE_DIR, check_layout, display_path, run_script


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize evaluation.csv of every run into evaluation_metrics.csv.")
    p.add_argument("--system", choices=[*rp.SYSTEM_KEYS, "all"], default="all", help="test system (default: all)")
    p.add_argument("--results-dir", default=str(REFERENCE_DIR),
                   help="results folder holding <system>/runs (default: the reference results in results/)")
    p.add_argument("--data-dir", default=str(DATA_DIR), help="processed input data (default: data/)")
    p.add_argument("--check", action="store_true",
                   help="compare with the existing evaluation_metrics.csv (relative tolerance 1e-9) "
                        "instead of writing it; the exit status is 1 when they differ or the file "
                        "is missing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = rp.resolve_dir(args.results_dir)
    data_dir = rp.resolve_dir(args.data_dir)
    check_layout([d for d in (results_dir, data_dir) if d in (REFERENCE_DIR, DATA_DIR)])
    systems = rp.SYSTEM_KEYS if args.system == "all" else (args.system,)
    status = 0
    done = 0
    for system in systems:
        summary = rp.summarize_system(results_dir, system, data_dir)
        if summary.empty:
            print(f"[{system}] skipped: no runs with evaluation.csv under "
                  f"{display_path(results_dir / system / 'runs')}")
            continue
        done += 1
        out = results_dir / system / "summary" / "evaluation_metrics.csv"
        print(f"[{system}] {len(summary)} runs: {', '.join(summary['run'])}")
        scaling_file = results_dir / system / "summary" / "scaling.csv"
        if scaling_file.is_file():
            problems = rp.check_scaling_evaluation(summary, pd.read_csv(scaling_file))
            print(f"[{system}] evaluation columns of summary/scaling.csv: "
                  + ("agree with the recomputed values" if not problems else f"{len(problems)} differences"))
            for line in problems:
                print(f"    {line}")
            status |= bool(problems)
        if args.check:
            if not out.is_file():
                print(f"[{system}] {display_path(out)} does not exist")
                status = 1
                continue
            problems = rp.compare_frames(summary, pd.read_csv(out))
            print(f"[{system}] {out.name}: " + ("identical within 1e-9" if not problems else f"{len(problems)} differences"))
            for line in problems:
                print(f"    {line}")
            status |= bool(problems)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out, index=False)
            print(f"[{system}] wrote {display_path(out)}")
        cols = ["run", "expected_total_cost", "unserved_energy_rate"]
        view = summary[cols].assign(unserved_energy_pct=100 * summary["unserved_energy_rate"])
        print(view.drop(columns="unserved_energy_rate").to_string(
            index=False, formatters={"expected_total_cost": "{:,.0f}".format, "unserved_energy_pct": "{:.6f}".format}))
    if done == 0:
        print("No system could be summarized.")
        return 1
    return status


if __name__ == "__main__":
    sys.exit(run_script(main))
