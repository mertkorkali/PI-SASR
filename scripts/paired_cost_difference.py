#!/usr/bin/env python
"""Paired evaluation-set cost difference between PI-SASR and the full model (Section 6.3).

Examples
--------
Reproduce the confidence intervals of the paper (S = 1,000, both systems)::

    python scripts/paired_cost_difference.py

Check the stored tables without writing, or use a rerun in ``outputs/``::

    python scripts/paired_cost_difference.py --check
    python scripts/paired_cost_difference.py --results-dir outputs --system tx123bt

The two commitments are evaluated on the same 2,000 evaluation scenarios under
the same outage draws, so their costs are paired scenario by scenario. For the
total cost (fixed commitment cost plus dispatch cost) and for the dispatch cost
alone, the script prints and writes the mean difference (PI-SASR minus full) and its
two-sided 95% Student-t interval, in dollars and as a percentage of the full
model's mean cost of the same kind. The result is written to
``<results-dir>/<system>/summary/paired_cost_difference_S<S>.csv``.

A system whose two ``evaluation.csv`` files are missing is skipped. The exit
status is 1 when no system could be compared or, with ``--check``, when a stored
table differs or is missing; otherwise it is 0.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from pisasr import reporting as rp
from pisasr.paths import REFERENCE_DIR, check_layout, display_path, run_script


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired cost difference, PI-SASR minus full model, on the evaluation set.")
    p.add_argument("--system", choices=[*rp.SYSTEM_KEYS, "all"], default="all", help="test system (default: all)")
    p.add_argument("--S", type=int, default=1000, help="scenario count of the two runs (default: 1000)")
    p.add_argument("--run-a", default=None, help="first run (default: pisasr_S<S>)")
    p.add_argument("--run-b", default=None, help="reference run (default: full_S<S>)")
    p.add_argument("--confidence", type=float, default=0.95, help="two-sided confidence level (default: 0.95)")
    p.add_argument("--results-dir", default=str(REFERENCE_DIR),
                   help="results folder holding <system>/runs (default: the reference results in results/)")
    p.add_argument("--check", action="store_true",
                   help="compare with the existing CSV (relative tolerance 1e-9) instead of writing it; "
                        "the exit status is 1 when they differ or the file is missing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = rp.resolve_dir(args.results_dir)
    check_layout([results_dir] if results_dir == REFERENCE_DIR else [])
    systems = rp.SYSTEM_KEYS if args.system == "all" else (args.system,)
    status = 0
    done = 0
    for system in systems:
        try:
            table = rp.paired_cost_difference(results_dir, system, args.S, args.confidence, args.run_a, args.run_b)
        except rp.MissingSource as exc:
            print(f"[{system}] skipped: missing {exc.relpath}")
            continue
        done += 1
        out = results_dir / system / "summary" / f"paired_cost_difference_S{args.S}.csv"
        for r in table.itertuples():
            print(f"[{system}] {r.quantity:13s} {r.run_a} - {r.run_b}: {r.mean_difference_pct:+.3f}% "
                  f"[{r.ci_low_pct:.3f}, {r.ci_high_pct:.3f}]  ({r.mean_difference:+,.0f} $ on "
                  f"{r.mean_b:,.0f} $, n = {r.n_pairs})")
        if args.check:
            if not out.is_file():
                print(f"[{system}] {display_path(out)} does not exist")
                status = 1
                continue
            problems = rp.compare_frames(table, pd.read_csv(out), key="quantity")
            print(f"[{system}] {out.name}: " + ("identical within 1e-9" if not problems else f"{len(problems)} differences"))
            for line in problems:
                print(f"    {line}")
            status |= bool(problems)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(out, index=False)
            print(f"[{system}] wrote {display_path(out)}")
    if done == 0:
        print("No system could be compared.")
        return 1
    return status


if __name__ == "__main__":
    sys.exit(run_script(main))
