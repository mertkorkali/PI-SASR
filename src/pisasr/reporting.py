"""Evaluation summaries, paper tables, paper figures and the register of paper numbers.

Every function in this module reads CSV files: the run folders, summaries and
experiment tables under ``results/`` (or a rerun under ``outputs/``) and the
processed inputs under ``data/``. No optimization solver is imported, so the
tables, figures and numbers of the paper can be rebuilt with pandas and
matplotlib alone.

Folder layout read here (``<root>`` is ``results/`` or ``outputs/``)::

    <root>/<system>/runs/<run>/commitment.csv, solve_metrics.csv, evaluation.csv, pisasr_log.csv
    <root>/<system>/summary/*.csv
    <root>/<system>/experiments/*.csv

where ``<system>`` is ``tx123bt`` or ``texas2k`` and ``<run>`` is ``deterministic``,
``full_S<S>`` or ``pisasr_S<S>``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import systems
from .constants import HOUR_COLS
from .paths import DATA_DIR, REFERENCE_DIR, REPO_ROOT
from .statistics import paired_difference_interval

SYSTEM_KEYS = ("tx123bt", "texas2k")
SHORT_NAMES = {"tx123bt": "TX-123BT", "texas2k": "Texas2k"}
LONG_NAMES = {"tx123bt": "TX-123BT", "texas2k": "Texas2k Series25"}
S_GRID = (100, 200, 500, 1000, 2500)
S_LABELS = ("100", "200", "500", "1k", "2.5k")
N_EVAL = 2000
ENS_THRESHOLD_MWH = 1e-4          # a scenario (or hour) sheds load when ENS exceeds this
TAU_LEVELS = (1.0, 0.5)           # benchmark-gap tolerances of Fig. 3(b), in percent
K_CRITICAL, K_COVER = 90, 60      # default active-set budget
ENRICHMENT_TOLERANCE_PCT = 0.2    # enrichment tolerance epsilon, in percent


class MissingSource(FileNotFoundError):
    """A CSV file (or a row of it) that a table, figure or number needs is absent."""

    def __init__(self, relpath: str):
        super().__init__(relpath)
        self.relpath = relpath


def resolve_dir(path) -> Path:
    """Resolve a directory argument.

    A relative path that does not exist in the working directory is taken
    relative to the repository root, so ``--results-dir outputs`` works from any
    working directory.
    """
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


# ----------------------------------------------------------------------------
# Run folders
# ----------------------------------------------------------------------------
_MODEL_RANK = {"deterministic": 0, "full": 1, "pisasr": 2}


def parse_run_name(run: str) -> tuple[str, int | None]:
    """Return ``(model, S)`` for a run folder name (:func:`pisasr.systems.parse_run_name`).

    ``S`` is 1 for the deterministic (expected-value) model, which uses a single
    net-load profile, and ``None`` for a folder name outside the naming scheme.
    """
    return systems.parse_run_name(run)


def _run_key(run: str):
    model, S = parse_run_name(run)
    return (_MODEL_RANK.get(model, len(_MODEL_RANK)), math.inf if S is None else S, run)


def list_runs(results_dir, system: str, filename: str = "evaluation.csv") -> list[str]:
    """Run folders of ``system`` that contain ``filename``, in table order."""
    runs_dir = Path(results_dir) / system / "runs"
    if not runs_dir.is_dir():
        return []
    names = [p.name for p in runs_dir.iterdir() if (p / filename).is_file()]
    return sorted(names, key=_run_key)


def load_portfolio(system: str, data_dir=DATA_DIR) -> pd.DataFrame:
    gen = pd.read_csv(Path(data_dir) / system / "generator_portfolio.csv")
    gen["unit"] = gen["unit"].astype(str)
    return gen


# ----------------------------------------------------------------------------
# Evaluation summaries (results/<system>/summary/evaluation_metrics.csv)
# ----------------------------------------------------------------------------
EVALUATION_COLUMNS = [
    "run", "model", "S",
    "fixed_commitment_cost", "expected_dispatch_cost", "expected_total_cost",
    "p95_total_cost", "worst_total_cost",
    "expected_ens_mwh", "total_ens_mwh", "ens_probability", "expected_ens_hours",
    "total_spill_mwh", "unserved_energy_rate", "n_evaluation_scenarios",
    "no_load_cost", "startup_cost", "shutdown_cost", "startup_count", "committed_unit_hours",
    "normalized_expected_cost", "normalized_worst_cost",
]
COMMITMENT_COLUMNS = ["no_load_cost", "startup_cost", "shutdown_cost", "startup_count",
                      "committed_unit_hours"]


def commitment_costs(commitment: pd.DataFrame, portfolio: pd.DataFrame) -> dict:
    """First-stage cost of a commitment and its counts, as computed by the evaluation code.

    Startups and shutdowns are counted against each unit's ``initial_status``
    before hour 0. The no-load cost multiplies the stored commitment values; the
    startup and shutdown costs use the values rounded to 0 or 1.
    """
    c = commitment.copy()
    c["unit"] = c["unit"].astype(str)
    c = c.set_index("unit")
    gen = portfolio.set_index("unit").loc[c.index]
    u = c[HOUR_COLS].to_numpy(dtype=float)
    u01 = np.rint(u).astype(int)
    previous = np.column_stack([gen["initial_status"].astype(int).to_numpy(), u01[:, :-1]])
    startup = np.maximum(u01 - previous, 0)
    shutdown = np.maximum(previous - u01, 0)
    no_load = float((gen["no_load_cost"].to_numpy(float)[:, None] * u).sum())
    start = float((gen["startup_cost"].to_numpy(float)[:, None] * startup).sum())
    stop = float((gen["shutdown_cost"].to_numpy(float)[:, None] * shutdown).sum())
    return {
        "no_load_cost": no_load,
        "startup_cost": start,
        "shutdown_cost": stop,
        "fixed_commitment_cost": no_load + start + stop,
        "startup_count": int(startup.sum()),
        "committed_unit_hours": int(round(float(u.sum()))),
    }


def evaluation_metrics(evaluation: pd.DataFrame) -> dict:
    """Summary of one ``evaluation.csv`` (one row for each evaluation scenario).

    The definitions are those of the evaluation code that produced the paper's
    results: means and sums over the evaluation scenarios; ``p95_total_cost`` is
    the pandas 0.95 quantile (linear interpolation); a scenario counts towards
    ``ens_probability`` when its unserved energy exceeds 1e-4 MWh; and
    ``unserved_energy_rate`` is total unserved energy divided by total net load
    (a fraction; multiply by 100 for the percentages printed in the paper). The
    fixed commitment cost is ``total_cost_with_fixed - dispatch_cost``, which is
    the same in every row.
    """
    ev = evaluation
    total = ev["total_cost_with_fixed"]
    dispatch = ev["dispatch_cost"]
    fixed = total - dispatch
    fixed_cost = float(fixed.mean())
    if float(fixed.max() - fixed.min()) > 1e-9 * max(abs(fixed_cost), 1.0):
        raise ValueError("total_cost_with_fixed - dispatch_cost differs between scenarios")
    return {
        "fixed_commitment_cost": fixed_cost,
        "expected_dispatch_cost": float(dispatch.mean()),
        "expected_total_cost": float(total.mean()),
        "p95_total_cost": float(total.quantile(0.95)),
        "worst_total_cost": float(total.max()),
        "expected_ens_mwh": float(ev["ens_mwh"].mean()),
        "total_ens_mwh": float(ev["ens_mwh"].sum()),
        "ens_probability": float((ev["ens_mwh"] > ENS_THRESHOLD_MWH).mean()),
        "expected_ens_hours": float(ev["ens_hours"].mean()),
        "total_spill_mwh": float(ev["spill_mwh"].sum()),
        "unserved_energy_rate": float(ev["ens_mwh"].sum() / max(ev["total_net_load_mwh"].sum(), 1e-6)),
        "n_evaluation_scenarios": int(len(ev)),
    }


def summarize_system(results_dir=REFERENCE_DIR, system: str = "tx123bt", data_dir=DATA_DIR) -> pd.DataFrame:
    """One row for each run of ``system`` that has an ``evaluation.csv``.

    The commitment columns (no-load, startup and shutdown costs and counts) are
    recomputed from ``commitment.csv`` and the generator portfolio, and the
    resulting fixed cost is checked against the one implied by ``evaluation.csv``.
    """
    results_dir = Path(results_dir)
    portfolio_file = Path(data_dir) / system / "generator_portfolio.csv"
    portfolio = load_portfolio(system, data_dir) if portfolio_file.is_file() else None
    rows = []
    for run in list_runs(results_dir, system):
        folder = results_dir / system / "runs" / run
        model, S = parse_run_name(run)
        row = {"run": run, "model": model, "S": S}
        row.update(evaluation_metrics(pd.read_csv(folder / "evaluation.csv")))
        if portfolio is not None and (folder / "commitment.csv").is_file():
            costs = commitment_costs(pd.read_csv(folder / "commitment.csv"), portfolio)
            a, b = costs["fixed_commitment_cost"], row["fixed_commitment_cost"]
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"{system}/{run}: fixed cost {a} from commitment.csv differs "
                                 f"from {b} implied by evaluation.csv")
            row.update({k: costs[k] for k in COMMITMENT_COLUMNS})
        rows.append(row)
    df = pd.DataFrame(rows).reindex(columns=EVALUATION_COLUMNS)
    if df.empty:
        return df
    det = df[df["model"] == "deterministic"]
    if not det.empty:
        df["normalized_expected_cost"] = df["expected_total_cost"] / det["expected_total_cost"].iloc[0]
        df["normalized_worst_cost"] = df["worst_total_cost"] / det["worst_total_cost"].iloc[0]
    for col in ("S", "startup_count", "committed_unit_hours", "n_evaluation_scenarios"):
        df[col] = df[col].astype("Int64")
    return df


def compare_frames(new: pd.DataFrame, old: pd.DataFrame, key: str = "run", rtol: float = 1e-9) -> list[str]:
    """Differences between two tables with the same key column (shared columns only)."""
    problems = []
    a, b = new.set_index(key), old.set_index(key)
    for k in sorted(set(a.index) ^ set(b.index), key=str):
        problems.append(f"{key}={k}: present in only one of the two tables")
    shared = [c for c in a.columns if c in b.columns]
    for k in a.index.intersection(b.index):
        for c in shared:
            x, y = a.at[k, c], b.at[k, c]
            if pd.isna(x) and pd.isna(y):
                continue
            if isinstance(x, (int, float, np.integer, np.floating)) and isinstance(y, (int, float, np.integer, np.floating)):
                if pd.isna(x) or pd.isna(y) or not math.isclose(float(x), float(y), rel_tol=rtol, abs_tol=1e-12):
                    problems.append(f"{key}={k}, {c}: {x!r} versus {y!r}")
            elif str(x) != str(y):
                problems.append(f"{key}={k}, {c}: {x!r} versus {y!r}")
    return problems


def check_scaling_evaluation(summary: pd.DataFrame, scaling: pd.DataFrame, rtol: float = 1e-9) -> list[str]:
    """Compare the evaluation columns of ``summary/scaling.csv`` with a recomputed summary.

    ``*_oos_cost`` is the evaluation-set expected total cost, ``*_oos_ens`` the
    total unserved energy (MWh) and ``*_oos_unserved`` the unserved-energy rate
    (fraction). Empty cells are skipped.
    """
    problems = []
    by_run = summary.set_index("run")
    pairs = (("oos_cost", "expected_total_cost"), ("oos_ens", "total_ens_mwh"),
             ("oos_unserved", "unserved_energy_rate"))
    for _, r in scaling.iterrows():
        S = int(r["S"])
        for prefix in ("full", "pisasr"):
            run = f"{prefix}_S{S}"
            for suffix, col in pairs:
                name = f"{prefix}_{suffix}"
                if name not in scaling.columns or pd.isna(r[name]):
                    continue
                if run not in by_run.index:
                    problems.append(f"{run}: no evaluation.csv for {name}")
                    continue
                x = float(by_run.at[run, col])
                if not math.isclose(x, float(r[name]), rel_tol=rtol, abs_tol=1e-15):
                    problems.append(f"{run}, {name}: scaling.csv {r[name]!r} versus recomputed {x!r}")
    return problems


# ----------------------------------------------------------------------------
# Paired cost difference on the evaluation set (Section 6.3)
# ----------------------------------------------------------------------------
PAIRED_QUANTITIES = (("total_cost", "total_cost_with_fixed"), ("dispatch_cost", "dispatch_cost"))


def paired_cost_difference(results_dir=REFERENCE_DIR, system: str = "tx123bt", S: int = 1000,
                           confidence: float = 0.95, run_a: str | None = None,
                           run_b: str | None = None) -> pd.DataFrame:
    """Paired difference of evaluation-set costs, ``run_a`` minus ``run_b``.

    By default ``run_a`` is PI-SASR and ``run_b`` the full model at the same
    ``S``. Each quantity (total cost including the fixed commitment cost, and
    dispatch cost alone) gets a two-sided Student-t interval, and the ``*_pct``
    columns divide by the mean of ``run_b`` for the same quantity.
    """
    run_a = run_a or f"pisasr_S{S}"
    run_b = run_b or f"full_S{S}"
    root = Path(results_dir) / system / "runs"
    frames = {}
    for run in (run_a, run_b):
        f = root / run / "evaluation.csv"
        if not f.is_file():
            raise MissingSource(f"{system}/runs/{run}/evaluation.csv")
        frames[run] = pd.read_csv(f)
    a, b = frames[run_a], frames[run_b]
    merged = a.merge(b, on="scenario", suffixes=("_a", "_b"), validate="one_to_one")
    if not (len(merged) == len(a) == len(b)):
        raise ValueError(f"{run_a} and {run_b} were not evaluated on the same scenarios")
    rows = []
    for quantity, col in PAIRED_QUANTITIES:
        r = paired_difference_interval(merged[f"{col}_a"], merged[f"{col}_b"], confidence=confidence)
        rows.append({"system": system, "S": S, "quantity": quantity, "run_a": run_a, "run_b": run_b,
                     **r.as_dict()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Access to the CSV files used by the tables, figures and numbers
# ----------------------------------------------------------------------------
class Sources:
    """Read-only access to the CSV files of one results folder and ``data/`` (each file is read once).

    A missing file raises :class:`MissingSource` with its path relative to the
    results folder (or starting with ``data/``), so that callers can name it.
    """

    def __init__(self, results_dir=REFERENCE_DIR, data_dir=DATA_DIR):
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self._frames: dict[str, pd.DataFrame] = {}
        self._metrics: dict[tuple, dict] = {}
        self._paired: dict[tuple, pd.DataFrame] = {}

    def _read(self, path: Path, label: str) -> pd.DataFrame:
        if label not in self._frames:
            if not path.is_file():
                raise MissingSource(label)
            self._frames[label] = pd.read_csv(path)
        return self._frames[label]

    def result(self, system: str, *parts: str) -> pd.DataFrame:
        return self._read(self.results_dir.joinpath(system, *parts), "/".join((system,) + parts))

    def data(self, system: str, name: str) -> pd.DataFrame:
        return self._read(self.data_dir / system / name, f"data/{system}/{name}")

    def experiment(self, system: str, name: str) -> pd.DataFrame:
        return self.result(system, "experiments", name)

    def solve(self, system: str, run: str) -> pd.Series:
        return self.result(system, "runs", run, "solve_metrics.csv").iloc[0]

    def evaluation(self, system: str, run: str) -> pd.DataFrame:
        return self.result(system, "runs", run, "evaluation.csv")

    def metrics(self, system: str, run: str) -> dict:
        key = (system, run)
        if key not in self._metrics:
            self._metrics[key] = evaluation_metrics(self.evaluation(system, run))
        return self._metrics[key]

    def ens_pct(self, system: str, run: str) -> float:
        """Evaluation-set unserved-energy rate in percent."""
        return 100.0 * self.metrics(system, run)["unserved_energy_rate"]

    def eval_cost(self, system: str, run: str) -> float:
        return self.metrics(system, run)["expected_total_cost"]

    def under_rep_pct(self, system: str, S: int) -> float:
        """Under-representation statistic (percent) of the first reduced UC."""
        return 100.0 * float(self.result(system, "runs", f"pisasr_S{S}", "pisasr_log.csv")["under_rep_gap"].iloc[0])

    def scaling_row(self, system: str, S: int) -> dict:
        full = self.solve(system, f"full_S{S}")
        pis = self.solve(system, f"pisasr_S{S}")
        return {
            "S": S,
            "full_obj": float(full["objective"]),
            "full_time_s": float(full["solve_time_seconds"]),
            "pisasr_obj": float(pis["objective"]),
            "pisasr_gap_pct": float(pis["opt_gap_pct_vs_full"]),
            "pisasr_time_s": float(pis["solve_time_seconds"]),
            "speedup_x": float(full["solve_time_seconds"]) / float(pis["solve_time_seconds"]),
            "K_used": int(pis["n_scenarios_used"]),
            "fixed_units": int(pis["n_fixed_baseload_units"]),
            "surrogate_r2": float(pis["surrogate_r2"]),
        }

    def scaling(self, system: str) -> pd.DataFrame:
        """Scaling rows for every ``S`` of the grid whose full and PI-SASR runs exist."""
        rows = []
        for S in S_GRID:
            try:
                rows.append(self.scaling_row(system, S))
            except MissingSource:
                continue
        if not rows:
            raise MissingSource(f"{system}/runs/full_S*/solve_metrics.csv")
        return pd.DataFrame(rows)

    def paired(self, system: str, S: int = 1000) -> pd.DataFrame:
        key = (system, S)
        if key not in self._paired:
            self._paired[key] = paired_cost_difference(self.results_dir, system, S).set_index("quantity")
        return self._paired[key]

    def scenario_pool(self, system: str) -> pd.DataFrame:
        """Training and test scenario files together (7,500 scenarios for both systems)."""
        train = self.data(system, "net_load_scenarios_train.csv")
        test = self.data(system, "net_load_scenarios_test.csv")
        return pd.concat([train, test], ignore_index=True)


def row_where(df: pd.DataFrame, column: str, value) -> pd.Series:
    sel = df[df[column] == value]
    if sel.empty:
        raise KeyError(f"no row with {column} = {value}")
    return sel.iloc[0]


def kmin_from_budget(d: pd.DataFrame, tau: float) -> dict[int, tuple[int | None, int, float]]:
    """Smallest active set whose benchmark gap meets ``tau`` percent, for each ``S``.

    Returns ``{S: (K_min or None, largest K tested, smallest gap)}``; ``None``
    means that no tested budget met the tolerance (a censored value).
    """
    out = {}
    for S, g in d.groupby("S"):
        g = g.sort_values("K")
        ok = g[g["gap_pct"] <= tau]
        out[int(S)] = (int(ok["K"].iloc[0]) if len(ok) else None, int(g["K"].max()), float(g["gap_pct"].min()))
    return out


# ----------------------------------------------------------------------------
# Number formatting shared by the tables and the register
# ----------------------------------------------------------------------------
def fmt(x: float, decimals: int, sign: bool = False, thousands: bool = False) -> str:
    spec = ("+" if sign else "") + ("," if thousands else "") + f".{decimals}f"
    s = format(float(x), spec)
    if float(s.replace(",", "")) == 0.0:  # no "-0.00"
        s = s.replace("-", "+" if sign else "")
    return s


def fmt_gap(x: float) -> str:
    """Benchmark gap as printed in Table 3: three decimals below 0.1%, else two."""
    return fmt(x, 3) if abs(x) < 0.1 else fmt(x, 2)


def fmt_speedup(x: float) -> str:
    return fmt(x, 1) if x >= 10 else fmt(x, 2)


def tex_number(s: str) -> str:
    """LaTeX form of a formatted number: thousands separators and math-mode signs."""
    t = s.replace(",", "{,}")
    return f"${t}$" if t[:1] in "+-" else t


def _bold_tex(s: str, bold: bool) -> str:
    return f"\\textbf{{{s}}}" if bold else s


def _bold_md(s: str, bold: bool) -> str:
    return f"**{s}**" if bold else s


def _safe(fn: Callable[[], str], default: str = "n/a") -> str:
    try:
        return fn()
    except (MissingSource, KeyError, IndexError):
        return default


_GENERATED = "Generated by scripts/make_paper_assets.py from the CSV files of this repository; do not edit by hand."


# ----------------------------------------------------------------------------
# Tables 1-4 (LaTeX with booktabs, and Markdown)
# ----------------------------------------------------------------------------
@dataclass
class TableText:
    stem: str
    tex: str
    md: str


def table_systems(src: Sources) -> TableText:
    """Table 1: the two test fleets."""
    vals = {}
    for s in SYSTEM_KEYS:
        v = {}
        v["units"] = _safe(lambda: str(len(src.data(s, "generator_portfolio.csv"))))
        v["cap"] = _safe(lambda: fmt(src.data(s, "generator_portfolio.csv")["p_max_mw"].sum() / 1000, 1))
        v["peak"] = _safe(lambda: fmt(src.scenario_pool(s)[HOUR_COLS].max(axis=1).max() / 1000, 1))
        v["train"] = _safe(lambda: fmt(len(src.data(s, "net_load_scenarios_train.csv")), 0, thousands=True))
        v["eval"] = _safe(lambda: fmt(src.metrics(s, "full_S1000")["n_evaluation_scenarios"], 0, thousands=True))
        vals[s] = v
    tx, tk = vals["tx123bt"], vals["texas2k"]
    caption = ("The two synthetic Texas test fleets. Capacities are comparable; Texas2k carries "
               "four times as many decision variables.")
    rows = [
        ("Committable units", tx["units"], tk["units"], "", ""),
        ("Committable capacity", tx["cap"], tk["cap"], "\\,GW", " GW"),
        ("Max.\\ peak net load", tx["peak"], tk["peak"], "\\,GW", " GW"),
        ("Training / evaluation sets", f"{tx['train']} / {tx['eval']}", f"{tk['train']} / {tk['eval']}", "", ""),
    ]
    body = "\n".join(f"{lab} & {tex_number(a) if '/' not in a else a.replace(',', '{,}')}{ut} & "
                     f"{tex_number(b) if '/' not in b else b.replace(',', '{,}')}{ut} \\\\"
                     for lab, a, b, ut, _ in rows)
    tex = (f"% Table 1. {_GENERATED}\n% Requires \\usepackage{{booktabs}}.\n"
           "\\begin{table}[t]\n"
           f"\\caption{{{caption}}}\n\\label{{tab:systems}}\n\\centering\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{2.4pt}\n\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
           " & \\textbf{TX-123BT} & \\textbf{Texas2k Series25} \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    md_rows = "\n".join(f"| {lab.replace(chr(92) + ' ', ' ').replace(chr(92), '')} | {a}{um} | {b}{um} |"
                        for lab, a, b, _, um in rows)
    md = (f"**Table 1.** {caption}\n\n"
          "| | TX-123BT | Texas2k Series25 |\n|---|---:|---:|\n"
          f"{md_rows}\n\n"
          "Committable capacity is the sum of `p_max_mw` over the generator portfolio. The maximum peak "
          "net load is the largest hourly net load over all 7,500 scenarios of the training and test files. "
          "The evaluation set is the first 2,000 scenarios of the test file.\n")
    return TableText("table1_systems", tex, md)


def _primary_rows(src: Sources, s: str) -> list[tuple[str, list[tuple[str, bool]]]]:
    det_obj = _safe(lambda: fmt(src.solve(s, "deterministic")["objective"] / 1e6, 2))
    det_time = _safe(lambda: fmt(src.solve(s, "deterministic")["solve_time_seconds"], 1))
    det_ens = _safe(lambda: format(src.ens_pct(s, "deterministic"), ".4g"))
    full_obj = _safe(lambda: fmt(src.solve(s, "full_S1000")["objective"] / 1e6, 2))
    full_time = _safe(lambda: fmt(src.solve(s, "full_S1000")["solve_time_seconds"], 0))
    full_S = _safe(lambda: str(int(src.solve(s, "full_S1000")["n_scenarios"])))
    full_ens = _safe(lambda: fmt(src.ens_pct(s, "full_S1000"), 4))
    p = lambda col: src.solve(s, "pisasr_S1000")[col]  # noqa: E731
    pis_obj = _safe(lambda: fmt(p("objective") / 1e6, 2))
    pis_gap = _safe(lambda: fmt(p("opt_gap_pct_vs_full"), 2))
    pis_time = _safe(lambda: fmt(p("solve_time_seconds"), 0))
    pis_K = _safe(lambda: str(int(p("n_scenarios_used"))))
    pis_ens = _safe(lambda: fmt(src.ens_pct(s, "pisasr_S1000"), 4))
    return [
        ("Deterministic", [(det_obj, False), ("n/a", False), (det_time, False), ("1", False), (det_ens, False)]),
        ("Full stochastic", [(full_obj, False), ("0.00", False), (full_time, False), (full_S, False), (full_ens, False)]),
        ("PI-SASR", [(pis_obj, False), (pis_gap, False), (pis_time, True), (pis_K, True), (pis_ens, False)]),
    ]


def table_primary(src: Sources) -> TableText:
    """Table 2: primary comparison at S = 1,000."""
    note_c = _safe(lambda: (
        f"{src.eval_cost('texas2k', 'deterministic') / 1e9:.1f}",
        f"{src.eval_cost('texas2k', 'deterministic') / src.eval_cost('texas2k', 'full_S1000'):.0f}"), None)
    det_b, det_x = note_c if note_c else ("n/a", "n/a")
    tex_rows, md_rows = [], []
    for s in SYSTEM_KEYS:
        tex_rows.append(f"\\multirow{{3}}{{*}}{{{SHORT_NAMES[s]}}}")
        for i, (method, cells) in enumerate(_primary_rows(src, s)):
            label = "\\textbf{PI-SASR}" if method == "PI-SASR" else method
            tcells = [_bold_tex(tex_number(v), b) for v, b in cells]
            if s == "texas2k" and method == "Deterministic":
                tcells[0] += "\\tnote{c}"
            tex_rows.append(f" & {label} & " + " & ".join(tcells) + " \\\\")
            mcells = [_bold_md(v, b) for v, b in cells]
            if s == "texas2k" and method == "Deterministic":
                mcells[0] += " (c)"
            md_rows.append(f"| {SHORT_NAMES[s] if i == 0 else ''} | {_bold_md(method, method == 'PI-SASR')} | "
                           + " | ".join(mcells) + " |")
        tex_rows.append("\\midrule")
    tex_rows[-1] = "\\bottomrule"
    notes = [
        ("a", "Deterministic UC: objective over its single expected-value scenario. Full model: objective, the "
              "expected total cost over the $S$ training scenarios. PI-SASR: exact expected total cost of its "
              "commitment over the same $S$ scenarios (fixed cost plus the mean ramp-coupled LP recourse).",
         "Deterministic UC: objective over its single expected-value scenario. Full model: objective, the "
         "expected total cost over the S training scenarios. PI-SASR: exact expected total cost of its "
         "commitment over the same S scenarios (fixed cost plus the mean ramp-coupled LP recourse)."),
        ("b", "Unserved-energy rate on the evaluation set ($2{,}000$ scenarios) under generator outages. "
              "Times are wall-clock times in seconds, defined as in Table~\\ref{tab:scaling}.",
         "Unserved-energy rate on the evaluation set (2,000 scenarios) under generator outages. "
         "Times are wall-clock times in seconds, defined as in Table 3."),
        ("c", f"The deterministic objective ignores uncertainty; its evaluation-set expected cost on Texas2k "
              f"is \\${det_b}B, ${det_x}\\times$ the stochastic schedules'.",
         f"The deterministic objective ignores uncertainty; its evaluation-set expected cost on Texas2k "
         f"is \\${det_b}B, {det_x}× the stochastic schedules'."),
    ]
    caption = "Primary comparison at $S=1{,}000$ on both systems."
    tex = (f"% Table 2. {_GENERATED}\n% Requires \\usepackage{{booktabs,multirow,threeparttable}}.\n"
           "\\begin{table}[t]\n"
           f"\\caption{{{caption}}}\n\\label{{tab:primary}}\n\\centering\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{2.6pt}\n\\begin{threeparttable}\n\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n"
           "\\textbf{System} & \\textbf{Method} & \\textbf{Obj.}\\tnote{a} & \\textbf{Gap} & \\textbf{Time} & "
           "\\textbf{Scen.} & \\textbf{Unserved}\\tnote{b} \\\\\n"
           " & & (\\$M) & (\\%) & (s) & used & (\\%) \\\\\n\\midrule\n"
           + "\n".join(tex_rows) + "\n\\end{tabular}\n\\begin{tablenotes}\\footnotesize\n"
           + "\n".join(f"\\item[{k}] {t}" for k, t, _ in notes)
           + "\n\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n")
    md = ("**Table 2.** Primary comparison at S = 1,000 on both systems.\n\n"
          "| System | Method | Obj. (\\$M) (a) | Gap (%) | Time (s) | Scen. used | Unserved (%) (b) |\n"
          "|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(md_rows) + "\n\n"
          + "\n".join(f"({k}) {m}  " for k, _, m in notes).rstrip() + "\n")
    return TableText("table2_primary", tex, md)


def table_scaling(src: Sources) -> TableText:
    """Table 3: scenario scaling of PI-SASR against the full model."""
    tex_rows, md_rows = [], []
    for s in SYSTEM_KEYS:
        try:
            sc = src.scaling(s)
        except MissingSource:
            continue
        tex_rows.append(f"\\multirow{{{len(sc)}}}{{*}}{{{SHORT_NAMES[s]}}}")
        for i, r in enumerate(sc.itertuples()):
            S = int(r.S)
            ur = _safe(lambda: fmt(src.under_rep_pct(s, S), 2, sign=True))
            bold = S == 2500
            t_full = fmt(r.full_time_s, 0)
            t_pis, sp = fmt(r.pisasr_time_s, 0), fmt_speedup(r.speedup_x)
            cnote = "\\tnote{c}" if (s == "texas2k" and S == 2500) else ""
            tex_rows.append(f" & {S} & {fmt(r.full_obj / 1e6, 2)} & {t_full}{cnote} & ${fmt_gap(r.pisasr_gap_pct)}$ & "
                            f"{tex_number(ur) if ur != 'n/a' else ur} & {_bold_tex(t_pis, bold)} & {r.K_used} & "
                            f"{_bold_tex(sp, bold)} \\\\")
            md_rows.append(f"| {SHORT_NAMES[s] if i == 0 else ''} | {S:,} | {fmt(r.full_obj / 1e6, 2)} | "
                           f"{t_full}{' (c)' if cnote else ''} | {fmt_gap(r.pisasr_gap_pct)} | {ur} | "
                           f"{_bold_md(t_pis, bold)} | {r.K_used} | {_bold_md(sp, bold)} |")
        tex_rows.append("\\midrule")
    if tex_rows:
        tex_rows[-1] = "\\bottomrule"
    solver_only = _safe(lambda: fmt(src.solve("texas2k", "full_S2500")["lean_solver_only_seconds"], 0, thousands=True))
    eps = f"{ENRICHMENT_TOLERANCE_PCT:g}"
    notes = [
        ("a", "Speedup $=$ full time $/$ PI-SASR time (unrounded). Gaps are benchmark gaps against the full "
              "model's incumbent ($0.1\\%$ MILP tolerance), not certificates; certified gaps are given in "
              f"Section~6.1. $\\widehat\\Delta$ is the under-representation statistic of the first reduced UC; "
              f"it never exceeded the enrichment tolerance $\\epsilon={eps}\\%$, so no enrichment round was performed.",
         "Speedup = full time / PI-SASR time (unrounded). Gaps are benchmark gaps against the full model's "
         "incumbent (0.1% MILP tolerance), not certificates; certified gaps are given in Section 6.1. "
         "Δ̂ is the under-representation statistic of the first reduced UC; it never exceeded the "
         f"enrichment tolerance ε = {eps}%, so no enrichment round was performed."),
        ("b", "Wall-clock times in seconds. Full model on TX-123BT: solver call only; full model on Texas2k: model "
              "construction and solver call, so the TX-123BT speedups are conservative. PI-SASR: online time from "
              "data loading to the end of the merit-order verification, excluding the deterministic UC solution "
              "that gives the reference commitment and the exact re-evaluation behind the gap column.",
         "Wall-clock times in seconds. Full model on TX-123BT: solver call only; full model on Texas2k: model "
         "construction and solver call, so the TX-123BT speedups are conservative. PI-SASR: online time from "
         "data loading to the end of the merit-order verification, excluding the deterministic UC solution "
         "that gives the reference commitment and the exact re-evaluation behind the gap column."),
        ("c", "The Texas2k $S=2{,}500$ extensive form exceeded the workstation's $192$\\,GB of memory and was "
              "therefore written to a model file and solved from it; its time covers model construction, writing "
              f"the model file and the solver call (solver call alone: {solver_only.replace(',', '{,}')}\\,s).",
         "The Texas2k S = 2,500 extensive form exceeded the workstation's 192 GB of memory and was therefore "
         "written to a model file and solved from it; its time covers model construction, writing the model "
         f"file and the solver call (solver call alone: {solver_only} s)."),
    ]
    caption = ("Scenario scaling on both systems: PI-SASR versus full stochastic UC. $\\widehat\\Delta$ is the "
               "realized under-representation statistic.")
    tex = (f"% Table 3. {_GENERATED}\n% Requires \\usepackage{{booktabs,multirow,threeparttable}}.\n"
           "\\begin{table}[t]\n"
           f"\\caption{{{caption}}}\n\\label{{tab:scaling}}\n\\centering\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{1.5pt}\n\\begin{threeparttable}\n\\begin{tabular}{@{}lrccccccc@{}}\n\\toprule\n"
           "\\textbf{System} & $S$ & \\multicolumn{2}{c}{\\textbf{Full}} & \\multicolumn{4}{c}{\\textbf{PI-SASR}} & "
           "\\textbf{Sp.}\\tnote{a} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n"
           " & & Obj.\\,(\\$M) & Time\\tnote{b} & Gap\\,(\\%) & $\\widehat\\Delta$\\,(\\%) & Time\\tnote{b} & $K$ & \\\\\n"
           "\\midrule\n" + "\n".join(tex_rows) + "\n\\end{tabular}\n\\begin{tablenotes}\\footnotesize\n"
           + "\n".join(f"\\item[{k}] {t}" for k, t, _ in notes)
           + "\n\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n")
    md = ("**Table 3.** Scenario scaling on both systems: PI-SASR versus full stochastic UC. "
          "Δ̂ is the realized under-representation statistic.\n\n"
          "| System | S | Full obj. (\\$M) | Full time (s) (b) | PI-SASR gap (%) | Δ̂ (%) | PI-SASR time (s) (b) "
          "| K | Speedup (a) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
          + "\n".join(md_rows) + "\n\n" + "\n".join(f"({k}) {m}  " for k, _, m in notes).rstrip() + "\n")
    return TableText("table3_scaling", tex, md)


COMPONENT_VARIANTS = [
    ("ffs_K148", "Fast-forward selection (Heitsch and R\\\"omisch)", "Fast-forward selection (Heitsch and Römisch)"),
    ("medoids_K148", "Coverage medoids only", "Coverage medoids only"),
    ("costffs_K148", "Cost-space forward sel.\\ (Werner et al.)", "Cost-space forward selection (Werner et al.)"),
    ("exact_crit_K148", "Exact criticality only", "Exact criticality only"),
    ("gp_crit_K148", "GP criticality only", "GP criticality only"),
    ("gp_crit_K188", "GP criticality only, $K{=}188$", "GP criticality only, K = 188"),
    ("exact_crit90_med60", "Exact criticality $\\cup$ coverage", "Exact criticality ∪ coverage"),
    ("pisasr_nofix", "PI-SASR, no baseload pre-commitment", "PI-SASR, no baseload pre-commitment"),
    ("pisasr", "\\textbf{PI-SASR}", "**PI-SASR**"),
]


def _component_cells(r: pd.Series) -> list[str]:
    return [str(int(r["K"])), fmt(r["t_select_s"], 1), fmt(r["t_milp_s"], 0), fmt(r["t_online_s"], 0),
            fmt(r["gap_pct"], 2, sign=True)]


def table_components(src: Sources) -> TableText:
    """Table 4: component analysis at equal active-set size (TX-123BT, S = 1,000)."""
    try:
        d = src.experiment("tx123bt", "component_timing_S1000.csv")
        have = {r["variant"]: r for _, r in d.iterrows()}
    except MissingSource:
        have = {}
    tex_rows, md_rows = [], []
    for key, tex_label, md_label in COMPONENT_VARIANTS:
        cells = _component_cells(have[key]) if key in have else ["n/a"] * 5
        bold = key == "pisasr"
        tex_rows.append(f"{tex_label} & " + " & ".join(
            tex_number(c) if c.startswith(("+", "-")) else _bold_tex(c, bold) for c in cells) + " \\\\")
        md_rows.append(f"| {md_label} | " + " | ".join(
            c if c.startswith(("+", "-")) else _bold_md(c, bold) for c in cells) + " |")
    note = ("Features, selection, reduced MILP, and merit-order verification over all $S$; the offline exact "
            "re-evaluation behind the gap column is excluded for all variants.")
    caption = ("Component analysis at equal active-set size on TX-123BT ($S{=}1{,}000$): benchmark gap and "
               "end-to-end online time of each variant (each timed in a single dedicated run, separately from "
               "the runs of Table~\\ref{tab:scaling}).")
    tex = (f"% Table 4. {_GENERATED}\n% Requires \\usepackage{{booktabs,graphicx}}.\n"
           "\\begin{table}[t]\n"
           f"\\caption{{{caption}}}\n\\label{{tab:ablation}}\n\\centering\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{2pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           "\\textbf{Variant} & $K$ & Sel.\\,(s) & MILP\\,(s) & Online\\,(s)$^{\\mathrm{a}}$ & \\textbf{Gap\\,(\\%)} \\\\\n"
           "\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}}\\par\\vspace{2pt}\n"
           f"\\parbox{{\\columnwidth}}{{\\footnotesize $^{{\\mathrm{{a}}}}$\\,{note}}}\n\\end{{table}}\n")
    md = ("**Table 4.** Component analysis at equal active-set size on TX-123BT (S = 1,000): benchmark gap and "
          "end-to-end online time of each variant (each timed in a single dedicated run, separately from the "
          "runs of Table 3).\n\n"
          "| Variant | K | Sel. (s) | MILP (s) | Online (s) (a) | Gap (%) |\n|---|---:|---:|---:|---:|---:|\n"
          + "\n".join(md_rows) + "\n\n"
          "(a) Features, selection, reduced MILP, and merit-order verification over all S; the offline exact "
          "re-evaluation behind the gap column is excluded for all variants. The two reference methods are "
          "the fast-forward selection of Heitsch and Römisch (2003) and the problem-driven forward selection "
          "in recourse-cost space of Werner et al. (2025).\n")
    return TableText("table4_components", tex, md)


def write_tables(src: Sources, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for t in (table_systems(src), table_primary(src), table_scaling(src), table_components(src)):
        for ext, text in (("tex", t.tex), ("md", t.md)):
            f = out_dir / f"{t.stem}.{ext}"
            f.write_text(text)
            written.append(f)
    return written


# ----------------------------------------------------------------------------
# Figures 2-4 (matplotlib; PDF and 300-dpi PNG)
# ----------------------------------------------------------------------------
C_FULL = "#4a4a4a"        # full model, lines
C_FULL_BAR = "#b3b3b3"    # full model, bars
C_PISASR = "#1f7a3a"      # PI-SASR
C_DET = "#b2222b"         # deterministic model
C_SYSTEM = {"tx123bt": "#c55a11", "texas2k": "#1f3a8a"}
M_SYSTEM = {"tx123bt": "s", "texas2k": "^"}
M_SPEEDUP = {"tx123bt": "o", "texas2k": "^"}   # markers of Fig. 3(a), as in the paper
LS_REFERENCE = "--"                            # breakeven and default-budget lines of Fig. 3
_AUTHORS = "M. Korkali and D. Andriniaina"
_CREATOR = "PI-SASR scripts/make_paper_assets.py"
_STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}


def _figure(width: float = 7.0, height: float = 2.75):
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), layout="constrained")


def _s_axis(ax) -> None:
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    ax.set_xscale("log")
    ax.set_xlim(78, 3200)
    ax.xaxis.set_major_locator(FixedLocator(S_GRID))
    ax.xaxis.set_major_formatter(FixedFormatter(S_LABELS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Number of scenarios $S$")


def _plain_log_ticks(axis, ticks=None) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

    axis.set_major_locator(FixedLocator(ticks) if ticks is not None else LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_locator(NullLocator())


def _save(fig, out_dir: Path, stem: str, title: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf, metadata={"Title": title, "Author": _AUTHORS, "Creator": _CREATOR,
                               "Producer": None, "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=300, metadata={"Title": title, "Author": _AUTHORS, "Software": None})
    return [pdf, png]


def figure_solution_time(src: Sources, out_dir: Path) -> list[Path]:
    """Figure 2: solution time of the full model and of PI-SASR versus S (log-log)."""
    import matplotlib

    with matplotlib.rc_context(_STYLE):
        fig = _figure()
        axes = fig.subplots(1, 2)
        for ax, s, letter in zip(axes, SYSTEM_KEYS, "ab"):
            sc = src.scaling(s)
            ax.plot(sc["S"], sc["full_time_s"], color=C_FULL, marker="o", ls="-", label="Full stochastic UC")
            ax.plot(sc["S"], sc["pisasr_time_s"], color=C_PISASR, marker="s", ls="--", label="PI-SASR")
            _s_axis(ax)
            ax.set_yscale("log")
            _plain_log_ticks(ax.yaxis)
            ax.grid(True, which="major")
            ax.set_title(f"({letter}) {SHORT_NAMES[s]}")
        axes[0].set_ylabel("Solution time (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2)
        return _save(fig, out_dir, "fig2_solution_time", "Solution time versus scenario count")


def figure_speedup_kmin(src: Sources, out_dir: Path) -> list[Path]:
    """Figure 3: (a) speedup versus S; (b) minimum active-set size K_min(tau) versus S."""
    import matplotlib
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    with matplotlib.rc_context(_STYLE):
        fig = _figure()
        ax_a, ax_b = fig.subplots(1, 2)
        for s in SYSTEM_KEYS:
            sc = src.scaling(s)
            ax_a.plot(sc["S"], sc["speedup_x"], color=C_SYSTEM[s], marker=M_SPEEDUP[s], label=SHORT_NAMES[s])
        ax_a.axhline(1.0, color="0.45", ls=LS_REFERENCE, lw=1.0, label="Breakeven (equal times)")
        _s_axis(ax_a)
        ax_a.set_yscale("log")
        ax_a.set_ylim(0.8, 45)
        _plain_log_ticks(ax_a.yaxis, ticks=(1, 3, 10, 30))
        ax_a.grid(True, which="major")
        ax_a.set_ylabel("Speedup (full time / PI-SASR time)")
        ax_a.set_title("(a) Speedup versus $S$")
        ax_a.legend(loc="upper left")

        handles, labels = [], []
        censored = []
        for s in SYSTEM_KEYS:
            d = src.experiment(s, "budget_sensitivity.csv")
            for tau, ls, filled in ((TAU_LEVELS[0], "-", True), (TAU_LEVELS[1], "--", False)):
                km = kmin_from_budget(d, tau)
                style = dict(color=C_SYSTEM[s], ls=ls, marker=M_SYSTEM[s], mec=C_SYSTEM[s],
                             mfc=C_SYSTEM[s] if filled else "white")
                # As in the paper, the curve joins the values that meet the tolerance. Where a
                # censored value lies between two of them, the joining segment is drawn lighter
                # and without markers; the censored value itself is drawn separately below.
                segments, current, bridges, gap = [], [], [], False
                for S, v in sorted(km.items()):
                    if v[0] is None:
                        censored.append((S, v[1], s))
                        gap = True
                        continue
                    if gap and current:
                        bridges.append((current[-1], (S, v[0])))
                        segments.append(current)
                        current = []
                    current.append((S, v[0]))
                    gap = False
                segments += [current] if current else []
                for (x0, y0), (x1, y1) in bridges:
                    ax_b.plot([x0, x1], [y0, y1], color=C_SYSTEM[s], ls=ls, alpha=0.4, lw=1.1)
                for seg in segments:
                    ax_b.plot([p[0] for p in seg], [p[1] for p in seg], **style)
                handles.append(Line2D([], [], **style))
                labels.append(f"{SHORT_NAMES[s]}, $\\tau = {tau:g}\\%$")
        # A censored value is drawn as a hollow marker at the largest tested budget with ">" above it;
        # the two systems are nudged apart horizontally so that coincident markers stay visible.
        for S, kmax, s in sorted(set(censored)):
            x = S * (0.92 if s == "tx123bt" else 1.08)
            ax_b.plot([x], [kmax], ls="none", marker=M_SYSTEM[s], ms=6, mfc="white", mec=C_SYSTEM[s], mew=1.1)
            ax_b.annotate(">", (x, kmax), xytext=(0, 3.5), textcoords="offset points", ha="center",
                          va="bottom", color=C_SYSTEM[s], fontsize=9)
        ax_b.axhline(K_CRITICAL + K_COVER, color="0.45", ls=LS_REFERENCE, lw=1.0)
        _s_axis(ax_b)
        ax_b.set_ylim(0, 290)
        ax_b.set_yticks([0, 100, 200])
        ax_b.grid(True, which="major")
        ax_b.set_ylabel("$K_{\\min}(\\tau)$")
        ax_b.set_title("(b) Minimum active-set size $K_{\\min}(\\tau)$")
        handles += [
            tuple(Line2D([], [], ls="none", marker=M_SYSTEM[s], ms=6, mfc="white", mec=C_SYSTEM[s], mew=1.1)
                  for s in SYSTEM_KEYS),
            Line2D([], [], color="0.45", ls=LS_REFERENCE, lw=1.0),
        ]
        labels += ["$>$: no tested budget met $\\tau$ (largest $K$ tested)",
                   f"Default budget $K_c + K_r = {K_CRITICAL + K_COVER}$"]
        fig.legend(handles, labels, loc="outside lower center", ncol=3,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})
        return _save(fig, out_dir, "fig3_speedup_and_kmin", "Speedup and minimum active-set size versus scenario count")


def figure_unserved_energy(src: Sources, out_dir: Path) -> list[Path]:
    """Figure 4: evaluation-set unserved-energy rate at every S (log scale)."""
    import matplotlib

    def rate(s, run):
        try:
            return src.ens_pct(s, run)
        except MissingSource:
            return np.nan

    with matplotlib.rc_context(_STYLE):
        fig = _figure()
        axes = fig.subplots(1, 2, sharey=True)
        x = np.arange(len(S_GRID))
        w = 0.34
        for ax, s, letter in zip(axes, SYSTEM_KEYS, "ab"):
            full = [rate(s, f"full_S{S}") for S in S_GRID]
            pis = [rate(s, f"pisasr_S{S}") for S in S_GRID]
            ax.bar(x - 0.18, full, w, color=C_FULL_BAR, label="Full stochastic UC", zorder=2)
            ax.bar(x + 0.18, pis, w, color=C_PISASR, label="PI-SASR", zorder=2)
            det = rate(s, "deterministic")
            if np.isfinite(det):
                ax.axhline(det, color=C_DET, ls="--", lw=1.3, label="Deterministic UC", zorder=3)
            ax.set_yscale("log")
            ax.set_ylim(5e-4, 40)
            _plain_log_ticks(ax.yaxis, ticks=(1e-3, 1e-2, 1e-1, 1, 10))
            ax.grid(True, which="major", axis="y")
            ax.set_xticks(x, S_LABELS)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.6, len(S_GRID) - 0.4)
            ax.set_xlabel("Number of scenarios $S$")
            ax.set_title(f"({letter}) {SHORT_NAMES[s]}")
        axes[0].set_ylabel("Unserved energy (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: ["Full stochastic UC", "PI-SASR", "Deterministic UC"].index(labels[i]))
        fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside lower center", ncol=3)
        return _save(fig, out_dir, "fig4_unserved_energy", "Evaluation-set unserved-energy rate")


def write_figures(src: Sources, out_dir: Path) -> list[Path]:
    written = []
    for fn in (figure_solution_time, figure_speedup_kmin, figure_unserved_energy):
        try:
            written += fn(src, out_dir)
        except MissingSource as exc:
            print(f"[assets] skipped {fn.__name__}: missing {exc.relpath}")
    return written


# ----------------------------------------------------------------------------
# Register of the numbers quoted in the paper (paper_assets/numbers.md)
# ----------------------------------------------------------------------------
@dataclass
class Claim:
    """One number printed in the paper and the value computed from the CSV files.

    ``kind`` says how the two are compared: ``round`` (the computed value rounded
    to the printed precision equals the printed value), ``le``/``lt``/``gt``
    (the printed value is an upper or lower bound), ``approx`` (within ``tol``
    of a number that the paper qualifies with "about"), ``censored`` (the paper
    prints ">K": no tested budget met the tolerance and K is the largest tested)
    and ``bool`` (a qualitative statement checked by a condition).

    ``timing`` marks a wall-clock time or a ratio of times (a speedup). Times
    are not reproducible, so a time that differs from the printed value gets
    the status ``time differs`` instead of ``NO``.
    """

    group: str
    quantity: str
    printed: str
    source: str
    kind: str = "round"
    tol: float | None = None
    computed: str = ""
    unrounded: str = ""
    status: str = ""
    timing: bool = False


def _parse_printed(p: str) -> tuple[float, int]:
    t = p.replace("{,}", "").replace(",", "").replace("$", "").replace("%", "").strip()
    t = t.lstrip(">")
    decimals = len(t.split(".")[1]) if "." in t else 0
    return float(t), decimals


def _unrounded(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5:
        return f"{v:,.1f}"
    return f"{v:.6g}" if a >= 1e-3 else f"{v:.4g}"


_TIME_SOURCE = re.compile(r"solve_time_seconds|\btime_s\b|\bt_[a-z_]+_s\b")
"""Source columns that hold wall-clock times (``solve_time_seconds``, ``time_s``, ``t_*_s``)."""


class Register:
    def __init__(self, src: Sources):
        self.src = src
        self.claims: list[Claim] = []

    def add(self, group: str, quantity: str, printed: str, fn: Callable, source: str,
            kind: str = "round", tol: float | None = None) -> None:
        c = Claim(group, quantity, printed, source, kind, tol,
                  timing=bool(_TIME_SOURCE.search(source)))
        try:
            value = fn()
        except MissingSource as exc:
            c.status, c.computed = "not yet generated", f"missing `{exc.relpath}`"
        except KeyError as exc:
            c.status, c.computed = "not yet generated", f"missing row ({exc.args[0]})"
        else:
            self._judge(c, value)
        self.claims.append(c)

    @staticmethod
    def _judge(c: Claim, value) -> None:
        if c.kind == "bool":
            ok, text = value
            c.computed, c.status = text, "yes" if ok else "NO"
            return
        if c.kind == "censored":
            kmin, kmax = value
            c.computed = f">{kmax}" if kmin is None else str(kmin)
            c.unrounded = "no tested budget met the tolerance" if kmin is None else ""
            p, _ = _parse_printed(c.printed)
            c.status = "yes" if (kmin is None and c.printed.lstrip().startswith(">") and kmax == int(p)) else "NO"
            return
        v = float(value)
        p, d = _parse_printed(c.printed)
        if not math.isfinite(v):
            c.computed, c.status = "none", "NO"
            return
        c.unrounded = _unrounded(v)
        if c.kind == "round":
            c.computed = fmt(v, d, sign=c.printed.strip().startswith("+"), thousands="," in c.printed)
            ok = abs(float(c.computed.replace(",", "")) - p) < 0.5 * 10 ** (-d - 2)
        elif c.kind in ("le", "lt", "ge", "gt"):
            c.computed = fmt(v, d + 1)
            ok = {"le": v <= p, "lt": v < p, "ge": v >= p, "gt": v > p}[c.kind]
        elif c.kind == "approx":
            c.computed = fmt(v, max(d, 1))
            ok = abs(v - p) <= (c.tol if c.tol is not None else 0.1 * abs(p))
        else:
            raise ValueError(f"unknown kind {c.kind}")
        c.status = "yes" if ok else ("time differs" if c.timing else "NO")


# Values printed in the paper (camera-ready version, HICSS-60). Macro names of the
# paper source are given in the quantity labels where a macro prints the value.
_PAPER_T1 = {"tx123bt": ("138", "76.4", "77.2", "2,500", "2,000"),
             "texas2k": ("568", "77.8", "76.8", "2,500", "2,000")}
_PAPER_T2 = {  # det: obj, time, ens; full: obj, time, S, ens; pisasr: obj, gap, time, K, ens
    "tx123bt": (("37.76", "1.2", "1.755"), ("38.11", "434", "1000", "0.0026"),
                ("38.37", "0.69", "74", "148", "0.0060")),
    "texas2k": (("22.88", "3.6", "12.13"), ("28.59", "4324", "1000", "0.0019"),
                ("28.79", "0.71", "671", "146", "0.0010")),
}
_PAPER_T3 = {  # S: full obj, full time, gap, under-representation, PI-SASR time, K, speedup
    "tx123bt": {100: ("37.96", "70", "0.004", "-0.28", "56", "95", "1.26"),
                200: ("38.04", "108", "0.67", "-0.40", "62", "122", "1.74"),
                500: ("38.02", "305", "0.61", "-0.47", "84", "143", "3.61"),
                1000: ("38.11", "434", "0.69", "-1.33", "74", "148", "5.89"),
                2500: ("38.18", "1347", "0.83", "-1.12", "63", "146", "21.3")},
    "texas2k": {100: ("27.59", "568", "0.017", "-0.39", "419", "99", "1.36"),
                200: ("28.25", "1441", "0.082", "-6.32", "554", "128", "2.60"),
                500: ("28.64", "3381", "0.40", "-12.98", "567", "143", "5.96"),
                1000: ("28.59", "4324", "0.71", "-15.41", "671", "146", "6.44"),
                2500: ("28.56", "14689", "1.29", "-15.87", "613", "147", "24.0")},
}
_PAPER_T4 = {  # K, selection time, MILP time, online time, gap
    "ffs_K148": ("148", "0.3", "69", "70", "+4.99"),
    "medoids_K148": ("148", "0.1", "75", "76", "+1.39"),
    "costffs_K148": ("148", "1.4", "82", "85", "+0.94"),
    "exact_crit_K148": ("148", "1.0", "92", "94", "+0.20"),
    "gp_crit_K148": ("148", "6.1", "87", "94", "+0.49"),
    "gp_crit_K188": ("188", "6.1", "111", "118", "+0.39"),
    "exact_crit90_med60": ("147", "1.2", "93", "95", "+0.67"),
    "pisasr_nofix": ("148", "5.6", "103", "109", "+0.56"),
    "pisasr": ("148", "4.6", "71", "77", "+0.69"),
}
_PAPER_F2 = {"tx123bt": (("70.5", "108.2", "304.7", "433.7", "1347"), ("56", "62", "84", "74", "63")),
             "texas2k": (("568", "1441", "3381", "4324", "14689"), ("419", "554", "567", "671", "613"))}
_PAPER_F3A = {"tx123bt": ("1.26", "1.74", "3.61", "5.89", "21.35"),
              "texas2k": ("1.36", "2.60", "5.96", "6.44", "23.97")}
_PAPER_F3B = {  # (system, tau): {S: printed K_min, ">K" when no tested budget met tau}
    ("tx123bt", 1.0): {200: "90", 500: "143", 1000: "75", 2500: "100"},
    ("tx123bt", 0.5): {200: "90", 500: "182", 1000: ">191", 2500: "198"},
    ("texas2k", 1.0): {500: "49", 1000: "98", 2500: ">198"},
    ("texas2k", 0.5): {500: "96", 1000: ">191", 2500: ">198"},
}
_PAPER_F4 = {"tx123bt": (("0.0172", "0.0083", "0.0039", "0.0026", "0.0012"),
                         ("0.0191", "0.0119", "0.0075", "0.0060", "0.0059"), "1.755"),
             "texas2k": (("0.0518", "0.0136", "0.0013", "0.0019", "0.0026"),
                         ("0.0524", "0.0135", "0.0010", "0.0010", "0.0043"), "12.13")}

# Section 5.1 numbers that need the raw source data (downloaded by scripts/data/*),
# not the processed files included in data/.
RAW_DATA_NUMBERS = [
    ("TX-123BT wind and solar units netted into demand", "154", "raw TX-123BT generator workbook"),
    ("Texas2k wind and solar units and capacity", "400 units, 74.1 GW", "Texas2k Series25 case file"),
    ("Texas2k units of fuel type \"other\", excluded", "131", "Texas2k Series25 case file"),
    ("Texas2k summer-peak load of the released case", "85.8 GW", "Texas2k Series25 case file"),
    ("Summer days of the ACTIVSg2000 series used for the copula fit", "92", "ACTIVSg2000 time series"),
]


def build_register(src: Sources) -> Register:
    """Every number of the paper's results with its source and the recomputed value."""
    R = Register(src)
    add = R.add
    tx, tk = "tx123bt", "texas2k"
    SN = SHORT_NAMES
    ex = src.experiment
    solve = src.solve

    # ---------------- Table 1 and Section 5.1 ----------------
    g = "Table 1 and Section 5.1 (test systems)"
    for s in SYSTEM_KEYS:
        units, cap, peak, ntrain, neval = _PAPER_T1[s]
        add(g, f"{SN[s]}: committable units", units, lambda s=s: len(src.data(s, "generator_portfolio.csv")),
            f"data/{s}/generator_portfolio.csv: number of rows")
        add(g, f"{SN[s]}: committable capacity (GW)", cap,
            lambda s=s: src.data(s, "generator_portfolio.csv")["p_max_mw"].sum() / 1000,
            f"data/{s}/generator_portfolio.csv: sum of p_max_mw / 1000")
        add(g, f"{SN[s]}: maximum peak net load over the 7,500 scenarios (GW)", peak,
            lambda s=s: src.scenario_pool(s)[HOUR_COLS].max(axis=1).max() / 1000,
            f"data/{s}/net_load_scenarios_{{train,test}}.csv: largest hourly value / 1000")
        add(g, f"{SN[s]}: training scenarios", ntrain, lambda s=s: len(src.data(s, "net_load_scenarios_train.csv")),
            f"data/{s}/net_load_scenarios_train.csv: number of rows")
        add(g, f"{SN[s]}: evaluation scenarios", neval,
            lambda s=s: src.metrics(s, "full_S1000")["n_evaluation_scenarios"],
            f"{s}/runs/full_S1000/evaluation.csv: number of rows (first rows of the test file)")
        add(g, f"{SN[s]}: size of the scenario set (training and test files)", "7,500",
            lambda s=s: len(src.scenario_pool(s)), f"data/{s}/net_load_scenarios_{{train,test}}.csv: rows")
    add(g, "Texas2k has about four times as many committable units as TX-123BT (Table 1 caption)", "4",
        lambda: len(src.data(tk, "generator_portfolio.csv")) / len(src.data(tx, "generator_portfolio.csv")),
        "ratio of the two portfolio row counts", kind="approx", tol=0.5)
    add(g, "TX-123BT hydro stations", "10",
        lambda: int((src.data(tx, "generator_portfolio.csv")["fuel_type"].str.lower() == "hydro").sum()),
        "data/tx123bt/generator_portfolio.csv: fuel_type = Hydro (printed as \"ten\")")
    for fuel, label, printed in (("ng", "natural-gas", "507"), ("coal", "coal", "21"), ("nuclear", "nuclear", "4"),
                                 ("hydro", "hydro", "22"), ("dfo", "diesel (fuel_type dfo)", "10"),
                                 ("wood", "wood", "4")):
        add(g, f"Texas2k {label} units", printed,
            lambda fuel=fuel: int((src.data(tk, "generator_portfolio.csv")["fuel_type"] == fuel).sum()),
            f"data/texas2k/generator_portfolio.csv: fuel_type = {fuel}")
    add(g, "Texas2k hydro capacity (GW)", "0.55",
        lambda: src.data(tk, "generator_portfolio.csv").query("fuel_type == 'hydro'")["p_max_mw"].sum() / 1000,
        "data/texas2k/generator_portfolio.csv: sum of p_max_mw for hydro / 1000")
    add(g, "Texas2k mean peak net load over the 7,500 scenarios (GW)", "56.2",
        lambda: src.scenario_pool(tk)[HOUR_COLS].max(axis=1).mean() / 1000,
        "data/texas2k/net_load_scenarios_{train,test}.csv: mean of the daily maxima / 1000")

    # ---------------- Table 2 ----------------
    g = "Table 2 (primary comparison, S = 1,000)"
    for s in SYSTEM_KEYS:
        (d_obj, d_t, d_ens), (f_obj, f_t, f_S, f_ens), (p_obj, p_gap, p_t, p_K, p_ens) = _PAPER_T2[s]
        rs = f"{s}/runs"
        add(g, f"{SN[s]} deterministic: objective ($M)", d_obj, lambda s=s: solve(s, "deterministic")["objective"] / 1e6,
            f"{rs}/deterministic/solve_metrics.csv: objective / 1e6")
        add(g, f"{SN[s]} deterministic: time (s)", d_t, lambda s=s: solve(s, "deterministic")["solve_time_seconds"],
            f"{rs}/deterministic/solve_metrics.csv: solve_time_seconds")
        add(g, f"{SN[s]} deterministic: unserved energy (%)", d_ens, lambda s=s: src.ens_pct(s, "deterministic"),
            f"{rs}/deterministic/evaluation.csv: sum ens_mwh / sum total_net_load_mwh × 100")
        add(g, f"{SN[s]} full: objective ($M)", f_obj, lambda s=s: solve(s, "full_S1000")["objective"] / 1e6,
            f"{rs}/full_S1000/solve_metrics.csv: objective / 1e6")
        add(g, f"{SN[s]} full: time (s)", f_t, lambda s=s: solve(s, "full_S1000")["solve_time_seconds"],
            f"{rs}/full_S1000/solve_metrics.csv: solve_time_seconds")
        add(g, f"{SN[s]} full: scenarios used", f_S, lambda s=s: solve(s, "full_S1000")["n_scenarios"],
            f"{rs}/full_S1000/solve_metrics.csv: n_scenarios")
        add(g, f"{SN[s]} full: unserved energy (%)", f_ens, lambda s=s: src.ens_pct(s, "full_S1000"),
            f"{rs}/full_S1000/evaluation.csv: sum ens_mwh / sum total_net_load_mwh × 100")
        add(g, f"{SN[s]} PI-SASR: objective ($M)", p_obj, lambda s=s: solve(s, "pisasr_S1000")["objective"] / 1e6,
            f"{rs}/pisasr_S1000/solve_metrics.csv: objective / 1e6")
        add(g, f"{SN[s]} PI-SASR: benchmark gap (%)", p_gap, lambda s=s: solve(s, "pisasr_S1000")["opt_gap_pct_vs_full"],
            f"{rs}/pisasr_S1000/solve_metrics.csv: opt_gap_pct_vs_full")
        add(g, f"{SN[s]} PI-SASR: time (s)", p_t, lambda s=s: solve(s, "pisasr_S1000")["solve_time_seconds"],
            f"{rs}/pisasr_S1000/solve_metrics.csv: solve_time_seconds")
        add(g, f"{SN[s]} PI-SASR: scenarios used K", p_K, lambda s=s: solve(s, "pisasr_S1000")["n_scenarios_used"],
            f"{rs}/pisasr_S1000/solve_metrics.csv: n_scenarios_used")
        add(g, f"{SN[s]} PI-SASR: unserved energy (%)", p_ens, lambda s=s: src.ens_pct(s, "pisasr_S1000"),
            f"{rs}/pisasr_S1000/evaluation.csv: sum ens_mwh / sum total_net_load_mwh × 100")
    add(g, "Note c: Texas2k deterministic evaluation-set expected cost ($B)", "1.0",
        lambda: src.eval_cost(tk, "deterministic") / 1e9,
        "texas2k/runs/deterministic/evaluation.csv: mean total_cost_with_fixed / 1e9")
    for run, name in (("full_S1000", "full model"), ("pisasr_S1000", "PI-SASR")):
        add(g, f"Note c: ratio of that cost to the {name}'s (×)", "35",
            lambda run=run: src.eval_cost(tk, "deterministic") / src.eval_cost(tk, run),
            f"texas2k/runs/{{deterministic,{run}}}/evaluation.csv: ratio of mean total_cost_with_fixed")

    # ---------------- Table 3 ----------------
    g = "Table 3 (scenario scaling)"
    for s in SYSTEM_KEYS:
        for S, (fo, ft, gap, ur, pt, K, sp) in _PAPER_T3[s].items():
            row = lambda s=s, S=S: src.scaling_row(s, S)  # noqa: E731
            rs = f"{s}/runs"
            add(g, f"{SN[s]}, S = {S:,}: full objective ($M)", fo, lambda row=row: row()["full_obj"] / 1e6,
                f"{rs}/full_S{S}/solve_metrics.csv: objective / 1e6")
            add(g, f"{SN[s]}, S = {S:,}: full time (s)", ft, lambda row=row: row()["full_time_s"],
                f"{rs}/full_S{S}/solve_metrics.csv: solve_time_seconds")
            add(g, f"{SN[s]}, S = {S:,}: PI-SASR benchmark gap (%)", gap, lambda row=row: row()["pisasr_gap_pct"],
                f"{rs}/pisasr_S{S}/solve_metrics.csv: opt_gap_pct_vs_full")
            add(g, f"{SN[s]}, S = {S:,}: under-representation statistic (%)", ur,
                lambda s=s, S=S: src.under_rep_pct(s, S), f"{rs}/pisasr_S{S}/pisasr_log.csv: under_rep_gap × 100")
            add(g, f"{SN[s]}, S = {S:,}: PI-SASR time (s)", pt, lambda row=row: row()["pisasr_time_s"],
                f"{rs}/pisasr_S{S}/solve_metrics.csv: solve_time_seconds")
            add(g, f"{SN[s]}, S = {S:,}: active-set size K", K, lambda row=row: row()["K_used"],
                f"{rs}/pisasr_S{S}/solve_metrics.csv: n_scenarios_used")
            add(g, f"{SN[s]}, S = {S:,}: speedup (×)", sp, lambda row=row: row()["speedup_x"],
                f"{rs}/{{full,pisasr}}_S{S}/solve_metrics.csv: ratio of solve_time_seconds")

    # ---------------- Table 4 ----------------
    g = "Table 4 (component analysis, TX-123BT, S = 1,000)"
    comp = lambda v, col: row_where(ex(tx, "component_timing_S1000.csv"), "variant", v)[col]  # noqa: E731
    for v, (K, sel, milp, onl, gap) in _PAPER_T4.items():
        f = "tx123bt/experiments/component_timing_S1000.csv"
        add(g, f"{v}: K", K, lambda v=v: comp(v, "K"), f"{f}: K (variant {v})")
        add(g, f"{v}: selection time (s)", sel, lambda v=v: comp(v, "t_select_s"), f"{f}: t_select_s")
        add(g, f"{v}: reduced MILP time (s)", milp, lambda v=v: comp(v, "t_milp_s"), f"{f}: t_milp_s")
        add(g, f"{v}: online time (s)", onl, lambda v=v: comp(v, "t_online_s"), f"{f}: t_online_s")
        add(g, f"{v}: benchmark gap (%)", gap, lambda v=v: comp(v, "gap_pct"), f"{f}: gap_pct")

    # ---------------- Figures 2-4 ----------------
    g = "Figure 2 (solution time)"
    for s in SYSTEM_KEYS:
        full_t, pis_t = _PAPER_F2[s]
        for S, pf, pp in zip(S_GRID, full_t, pis_t):
            add(g, f"{SN[s]}, S = {S:,}: full time (s)", pf, lambda s=s, S=S: src.scaling_row(s, S)["full_time_s"],
                f"{s}/runs/full_S{S}/solve_metrics.csv: solve_time_seconds")
            add(g, f"{SN[s]}, S = {S:,}: PI-SASR time (s)", pp, lambda s=s, S=S: src.scaling_row(s, S)["pisasr_time_s"],
                f"{s}/runs/pisasr_S{S}/solve_metrics.csv: solve_time_seconds")
    g = "Figure 3 (speedup and minimum active-set size)"
    for s in SYSTEM_KEYS:
        for S, sp in zip(S_GRID, _PAPER_F3A[s]):
            add(g, f"(a) {SN[s]}, S = {S:,}: speedup (×)", sp, lambda s=s, S=S: src.scaling_row(s, S)["speedup_x"],
                f"{s}/runs/{{full,pisasr}}_S{S}/solve_metrics.csv: ratio of solve_time_seconds")
    for (s, tau), pts in _PAPER_F3B.items():
        for S, printed in pts.items():
            kind = "censored" if printed.startswith(">") else "round"
            fn = (lambda s=s, tau=tau, S=S: kmin_from_budget(ex(s, "budget_sensitivity.csv"), tau)[S][:2]) \
                if kind == "censored" else \
                (lambda s=s, tau=tau, S=S: _need(kmin_from_budget(ex(s, "budget_sensitivity.csv"), tau)[S][0]))
            add(g, f"(b) {SN[s]}, tau = {tau:g}%, S = {S:,}: K_min", printed, fn,
                f"{s}/experiments/budget_sensitivity.csv: smallest K with gap_pct <= {tau:g}", kind=kind)
    g = "Figure 4 (evaluation-set unserved energy, %)"
    for s in SYSTEM_KEYS:
        full_e, pis_e, det_e = _PAPER_F4[s]
        for S, pf, pp in zip(S_GRID, full_e, pis_e):
            add(g, f"{SN[s]}, S = {S:,}: full model", pf, lambda s=s, S=S: src.ens_pct(s, f"full_S{S}"),
                f"{s}/runs/full_S{S}/evaluation.csv: sum ens_mwh / sum total_net_load_mwh × 100")
            add(g, f"{SN[s]}, S = {S:,}: PI-SASR", pp, lambda s=s, S=S: src.ens_pct(s, f"pisasr_S{S}"),
                f"{s}/runs/pisasr_S{S}/evaluation.csv: same definition")
        add(g, f"{SN[s]}: deterministic (dashed line)", det_e, lambda s=s: src.ens_pct(s, "deterministic"),
            f"{s}/runs/deterministic/evaluation.csv: same definition")

    # ---------------- Section 4.1 ----------------
    g = "Section 4.1 (low-rank embedding)"
    spec = lambda s, r: row_where(ex(s, "embedding_spectrum_S1000.csv"), "rank", r)["cum_explained"] * 100  # noqa: E731
    add(g, "Texas2k: variance share of five modes (%) (\\tkSvdCumFive)", "83.1", lambda: spec(tk, 5),
        "texas2k/experiments/embedding_spectrum_S1000.csv: cum_explained at rank 5 × 100")
    add(g, "TX-123BT: variance share of five modes (%) (\\txSvdCumFive)", "30.9", lambda: spec(tx, 5),
        "tx123bt/experiments/embedding_spectrum_S1000.csv: cum_explained at rank 5 × 100")
    for space, printed, macro, label in (("svd5", "+0.69", "txSpaceFiveGap", "rank-5 embedding"),
                                         ("raw24", "+0.85", "txSpaceRawGap", "raw profiles"),
                                         ("svd2", "+0.94", "txSpaceTwoGap", "rank-2 embedding"),
                                         ("svd10", "+0.66", "txSpaceTenGap", "rank-10 embedding")):
        add(g, f"TX-123BT S = 1,000 benchmark gap, medoids clustered on the {label} (%) (\\{macro})", printed,
            lambda space=space: row_where(ex(tx, "embedding_space_S1000.csv"), "space", space)["gap_pct"],
            f"tx123bt/experiments/embedding_space_S1000.csv: gap_pct (space {space})")

    # ---------------- Section 4.2 ----------------
    g = "Section 4.2 (GP kernel choice)"
    kern = lambda s, k, col: _mean_rows(ex(s, "gp_kernels_S1000.csv"), "kernel", k, col)  # noqa: E731
    for k, name, mk, rho, top in (("matern52_ard", "Matérn-5/2", "MatFive", "0.70", "40"),
                                  ("matern32_ard", "Matérn-3/2", "MatThree", "0.72", "43"),
                                  ("matern12_ard", "Matérn-1/2", "MatOne", "0.75", "49"),
                                  ("rbf_ard", "squared-exponential", "Rbf", "0.70", "38"),
                                  ("matern52_iso", "isotropic Matérn-5/2", "Iso", "0.76", "46")):
        f = "tx123bt/experiments/gp_kernels_S1000.csv"
        add(g, f"TX-123BT {name}: Spearman rank correlation (\\txK{mk}Rho)", rho,
            lambda k=k: kern(tx, k, "spearman_allS"), f"{f}: mean of spearman_allS over the design-set draws")
        add(g, f"TX-123BT {name}: share of the exact top 90 recovered (%) (\\txK{mk}Top)", top,
            lambda k=k: 100 * kern(tx, k, "top90_overlap"), f"{f}: mean of top90_overlap × 100")
    add(g, "Number of random draws of the design set", "3",
        lambda: ex(tx, "gp_kernels_S1000.csv").query("kernel == 'matern52_ard'")["seed"].nunique(),
        "tx123bt/experiments/gp_kernels_S1000.csv: distinct seeds (printed as \"three\")")
    add(g, "Texas2k: every kernel exceeds this rank correlation", "0.98",
        lambda: ex(tk, "gp_kernels_S1000.csv").groupby("kernel")["spearman_allS"].mean().min(),
        "texas2k/experiments/gp_kernels_S1000.csv: smallest kernel mean of spearman_allS", kind="gt")

    # ---------------- Section 4.4 ----------------
    g = "Section 4.4 (cost of the certified gap)"
    lbt = lambda s: ex(s, "lower_bound_batches.csv").query("batch_size == 200")["time_s"].sum()  # noqa: E731
    add(g, "TX-123BT lower-bound batches, S' = 200 (min, \"about 15 min\")", "15", lambda: lbt(tx) / 60,
        "tx123bt/experiments/lower_bound_batches.csv: sum of time_s (batch_size 200) / 60", kind="approx", tol=2.0)
    add(g, "Texas2k lower-bound batches, S' = 200 (h, \"2 h\")", "2", lambda: lbt(tk) / 3600,
        "texas2k/experiments/lower_bound_batches.csv: sum of time_s (batch_size 200) / 3600", kind="approx", tol=0.25)

    # ---------------- Section 6.1 ----------------
    g = "Section 6.1 (optimality, certification, surrogate accuracy)"
    add(g, "TX-123BT benchmark gap at S = 1,000 (%) (\\txGapK)", "0.69",
        lambda: solve(tx, "pisasr_S1000")["opt_gap_pct_vs_full"], "tx123bt/runs/pisasr_S1000/solve_metrics.csv: opt_gap_pct_vs_full")
    add(g, "TX-123BT active-set size at S = 1,000 (\\txKred)", "148",
        lambda: solve(tx, "pisasr_S1000")["n_scenarios_used"], "tx123bt/runs/pisasr_S1000/solve_metrics.csv: n_scenarios_used")
    add(g, "Texas2k benchmark gap at S = 1,000 (%) (\\tkGapK)", "0.71",
        lambda: solve(tk, "pisasr_S1000")["opt_gap_pct_vs_full"], "texas2k/runs/pisasr_S1000/solve_metrics.csv: opt_gap_pct_vs_full")
    add(g, "Texas2k active-set size at S = 1,000 (\\tkKred)", "146",
        lambda: solve(tk, "pisasr_S1000")["n_scenarios_used"], "texas2k/runs/pisasr_S1000/solve_metrics.csv: n_scenarios_used")
    urs = lambda s: [src.under_rep_pct(s, S) for S in S_GRID]  # noqa: E731
    for s, lo, hi, mlo, mhi in ((tx, "-1.33", "-0.28", "txUrMin", "txUrMax"), (tk, "-15.87", "-0.39", "tkUrMin", "tkUrMax")):
        add(g, f"{SN[s]}: smallest under-representation statistic (%) (\\{mlo}; also Sec. 6.2)", lo,
            lambda s=s: min(urs(s)), f"{s}/runs/pisasr_S*/pisasr_log.csv: min of under_rep_gap × 100")
        add(g, f"{SN[s]}: largest under-representation statistic (%) (\\{mhi})", hi,
            lambda s=s: max(urs(s)), f"{s}/runs/pisasr_S*/pisasr_log.csv: max of under_rep_gap × 100")
    add(g, "Every under-representation statistic lies below the enrichment tolerance epsilon (%)", "0.2",
        lambda: max(urs(tx) + urs(tk)), "*/runs/pisasr_S*/pisasr_log.csv: max of under_rep_gap × 100", kind="lt")
    add(g, "Texas2k: reduced objective exceeds the verification cost by up to (%)", "19",
        lambda: max(100 * (1 / (1 + u / 100) - 1) for u in urs(tk)),
        "texas2k/runs/pisasr_S*/pisasr_log.csv: max of 1/(1 + under_rep_gap) - 1, × 100")
    for s, printed, macro in ((tx, "0.83", "txGapMax"), (tk, "1.29", "tkGapMax")):
        add(g, f"{SN[s]}: largest benchmark gap over S (%) (\\{macro}; also Sec. 8)", printed,
            lambda s=s: src.scaling(s)["pisasr_gap_pct"].max(), f"{s}/runs/pisasr_S*/solve_metrics.csv: max of opt_gap_pct_vs_full")
    cert = lambda s, case, col: row_where(ex(s, "certified_gap.csv").query("batch_size == 200"), "case", case)[col]  # noqa: E731
    for s, lb, cp, cf, diff, mk in ((tx, "37.90", "2.79", "1.68", "1.1", "tx"), (tk, "27.87", "3.76", "2.75", "1.0", "tk")):
        f = f"{s}/experiments/certified_gap.csv (batch_size 200)"
        add(g, f"{SN[s]}: lower confidence bound on the optimal expected cost ($M) (\\{mk}LbTwo)", lb,
            lambda s=s: cert(s, "pisasr", "lb_95") / 1e6, f"{f}: lb_95 / 1e6")
        add(g, f"{SN[s]}: certified gap of PI-SASR (%) (\\{mk}CertTwoPis)", cp,
            lambda s=s: cert(s, "pisasr", "gap_cert95_pct"), f"{f}: gap_cert95_pct (case pisasr)")
        add(g, f"{SN[s]}: certified gap of the full model (%) (\\{mk}CertTwoFull)", cf,
            lambda s=s: cert(s, "full", "gap_cert95_pct"), f"{f}: gap_cert95_pct (case full)")
        add(g, f"{SN[s]}: difference of the two certified gaps (pp) (\\{mk}CertTwoDiff)", diff,
            lambda s=s: cert(s, "pisasr", "gap_cert95_pct") - cert(s, "full", "gap_cert95_pct"),
            f"{f}: difference of gap_cert95_pct")
        add(g, f"{SN[s]}: number of batches M", "5", lambda s=s: cert(s, "pisasr", "M"), f"{f}: M")
        add(g, f"{SN[s]}: evaluation scenarios in the upper bound", "2,000", lambda s=s: cert(s, "pisasr", "n_eval"),
            f"{f}: n_eval")
    add(g, "Batch size S'", "200", lambda: ex(tx, "lower_bound_batches.csv")["batch_size"].min(),
        "tx123bt/experiments/lower_bound_batches.csv: batch_size")
    add(g, "TX-123BT GP coefficient of determination in the run of Table 2", "0.51",
        lambda: solve(tx, "pisasr_S1000")["surrogate_r2"], "tx123bt/runs/pisasr_S1000/solve_metrics.csv: surrogate_r2")
    add(g, "TX-123BT GP coefficient of determination, mean over three design-set draws (\\txKMatFiveR)", "0.18",
        lambda: kern(tx, "matern52_ard", "r2_holdout"),
        "tx123bt/experiments/gp_kernels_S1000.csv: mean of r2_holdout (matern52_ard)")
    add(g, "Texas2k GP coefficient of determination (\\tkRtwo)", "0.999",
        lambda: solve(tk, "pisasr_S1000")["surrogate_r2"], "texas2k/runs/pisasr_S1000/solve_metrics.csv: surrogate_r2")
    for s, printed in ((tx, "0.9"), (tk, "25")):
        add(g, f"{SN[s]}: stochastic over deterministic objective, S = 1,000 (%)", printed,
            lambda s=s: 100 * (solve(s, "full_S1000")["objective"] / solve(s, "deterministic")["objective"] - 1),
            f"{s}/runs/{{full_S1000,deterministic}}/solve_metrics.csv: objective")

    # ---------------- Section 6.2 ----------------
    g = "Section 6.2 (scaling with the number of scenarios)"
    for s, p100, pk, pmax, mk in ((tx, "1.3", "5.9", "21.3", "tx"), (tk, "1.4", "6.4", "24.0", "tk")):
        f = f"{s}/runs/{{{{full,pisasr}}}}_S{{S}}/solve_metrics.csv: ratio of solve_time_seconds"
        add(g, f"{SN[s]}: speedup at S = 100 (×)", p100, lambda s=s: src.scaling_row(s, 100)["speedup_x"],
            f.format(S=100))
        add(g, f"{SN[s]}: speedup at S = 1,000 (×) (\\{mk}SpdK)", pk,
            lambda s=s: src.scaling_row(s, 1000)["speedup_x"], f.format(S=1000))
        add(g, f"{SN[s]}: speedup at S = 2,500 (×) (\\{mk}SpdMax; also abstract)", pmax,
            lambda s=s: src.scaling_row(s, 2500)["speedup_x"], f.format(S=2500))
    add(g, "TX-123BT full model at S = 2,500 (min)", "22", lambda: src.scaling_row(tx, 2500)["full_time_s"] / 60,
        "tx123bt/runs/full_S2500/solve_metrics.csv: solve_time_seconds / 60")
    add(g, "TX-123BT PI-SASR at S = 2,500 (s)", "63", lambda: src.scaling_row(tx, 2500)["pisasr_time_s"],
        "tx123bt/runs/pisasr_S2500/solve_metrics.csv: solve_time_seconds")
    add(g, "Texas2k full model at S = 2,500 (h)", "4.1", lambda: src.scaling_row(tk, 2500)["full_time_s"] / 3600,
        "texas2k/runs/full_S2500/solve_metrics.csv: solve_time_seconds / 3600")
    add(g, "Texas2k PI-SASR at S = 2,500 (min)", "10",
        lambda: src.scaling_row(tk, 2500)["pisasr_time_s"] / 60,
        "texas2k/runs/pisasr_S2500/solve_metrics.csv: solve_time_seconds / 60")
    rep = lambda kind: ex(tx, "design_set_replication_S1000.csv").query(f"kind == '{kind}'")  # noqa: E731
    f = "tx123bt/experiments/design_set_replication_S1000.csv"
    add(g, "Design-set draws at S = 1,000 (printed as \"five\")", "5", lambda: len(rep("pisasr")), f"{f}: rows with kind pisasr")
    add(g, "Benchmark gap over the draws, mean (%)", "0.57", lambda: rep("pisasr")["gap_pct"].mean(), f"{f}: gap_pct (kind pisasr)")
    add(g, "Benchmark gap over the draws, standard deviation (%)", "0.18",
        lambda: rep("pisasr")["gap_pct"].std(ddof=1), f"{f}: gap_pct, sample standard deviation")
    add(g, "PI-SASR time over the draws, mean (s)", "106", lambda: rep("pisasr")["time_s"].mean(), f"{f}: time_s (kind pisasr)")
    add(g, "PI-SASR time over the draws, standard deviation (s)", "47", lambda: rep("pisasr")["time_s"].std(ddof=1),
        f"{f}: time_s, sample standard deviation")
    add(g, "Full model in separate runs, including model construction, mean (s)", "525", lambda: rep("full")["time_s"].mean(), f"{f}: time_s (kind full)")
    add(g, "Full model in separate runs, including model construction, standard deviation (s)", "11", lambda: rep("full")["time_s"].std(ddof=1),
        f"{f}: time_s, sample standard deviation")
    add(g, "Draws that needed one enrichment round (K above the budget of 150)", "1",
        lambda: int((rep("pisasr")["K"] > K_CRITICAL + K_COVER).sum()), f"{f}: rows with K > 150")
    kmid = lambda: [src.scaling_row(s, S)["K_used"] for s in SYSTEM_KEYS for S in S_GRID if S >= 500]  # noqa: E731
    add(g, "Smallest K for S >= 500, both systems", "143", lambda: min(kmid()), "*/runs/pisasr_S*/solve_metrics.csv: n_scenarios_used")
    add(g, "Largest K for S >= 500, both systems", "148", lambda: max(kmid()), "*/runs/pisasr_S*/solve_metrics.csv: n_scenarios_used")
    bs = lambda s: ex(s, "budget_sensitivity.csv")  # noqa: E731
    for s, smin, kc0, kr0 in ((tx, "200", "15", "10"), (tk, "500", "30", "20")):
        f = f"{s}/experiments/budget_sensitivity.csv"
        add(g, f"{SN[s]}: smallest S of the budget sensitivity analysis", smin, lambda s=s: bs(s)["S"].min(), f"{f}: S")
        add(g, f"{SN[s]}: smallest budget K_c", kc0, lambda s=s: bs(s)["k_critical"].min(), f"{f}: k_critical")
        add(g, f"{SN[s]}: smallest budget K_r", kr0, lambda s=s: bs(s)["k_cover"].min(), f"{f}: k_cover")
        add(g, f"{SN[s]}: largest budget K_c", "120", lambda s=s: bs(s)["k_critical"].max(), f"{f}: k_critical")
        add(g, f"{SN[s]}: largest budget K_r", "80", lambda s=s: bs(s)["k_cover"].max(), f"{f}: k_cover")
    km = lambda s, tau, S: _need(kmin_from_budget(bs(s), tau)[S][0])  # noqa: E731
    for s, tau, pts, mk in ((tx, 1.0, {200: "90", 500: "143", 1000: "75", 2500: "100"}, "txKminOne"),
                            (tk, 1.0, {500: "49", 1000: "98"}, "tkKminOne"),
                            (tx, 0.5, {200: "90", 500: "182", 2500: "198"}, "txKminHalf")):
        for S, printed in pts.items():
            add(g, f"{SN[s]}: K_min at tau = {tau:g}%, S = {S:,} (\\{mk}S{'abcde'[S_GRID.index(S)]})", printed,
                lambda s=s, tau=tau, S=S: km(s, tau, S),
                f"{s}/experiments/budget_sensitivity.csv: smallest K with gap_pct <= {tau:g}")
    add(g, "Texas2k, tau = 1%, S = 2,500: no tested budget met the tolerance (\\tkKminOneSe)", ">198",
        lambda: kmin_from_budget(bs(tk), 1.0)[2500][:2], "texas2k/experiments/budget_sensitivity.csv", kind="censored")
    add(g, "TX-123BT, tau = 0.5%, S = 1,000: no tested budget met the tolerance (\\txKminHalfSd)", ">191",
        lambda: kmin_from_budget(bs(tx), 0.5)[1000][:2], "tx123bt/experiments/budget_sensitivity.csv", kind="censored")
    big = lambda: row_where(bs(tk).query("S == 2500").sort_values("K"), "K", bs(tk).query("S == 2500")["K"].max())  # noqa: E731
    add(g, "Texas2k, S = 2,500: largest tested active set K", "198", lambda: big()["K"],
        "texas2k/experiments/budget_sensitivity.csv: max K at S = 2500")
    add(g, "Texas2k, S = 2,500: benchmark gap at that K (%)", "1.08", lambda: big()["gap_pct"],
        "texas2k/experiments/budget_sensitivity.csv: gap_pct")
    wt = lambda s, name, scheme, col: row_where(ex(s, name), "scheme", scheme)[col]  # noqa: E731
    for s in SYSTEM_KEYS:
        add(g, f"{SN[s]}: uniform share of the critical scenarios in the reduced mass (%, \"about 60%\")", "60",
            lambda s=s: 100 * wt(s, "scenario_weights_S1000.csv", "uniform", "critical_mass"),
            f"{s}/experiments/scenario_weights_S1000.csv: critical_mass (uniform) × 100", kind="approx", tol=5)
    f25 = "experiments/scenario_weights_S2500.csv"
    for s, mk, uni_gap, vor_gap in ((tk, "tk", "1.29", "0.97"), (tx, "tx", "0.90", "0.89")):
        add(g, f"{SN[s]} S = 2,500: gap under uniform weights, re-solved (%) (\\{mk}WtUniGap)", uni_gap,
            lambda s=s: wt(s, "scenario_weights_S2500.csv", "uniform", "gap_pct"), f"{s}/{f25}: gap_pct (uniform)")
        add(g, f"{SN[s]} S = 2,500: gap under nearest-neighbor redistribution (%) (\\{mk}WtVorGap)", vor_gap,
            lambda s=s: wt(s, "scenario_weights_S2500.csv", "voronoi_nn", "gap_pct"), f"{s}/{f25}: gap_pct (voronoi_nn)")
    add(g, "Texas2k S = 2,500: active-set size (\\tkWtVorK)", "147",
        lambda: wt(tk, "scenario_weights_S2500.csv", "voronoi_nn", "K"), f"texas2k/{f25}: K")
    add(g, "Texas2k S = 2,500: under-representation statistic under redistribution (%) (\\tkWtVorUr)", "+2.00",
        lambda: wt(tk, "scenario_weights_S2500.csv", "voronoi_nn", "under_rep_pct"), f"texas2k/{f25}: under_rep_pct")
    add(g, "TX-123BT: every under-representation statistic stays above (%)", "-1.4",
        lambda: min(urs(tx)), "tx123bt/runs/pisasr_S*/pisasr_log.csv: min of under_rep_gap × 100", kind="gt")

    # ---------------- Section 6.3 ----------------
    g = "Section 6.3 (resilience to generator outages)"
    cdiff = lambda s, S: 100 * (src.eval_cost(s, f"pisasr_S{S}") / src.eval_cost(s, f"full_S{S}") - 1)  # noqa: E731
    add(g, "Evaluation-set expected cost of PI-SASR within this of the full model's, every S (%)", "1.5",
        lambda: max(abs(cdiff(s, S)) for s in SYSTEM_KEYS for S in S_GRID),
        "*/runs/{pisasr,full}_S*/evaluation.csv: mean total_cost_with_fixed", kind="le")
    add(g, "The same for S <= 1,000 (%, \"about 1%\")", "1",
        lambda: max(abs(cdiff(s, S)) for s in SYSTEM_KEYS for S in S_GRID if S <= 1000),
        "*/runs/{pisasr,full}_S*/evaluation.csv: mean total_cost_with_fixed", kind="approx", tol=0.15)
    for s, pp, pf, d in ((tk, "0.0010", "0.0019", None), (tx, "0.006", "0.003", None)):
        add(g, f"{SN[s]} S = 1,000: PI-SASR unserved energy (%)", pp, lambda s=s: src.ens_pct(s, "pisasr_S1000"),
            f"{s}/runs/pisasr_S1000/evaluation.csv: sum ens_mwh / sum total_net_load_mwh × 100")
        add(g, f"{SN[s]} S = 1,000: full-model unserved energy (%)", pf, lambda s=s: src.ens_pct(s, "full_S1000"),
            f"{s}/runs/full_S1000/evaluation.csv: same definition")
    add(g, "Paired evaluation scenarios N", "2,000", lambda: src.paired(tx).loc["total_cost", "n_pairs"],
        "*/runs/{pisasr,full}_S1000/evaluation.csv: scenarios common to both")
    ci = lambda s, q, col: src.paired(s).loc[q, col]  # noqa: E731
    pf_src = "{s}/runs/{{pisasr,full}}_S1000/evaluation.csv, paired Student-t interval " \
             "(summary/paired_cost_difference_S1000.csv: {c})"
    for s, q, trio, macro in ((tx, "total_cost", ("0.9", "0.6", "1.2"), "txCI"),
                              (tk, "total_cost", ("1.1", "0.9", "1.3"), "tkCI"),
                              (tk, "dispatch_cost", ("+0.01", "-0.24", "0.26"), "tkCI")):
        what = "total cost" if q == "total_cost" else "dispatch cost alone"
        for printed, col, lab in zip(trio, ("mean_difference_pct", "ci_low_pct", "ci_high_pct"),
                                     ("mean difference", "95% interval, lower end", "95% interval, upper end")):
            add(g, f"{SN[s]} PI-SASR minus full, {what}: {lab} (%) (\\{macro})", printed,
                lambda s=s, q=q, col=col: ci(s, q, col), pf_src.format(s=s, c=col))
    ratios = lambda: [np.log10(src.ens_pct(s, "deterministic") / src.ens_pct(s, f"{m}_S{S}"))  # noqa: E731
                      for s in SYSTEM_KEYS for m in ("full", "pisasr") for S in S_GRID]
    add(g, "Unserved energy of both commitments is about two to four orders of magnitude below the deterministic "
           "schedule's (also abstract, Fig. 4 caption)", "2 to 4",
        lambda: (round(min(ratios())) >= 2 and round(max(ratios())) <= 4,
                 f"10^{min(ratios()):.2f} to 10^{max(ratios()):.2f}"),
        "*/runs/*/evaluation.csv: log10 of the deterministic rate over each full and PI-SASR rate "
        "(yes when the extremes round to 2 and 4)", kind="bool")
    add(g, "TX-123BT deterministic schedule sheds (%)", "1.75", lambda: src.ens_pct(tx, "deterministic"),
        "tx123bt/runs/deterministic/evaluation.csv: sum ens_mwh / sum total_net_load_mwh × 100")
    add(g, "Texas2k deterministic schedule sheds (%)", "12.1", lambda: src.ens_pct(tk, "deterministic"),
        "texas2k/runs/deterministic/evaluation.csv: same definition")
    add(g, "Full-model unserved energy falls from S = 100 to 200 to 500 on both systems", "yes",
        lambda: (all(src.ens_pct(s, "full_S100") > src.ens_pct(s, "full_S200") > src.ens_pct(s, "full_S500")
                     for s in SYSTEM_KEYS), "decreasing on both systems"),
        "*/runs/full_S{100,200,500}/evaluation.csv", kind="bool")
    add(g, "PI-SASR within one order of magnitude of the full model at every S", "yes",
        lambda: (all(0.1 <= src.ens_pct(s, f"pisasr_S{S}") / src.ens_pct(s, f"full_S{S}") <= 10
                     for s in SYSTEM_KEYS for S in S_GRID),
                 "ratios from {:.2f} to {:.2f}".format(
                     min(src.ens_pct(s, f"pisasr_S{S}") / src.ens_pct(s, f"full_S{S}") for s in SYSTEM_KEYS for S in S_GRID),
                     max(src.ens_pct(s, f"pisasr_S{S}") / src.ens_pct(s, f"full_S{S}") for s in SYSTEM_KEYS for S in S_GRID))),
        "*/runs/{pisasr,full}_S*/evaluation.csv", kind="bool")

    # ---------------- Section 6.4 ----------------
    g = "Section 6.4 (scenario weights, S = 1,000)"
    w1 = lambda s, scheme, col: wt(s, "scenario_weights_S1000.csv", scheme, col)  # noqa: E731
    dcost = lambda s, scheme: 100 * (w1(s, scheme, "eval_expected_cost") / w1(s, "uniform", "eval_expected_cost") - 1)  # noqa: E731
    names = {"uniform": ("Uni", "uniform weights"), "voronoi_nn": ("Vor", "nearest-neighbor redistribution"),
             "cluster_mass": ("Cl", "cluster-mass weights")}
    items = [
        (tx, "uniform", "crit", "61"), (tx, "voronoi_nn", "crit", "35"), (tx, "cluster_mass", "crit", "11"),
        (tx, "uniform", "gap", "+0.79"), (tx, "voronoi_nn", "gap", "+0.51"), (tx, "voronoi_nn", "dcost", "-0.25"),
        (tx, "uniform", "ens", "0.0063"), (tx, "voronoi_nn", "ens", "0.0054"),
        (tx, "cluster_mass", "gap", "+0.82"), (tx, "cluster_mass", "dcost", "+0.08"), (tx, "cluster_mass", "ens", "0.0065"),
        (tx, "cluster_mass", "ur", "+0.86"),
        (tk, "uniform", "gap", "+0.71"), (tk, "voronoi_nn", "gap", "+0.10"), (tk, "cluster_mass", "gap", "+0.08"),
        (tk, "voronoi_nn", "dcost", "-1.01"), (tk, "cluster_mass", "dcost", "-1.02"),
        (tk, "voronoi_nn", "hours", "8,213"), (tk, "cluster_mass", "hours", "8,200"), (tk, "uniform", "hours", "8,750"),
        (tk, "uniform", "ens", "0.0010"), (tk, "voronoi_nn", "ens", "0.0023"), (tk, "cluster_mass", "ens", "0.0025"),
    ]
    what = {"crit": ("share of the critical scenarios in the reduced mass (%)", "critical_mass × 100", "Crit"),
            "gap": ("benchmark gap (%)", "gap_pct", "Gap"),
            "dcost": ("change of the evaluation-set expected cost against uniform weights (%)",
                      "eval_expected_cost relative to the uniform row", "DCost"),
            "ens": ("evaluation-set unserved energy (%)", "eval_ens_rate_pct", "Ens"),
            "ur": ("under-representation statistic (%)", "under_rep_pct", "Ur"),
            "hours": ("committed unit-hours", "unit_hours_on", "Hours")}
    fn_of = {"crit": lambda s, sc: 100 * w1(s, sc, "critical_mass"), "gap": lambda s, sc: w1(s, sc, "gap_pct"),
             "dcost": dcost, "ens": lambda s, sc: w1(s, sc, "eval_ens_rate_pct"),
             "ur": lambda s, sc: w1(s, sc, "under_rep_pct"), "hours": lambda s, sc: w1(s, sc, "unit_hours_on")}
    for s, scheme, key, printed in items:
        label, col, mk = what[key]
        prefix = "tx" if s == tx else "tk"
        add(g, f"{SN[s]}, {names[scheme][1]}: {label} (\\{prefix}W{names[scheme][0]}{mk})", printed,
            lambda s=s, scheme=scheme, key=key: fn_of[key](s, scheme),
            f"{s}/experiments/scenario_weights_S1000.csv: {col} (scheme {scheme})")
    add(g, "Texas2k: the uniform rule raises the evaluation-set expected cost by about (%)", "1",
        lambda: -dcost(tk, "voronoi_nn"), "texas2k/experiments/scenario_weights_S1000.csv: eval_expected_cost",
        kind="approx", tol=0.2)

    # ---------------- Section 6.5 ----------------
    g = "Section 6.5 (accuracy of the merit-order verification)"
    for s, r2, err, mx, mk in ((tx, "0.9996", "0.66", "5.0", "txMoerr"), (tk, "0.9999", "0.57", "8.2", "tkMoerr")):
        f = f"{s}/summary/merit_order_validation.csv"
        mo = lambda s=s: src.result(s, "summary", "merit_order_validation.csv")  # noqa: E731
        add(g, f"{SN[s]}: squared correlation of the merit-order and LP values", r2,
            lambda mo=mo: np.corrcoef(mo()["merit_order"], mo()["lp"])[0, 1] ** 2,
            f"{f}: squared correlation of merit_order and lp")
        add(g, f"{SN[s]}: mean absolute error (%) (\\{mk})", err, lambda mo=mo: mo()["rel_err_pct"].abs().mean(),
            f"{f}: mean of |rel_err_pct|")
        add(g, f"{SN[s]}: largest single-scenario deviation (%)", mx, lambda mo=mo: mo()["rel_err_pct"].abs().max(),
            f"{f}: max of |rel_err_pct|")
        add(g, f"{SN[s]}: every deviation is an under-estimate", "yes",
            lambda mo=mo: (bool((mo()["rel_err_pct"] <= 0).all()), f"{int((mo()['rel_err_pct'] <= 0).sum())} of {len(mo())} rows"),
            f"{f}: rel_err_pct <= 0", kind="bool")
        add(g, f"{SN[s]}: contribution of the largest deviation to the error of the scenario average (pp)", "0.1",
            lambda mo=mo: _largest_share(mo()), f"{f}: (merit_order - lp) of that row / sum of lp × 100", kind="lt")

    # ---------------- Section 7.3 ----------------
    g = "Section 7.3 (contribution of each component)"
    for v, col, printed, mk, lab in (
            ("medoids_K148", "gap_pct", "+1.39", "timMedGap", "coverage medoids alone: gap (%)"),
            ("ffs_K148", "gap_pct", "+4.99", "timFfsGap", "fast-forward selection: gap (%)"),
            ("costffs_K148", "gap_pct", "+0.94", "timCffsGap", "cost-space forward selection: gap (%)"),
            ("exact_crit_K148", "t_select_s", "1.0", "timExKSel", "exact ranking of all scenarios: time (s)"),
            ("pisasr", "t_select_s", "4.6", "timPisSel", "design dispatches and GP fit: time (s)"),
            ("exact_crit_K148", "gap_pct", "+0.20", "timExKGap", "exact criticality only: gap (%)"),
            ("gp_crit_K148", "gap_pct", "+0.49", "timGpKGap", "GP criticality only: gap (%)"),
            ("exact_crit_K148", "t_online_s", "94", "timExKOn", "exact criticality only: online time (s)"),
            ("gp_crit_K148", "t_online_s", "94", "timGpKOn", "GP criticality only: online time (s) (\"in both cases\")"),
            ("pisasr", "gap_pct", "+0.69", "timPisGap", "PI-SASR: gap (%)"),
            ("pisasr", "t_online_s", "77", "timPisOn", "PI-SASR: online time (s)"),
            ("exact_crit90_med60", "gap_pct", "+0.67", "timExCovGap", "exact criticality with coverage: gap (%)"),
            ("pisasr_nofix", "gap_pct", "+0.56", "timNoFixGap", "no baseload pre-commitment: gap (%)"),
            ("pisasr", "t_milp_s", "71", "timPisMilp", "PI-SASR: reduced MILP time (s)"),
            ("pisasr_nofix", "t_milp_s", "103", "timNoFixMilp", "no baseload pre-commitment: reduced MILP time (s)")):
        add(g, f"TX-123BT {lab} (\\{mk})", printed, lambda v=v, col=col: comp(v, col),
            f"tx123bt/experiments/component_timing_S1000.csv: {col} (variant {v})")
    add(g, "Common active-set size of the variants", "148", lambda: comp("pisasr", "K"),
        "tx123bt/experiments/component_timing_S1000.csv: K")
    add(g, "Gap change when the GP is replaced by the exact ranking (pp)", "0.02",
        lambda: comp("pisasr", "gap_pct") - comp("exact_crit90_med60", "gap_pct"),
        "tx123bt/experiments/component_timing_S1000.csv: gap_pct (pisasr minus exact_crit90_med60)")
    cp = lambda s, v, col: row_where(ex(s, "criticality_only_S1000.csv"), "variant", v)[col]  # noqa: E731
    cpd = lambda s, v: 100 * (cp(s, v, "eval_expected_cost") / cp(s, "pisasr", "eval_expected_cost") - 1)  # noqa: E731
    for s, v, key, printed, mk in (
            (tx, "exact_crit_K148", "dcost", "-0.65", "txCpExKDCost"), (tx, "exact_crit_K148", "ens", "0.0035", "txCpExKEns"),
            (tx, "pisasr", "ens", "0.0060", "txCpPisEns"), (tx, "gp_crit_K148", "dcost", "-0.25", "txCpGpKDCost"),
            (tx, "gp_crit_K148", "ens", "0.0049", "txCpGpKEns"), (tx, "medoids_K148", "dcost", "+0.82", "txCpMedDCost"),
            (tx, "medoids_K148", "ens", "0.0089", "txCpMedEns"),
            (tk, "exact_crit_K148", "gap", "+1.92", "tkCpExKGap"), (tk, "exact_crit_K148", "dcost", "+1.18", "tkCpExKDCost"),
            (tk, "exact_crit_K148", "ens", "0.0007", "tkCpExKEns"), (tk, "pisasr", "ens", "0.0010", "tkCpPisEns"),
            (tk, "gp_crit_K148", "dcost", "+1.18", "tkCpGpKDCost"), (tk, "gp_crit_K148", "ens", "0.0007", "tkCpGpKEns")):
        f = f"{s}/experiments/criticality_only_S1000.csv"
        if key == "dcost":
            fn, col, lab = (lambda s=s, v=v: cpd(s, v)), "eval_expected_cost relative to pisasr", \
                "evaluation-set expected cost relative to PI-SASR (%)"
        elif key == "ens":
            fn, col, lab = (lambda s=s, v=v: cp(s, v, "eval_ens_rate_pct")), "eval_ens_rate_pct", \
                "evaluation-set unserved energy (%)"
        else:
            fn, col, lab = (lambda s=s, v=v: cp(s, v, "gap_pct")), "gap_pct", "benchmark gap (%)"
        add(g, f"{SN[s]} {v}: {lab} (\\{mk})", printed, fn, f"{f}: {col} (variant {v})")

    # ---------------- Abstract, Section 7.1 and Section 8 ----------------
    g = "Abstract, Section 7.1 and Section 8 (bounds restated)"
    add(g, "TX-123BT benchmark gap stays within (%) at every S (abstract)", "0.9",
        lambda: src.scaling(tx)["pisasr_gap_pct"].max(), "tx123bt/runs/pisasr_S*/solve_metrics.csv: max of opt_gap_pct_vs_full",
        kind="le")
    add(g, "Texas2k benchmark gap stays within (%) at every S (abstract, Sec. 7.1)", "1.3",
        lambda: src.scaling(tk)["pisasr_gap_pct"].max(), "texas2k/runs/pisasr_S*/solve_metrics.csv: max of opt_gap_pct_vs_full",
        kind="le")
    return R


GROUP_NOTES = {
    "Table 3 (scenario scaling)":
        "The under-representation statistic is the first row of `pisasr_log.csv` (the first reduced UC); every "
        "run of Table 3 has a single row because no enrichment round was needed.",
    "Figure 3 (speedup and minimum active-set size)":
        "A printed value \">K\" means that no tested budget met the tolerance at that S and K is the largest "
        "active set tested. Most run times in `budget_sensitivity.csv` were recorded while other experiments "
        "shared the workstation (`docs/REPRODUCIBILITY.md`, Section 4); the paper uses only its gaps and "
        "active-set sizes.",
    "Section 6.4 (scenario weights, S = 1,000)":
        "Each scheme re-solves the reduced UC on the active set of the S = 1,000 run; the evaluation-set columns "
        "come from the exact evaluation of each resulting commitment on the 2,000 evaluation scenarios.",
    "Section 6.5 (accuracy of the merit-order verification)":
        "Both validation files compare the merit-order recourse with a single-scenario ramp-coupled dispatch LP "
        "on the first 70 scenarios of the training set under the deterministic reference commitment "
        "(`scripts/validate_merit_order.py`), as stated in Section 6.5 of the paper. The ramp limits of this LP "
        "have no startup or shutdown allowance, so its errors are upper bounds on those relative to the "
        "evaluation LP (`docs/MODEL.md`, Section 7).",
    "Section 7.3 (contribution of each component)":
        "Times and gaps come from single dedicated runs on an otherwise idle workstation "
        "(`component_timing_S1000.csv`); the evaluation-set costs and unserved energy come from "
        "`criticality_only_S1000.csv`, whose run times are not used.",
}


def _need(value):
    """A censored K_min where the paper prints a number becomes NaN (counted as a difference)."""
    return math.nan if value is None else value


def _mean_rows(df: pd.DataFrame, column: str, value, target: str) -> float:
    sel = df[df[column] == value]
    if sel.empty:
        raise KeyError(f"no row with {column} = {value}")
    return float(sel[target].mean())


def _largest_share(mo: pd.DataFrame) -> float:
    i = mo["rel_err_pct"].abs().idxmax()
    return abs(100.0 * (mo.at[i, "merit_order"] - mo.at[i, "lp"]) / mo["lp"].sum())


def _md_escape(text: str) -> str:
    """Escape a table cell for GitHub Markdown: pipes, dollar signs and paper macro names."""
    text = str(text).replace("|", "\\|").replace("$", "\\$")
    return re.sub(r"\\([A-Za-z]+)", r"`\\\1`", text)


def render_numbers(register: Register, results_label: str) -> str:
    claims = register.claims
    n = len(claims)
    ok = sum(c.status == "yes" for c in claims)
    bad = [c for c in claims if c.status == "NO"]
    timed = [c for c in claims if c.status == "time differs"]
    missing = [c for c in claims if c.status == "not yet generated"]
    summary = f"**Summary.** {n} numbers checked: {ok} agree with the paper, {len(bad)} differ, "
    if timed:
        summary += f"{len(timed)} times or speedups differ (expected for a rerun), "
    summary += f"{len(missing)} not yet generated."
    lines = [
        "# Numbers quoted in the paper",
        "",
        f"Generated by `python scripts/make_paper_assets.py` from `{results_label}/` and `data/`; do not edit by hand.",
        "",
        "Every number printed in the results of the paper (Tables 1 to 4, Figures 2 to 4 and the numbers in "
        "Sections 4.1, 4.2, 4.4, 5.1, 6.1 to 6.5 and 7.3) is listed below with the file and column it comes "
        f"from (paths relative to `{results_label}/`, or to the repository root when they start with `data/`), "
        "the value computed from that file, and whether the two agree.",
        "",
        "* **Paper** is the value as printed in the paper; the paper's macro name is given in parentheses when a "
        "macro prints it.",
        "* **From CSV** is the computed value, rounded to the precision printed in the paper; **Unrounded** is the "
        "same value before rounding.",
        "* **Agrees** is `yes` when the rounded value equals the printed value. When the paper states a bound "
        "(\"within 1.5%\", \"exceeds 0.98\"), an approximate value (\"about 15 min\") or a qualitative statement, "
        "`yes` means that the computed value satisfies it. `NO` marks a difference; `not yet generated` means "
        "that the source file does not exist yet.",
        "* Wall-clock times, and speedups computed from them, cannot be reproduced exactly. When one of them "
        "differs from the paper, **Agrees** reads `time differs` rather than `NO`. In a rerun, objectives of "
        "MILPs solved again from scratch can also differ within the 0.1% MILP tolerance "
        "(`docs/REPRODUCIBILITY.md`, Section 8).",
        "",
        summary + " Section 7.2 of the paper summarizes a companion study whose code, data and results are "
        "not part of this repository.",
        "",
    ]
    header = ["| Quantity | Paper | From CSV | Unrounded | Agrees | Source |", "|---|---:|---:|---:|:---:|---|"]

    def row(c: Claim) -> str:
        agrees = {"yes": "yes", "NO": "**NO**"}.get(c.status, c.status)
        return (f"| {_md_escape(c.quantity)} | {_md_escape(c.printed)} | {_md_escape(c.computed)} | "
                f"{_md_escape(c.unrounded)} | {agrees} | {_md_escape(c.source)} |")

    if bad or missing:
        lines += ["## Values that differ from the paper or are not yet available", ""]
        lines += ["| Location | " + header[0][2:], "|---" + header[1]]
        for c in bad + missing:
            lines.append(f"| {_md_escape(c.group.split(' (')[0])} " + row(c))
        lines.append("")
    groups: dict[str, list[Claim]] = {}
    for c in claims:
        groups.setdefault(c.group, []).append(c)
    for gname, cs in groups.items():
        lines += [f"## {gname}", ""] + header + [row(c) for c in cs] + [""]
        if gname in GROUP_NOTES:
            lines += [GROUP_NOTES[gname], ""]
        if gname.startswith("Table 1"):
            lines += ["Numbers of Section 5.1 that need the raw source data (see `scripts/data/`) and are therefore "
                      "not recomputed here:", ""]
            lines += [f"* {q}: {v} ({where})" for q, v, where in RAW_DATA_NUMBERS] + [""]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_numbers(src: Sources, out_dir: Path, results_label: str = "results") -> tuple[Path, Register]:
    register = build_register(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "numbers.md"
    path.write_text(render_numbers(register, results_label))
    return path, register
