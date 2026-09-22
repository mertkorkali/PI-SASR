"""Additional experiments of the paper.

Each stage below writes one CSV file (two for ``embedding`` and ``lower-bound``) to
the ``experiments`` folder of the active output root, ``outputs/<system>/experiments``
by default. Each row is identified by its variant, kernel and seed, scheme, batch,
or scenario count and budget; rows already in the file are skipped, so an interrupted
stage resumes where it stopped, and ``force=True`` computes them again and replaces
them.

======================  ====================================  ==============================
Stage                   File(s)                               Paper
======================  ====================================  ==============================
``component-timing``    ``component_timing_S<S>.csv``         Table 4, Section 7.3
``criticality-only``    ``criticality_only_S<S>.csv``         Section 7.3
``gp-kernels``          ``gp_kernels_S<S>.csv``               Section 4.2
``embedding``           ``embedding_spectrum_S<S>.csv``,      Section 4.1
                        ``embedding_space_S<S>.csv``
``scenario-weights``    ``scenario_weights_S<S>.csv``         Section 6.4
``weights-s2500``       ``scenario_weights_S2500.csv``        Section 6.2
``lower-bound``         ``lower_bound_batches.csv``,          Sections 4.4 and 6.1
                        ``certified_gap.csv``
``budget-sensitivity``  ``budget_sensitivity.csv``            Figure 3(b), Section 6.2
(replication)           ``design_set_replication_S<S>.csv``   Section 6.2
(merit-order check)     ``merit_order_validation.csv``        Section 6.5
======================  ====================================  ==============================

The last two are run by ``scripts/run_replication.py`` and
``scripts/validate_merit_order.py``; the merit-order check writes to the
``summary`` folder of the output root instead.

Every stage uses the settings of PI-SASR in the paper (K_c = 90 critical scenarios,
K_r = 60 coverage scenarios, a design set of 140 scenarios, rank-5 embedding and
random seed 7), the reference commitment u_ref (by default the stored schedule
``results/<system>/runs/deterministic/commitment.csv``) and, for the benchmark gap,
the full-model objective of :func:`pisasr.systems.resolve_full_objective`.

Quantities written by several stages:

* ``gap_pct``: benchmark gap (%) of the exact expected cost of the commitment over
  the S training scenarios (ramp-coupled LP recourse) against the full-model
  objective;
* ``under_rep_pct``: under-representation statistic (%): merit-order verification
  cost over all S scenarios relative to the reduced objective;
* ``reduced_obj``, ``approx_obj``, ``objective``: reduced MILP objective, verified
  (merit-order) objective and exact expected cost ($);
* ``t_*_s``: wall-clock times (s). Only the component-timing stage of the paper ran
  on an otherwise idle workstation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from . import evaluation as ev
from . import systems
from . import uc_model as r
from .constants import HOUR_COLS, PISASR_DEFAULTS
from .evaluation import compute_fixed_commitment_cost, evaluate_commitment_batch
from .features import build_feature_matrix, lowrank_embedding
from .merit_order import MeritOrderDispatch, validate_against_lp
from .outages import GEN_FORCED_OUTAGE_RATE, draw_scenario_availability
from .paths import display_path
from .pisasr import _avail_vec, _exact_recourse_mean, read_commitment, run_pisasr
from .statistics import certified_gap
from .surrogate import GPSurrogate

__all__ = [
    "STAGES",
    "FILE_NAMES",
    "ExperimentContext",
    "ffs_select",
    "cost_ffs_select",
    "gp_with_kernel",
    "done_keys",
    "upsert_row",
    "stage_component_timing",
    "stage_criticality_only",
    "stage_gp_kernels",
    "embedding_spectrum",
    "stage_embedding",
    "stage_scenario_weights",
    "solve_lower_bound_batches",
    "certified_gap_table",
    "stage_lower_bound",
    "stage_budget_sensitivity",
    "design_set_replication",
    "merit_order_validation",
    "merit_order_summary",
    "run_stage",
]

# ----------------------------------------------------------------------------
# Settings of the paper
# ----------------------------------------------------------------------------

SEED = PISASR_DEFAULTS["random_seed"]        # 7
DESIGN = PISASR_DEFAULTS["design_size"]      # 140
KC = PISASR_DEFAULTS["k_critical"]           # 90
KR = PISASR_DEFAULTS["k_cover"]              # 60
RANK = PISASR_DEFAULTS["svd_rank"]           # 5

STAGES = (
    "component-timing",
    "criticality-only",
    "gp-kernels",
    "embedding",
    "scenario-weights",
    "weights-s2500",
    "lower-bound",
    "budget-sensitivity",
)
"""Stages of :func:`run_stage` (``scripts/run_experiments.py --stage``)."""

FILE_NAMES = {
    "component_timing": "component_timing_S{S}.csv",
    "criticality_only": "criticality_only_S{S}.csv",
    "gp_kernels": "gp_kernels_S{S}.csv",
    "embedding_spectrum": "embedding_spectrum_S{S}.csv",
    "embedding_space": "embedding_space_S{S}.csv",
    "scenario_weights": "scenario_weights_S{S}.csv",
    "lower_bound_batches": "lower_bound_batches.csv",
    "certified_gap": "certified_gap.csv",
    "budget_sensitivity": "budget_sensitivity.csv",
    "design_set_replication": "design_set_replication_S{S}.csv",
}
"""File names below ``<output root>/experiments/``."""

MERIT_ORDER_FILE = "merit_order_validation.csv"
"""File name of the merit-order check below ``<output root>/summary/``."""

MERIT_ORDER_SCENARIOS = 70
"""Number of training scenarios (the first rows of the training set) in the
merit-order check of Section 6.5."""

COMPONENT_VARIANTS = [
    "pisasr", "gp_crit_K148", "exact_crit_K148", "exact_crit90_med60",
    "gp_crit_K188", "medoids_K148", "ffs_K148", "costffs_K148", "pisasr_nofix",
]
"""Selection variants of the component analysis (Table 4). The fixed sizes 148 and
188 are the PI-SASR active-set size on TX-123BT at S = 1,000 and that size plus
the enrichment step of 40."""

CRITICALITY_ONLY_VARIANTS = {
    "tx123bt": ["pisasr", "gp_crit_K148", "exact_crit_K148", "medoids_K148", "gp_crit_K188"],
    "texas2k": ["pisasr", "exact_crit_K148", "gp_crit_K148"],
}
"""Variants evaluated on the evaluation set in Section 7.3, for each system."""

CRITICALITY_ONLY_CHOICES = ["pisasr", "gp_crit_K148", "exact_crit_K148", "medoids_K148", "gp_crit_K188"]
"""Variants accepted by the criticality-only stage."""

KERNELS = ["matern52_ard", "matern32_ard", "matern12_ard", "rbf_ard", "matern52_iso"]
"""GP kernels compared in Section 4.2."""

KERNEL_SEEDS = (7, 17, 27)
"""Random seeds of the design sets of the kernel comparison."""

EMBEDDING_SPACES = {
    "tx123bt": ["svd5", "raw24", "svd2", "svd10"],
    "texas2k": [],
}
"""Clustering spaces of the coverage medoids compared in Section 4.1 (TX-123BT
only; for Texas2k the paper gives the spectrum alone)."""

WEIGHT_SCHEMES = ["uniform", "voronoi_nn", "cluster_mass"]
"""Scenario-weighting schemes of Section 6.4."""

WEIGHT_SCHEMES_S2500 = ["uniform", "voronoi_nn"]
"""Schemes compared at S = 2,500 (Section 6.2), without the evaluation set."""

BUDGETS = {
    "tx123bt": [(15, 10), (30, 20), (45, 30), (60, 40), (90, 60), (120, 80)],
    "texas2k": [(30, 20), (45, 30), (60, 40), (90, 60), (120, 80)],
}
"""Budgets (K_c, K_r) of the budget-sensitivity study, for each system."""

DEFAULT_S = {
    "component-timing": {"tx123bt": [1000], "texas2k": [1000]},
    "criticality-only": {"tx123bt": [1000], "texas2k": [1000]},
    "gp-kernels": {"tx123bt": [1000], "texas2k": [1000]},
    "embedding": {"tx123bt": [1000], "texas2k": [1000]},
    "scenario-weights": {"tx123bt": [1000], "texas2k": [1000]},
    "weights-s2500": {"tx123bt": [2500], "texas2k": [2500]},
    "lower-bound": {"tx123bt": [200, 500], "texas2k": [200]},
    "budget-sensitivity": {"tx123bt": [200, 500, 1000, 2500], "texas2k": [500, 1000, 2500]},
}
"""Scenario counts of the paper for each stage and system. For ``lower-bound``
they are the batch sizes S'."""

LOWER_BOUND_BATCHES = 5
"""Number M of independent batches of the lower bound."""

LOWER_BOUND_S_REF = 1000
"""Scenario count of the two commitments whose certified gap is computed."""

LOWER_BOUND_MIP_GAP = 1e-3
"""Relative MILP tolerance of the batch problems (deflates each batch incumbent)."""

REPLICATION_SEEDS = (7, 17, 27, 37, 47)
"""Random seeds of the five design-set draws of Section 6.2."""

FULL_TIMING_REPEATS = 3
"""Number of separate solutions of the full model timed in the replication study."""

EVALUATION_BATCH = 100
"""Evaluation scenarios in one LP in this module (the LP separates across
scenarios, so the batch size does not change the cost of any scenario)."""


# ----------------------------------------------------------------------------
# Result files: skip rows already present, replace them with force
# ----------------------------------------------------------------------------

def _row_keys(df: pd.DataFrame, key_cols: list[str]) -> list[tuple]:
    return [tuple(str(v) for v in rec) for rec in df[key_cols].itertuples(index=False)]


def done_keys(path: Path, key_cols: list[str]) -> set:
    """Keys (as tuples of strings) of the rows already in ``path``."""
    path = Path(path)
    if not path.exists():
        return set()
    return set(_row_keys(pd.read_csv(path), key_cols))


def upsert_row(path: Path, row: dict, key_cols: list[str]) -> None:
    """Append ``row`` to the CSV file ``path``, replacing a row with the same key in place."""
    path = Path(path)
    new = pd.DataFrame([row])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(path, index=False)
        return
    df = pd.read_csv(path)
    keys = _row_keys(df, key_cols)
    key = tuple(str(row[c]) for c in key_cols)
    if key in keys:
        i = keys.index(key)
        before = [j for j, k in enumerate(keys) if j < i and k != key]
        after = [j for j, k in enumerate(keys) if j > i and k != key]
        df = pd.concat([df.iloc[before], new, df.iloc[after]], ignore_index=True)
    else:
        df = pd.concat([df, new], ignore_index=True)
    df.to_csv(path, index=False)


def _skip(key: tuple, done: set, force: bool, label: str) -> bool:
    if key in done and not force:
        print(f"[{label}] {', '.join(key)} already in the file; skipping (use --force to recompute)")
        return True
    return False


def _check_names(names, allowed, what: str) -> None:
    """Raise ``ValueError`` before any computation if a requested name is unknown."""
    unknown = [n for n in (names or []) if n not in allowed]
    if unknown:
        raise ValueError(f"Unknown {what} {unknown}; choose from {list(allowed)}")


def _output_path(out: Path, name: str, **fmt) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return out / FILE_NAMES[name].format(**fmt)


# ----------------------------------------------------------------------------
# Shared context
# ----------------------------------------------------------------------------

class ExperimentContext:
    """Data and helpers shared by the selection variants of one system and S.

    Parameters
    ----------
    S
        Scenario count; the S scenarios are the nested subset of the training set
        used by the full model.
    seed
        Random seed of the design set, of the split of the design set, of the GP
        restarts and of k-means.
    reference_commitment
        Path of u_ref (default: the stored schedule of the paper).
    """

    def __init__(self, S: int, seed: int = SEED, reference_commitment: str | Path | None = None):
        self.S = S
        self.seed = seed
        self.gen = r.load_generators()
        self.net = r.load_net_load_scenarios()
        self.units = [str(u) for u in self.gen["unit"]]
        self.target = r.select_stochastic_scenarios(self.net, S)
        self.tids = list(self.target.index)
        self.full_obj, self.full_obj_source = systems.resolve_full_objective(S)
        if reference_commitment is None:
            reference_commitment = systems.resolve_reference_commitment("results")
        self.reference_path = Path(reference_commitment)
        det = read_commitment(self.reference_path)
        self.det = det
        on_all = det.set_index("unit")[HOUR_COLS].min(axis=1)
        self.baseload = [str(u) for u, v in on_all.items() if int(round(v)) == 1]
        self.md_det = MeritOrderDispatch(self.gen, det)
        t = time.time()
        self.feats = build_feature_matrix(self.target, self.gen, rank=RANK,
                                          forced_outage_rate=GEN_FORCED_OUTAGE_RATE)
        self.t_features = time.time() - t
        self.feat_cols = list(self.feats.columns)
        self.X = self.feats[self.feat_cols].to_numpy(float)
        self.emb_cols = [c for c in self.feat_cols if c.startswith("svd_")]
        self.E = self.feats[self.emb_cols].to_numpy(float)
        self.raw = self.target[HOUR_COLS].to_numpy(float)
        self._mo_all = None

    # ---- exact criticality: merit-order recourse under u_ref of all S scenarios ----
    def exact_rank(self):
        """Merit-order recourse under u_ref of all S scenarios, and its time (s)."""
        t = time.time()
        if self._mo_all is None:
            self._mo_all = self.md_det.recourse_over(
                self.net, self.tids, avail_fn=lambda s: _avail_vec(self.units, s))
        return self._mo_all, time.time() - t

    # ---- GP criticality: design dispatches, fit and prediction, as in run_pisasr ----
    def gp_rank(self, seed=None, design=DESIGN):
        """GP-predicted recourse of all S scenarios (steps 2 and 3 of PI-SASR).

        The random-number calls (design set, then the split of the design set) are those of
        :func:`pisasr.pisasr.select_active_set`, so the prediction is identical.
        """
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        t = time.time()
        design_ids = [self.tids[i] for i in
                      rng.choice(len(self.tids), size=min(design, len(self.tids)), replace=False)]
        y = self.md_det.recourse_over(self.net, design_ids, avail_fn=lambda s: _avail_vec(self.units, s))
        t_design = time.time() - t
        Xd = self.feats.loc[design_ids, self.feat_cols].to_numpy(float)
        perm = rng.permutation(len(design_ids))
        n_hold = max(10, int(0.25 * len(design_ids)))
        tr, ho = perm[n_hold:], perm[:n_hold]
        t = time.time()
        gp = GPSurrogate(length_scale_dim=len(self.feat_cols), random_state=seed)
        gp.fit(Xd[tr], y[tr])
        acc = gp.score(Xd[ho], y[ho])
        gp.fit(Xd, y)
        pred = gp.predict(self.X)
        t_fit = time.time() - t
        return pred, {"t_design_s": t_design, "t_gp_s": t_fit, "r2_holdout": acc["r2"],
                      "design_ids": design_ids, "y_design": y}

    def medoids(self, k, space=None, seed=None):
        """k-means medoids of the scenarios in ``space`` (default: the rank-5 embedding)."""
        seed = self.seed if seed is None else seed
        E = self.E if space is None else space
        t = time.time()
        km = KMeans(n_clusters=min(k, len(self.tids)), n_init=4, random_state=seed)
        labels = km.fit_predict(E)
        ids = []
        for c in range(km.n_clusters):
            mem = np.where(labels == c)[0]
            if len(mem):
                d2 = ((E[mem] - km.cluster_centers_[c]) ** 2).sum(1)
                ids.append(self.tids[mem[int(np.argmin(d2))]])
        return ids, labels, time.time() - t

    def topk(self, score, k):
        """The k scenarios with the largest ``score``."""
        return [self.tids[i] for i in np.argsort(score)[::-1][:k]]

    # ---- reduced UC over an active set ----
    def solve_reduced(self, ids, probs=None, fix=True):
        """Reduced UC over ``ids``; baseload units of u_ref are fixed on when ``fix``."""
        ds = {str(s): self.net.loc[s, HOUR_COLS].to_numpy(float) for s in ids}
        av = {}
        for s in ids:
            a = draw_scenario_availability(self.units, s)
            for g in self.units:
                av[(str(s), g)] = a[g]
        if probs is None:
            probs = {s: 1.0 / len(ds) for s in ds}
        t = time.time()
        model = r.build_uc_master_model(self.gen, ds, probs, "stochastic", availability=av)
        if fix:
            for g in self.baseload:
                for tt in model.T:
                    model.u[g, tt].fix(1)
        metrics = r.solve_pyomo_model(model)
        commitment = r.get_commitment_from_model(model)
        return commitment, float(metrics["objective"]), time.time() - t

    # ---- merit-order verification over all S scenarios (online) ----
    def verify(self, commitment, reduced_obj):
        """Verified objective, under-representation statistic (fraction) and time (s)."""
        t = time.time()
        md = MeritOrderDispatch(self.gen, commitment)
        rec = md.recourse_over(self.net, self.tids, avail_fn=lambda s: _avail_vec(self.units, s))
        fixed = compute_fixed_commitment_cost(commitment, self.gen)["fixed_commitment_cost"]
        approx = fixed + float(rec.mean())
        return approx, (approx - reduced_obj) / reduced_obj, time.time() - t

    # ---- exact ramp-coupled objective over all S scenarios (offline) ----
    def exact_gap(self, commitment):
        """Exact expected cost over the S scenarios, benchmark gap (%) and time (s)."""
        t = time.time()
        exact = _exact_recourse_mean(commitment, self.gen, self.net, self.tids)
        fixed = compute_fixed_commitment_cost(commitment, self.gen)["fixed_commitment_cost"]
        obj = fixed + exact
        return obj, 100.0 * (obj - self.full_obj) / self.full_obj, time.time() - t

    # ---- evaluation set (2,000 scenarios), exact LP recourse ----
    def eval_pool(self, commitment, batch=EVALUATION_BATCH):
        """Mean and standard deviation of the total cost and the unserved energy on the evaluation set."""
        test = ev.load_net_load_scenarios()
        gen_ev = ev.load_generators()
        fixed = compute_fixed_commitment_cost(commitment, gen_ev)["fixed_commitment_cost"]
        parts = []
        ids = list(test.index)
        for st in range(0, len(ids), batch):
            parts.append(evaluate_commitment_batch(commitment, gen_ev, test.loc[ids[st:st + batch]]))
        res = pd.concat(parts, ignore_index=True)
        tot = fixed + res["dispatch_cost"]
        return {
            "eval_expected_cost": float(tot.mean()),
            "eval_cost_sd": float(tot.std(ddof=1)),
            "eval_ens_rate_pct": 100.0 * float(res["ens_mwh"].sum() / max(res["total_net_load_mwh"].sum(), 1e-6)),
            "eval_ens_mwh": float(res["ens_mwh"].mean()),
            "eval_n": int(len(res)),
        }


# ----------------------------------------------------------------------------
# Selection rules used only for comparison
# ----------------------------------------------------------------------------

def ffs_select(X: np.ndarray, k: int) -> list[int]:
    """Fast-forward selection of Heitsch and Roemisch (Euclidean distance, equal weights).

    Returns the row indices of the ``k`` selected scenarios in order of selection.
    """
    S = X.shape[0]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
    sel, dmin = [], np.full(S, np.inf)
    remaining = list(range(S))
    for _ in range(k):
        vals = [(np.minimum(dmin, D[:, c]).sum(), c) for c in remaining]
        best = min(vals)[1]
        sel.append(best)
        remaining.remove(best)
        dmin = np.minimum(dmin, D[:, best])
    return sel


def cost_ffs_select(mo: np.ndarray, k: int) -> list[int]:
    """Forward selection with the distance |Q_s - Q_s'| between recourse costs."""
    S = len(mo)
    sel, dmin = [], np.full(S, np.inf)
    for _ in range(k):
        best, best_val = None, np.inf
        for c in range(S):
            if c in sel:
                continue
            val = np.minimum(dmin, np.abs(mo - mo[c])).sum()
            if val < best_val:
                best, best_val = c, val
        sel.append(best)
        dmin = np.minimum(dmin, np.abs(mo - mo[best]))
    return sel


def gp_with_kernel(name: str, dim: int, seed: int) -> GaussianProcessRegressor:
    """GP regressor with one of the kernels of :data:`KERNELS` (Section 4.2).

    ``*_ard`` kernels have one length scale for each of the ``dim`` features;
    ``matern52_iso`` has a single length scale. Every kernel is scaled by a
    constant and has a white-noise term, as in :class:`pisasr.surrogate.GPSurrogate`.
    """
    ls = np.ones(dim)
    if name == "matern52_ard":
        k = Matern(length_scale=ls, length_scale_bounds=(1e-3, 1e6), nu=2.5)
    elif name == "matern32_ard":
        k = Matern(length_scale=ls, length_scale_bounds=(1e-3, 1e6), nu=1.5)
    elif name == "matern12_ard":
        k = Matern(length_scale=ls, length_scale_bounds=(1e-3, 1e6), nu=0.5)
    elif name == "rbf_ard":
        k = RBF(length_scale=ls, length_scale_bounds=(1e-3, 1e6))
    elif name == "matern52_iso":
        k = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e6), nu=2.5)
    else:
        raise ValueError(f"Unknown kernel {name!r}; choose from {KERNELS}")
    kernel = ConstantKernel(1.0, (1e-3, 1e6)) * k + WhiteKernel(1e-2, (1e-10, 1e2))
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=4, random_state=seed)


# ============================================================================
# Stage: component-timing (Table 4)
# ============================================================================

def stage_component_timing(out: Path, S: int = 1000, variants=None, seed: int = SEED,
                           reference_commitment=None, force: bool = False) -> Path:
    """Online time and benchmark gap of each selection variant at equal active-set size.

    For each variant the online time is the sum of the feature computation, the
    selection (design dispatches and GP fit, or the exact merit-order ranking), the
    reduced MILP and the merit-order verification; the exact recomputation behind
    ``gap_pct`` is timed separately (``t_exact_eval_s``).
    """
    _check_names(variants, COMPONENT_VARIANTS, "variant(s)")
    path = _output_path(out, "component_timing", S=S)
    done = done_keys(path, ["variant"])
    c = ExperimentContext(S, seed=seed, reference_commitment=reference_commitment)
    print(f"[component-timing] S={S} features {c.t_features:.2f} s, baseload {len(c.baseload)} units")

    for v in variants or COMPONENT_VARIANTS:
        if _skip((v,), done, force, "component-timing"):
            continue
        t_sel = 0.0
        fix = True
        extra = {}
        if v in ("pisasr", "pisasr_nofix"):
            pred, info = c.gp_rank()
            t_sel += info["t_design_s"] + info["t_gp_s"]
            med, _, tm = c.medoids(KR)
            t_sel += tm
            ids = list(dict.fromkeys(c.topk(pred, KC) + med))
            fix = (v == "pisasr")
            extra["r2_holdout"] = info["r2_holdout"]
        elif v == "gp_crit_K148":
            pred, info = c.gp_rank()
            t_sel += info["t_design_s"] + info["t_gp_s"]
            ids = c.topk(pred, 148)
            extra["r2_holdout"] = info["r2_holdout"]
        elif v == "gp_crit_K188":
            pred, info = c.gp_rank()
            t_sel += info["t_design_s"] + info["t_gp_s"]
            ids = c.topk(pred, 188)
            extra["r2_holdout"] = info["r2_holdout"]
        elif v == "exact_crit_K148":
            mo, tr = c.exact_rank()
            c._mo_all = None
            t_sel += tr
            ids = c.topk(mo, 148)
        elif v == "exact_crit90_med60":
            mo, tr = c.exact_rank()
            c._mo_all = None
            t_sel += tr
            med, _, tm = c.medoids(KR)
            t_sel += tm
            ids = list(dict.fromkeys(c.topk(mo, KC) + med))
        elif v == "medoids_K148":
            med, _, tm = c.medoids(148)
            t_sel += tm
            ids = med
        elif v == "ffs_K148":
            t = time.time()
            sel = ffs_select(c.raw, 148)
            t_sel += time.time() - t
            ids = [c.tids[i] for i in sel]
        elif v == "costffs_K148":
            mo, tr = c.exact_rank()
            c._mo_all = None
            t_sel += tr
            t = time.time()
            sel = cost_ffs_select(mo, 148)
            t_sel += time.time() - t
            ids = [c.tids[i] for i in sel]
        else:
            raise ValueError(f"Unknown variant {v!r}; choose from {COMPONENT_VARIANTS}")
        commitment, red_obj, t_milp = c.solve_reduced(ids, fix=fix)
        approx, under_rep, t_ver = c.verify(commitment, red_obj)
        obj, gap, t_exact = c.exact_gap(commitment)
        row = {"variant": v, "K": len(ids), "t_features_s": c.t_features, "t_select_s": t_sel,
               "t_milp_s": t_milp, "t_verify_s": t_ver,
               "t_online_s": c.t_features + t_sel + t_milp + t_ver,
               "t_exact_eval_s": t_exact, "reduced_obj": red_obj, "approx_obj": approx,
               "under_rep_pct": 100 * under_rep, "objective": obj, "gap_pct": gap, **extra}
        upsert_row(path, row, ["variant"])
        print(f"[component-timing] {v}: K={len(ids)} gap={gap:+.3f}% online={row['t_online_s']:.0f} s "
              f"(selection {t_sel:.1f} s, MILP {t_milp:.0f} s, verification {t_ver:.1f} s)")
    return path


# ============================================================================
# Stage: criticality-only (Section 7.3)
# ============================================================================

def stage_criticality_only(out: Path, S: int = 1000, variants=None, seed: int = SEED,
                           reference_commitment=None, force: bool = False) -> Path:
    """Evaluation-set cost and unserved energy of criticality-only selections.

    The criticality-only variants take the top of the GP or exact ranking at the
    PI-SASR active-set size, without coverage scenarios; PI-SASR and the coverage
    medoids alone are included for comparison. The benchmark gap alone cannot show
    the value of the coverage scenarios, so each commitment is also evaluated on
    the 2,000 scenarios of the evaluation set.
    """
    _check_names(variants, CRITICALITY_ONLY_CHOICES, "variant(s)")
    path = _output_path(out, "criticality_only", S=S)
    done = done_keys(path, ["variant"])
    c = ExperimentContext(S, seed=seed, reference_commitment=reference_commitment)
    variants = variants or CRITICALITY_ONLY_VARIANTS[systems.active_system().key]
    for v in variants:
        if _skip((v,), done, force, "criticality-only"):
            continue
        if v == "pisasr":
            pred, _ = c.gp_rank()
            med, _, _ = c.medoids(KR)
            ids = list(dict.fromkeys(c.topk(pred, KC) + med))
        elif v == "gp_crit_K148":
            pred, _ = c.gp_rank()
            ids = c.topk(pred, 148)
        elif v == "gp_crit_K188":
            pred, _ = c.gp_rank()
            ids = c.topk(pred, 188)
        elif v == "exact_crit_K148":
            mo, _ = c.exact_rank()
            ids = c.topk(mo, 148)
        elif v == "medoids_K148":
            ids, _, _ = c.medoids(148)
        else:
            raise ValueError(f"Unknown variant {v!r}; choose from {CRITICALITY_ONLY_CHOICES}")
        commitment, red_obj, t_milp = c.solve_reduced(ids)
        approx, under_rep, _ = c.verify(commitment, red_obj)
        obj, gap, _ = c.exact_gap(commitment)
        evm = c.eval_pool(commitment)
        row = {"variant": v, "K": len(ids), "gap_pct": gap, "under_rep_pct": 100 * under_rep,
               "t_milp_s": t_milp, "unit_hours_on": int(commitment[HOUR_COLS].to_numpy().sum()), **evm}
        upsert_row(path, row, ["variant"])
        print(f"[criticality-only] {v}: gap={gap:+.3f}% evaluation cost={evm['eval_expected_cost']:,.0f} "
              f"unserved energy={evm['eval_ens_rate_pct']:.4f}%")
    return path


# ============================================================================
# Stage: gp-kernels (Section 4.2)
# ============================================================================

def stage_gp_kernels(out: Path, S: int = 1000, kernels=None, seeds=KERNEL_SEEDS,
                     reference_commitment=None, force: bool = False) -> Path:
    """Accuracy of GP kernels as criticality surrogates (no MILP).

    For each design-set seed, 140 scenarios are drawn and a quarter of them is kept
    aside; the target is the exact criticality (merit-order recourse under u_ref).
    Each kernel is fitted on the rest, scored on the quarter kept aside (R^2), fitted
    again on the whole design set and used to rank all S scenarios, which is compared
    with the exact ranking (Spearman correlation and overlap of the top 90 and top 148).
    """
    _check_names(kernels, KERNELS, "kernel(s)")
    path = _output_path(out, "gp_kernels", S=S)
    done = done_keys(path, ["kernel", "seed"])
    c = ExperimentContext(S, reference_commitment=reference_commitment)
    mo, _ = c.exact_rank()
    top90 = set(np.argsort(mo)[::-1][:KC])
    top148 = set(np.argsort(mo)[::-1][:148])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        design_idx = rng.choice(len(c.tids), size=DESIGN, replace=False)
        Xd, yd = c.X[design_idx], mo[design_idx]
        perm = rng.permutation(len(design_idx))
        n_hold = max(10, int(0.25 * len(design_idx)))
        tr, ho = perm[n_hold:], perm[:n_hold]
        for kn in kernels or KERNELS:
            if _skip((kn, str(seed)), done, force, "gp-kernels"):
                continue
            xs = StandardScaler().fit(Xd)
            ym, ys = float(yd.mean()), float(yd.std()) or 1.0
            t = time.time()
            gp = gp_with_kernel(kn, c.X.shape[1], seed).fit(xs.transform(Xd[tr]), (yd[tr] - ym) / ys)
            r2 = r2_score(yd[ho], gp.predict(xs.transform(Xd[ho])) * ys + ym)
            gp = gp_with_kernel(kn, c.X.shape[1], seed).fit(xs.transform(Xd), (yd - ym) / ys)
            pred = gp.predict(xs.transform(c.X)) * ys + ym
            t_fit = time.time() - t
            rho = stats.spearmanr(pred, mo).correlation
            p90 = set(np.argsort(pred)[::-1][:KC])
            p148 = set(np.argsort(pred)[::-1][:148])
            row = {"kernel": kn, "seed": seed, "r2_holdout": r2, "spearman_allS": rho,
                   "top90_overlap": len(p90 & top90) / KC, "top148_overlap": len(p148 & top148) / 148,
                   "t_fit_s": t_fit}
            upsert_row(path, row, ["kernel", "seed"])
            print(f"[gp-kernels] {kn} seed {seed}: R^2={r2:.3f} Spearman={rho:.3f} "
                  f"top-90 overlap={row['top90_overlap']:.2f}")
    return path


# ============================================================================
# Stage: embedding (Section 4.1)
# ============================================================================

def embedding_spectrum(target: pd.DataFrame) -> pd.DataFrame:
    """Share of variance of each singular direction of the centered S × 24 net-load matrix."""
    raw = target[HOUR_COLS].to_numpy(float)
    Xc = raw - raw.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    ev_ = sv ** 2 / (sv ** 2).sum()
    return pd.DataFrame({"rank": np.arange(1, len(sv) + 1), "explained": ev_,
                         "cum_explained": np.cumsum(ev_)})


def stage_embedding(out: Path, S: int = 1000, spaces=None, spectrum_only: bool = False,
                    seed: int = SEED, reference_commitment=None, force: bool = False) -> list[Path]:
    """Spectrum of the net-load matrix and the clustering space of the coverage medoids.

    Writes the spectrum (``embedding_spectrum_S<S>.csv``) and then, unless
    ``spectrum_only``, runs PI-SASR with the coverage medoids clustered in each space
    (``svd5`` default, ``raw24`` raw 24-hour profiles, ``svd2``, ``svd10``) and
    records the benchmark gap (``embedding_space_S<S>.csv``). The critical scenarios
    are the same in every space.
    """
    written = []
    c = ExperimentContext(S, seed=seed, reference_commitment=reference_commitment)
    spath = _output_path(out, "embedding_spectrum", S=S)
    if spath.exists() and not force:
        print(f"[embedding] {display_path(spath)} exists; skipping the spectrum (use --force to recompute)")
    else:
        spec = embedding_spectrum(c.target)
        spec.to_csv(spath, index=False)
        print(f"[embedding] cumulative variance share, ranks 1-6: "
              f"{np.round(spec['cum_explained'].to_numpy()[:6], 4)}")
    written.append(spath)

    if spectrum_only:
        return written
    spaces = EMBEDDING_SPACES[systems.active_system().key] if spaces is None else spaces
    if not spaces:
        print("[embedding] no clustering spaces requested for this system; spectrum only")
        return written

    path = _output_path(out, "embedding_space", S=S)
    done = done_keys(path, ["space"])
    todo = [name for name in spaces if not _skip((name,), done, force, "embedding")]
    if not todo:
        return written + [path]
    available = {"svd5": lambda: c.E, "raw24": lambda: c.raw,
                 "svd2": lambda: lowrank_embedding(c.target, 2).to_numpy(float),
                 "svd10": lambda: lowrank_embedding(c.target, 10).to_numpy(float)}
    unknown = [name for name in todo if name not in available]
    if unknown:
        raise ValueError(f"Unknown space(s) {unknown}; choose from {list(available)}")
    pred, info = c.gp_rank()
    for name in todo:
        E = available[name]()
        med, _, tm = c.medoids(KR, space=E)
        ids = list(dict.fromkeys(c.topk(pred, KC) + med))
        commitment, red_obj, t_milp = c.solve_reduced(ids)
        approx, under_rep, _ = c.verify(commitment, red_obj)
        obj, gap, _ = c.exact_gap(commitment)
        row = {"space": name, "K": len(ids), "t_kmeans_s": tm, "t_milp_s": t_milp,
               "under_rep_pct": 100 * under_rep, "objective": obj, "gap_pct": gap}
        upsert_row(path, row, ["space"])
        print(f"[embedding] {name}: K={len(ids)} gap={gap:+.3f}%")
    return written + [path]


# ============================================================================
# Stages: scenario-weights (Section 6.4) and weights-s2500 (Section 6.2)
# ============================================================================

def stage_scenario_weights(out: Path, S: int = 1000, evaluation: bool = True, schemes=None,
                           seed: int = SEED, reference_commitment=None, force: bool = False) -> Path:
    """Reduced UC on the PI-SASR active set with three weighting schemes.

    * ``uniform``: weight 1/K for every active-set scenario (PI-SASR);
    * ``voronoi_nn``: each of the S scenarios passes its probability 1/S to the
      nearest active-set scenario in the embedding (Dupacova, Growe-Kuska and
      Roemisch);
    * ``cluster_mass``: each critical scenario keeps its own 1/S and the mass of
      every k-means cluster goes to its medoid (to the nearest active-set scenario
      when the medoid is not in the active set).

    With ``evaluation`` each commitment is also evaluated on the evaluation set;
    otherwise the ``eval_*`` columns are NaN and ``eval_n`` is 0.
    """
    path = _output_path(out, "scenario_weights", S=S)
    done = done_keys(path, ["scheme"])
    c = ExperimentContext(S, seed=seed, reference_commitment=reference_commitment)
    pred, info = c.gp_rank()
    med, labels, _ = c.medoids(KR)
    crit = c.topk(pred, KC)
    ids = list(dict.fromkeys(crit + med))
    pos = {s: i for i, s in enumerate(c.tids)}
    ret_idx = np.array([pos[s] for s in ids])

    all_schemes = {}
    all_schemes["uniform"] = {str(s): 1.0 / len(ids) for s in ids}
    # (a) nearest active-set scenario in the embedding space
    d2 = ((c.E[:, None, :] - c.E[ret_idx][None, :, :]) ** 2).sum(2)
    nearest = d2.argmin(1)
    counts = np.bincount(nearest, minlength=len(ids))
    all_schemes["voronoi_nn"] = {str(ids[j]): counts[j] / S for j in range(len(ids))}
    # (b) cluster mass: the mass of each k-means cluster goes to its medoid, while the
    #     critical scenarios keep their own 1/S (removed from their cluster's mass)
    w = np.zeros(len(ids))
    med_pos = {m: j for j, m in enumerate(ids) if m in set(med)}
    clus_of = {}
    for cl in range(labels.max() + 1):
        mem = np.where(labels == cl)[0]
        if not len(mem):
            continue
        m_id = None
        for j in mem:
            if c.tids[j] in med_pos:
                m_id = c.tids[j]
                break
        clus_of[cl] = m_id
    for i, s in enumerate(c.tids):
        j = ids.index(s) if s in set(ids) else None
        if j is not None:
            w[j] += 1.0 / S
        else:
            m_id = clus_of.get(labels[i])
            if m_id is None:  # cluster without an active-set medoid: nearest active-set scenario
                w[nearest[i]] += 1.0 / S
            else:
                w[ids.index(m_id)] += 1.0 / S
    all_schemes["cluster_mass"] = {str(ids[j]): float(w[j]) for j in range(len(ids))}

    requested = list(schemes) if schemes else list(all_schemes)
    unknown = [name for name in requested if name not in all_schemes]
    if unknown:
        raise ValueError(f"Unknown scheme(s) {unknown}; choose from {list(all_schemes)}")
    for name in requested:
        probs = all_schemes[name]
        if _skip((name,), done, force, "scenario-weights"):
            continue
        tot = sum(probs.values())
        probs = {k: v / tot for k, v in probs.items()}
        pv = np.array(list(probs.values()))
        crit_mass = sum(probs[str(s)] for s in crit)
        commitment, red_obj, t_milp = c.solve_reduced(ids, probs=probs)
        approx, under_rep, _ = c.verify(commitment, red_obj)
        obj, gap, _ = c.exact_gap(commitment)
        if evaluation:
            evm = c.eval_pool(commitment)
        else:
            evm = {"eval_expected_cost": float("nan"), "eval_cost_sd": float("nan"),
                   "eval_ens_rate_pct": float("nan"), "eval_ens_mwh": float("nan"), "eval_n": 0}
        on_hours = int(commitment[HOUR_COLS].to_numpy().sum())
        row = {"scheme": name, "K": len(ids), "w_max": pv.max(), "w_min": pv.min(),
               "critical_mass": crit_mass, "reduced_obj": red_obj, "approx_obj": approx,
               "under_rep_pct": 100 * under_rep, "objective": obj, "gap_pct": gap,
               "t_milp_s": t_milp, "unit_hours_on": on_hours, **evm}
        upsert_row(path, row, ["scheme"])
        print(f"[scenario-weights] {name}: gap={gap:+.3f}% evaluation cost={evm['eval_expected_cost']:,.0f} "
              f"unserved energy={evm['eval_ens_rate_pct']:.4f}% critical mass={crit_mass:.3f}")
    return path


# ============================================================================
# Stage: lower-bound (Sections 4.4 and 6.1)
# ============================================================================

def solve_lower_bound_batches(path: Path, batch_sizes, n_batches: int = LOWER_BOUND_BATCHES,
                              force: bool = False) -> Path:
    """Solve M independent batch problems of each size S' for the lower bound.

    The batches of size S' are consecutive, non-overlapping blocks of one random
    permutation of the training set (random seed 1000 + S'); each batch problem is
    the full model over its S' scenarios with equal weights, solved to the 0.1%
    MILP tolerance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(path, ["batch_size", "m"])
    gen = r.load_generators()
    net = r.load_net_load_scenarios()
    units = [str(u) for u in gen["unit"]]
    for Sb in batch_sizes:
        rng = np.random.default_rng(1000 + Sb)
        order = rng.permutation(len(net))
        for m in range(n_batches):
            if _skip((str(Sb), str(m)), done, force, "lower-bound"):
                continue
            ids = list(net.index[order[m * Sb:(m + 1) * Sb]])
            ds = {str(s): net.loc[s, HOUR_COLS].to_numpy(float) for s in ids}
            av = {}
            for s in ids:
                a = draw_scenario_availability(units, s)
                for g in units:
                    av[(str(s), g)] = a[g]
            t = time.time()
            model = r.build_uc_master_model(gen, ds, {s: 1.0 / Sb for s in ds}, "stochastic", availability=av)
            met = r.solve_pyomo_model(model)
            row = {"batch_size": Sb, "m": m, "objective": float(met["objective"]),
                   "time_s": time.time() - t, "termination": met["termination"]}
            upsert_row(path, row, ["batch_size", "m"])
            print(f"[lower-bound] S'={Sb} batch {m}: objective={row['objective']:,.0f} ({row['time_s']:.0f} s)")
    return path


def _run_folder_with(name: str, filename: str = "commitment.csv") -> tuple[Path, str]:
    """Run folder ``name`` of the active output root if it holds ``filename``, else of ``results/``."""
    rerun = systems.run_dir(name)
    if (rerun / filename).exists():
        return rerun, "outputs"
    ref = systems.reference_run_dir(name)
    if (ref / filename).exists():
        return ref, "results"
    raise FileNotFoundError(f"No {filename} in {display_path(rerun)} or {display_path(ref)}")


def _evaluation_costs(folder: Path, commitment: pd.DataFrame, gen_ev: pd.DataFrame,
                      n_eval: int) -> np.ndarray:
    """Total cost of a commitment on the evaluation set (first-stage cost plus recourse).

    The recourse is read from ``evaluation.csv`` in the run folder when it holds at
    least ``n_eval`` scenarios; otherwise it is computed with the exact LP.
    """
    fixed = compute_fixed_commitment_cost(commitment, gen_ev)["fixed_commitment_cost"]
    ef = folder / "evaluation.csv"
    per = pd.read_csv(ef) if ef.exists() else None
    if per is not None and len(per) >= n_eval:
        per = per.iloc[:n_eval]
        print(f"[lower-bound] evaluation costs from {display_path(ef)}")
    else:
        print(f"[lower-bound] no complete evaluation.csv in {display_path(folder)}; "
              f"evaluating the commitment on {n_eval} scenarios")
        test = ev.load_net_load_scenarios()
        parts = []
        ids = list(test.index)
        for st in range(0, len(ids), EVALUATION_BATCH):
            parts.append(evaluate_commitment_batch(commitment, gen_ev, test.loc[ids[st:st + EVALUATION_BATCH]]))
        per = pd.concat(parts, ignore_index=True)
    return fixed + per["dispatch_cost"].to_numpy(float)


def certified_gap_table(batches: pd.DataFrame, S_ref: int = LOWER_BOUND_S_REF,
                        mip_gap: float = LOWER_BOUND_MIP_GAP, confidence: float = 0.95) -> pd.DataFrame:
    """Certified gaps of the full-model and PI-SASR commitments at ``S_ref``.

    For each case (``full``, ``pisasr``) and each batch size in ``batches``, the
    lower bound comes from the batch incumbents and the upper bound from the total
    cost of the commitment on the evaluation set
    (:func:`pisasr.statistics.certified_gap`). The commitments are taken from the
    active output root when it holds them, otherwise from ``results/``.
    """
    rows = []
    gen_ev = ev.load_generators()
    n_eval = int(ev.EVALUATION_SCENARIOS)
    for case, run in (("full", f"full_S{S_ref}"), ("pisasr", f"pisasr_S{S_ref}")):
        folder, source = _run_folder_with(run)
        print(f"[lower-bound] {case}: commitment from {display_path(folder)} ({source})")
        com = read_commitment(folder / "commitment.csv")
        tot = _evaluation_costs(folder, com, gen_ev, n_eval)
        insample = float(pd.read_csv(folder / "solve_metrics.csv")["objective"].iloc[0])
        for Sb, grp in batches.groupby("batch_size"):
            res = certified_gap(grp["objective"].to_numpy(float), tot, insample,
                                mip_gap=mip_gap, confidence=confidence)
            rows.append({"case": case, "S_ref": S_ref, "batch_size": Sb, **res})
    return pd.DataFrame(rows)


def stage_lower_bound(out: Path, batch_sizes, n_batches: int = LOWER_BOUND_BATCHES,
                      S_ref: int = LOWER_BOUND_S_REF, batches_file: str | Path | None = None,
                      force: bool = False) -> list[Path]:
    """Lower-bound batches and certified gaps.

    Solves the batches missing from ``lower_bound_batches.csv`` and then rewrites
    ``certified_gap.csv`` from all batches in the file. With ``batches_file`` no
    MILP is solved: the certified gaps are computed from that file (for example the
    reference file ``results/<system>/experiments/lower_bound_batches.csv``).
    """
    if batches_file is None:
        source = solve_lower_bound_batches(_output_path(out, "lower_bound_batches"),
                                           batch_sizes, n_batches, force=force)
    else:
        source = Path(batches_file)
        print(f"[lower-bound] batch incumbents from {display_path(source)} (no MILP solved)")
    table = certified_gap_table(pd.read_csv(source), S_ref)
    cpath = _output_path(out, "certified_gap")
    table.to_csv(cpath, index=False)
    print(table[["case", "batch_size", "lb_95", "ub_mean", "gap_point_pct", "gap_cert95_pct"]].to_string())
    return [source, cpath]


# ============================================================================
# Stage: budget-sensitivity (Figure 3(b), Section 6.2)
# ============================================================================

def stage_budget_sensitivity(out: Path, S_list, budgets=None, seed: int = SEED,
                             reference_commitment=None, force: bool = False) -> Path:
    """Benchmark gap of PI-SASR (no enrichment) for each scenario count and budget (K_c, K_r).

    The GP ranking is computed once for each S; for each budget the active set is
    the top K_c of the ranking and the K_r k-means medoids.
    """
    path = _output_path(out, "budget_sensitivity")
    done = done_keys(path, ["S", "k_critical", "k_cover"])
    budgets = budgets or BUDGETS[systems.active_system().key]
    for S in S_list:
        todo = [(kc, kr) for kc, kr in budgets
                if not _skip((str(S), str(kc), str(kr)), done, force, "budget-sensitivity")]
        if not todo:
            continue
        c = ExperimentContext(S, seed=seed, reference_commitment=reference_commitment)
        pred, info = c.gp_rank()
        for kc, kr in todo:
            med, _, tm = c.medoids(kr)
            ids = list(dict.fromkeys(c.topk(pred, kc) + med))
            commitment, red_obj, t_milp = c.solve_reduced(ids)
            approx, under_rep, t_ver = c.verify(commitment, red_obj)
            obj, gap, t_exact = c.exact_gap(commitment)
            row = {"S": S, "k_critical": kc, "k_cover": kr, "K": len(ids),
                   "objective": obj, "gap_pct": gap, "under_rep_pct": 100 * under_rep,
                   "t_milp_s": t_milp,
                   "t_online_s": c.t_features + info["t_design_s"] + info["t_gp_s"] + tm + t_milp + t_ver,
                   "t_exact_eval_s": t_exact}
            upsert_row(path, row, ["S", "k_critical", "k_cover"])
            print(f"[budget-sensitivity] S={S} (K_c, K_r)=({kc}, {kr}) K={len(ids)} "
                  f"gap={gap:+.3f}% MILP {t_milp:.0f} s")
    return path


# ============================================================================
# Design-set replication (Section 6.2; scripts/run_replication.py)
# ============================================================================

def design_set_replication(out: Path, S: int = 1000, seeds=REPLICATION_SEEDS,
                           full_repeats: int = FULL_TIMING_REPEATS, reference_commitment=None,
                           force: bool = False) -> Path:
    """PI-SASR with several random draws of the design set, and repeated full-model timings.

    Rows ``kind = "pisasr"``: PI-SASR at S with the settings of the paper and random
    seed ``seed`` (benchmark gap, online time and active-set size). Rows
    ``kind = "full"``: the full model at S solved ``full_repeats`` times (``seed`` is
    the repetition number, ``time_s`` covers model construction and solution); the
    objective does not change, only the time.
    """
    path = _output_path(out, "design_set_replication", S=S)
    done = done_keys(path, ["kind", "seed"])
    d = PISASR_DEFAULTS
    for seed in seeds:
        if _skip(("pisasr", str(seed)), done, force, "replication"):
            continue
        res = run_pisasr(n_target=S, k_critical=d["k_critical"], k_cover=d["k_cover"],
                         design_size=d["design_size"], cg_rounds=d["cg_rounds"], cg_add=d["cg_add"],
                         random_seed=seed, output_folder=None,
                         reference_commitment=reference_commitment)
        row = {"kind": "pisasr", "seed": seed, "gap_pct": res["opt_gap_pct_vs_full"],
               "time_s": res["solve_time_seconds"], "K": res["n_scenarios_used"]}
        upsert_row(path, row, ["kind", "seed"])
        print(f"[replication] seed {seed}: gap={res['opt_gap_pct_vs_full']:+.3f}% "
              f"time={res['solve_time_seconds']:.0f} s K={res['n_scenarios_used']}")

    reps = [rep for rep in range(full_repeats)
            if not _skip(("full", str(rep)), done, force, "replication")]
    if not reps:
        return path
    gen = r.load_generators()
    net = r.load_net_load_scenarios()
    units = [str(u) for u in gen["unit"]]
    sel = r.select_stochastic_scenarios(net, S)
    ds = {str(s): sel.loc[s, HOUR_COLS].to_numpy(float) for s in sel.index}
    av = {}
    for s in sel.index:
        a = draw_scenario_availability(units, s)
        for g in units:
            av[(str(s), g)] = a[g]
    for rep in reps:
        t = time.time()
        m = r.build_uc_master_model(gen, ds, {s: 1.0 / S for s in ds}, "stochastic", availability=av)
        r.solve_pyomo_model(m)
        row = {"kind": "full", "seed": rep, "gap_pct": 0.0, "time_s": time.time() - t, "K": S}
        upsert_row(path, row, ["kind", "seed"])
        print(f"[replication] full model, repetition {rep}: time={row['time_s']:.0f} s")
    return path


# ============================================================================
# Merit-order check (Section 6.5; scripts/validate_merit_order.py)
# ============================================================================

def merit_order_validation(n_scenarios: int = MERIT_ORDER_SCENARIOS,
                           reference_commitment=None) -> pd.DataFrame:
    """Merit-order recourse against the single-scenario dispatch LP (Section 6.5).

    The recourse of the first ``n_scenarios`` training scenarios of the active
    system is computed under the reference commitment u_ref (by default the
    stored schedule ``results/<system>/runs/deterministic/commitment.csv``) with
    the merit-order dispatch and with the dispatch LP of
    :func:`pisasr.uc_model.evaluate_fixed_commitment_dispatch_cost`, each scenario
    with its own unit outage. Gurobi solves one LP for each scenario.

    Returns
    -------
    pandas.DataFrame
        Columns ``scenario``, ``merit_order``, ``lp`` and ``rel_err_pct``
        (:func:`pisasr.merit_order.validate_against_lp`).
    """
    gen = r.load_generators()
    net = r.load_net_load_scenarios()
    units = [str(u) for u in gen["unit"]]
    if reference_commitment is None:
        reference_commitment = systems.resolve_reference_commitment("results")
    det = read_commitment(reference_commitment)
    ids = list(net.index[:n_scenarios])
    return validate_against_lp(gen, det, net, ids,
                               draw_avail_fn=lambda s: draw_scenario_availability(units, s))


def merit_order_summary(table: pd.DataFrame) -> dict:
    """Summary of a merit-order check (the numbers quoted in Section 6.5).

    Returns the number of scenarios ``n``, the mean and largest absolute
    relative error (%), the squared correlation ``r2`` of the merit-order and LP
    costs, and ``n_not_above_lp``, the number of scenarios whose merit-order cost
    does not exceed the LP cost (the merit order omits the ramp limits, so it
    should never exceed the LP).
    """
    err = table["rel_err_pct"].abs()
    return {
        "n": int(len(table)),
        "mean_abs_err_pct": float(err.mean()),
        "max_abs_err_pct": float(err.max()),
        "r2": float(np.corrcoef(table["merit_order"], table["lp"])[0, 1] ** 2),
        "n_not_above_lp": int((table["rel_err_pct"] <= 0).sum()),
    }


# ============================================================================
# Dispatcher
# ============================================================================

def run_stage(stage: str, out: Path, S_list=None, variants=None, seed: int = SEED,
              reference_commitment=None, force: bool = False, budgets=None,
              kernel_seeds=KERNEL_SEEDS, spectrum_only: bool = False,
              n_batches: int = LOWER_BOUND_BATCHES, S_ref: int = LOWER_BOUND_S_REF,
              batches_file=None) -> list[Path]:
    """Run one stage on the active system (see the module docstring).

    ``S_list`` defaults to the scenario counts of the paper (:data:`DEFAULT_S`);
    single-S stages run once for each S in it. ``variants`` selects variants
    (component-timing, criticality-only), kernels (gp-kernels), spaces (embedding)
    or schemes (scenario-weights, weights-s2500).
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}")
    key = systems.active_system().key
    S_list = list(S_list) if S_list else DEFAULT_S[stage][key]
    common = {"seed": seed, "reference_commitment": reference_commitment, "force": force}
    written: list[Path] = []
    if stage == "lower-bound":
        return stage_lower_bound(out, S_list, n_batches=n_batches, S_ref=S_ref,
                                 batches_file=batches_file, force=force)
    if stage == "budget-sensitivity":
        return [stage_budget_sensitivity(out, S_list, budgets=budgets, **common)]
    for S in S_list:
        if stage == "component-timing":
            written.append(stage_component_timing(out, S, variants=variants, **common))
        elif stage == "criticality-only":
            written.append(stage_criticality_only(out, S, variants=variants, **common))
        elif stage == "gp-kernels":
            written.append(stage_gp_kernels(out, S, kernels=variants, seeds=kernel_seeds,
                                            reference_commitment=reference_commitment, force=force))
        elif stage == "embedding":
            written += stage_embedding(out, S, spaces=variants, spectrum_only=spectrum_only, **common)
        elif stage == "scenario-weights":
            written.append(stage_scenario_weights(out, S, evaluation=True, schemes=variants, **common))
        elif stage == "weights-s2500":
            written.append(stage_scenario_weights(out, S, evaluation=False,
                                                  schemes=variants or WEIGHT_SCHEMES_S2500, **common))
    return written
