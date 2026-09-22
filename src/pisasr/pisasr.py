"""PI-SASR: physics-informed surrogate-assisted scenario reduction.

The full two-stage stochastic UC over S scenarios replicates the recourse stage
S times, and its solution time grows faster than linearly in S. PI-SASR
replaces it by a reduced UC over an active set whose size does not depend on S:

1. Features. Physics-informed adequacy and ramp features and a rank-r
   embedding are computed for all S scenarios (:mod:`pisasr.features`).
2. Design set and GP surrogate. The recourse cost under the reference
   commitment u_ref (the deterministic UC schedule) is computed with the
   merit-order dispatch for a random design set, and a GP surrogate
   (:mod:`pisasr.surrogate`) is fitted to it; a quarter of the design set is
   kept aside once to measure the surrogate's R^2.
3. Active set. The K_c scenarios with the largest predicted recourse cost
   (critical scenarios) are combined with the K_r medoids of a k-means
   clustering of the embedding (coverage scenarios).
4. Baseload pre-commitment. Units committed in every hour of u_ref are fixed
   on in the reduced UC.
5. Reduced UC and merit-order verification. The reduced UC is solved with
   uniform weights over the active set. Its commitment is verified with the
   merit-order recourse over all S scenarios; the relative excess of this
   verification cost over the reduced objective is the under-representation
   statistic. If it exceeds the enrichment tolerance epsilon, the scenarios
   with the largest verified recourse outside the active set are added and the
   reduced UC is solved again (enrichment round).
6. Exact objective. The objective of the selected commitment is recomputed
   once with the exact ramp-coupled LP (:mod:`pisasr.evaluation`), and the
   benchmark gap against the full-model objective is computed.

The solution time (``solve_time_seconds``) covers steps 1 to 5, from data
loading to the end of the merit-order verification. It excludes the solution of
the deterministic UC that gives u_ref and the exact recomputation of step 6,
whose time is recorded separately (``exact_validation_s``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import systems
from . import uc_model as r
from .constants import HOUR_COLS
from .evaluation import compute_fixed_commitment_cost, evaluate_commitment_batch
from .features import build_feature_matrix
from .merit_order import MeritOrderDispatch
from .outages import GEN_FORCED_OUTAGE_RATE, draw_scenario_availability
from .surrogate import GPSurrogate

__all__ = [
    "ActiveSetSelection",
    "select_active_set",
    "run_pisasr",
    "read_commitment",
]


def _avail_vec(units, s):
    """Availability vector of the units (portfolio order) in scenario ``s``."""
    a = draw_scenario_availability(units, s)
    return np.array([a[u] for u in units])


def _exact_recourse_mean(commitment, gen, net_df, ids, batch=100):
    """Mean exact ramp-coupled recourse of a commitment over the scenarios ``ids``."""
    tot, n = 0.0, 0
    for st in range(0, len(ids), batch):
        b = net_df.loc[ids[st:st + batch]]
        res = evaluate_commitment_batch(commitment, gen, b)
        tot += float(res["dispatch_cost"].sum())
        n += len(res)
    return tot / n


def read_commitment(path: str | Path) -> pd.DataFrame:
    """Read a ``commitment.csv`` with the unit identifiers as strings."""
    commitment = pd.read_csv(path)
    commitment["unit"] = commitment["unit"].astype(str)
    return commitment


@dataclass
class ActiveSetSelection:
    """Result of the selection stage of PI-SASR (steps 1 to 4)."""

    target_ids: list
    """The S scenarios of the full model (nested subset of the training set)."""
    features: pd.DataFrame
    """Feature matrix of the S scenarios (physics features and embedding)."""
    reference: pd.DataFrame
    """Reference commitment u_ref."""
    design_ids: list
    """Design set."""
    y_design: np.ndarray
    """Merit-order recourse of the design set under u_ref."""
    surrogate_accuracy: dict
    """R^2 and mean absolute error of the GP on the design scenarios kept aside."""
    predicted: np.ndarray
    """GP posterior mean of the recourse cost of all S scenarios."""
    critical: list
    """The K_c critical scenarios."""
    cover: list
    """The K_r coverage scenarios (k-means medoids of the embedding)."""
    active_set: list
    """Critical scenarios followed by the coverage scenarios not already included."""
    baseload: list
    """Units committed in every hour of u_ref (pre-committed in the reduced UC)."""
    timing: dict = field(default_factory=dict)
    """Stage times: ``design_eval_s`` and ``gp_fit_s``."""


def select_active_set(
    gen: pd.DataFrame,
    net: pd.DataFrame,
    reference_commitment: str | Path | pd.DataFrame,
    n_target: int = 1000,
    k_critical: int = 90,
    k_cover: int = 60,
    design_size: int = 140,
    svd_rank: int = 5,
    random_seed: int = 7,
) -> ActiveSetSelection:
    """Selection stage of PI-SASR: surrogate, active set and baseload units.

    The random-number generator seeded with ``random_seed`` is called first for
    the design set and then for the split of the design set that measures the
    GP accuracy; this order reproduces the
    reference results.
    """
    rng = np.random.default_rng(random_seed)
    timing = {}
    units = [str(u) for u in gen["unit"]]

    target = r.select_stochastic_scenarios(net, n_target)
    target_ids = list(target.index)

    # ---- physics-informed features and low-rank embedding ----
    feats = build_feature_matrix(target, gen, rank=svd_rank,
                                 forced_outage_rate=GEN_FORCED_OUTAGE_RATE)
    feat_cols = list(feats.columns)
    X_all = feats[feat_cols].to_numpy(float)

    # ---- design-set recourse under the reference commitment (merit order) ----
    ts = time.time()
    if isinstance(reference_commitment, pd.DataFrame):
        det = reference_commitment.copy()
        det["unit"] = det["unit"].astype(str)
    else:
        det = read_commitment(reference_commitment)
    md_det = MeritOrderDispatch(gen, det)

    design_ids = [target_ids[i] for i in
                  rng.choice(len(target_ids), size=min(design_size, len(target_ids)), replace=False)]
    y_design = md_det.recourse_over(net, design_ids, avail_fn=lambda s: _avail_vec(units, s))
    timing["design_eval_s"] = time.time() - ts

    # ---- GP surrogate (features -> recourse); accuracy on a quarter kept aside ----
    ts = time.time()
    Xd = feats.loc[design_ids, feat_cols].to_numpy(float)
    perm = rng.permutation(len(design_ids))
    n_hold = max(10, int(0.25 * len(design_ids)))
    tr, ho = perm[n_hold:], perm[:n_hold]
    gp = GPSurrogate(length_scale_dim=len(feat_cols), random_state=random_seed)
    gp.fit(Xd[tr], y_design[tr])
    acc = gp.score(Xd[ho], y_design[ho])
    gp.fit(Xd, y_design)
    pred = gp.predict(X_all)
    timing["gp_fit_s"] = time.time() - ts

    # ---- active set: critical and coverage scenarios ----
    order = np.argsort(pred)[::-1]
    critical = [target_ids[i] for i in order[:k_critical]]
    emb_cols = [c for c in feat_cols if c.startswith("svd_")]
    km = KMeans(n_clusters=min(k_cover, len(target_ids)), n_init=4, random_state=random_seed)
    labels = km.fit_predict(feats[emb_cols].to_numpy(float))
    cover = []
    E = feats[emb_cols].to_numpy(float)
    for c in range(km.n_clusters):
        mem = np.where(labels == c)[0]
        if len(mem):
            d2 = ((E[mem] - km.cluster_centers_[c]) ** 2).sum(1)
            cover.append(target_ids[mem[int(np.argmin(d2))]])
    reduced_ids = list(dict.fromkeys(critical + cover))

    # ---- baseload pre-commitment ----
    on_all = det.set_index("unit")[HOUR_COLS].min(axis=1)
    fixed_on = [str(u) for u, v in on_all.items() if int(round(v)) == 1]

    return ActiveSetSelection(
        target_ids=target_ids,
        features=feats,
        reference=det,
        design_ids=design_ids,
        y_design=y_design,
        surrogate_accuracy=acc,
        predicted=pred,
        critical=critical,
        cover=cover,
        active_set=reduced_ids,
        baseload=fixed_on,
        timing=timing,
    )


def _reference_label(reference_commitment) -> str:
    """Short description of where u_ref came from (stored in solve_metrics.csv)."""
    if isinstance(reference_commitment, pd.DataFrame):
        return "dataframe"
    path = Path(reference_commitment).resolve()
    try:
        path.relative_to(systems.active_system().reference_dir.resolve())
        return "results"
    except ValueError:
        pass
    try:
        path.relative_to(systems.active_root().resolve())
        return "outputs"
    except ValueError:
        return "file"


def _resolve_output_folder(output_folder) -> Path:
    """Run folder: an absolute path is used as it is, a bare name such as
    ``pisasr_S1000`` is placed below ``uc_model.OUTPUT_ROOT / "runs"``."""
    folder = Path(output_folder).expanduser()
    if folder.is_absolute():
        return folder
    return Path(r.OUTPUT_ROOT).resolve() / "runs" / folder


def run_pisasr(n_target=1000, k_critical=90, k_cover=60, design_size=140,
               cg_rounds=1, cg_add=40, svd_rank=5, random_seed=7,
               cg_tol=0.002, output_folder=None, reference_commitment=None,
               full_objective=None) -> dict:
    """Run PI-SASR on the active system at scenario count ``n_target``.

    Parameters
    ----------
    n_target
        Scenario count S of the full model being approximated.
    k_critical, k_cover
        Numbers of critical (K_c) and coverage (K_r) scenarios.
    design_size
        Size of the design set of the GP surrogate.
    cg_rounds
        Enrichment budget: maximum number of enrichment rounds.
    cg_add
        Scenarios added to the active set in one enrichment round.
    svd_rank
        Rank r of the low-rank embedding.
    random_seed
        Random seed of the design set, of the split of the design set, of the GP optimizer
        restarts and k-means.
    cg_tol
        Enrichment tolerance epsilon of the paper. The loop stops once the
        under-representation statistic (merit-order verification cost over all
        S scenarios relative to the reduced objective) is at most ``cg_tol``;
        the statistic never requires the full model.
    output_folder
        If given, ``commitment.csv``, ``solve_metrics.csv`` and ``pisasr_log.csv``
        are written there. An absolute path is used as it is; a relative path
        (normally a run-folder name such as ``pisasr_S1000``) is taken below
        ``uc_model.OUTPUT_ROOT / "runs"``.
    reference_commitment
        Path or table of u_ref. Default: the stored schedule of the paper,
        ``results/<system>/runs/deterministic/commitment.csv``.
    full_objective
        Full-model objective for the benchmark gap. Default: read from
        ``<output root>/runs/full_S<S>`` if present, else from ``results/``.

    Returns
    -------
    dict
        The row written to ``solve_metrics.csv``.
    """
    t0 = time.time()

    gen = r.load_generators()
    net = r.load_net_load_scenarios()
    units = [str(u) for u in gen["unit"]]

    if reference_commitment is None:
        reference_commitment = systems.resolve_reference_commitment("results")
    reference_source = _reference_label(reference_commitment)

    sel = select_active_set(
        gen, net, reference_commitment,
        n_target=n_target, k_critical=k_critical, k_cover=k_cover,
        design_size=design_size, svd_rank=svd_rank, random_seed=random_seed,
    )
    timing = dict(sel.timing)
    target_ids = sel.target_ids
    reduced_ids = list(sel.active_set)
    fixed_on = sel.baseload
    acc = sel.surrogate_accuracy

    # ---- reduced UC, merit-order verification and enrichment ----
    solve_t = 0.0
    best = None
    log = []
    if full_objective is None:
        full_obj, full_source = systems.resolve_full_objective(n_target)
    else:
        full_obj, full_source = float(full_objective), "given"

    for rnd in range(cg_rounds + 1):
        ds = {str(s): net.loc[s, HOUR_COLS].to_numpy(float) for s in reduced_ids}
        av = {}
        for s in reduced_ids:
            a = draw_scenario_availability(units, s)
            for g in units:
                av[(str(s), g)] = a[g]

        ts = time.time()
        model = r.build_uc_master_model(gen, ds,
                    {s: 1.0 / len(ds) for s in ds}, "stochastic", availability=av)
        for g in fixed_on:
            for t in model.T:
                model.u[g, t].fix(1)
        metrics = r.solve_pyomo_model(model)
        commitment = r.get_commitment_from_model(model)
        solve_t += time.time() - ts

        md = MeritOrderDispatch(gen, commitment)
        all_rec = md.recourse_over(net, target_ids, avail_fn=lambda s: _avail_vec(units, s))
        approx_obj = compute_fixed_commitment_cost(commitment, gen)["fixed_commitment_cost"] + float(all_rec.mean())
        # Under-representation statistic: if the verification cost over all S
        # scenarios does not exceed the reduced objective by more than the
        # enrichment tolerance, the active set is representative and no
        # enrichment round is needed.
        under_rep_gap = (approx_obj - metrics["objective"]) / metrics["objective"]
        log.append({"round": rnd, "n_scenarios": len(reduced_ids),
                    "approx_true_obj": approx_obj,
                    "under_rep_gap": under_rep_gap,
                    "elapsed_s": time.time() - t0})
        print(f"[PI-SASR S={n_target}] round {rnd}: K={len(reduced_ids)} "
              f"verified objective={approx_obj:,.0f} "
              f"under-representation statistic={round(under_rep_gap, 4) + 0.0:+.4f} "
              f"elapsed={time.time() - t0:,.0f}s")

        if best is None or approx_obj < best["approx_obj"]:
            best = {"commitment": commitment, "approx_obj": approx_obj,
                    "n_scenarios": len(reduced_ids), "all_rec": all_rec}
        if under_rep_gap <= cg_tol:
            print(f"[PI-SASR S={n_target}] statistic within the enrichment "
                  f"tolerance at round {rnd}; no enrichment round needed")
            break
        if rnd == cg_rounds:
            break
        rep = set(reduced_ids)
        rank = [target_ids[i] for i in np.argsort(best["all_rec"])[::-1] if target_ids[i] not in rep]
        reduced_ids = list(dict.fromkeys(reduced_ids + rank[:cg_add]))

    timing["milp_solve_s"] = solve_t

    # The online computation ends with the merit-order verification. The exact
    # ramp-coupled pass below validates the result and is timed separately.
    online_time = time.time() - t0
    approx_obj = best["approx_obj"]
    approx_gap = 100.0 * (approx_obj - full_obj) / full_obj if full_obj else float("nan")

    # ---- exact objective (one ramp-coupled pass over all S scenarios) ----
    ts = time.time()
    exact_mean_rec = _exact_recourse_mean(best["commitment"], gen, net, target_ids)
    fixed_cost = compute_fixed_commitment_cost(best["commitment"], gen)["fixed_commitment_cost"]
    true_obj = fixed_cost + exact_mean_rec
    timing["exact_cert_s"] = time.time() - ts
    gap = 100.0 * (true_obj - full_obj) / full_obj if full_obj else float("nan")

    result = {
        "method": "PI-SASR", "problem": f"stochastic_{n_target}",
        "objective": true_obj, "opt_gap_pct_vs_full": gap, "full_obj": full_obj,
        "approx_objective": approx_obj, "approx_gap_pct": approx_gap,
        "solve_time_seconds": online_time, "exact_validation_s": timing["exact_cert_s"],
        "n_scenarios_used": best["n_scenarios"],
        "n_scenarios_full": n_target, "n_fixed_baseload_units": len(fixed_on),
        "surrogate_r2": acc["r2"], "surrogate_mae": acc["mae"],
        **{f"t_{k}": v for k, v in timing.items()},
        "random_seed": random_seed,
        "reference_source": reference_source,
        "full_objective_source": full_source,
    }

    if output_folder is not None:
        folder = _resolve_output_folder(output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        best["commitment"].to_csv(folder / "commitment.csv", index=False)
        pd.DataFrame([result]).to_csv(folder / "solve_metrics.csv", index=False)
        pd.DataFrame(log).to_csv(folder / "pisasr_log.csv", index=False)

    return result
