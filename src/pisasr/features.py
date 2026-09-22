"""Physics-informed scenario features and low-rank temporal embedding.

For a set of 24-hour net-load scenarios and a generator fleet, the feature
matrix combines

* physics-informed adequacy and ramp features, which are closed-form functions
  of the net-load profile and the fleet capacity (no dispatch, no MILP); and
* a rank-r truncated-SVD embedding of the centered scenario profiles, which
  gives the coordinates in which the coverage scenarios are clustered.

The features are the inputs of the Gaussian-process criticality surrogate.
Computing them takes O(ST + STr) operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import HOUR_COLS
from .outages import GEN_FORCED_OUTAGE_RATE

__all__ = [
    "HOUR_COLS",
    "fleet_capacity",
    "physics_features",
    "lowrank_embedding",
    "build_feature_matrix",
]


def fleet_capacity(gen: pd.DataFrame) -> float:
    """Total installed capacity of the committable fleet (MW)."""
    return float(gen["p_max_mw"].sum())


def physics_features(
    net_load_df: pd.DataFrame,
    gen: pd.DataFrame,
    forced_outage_rate: float = GEN_FORCED_OUTAGE_RATE,
) -> pd.DataFrame:
    """Adequacy and ramp features for each scenario.

    The adequacy features compare the profile with the outage-adjusted fleet
    capacity ``(1 - forced_outage_rate) * sum(P_max)``.
    """
    d = net_load_df[HOUR_COLS].to_numpy(dtype=float)        # (S, 24)
    cap = fleet_capacity(gen)
    avail_cap = (1.0 - forced_outage_rate) * cap

    peak = d.max(axis=1)
    trough = d.min(axis=1)
    energy = d.sum(axis=1)
    mean = d.mean(axis=1)
    std = d.std(axis=1)

    ramp = np.diff(d, axis=1)
    max_ramp_up = np.maximum(ramp, 0.0).max(axis=1)
    max_ramp_down = np.maximum(-ramp, 0.0).max(axis=1)
    ramp_energy = np.abs(ramp).sum(axis=1)

    load_factor = mean / np.maximum(peak, 1e-9)

    # Adequacy stress relative to the outage-adjusted fleet capacity.
    margin_deficit = peak - avail_cap                      # > 0: expected shortfall
    util_peak = peak / np.maximum(cap, 1e-9)               # peak utilization
    hours_above_85 = (d > 0.85 * avail_cap).sum(axis=1).astype(float)
    hours_above_95 = (d > 0.95 * avail_cap).sum(axis=1).astype(float)

    feats = pd.DataFrame(
        {
            "peak": peak,
            "trough": trough,
            "energy": energy,
            "mean": mean,
            "std": std,
            "max_ramp_up": max_ramp_up,
            "max_ramp_down": max_ramp_down,
            "ramp_energy": ramp_energy,
            "load_factor": load_factor,
            "margin_deficit": margin_deficit,
            "util_peak": util_peak,
            "hours_above_85pct_cap": hours_above_85,
            "hours_above_95pct_cap": hours_above_95,
        },
        index=net_load_df.index,
    )
    return feats


def lowrank_embedding(net_load_df: pd.DataFrame, rank: int = 5) -> pd.DataFrame:
    """Rank-``rank`` truncated-SVD embedding of the scenario profiles.

    The (S, 24) scenario matrix is centered and each scenario is projected onto
    the leading ``rank`` right singular vectors, which gives a coordinate that
    summarizes the dominant temporal modes of net-load variation.
    """
    X = net_load_df[HOUR_COLS].to_numpy(dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # Economy SVD: Xc = U S Vt; embedding = U[:, :r] S[:r] = Xc Vt[:r].T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(rank, Vt.shape[0])
    emb = Xc @ Vt[:r].T                                     # (S, r)

    cols = [f"svd_{i+1}" for i in range(r)]
    return pd.DataFrame(emb, index=net_load_df.index, columns=cols)


def build_feature_matrix(
    net_load_df: pd.DataFrame,
    gen: pd.DataFrame,
    rank: int = 5,
    forced_outage_rate: float = GEN_FORCED_OUTAGE_RATE,
) -> pd.DataFrame:
    """Physics-informed features followed by the embedding (one row for each scenario)."""
    phys = physics_features(net_load_df, gen, forced_outage_rate)
    emb = lowrank_embedding(net_load_df, rank)
    return pd.concat([phys, emb], axis=1)
