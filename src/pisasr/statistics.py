"""Statistical summaries used in the paper.

This module depends only on NumPy and SciPy, so it can be imported without an
optimization solver.

``paired_difference_interval`` gives the confidence interval of Section 6.3 of
the paper: the mean difference in cost between two commitments evaluated on the
same evaluation scenarios, with a two-sided Student-t interval, expressed
relative to the mean cost of the second (reference) commitment.

``batch_lower_bound``, ``evaluation_upper_bound`` and ``certified_gap`` give the
statistical lower bound of Mak, Morton and Wood (1999) and the certified gap of
Sections 4.4 and 6.1 of the paper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairedDifference:
    """Mean of the paired differences ``a - b`` and its confidence interval.

    Absolute quantities are in the units of the inputs (dollars in this
    repository). The ``*_pct`` fields are expressed as a percentage of
    ``mean_b``, the mean of the reference sample.
    """

    n_pairs: int
    confidence: float
    distribution: str
    mean_a: float
    mean_b: float
    mean_difference: float
    sd_difference: float
    standard_error: float
    quantile: float
    ci_low: float
    ci_high: float
    mean_difference_pct: float
    ci_low_pct: float
    ci_high_pct: float

    def as_dict(self) -> dict:
        return asdict(self)


def paired_difference_interval(
    a,
    b,
    confidence: float = 0.95,
    distribution: str = "t",
) -> PairedDifference:
    """Confidence interval for the mean of the paired differences ``a - b``.

    Parameters
    ----------
    a, b
        Costs of the two commitments on the same scenarios, in the same order.
    confidence
        Two-sided confidence level (0.95 in the paper).
    distribution
        ``"t"`` uses the Student-t quantile with ``n - 1`` degrees of freedom
        (the interval printed in the paper); ``"normal"`` uses the standard
        normal quantile and is provided only for comparison.

    The interval is ``mean(d) +/- q * sd(d) / sqrt(n)`` with ``d = a - b`` and
    the sample standard deviation ``sd`` (``ddof=1``). Percentages divide the
    difference and both interval ends by ``mean(b)``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be one-dimensional arrays of equal length")
    n = a.size
    if n < 2:
        raise ValueError("at least two pairs are needed")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    d = a - b
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    se = sd_d / np.sqrt(n)
    upper = 0.5 + confidence / 2.0
    if distribution == "t":
        q = float(stats.t.ppf(upper, n - 1))
    elif distribution == "normal":
        q = float(stats.norm.ppf(upper))
    else:
        raise ValueError("distribution must be 't' or 'normal'")

    lo, hi = mean_d - q * se, mean_d + q * se
    ref = float(b.mean())
    return PairedDifference(
        n_pairs=int(n),
        confidence=float(confidence),
        distribution=distribution,
        mean_a=float(a.mean()),
        mean_b=ref,
        mean_difference=mean_d,
        sd_difference=sd_d,
        standard_error=float(se),
        quantile=q,
        ci_low=float(lo),
        ci_high=float(hi),
        mean_difference_pct=100.0 * mean_d / ref,
        ci_low_pct=100.0 * lo / ref,
        ci_high_pct=100.0 * hi / ref,
    )


# ---------------------------------------------------------------------------
# Lower bound and certified gap (Sections 4.4 and 6.1 of the paper)
# ---------------------------------------------------------------------------

def _upper_quantile(confidence: float, dof: int) -> float:
    """Student-t quantile of level ``0.5 + confidence / 2`` (0.975 for 0.95)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    return float(stats.t.ppf(0.5 + confidence / 2.0, dof))


def batch_lower_bound(batch_objectives, mip_gap: float = 1e-3, confidence: float = 0.95) -> dict:
    """Statistical lower bound on the optimal expected cost from independent batches.

    Mak, Morton and Wood (1999): the optimal objective of a sample-average problem
    over S' scenarios drawn from the scenario distribution has an expected value no
    larger than the optimal expected cost z*. The mean of the optimal objectives of
    M independent batches, less a Student-t margin, is therefore a lower confidence
    bound on z*.

    Each batch problem is a MILP solved to the relative tolerance ``mip_gap``; its
    incumbent exceeds the batch optimum by at most that fraction, so every incumbent
    is first multiplied by ``1 - mip_gap``, which gives a valid lower bound on the
    batch optimum.

    Parameters
    ----------
    batch_objectives
        MILP incumbents of the M batch problems (M >= 2).
    mip_gap
        Relative MILP tolerance of the batch problems (0.1% in the paper).
    confidence
        The bound uses the Student-t quantile of level ``0.5 + confidence / 2``
        with M - 1 degrees of freedom (0.975 for the default 0.95), that is, it
        is a one-sided 97.5% lower confidence bound.

    Returns
    -------
    dict
        ``M``, ``lb_mean`` and ``lb_sd`` (mean and sample standard deviation of
        the reduced incumbents) and ``lb_95`` (the lower confidence bound).
    """
    z = np.asarray(batch_objectives, dtype=float) * (1.0 - mip_gap)
    m = int(z.size)
    if z.ndim != 1 or m < 2:
        raise ValueError("at least two batch objectives are needed")
    lb_mean, lb_sd = float(z.mean()), float(z.std(ddof=1))
    q = _upper_quantile(confidence, m - 1)
    return {"M": m, "lb_mean": lb_mean, "lb_sd": lb_sd,
            "lb_95": lb_mean - q * lb_sd / np.sqrt(m)}


def evaluation_upper_bound(costs, confidence: float = 0.95) -> dict:
    """Upper confidence bound on the expected cost of one commitment.

    ``costs`` are the total costs (first-stage cost plus recourse cost) of the
    commitment on the scenarios of the evaluation set. The bound is the sample
    mean plus the Student-t quantile of level ``0.5 + confidence / 2`` (with
    n - 1 degrees of freedom) times the standard error.

    Returns
    -------
    dict
        ``ub_mean``, ``ub_sd``, ``ub_95`` and ``n_eval``.
    """
    tot = np.asarray(costs, dtype=float)
    n = int(tot.size)
    if tot.ndim != 1 or n < 2:
        raise ValueError("at least two evaluation costs are needed")
    ub_mean, ub_sd = float(tot.mean()), float(tot.std(ddof=1))
    q = _upper_quantile(confidence, n - 1)
    return {"ub_mean": ub_mean, "ub_sd": ub_sd,
            "ub_95": ub_mean + q * ub_sd / np.sqrt(n), "n_eval": n}


def certified_gap(
    batch_objectives,
    evaluation_costs,
    insample_objective: float = float("nan"),
    mip_gap: float = 1e-3,
    confidence: float = 0.95,
) -> dict:
    """Certified optimality gap of a commitment (Section 4.4 of the paper).

    The gap compares the upper confidence bound on the expected cost of the
    commitment (:func:`evaluation_upper_bound`, from the evaluation set) with
    the lower confidence bound on the optimal expected cost
    (:func:`batch_lower_bound`, from independent batches of training
    scenarios). Each bound holds with probability at least 97.5% for the
    default ``confidence``, so the certified gap holds with probability at
    least 95%.

    Returns
    -------
    dict
        The columns of ``certified_gap.csv`` from ``M`` to
        ``gap_insample_vs_lb95_pct``:

        * ``gap_point_pct`` = 100 (``ub_mean`` - ``lb_mean``) / ``lb_mean``;
        * ``gap_cert95_pct`` = 100 (``ub_95`` - ``lb_95``) / ``lb_95``, the
          certified gap;
        * ``gap_insample_vs_lb95_pct`` = 100 (``insample_obj`` - ``lb_95``) /
          ``lb_95``, with the objective of the commitment over its training
          scenarios.
    """
    lb = batch_lower_bound(batch_objectives, mip_gap=mip_gap, confidence=confidence)
    ub = evaluation_upper_bound(evaluation_costs, confidence=confidence)
    insample = float(insample_objective)
    return {
        **lb,
        **ub,
        "insample_obj": insample,
        "gap_point_pct": 100 * (ub["ub_mean"] - lb["lb_mean"]) / lb["lb_mean"],
        "gap_cert95_pct": 100 * (ub["ub_95"] - lb["lb_95"]) / lb["lb_95"],
        "gap_insample_vs_lb95_pct": 100 * (insample - lb["lb_95"]) / lb["lb_95"],
    }
