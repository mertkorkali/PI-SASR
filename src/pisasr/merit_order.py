"""Merit-order recourse of a fixed commitment.

For a commitment u[g, t] and the availability of each unit in a scenario, the
dispatch LP separates by hour once the ramp coupling is relaxed. In each hour
every committed and available unit is held at its minimum output, the remaining
net load is assigned to the cheapest headroom in merit order, surplus minimum
output is spilled at the spill penalty and any shortfall is priced at the value
of lost load. This is the exact optimum of the hourly LP without ramp limits,
so it is a lower bound on the ramp-coupled recourse, and it takes a small
fraction of the time of an LP solution. PI-SASR uses it for the design-set
dispatches and for the merit-order verification over all S scenarios.

:func:`validate_against_lp` compares the merit-order recourse with a
single-scenario dispatch LP to measure its error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import HOUR_COLS, SPILL_PENALTY, VOLL


class MeritOrderDispatch:
    """Merit-order recourse evaluator for one commitment.

    Parameters
    ----------
    gen
        Generator portfolio (the unit order is taken from it).
    commitment
        Commitment table with a ``unit`` column and ``hour_00`` ... ``hour_23``.
    """

    def __init__(self, gen: pd.DataFrame, commitment: pd.DataFrame):
        g = gen.set_index("unit")
        self.units = [str(u) for u in gen["unit"]]
        self.pmin = g["p_min_mw"].to_numpy(float)
        self.pmax = g["p_max_mw"].to_numpy(float)
        self.vcost = g["variable_cost"].to_numpy(float)

        cm = commitment.copy()
        cm["unit"] = cm["unit"].astype(str)
        cm = cm.set_index("unit").reindex(self.units)
        self.u = cm[HOUR_COLS].to_numpy(float)              # (G, 24) in {0, 1}

        self.order = np.argsort(self.vcost)                 # merit order, cheapest first

    def recourse_cost(self, demand: np.ndarray, avail: np.ndarray | None = None) -> float:
        """Second-stage cost of one scenario (variable cost, unserved energy, spill)."""
        G = len(self.units)
        if avail is None:
            avail = np.ones(G)
        active = self.u * avail[:, None]                    # (G, 24) committed and available
        total = 0.0
        for t in range(24):
            on = active[:, t] > 0.5
            if not on.any():
                total += VOLL * max(demand[t], 0.0)
                continue
            base = self.pmin[on].sum()
            d = demand[t]
            # cost of holding the committed units at their minimum outputs
            cost_t = float((self.vcost[on] * self.pmin[on]).sum())
            residual = d - base
            if residual <= 0:
                # the committed minimum outputs exceed the net load: spill the surplus
                total += cost_t + SPILL_PENALTY * (-residual)
                continue
            # assign the remaining net load to the headroom (pmax - pmin) in merit order
            head = (self.pmax - self.pmin)
            for gi in self.order:
                if not on[gi]:
                    continue
                take = min(head[gi], residual)
                if take <= 0:
                    continue
                cost_t += self.vcost[gi] * take
                residual -= take
                if residual <= 1e-6:
                    break
            if residual > 1e-6:
                cost_t += VOLL * residual                    # unserved energy
            total += cost_t
        return total

    def recourse_over(self, net_df: pd.DataFrame, scenario_ids,
                      avail_fn=None) -> np.ndarray:
        """Merit-order recourse of each scenario in ``scenario_ids``.

        ``avail_fn(scenario_id)`` returns the availability vector of the units
        (in portfolio order); all units are available when it is ``None``.
        """
        out = np.empty(len(scenario_ids))
        for i, s in enumerate(scenario_ids):
            d = net_df.loc[s, HOUR_COLS].to_numpy(float)
            av = avail_fn(s) if avail_fn is not None else None
            out[i] = self.recourse_cost(d, av)
        return out


def validate_against_lp(gen, commitment, net_df, scenario_ids, draw_avail_fn=None):
    """Compare the merit-order recourse with the single-scenario dispatch LP.

    Parameters
    ----------
    gen, commitment
        Generator portfolio and fixed commitment.
    net_df, scenario_ids
        Scenario table and the scenarios to compare.
    draw_avail_fn
        ``draw_avail_fn(scenario_id)`` returns ``{unit: availability}``; the
        outaged unit (if any) is removed in both evaluations.

    Returns
    -------
    pandas.DataFrame
        Columns ``scenario``, ``merit_order``, ``lp`` and ``rel_err_pct``
        (100 (merit order - LP) / LP).
    """
    from .uc_model import evaluate_fixed_commitment_dispatch_cost

    md = MeritOrderDispatch(gen, commitment)
    units = [str(u) for u in gen["unit"]]
    rows = []
    for s in scenario_ids:
        d = net_df.loc[s, HOUR_COLS].to_numpy(float)
        if draw_avail_fn is not None:
            a = draw_avail_fn(s)
            avail = np.array([a[u] for u in units])
            outaged = [u for u in units if a[u] == 0.0]
        else:
            avail, outaged = None, []
        mo = md.recourse_cost(d, avail)
        lp = evaluate_fixed_commitment_dispatch_cost(
            gen, commitment, d, outaged_unit=(outaged[0] if outaged else None))
        rows.append({"scenario": s, "merit_order": mo, "lp": lp,
                     "rel_err_pct": 100.0 * (mo - lp) / lp if lp else 0.0})
    return pd.DataFrame(rows)
