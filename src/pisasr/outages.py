"""Reproducible N-1 generator-outage draw for each scenario.

This module is the single definition of the forced-outage model. It is used by
the unit-commitment model, by PI-SASR and by the exact evaluator, so that every
component applies the same outages to a given scenario.

Model: each unit is forced out independently with probability ``for_rate``; if
more than ``max_out`` units are drawn for one scenario, ``max_out`` of them are
kept at random. An outaged unit has availability 0.0 (it cannot produce); every
other unit has availability 1.0.

The draw for a scenario is seeded by a stable function of the scenario
identifier (``zlib.crc32`` of the identifier XOR a base seed; Python's salted
``hash`` is not used). Consequently

* a scenario always receives the same outage, however many scenarios are
  selected, which preserves the nested-subset property of the scenario sets;
* every commitment is evaluated under identical outages, so differences in
  unserved energy reflect commitment decisions and not the sampled draws.
"""

from __future__ import annotations

import zlib

import numpy as np

GEN_FORCED_OUTAGE_RATE = 0.05
"""Forced-outage rate of every unit."""

MAX_OUTAGES_PER_SCENARIO = 1
"""At most one generator outage in each scenario (N-1)."""

OUTAGE_BASE_SEED = 12345
"""Base seed combined with the scenario identifier."""


def _stable_seed(scenario_id, base_seed: int) -> int:
    """Process-independent seed derived from a scenario identifier."""
    digest = zlib.crc32(str(scenario_id).encode("utf-8"))
    return (digest ^ (base_seed & 0xFFFFFFFF)) & 0xFFFFFFFF


def draw_scenario_availability(
    units,
    scenario_id,
    for_rate: float = GEN_FORCED_OUTAGE_RATE,
    max_out: int | None = MAX_OUTAGES_PER_SCENARIO,
    base_seed: int = OUTAGE_BASE_SEED,
) -> dict:
    """Return the availability ``{unit: 1.0 or 0.0}`` of one scenario.

    Parameters
    ----------
    units
        Iterable of unit identifiers. The draw visits the units in this order,
        so the row order of the generator file matters.
    scenario_id
        Scenario identifier (for example ``"scenario_1018"``).
    for_rate, max_out, base_seed
        Forced-outage rate, maximum number of outages and base seed.
    """
    units = list(units)
    rng = np.random.default_rng(_stable_seed(scenario_id, base_seed))

    availability = {u: 1.0 for u in units}

    outaged = [u for u in units if rng.random() < for_rate]

    if max_out is not None and len(outaged) > max_out:
        keep_idx = rng.choice(len(outaged), size=max_out, replace=False)
        outaged = [outaged[i] for i in keep_idx]

    for u in outaged:
        availability[u] = 0.0

    return availability
