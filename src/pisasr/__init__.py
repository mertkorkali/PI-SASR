"""PI-SASR: physics-informed surrogate-assisted scenario reduction.

Reference implementation of

    M. Korkali and D. Andriniaina, "Physics-Informed Surrogate-Assisted Scenario
    Reduction for Scalable Two-Stage Stochastic Unit Commitment: A Cross-System
    Validation on Synthetic Texas Grids," Proc. 60th Hawaii International
    Conference on System Sciences (HICSS-60), January 2027.

Modules
-------
paths        repository folders (data, reference results, outputs, raw data)
constants    shared constants (hours, value of lost load, solver options)
systems      the two test systems and the function that selects one
outages      reproducible N-1 generator-outage draw for each scenario
uc_model     unit-commitment model (deterministic and two-stage stochastic)
evaluation   exact ramp-coupled LP evaluation of a fixed commitment
features     physics-informed scenario features and low-rank embedding
surrogate    Gaussian-process (GP) criticality surrogate
merit_order  merit-order recourse used for the verification over all scenarios
pisasr       the PI-SASR algorithm
baselines    deterministic (expected-value) UC and full (extensive-form) model
experiments  additional experiments of the paper (component timing, kernels,
             embedding, scenario weights, lower bound, active-set budget)
statistics   paired cost-difference confidence intervals and the lower bound
reporting    evaluation summaries, tables and figures of the paper
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
