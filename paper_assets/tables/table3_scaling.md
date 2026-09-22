**Table 3.** Scenario scaling on both systems: PI-SASR versus full stochastic UC. Δ̂ is the realized under-representation statistic.

| System | S | Full obj. (\$M) | Full time (s) (b) | PI-SASR gap (%) | Δ̂ (%) | PI-SASR time (s) (b) | K | Speedup (a) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TX-123BT | 100 | 37.96 | 70 | 0.004 | -0.28 | 56 | 95 | 1.26 |
|  | 200 | 38.04 | 108 | 0.67 | -0.40 | 62 | 122 | 1.74 |
|  | 500 | 38.02 | 305 | 0.61 | -0.47 | 84 | 143 | 3.61 |
|  | 1,000 | 38.11 | 434 | 0.69 | -1.33 | 74 | 148 | 5.89 |
|  | 2,500 | 38.18 | 1347 | 0.83 | -1.12 | **63** | 146 | **21.3** |
| Texas2k | 100 | 27.59 | 568 | 0.017 | -0.39 | 419 | 99 | 1.36 |
|  | 200 | 28.25 | 1441 | 0.082 | -6.32 | 554 | 128 | 2.60 |
|  | 500 | 28.64 | 3381 | 0.40 | -12.98 | 567 | 143 | 5.96 |
|  | 1,000 | 28.59 | 4324 | 0.71 | -15.41 | 671 | 146 | 6.44 |
|  | 2,500 | 28.56 | 14689 (c) | 1.29 | -15.87 | **613** | 147 | **24.0** |

(a) Speedup = full time / PI-SASR time (unrounded). Gaps are benchmark gaps against the full model's incumbent (0.1% MILP tolerance), not certificates; certified gaps are given in Section 6.1. Δ̂ is the under-representation statistic of the first reduced UC; it never exceeded the enrichment tolerance ε = 0.2%, so no enrichment round was performed.  
(b) Wall-clock times in seconds. Full model on TX-123BT: solver call only; full model on Texas2k: model construction and solver call, so the TX-123BT speedups are conservative. PI-SASR: online time from data loading to the end of the merit-order verification, excluding the deterministic UC solution that gives the reference commitment and the exact re-evaluation behind the gap column.  
(c) The Texas2k S = 2,500 extensive form exceeded the workstation's 192 GB of memory and was therefore written to a model file and solved from it; its time covers model construction, writing the model file and the solver call (solver call alone: 11,594 s).
