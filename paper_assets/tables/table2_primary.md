**Table 2.** Primary comparison at S = 1,000 on both systems.

| System | Method | Obj. (\$M) (a) | Gap (%) | Time (s) | Scen. used | Unserved (%) (b) |
|---|---|---:|---:|---:|---:|---:|
| TX-123BT | Deterministic | 37.76 | n/a | 1.2 | 1 | 1.755 |
|  | Full stochastic | 38.11 | 0.00 | 434 | 1000 | 0.0026 |
|  | **PI-SASR** | 38.37 | 0.69 | **74** | **148** | 0.0060 |
| Texas2k | Deterministic | 22.88 (c) | n/a | 3.6 | 1 | 12.13 |
|  | Full stochastic | 28.59 | 0.00 | 4324 | 1000 | 0.0019 |
|  | **PI-SASR** | 28.79 | 0.71 | **671** | **146** | 0.0010 |

(a) Deterministic UC: objective over its single expected-value scenario. Full model: objective, the expected total cost over the S training scenarios. PI-SASR: exact expected total cost of its commitment over the same S scenarios (fixed cost plus the mean ramp-coupled LP recourse).  
(b) Unserved-energy rate on the evaluation set (2,000 scenarios) under generator outages. Times are wall-clock times in seconds, defined as in Table 3.  
(c) The deterministic objective ignores uncertainty; its evaluation-set expected cost on Texas2k is \$1.0B, 35× the stochastic schedules'.
