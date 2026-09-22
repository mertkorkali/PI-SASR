**Table 4.** Component analysis at equal active-set size on TX-123BT (S = 1,000): benchmark gap and end-to-end online time of each variant (each timed in a single dedicated run, separately from the runs of Table 3).

| Variant | K | Sel. (s) | MILP (s) | Online (s) (a) | Gap (%) |
|---|---:|---:|---:|---:|---:|
| Fast-forward selection (Heitsch and Römisch) | 148 | 0.3 | 69 | 70 | +4.99 |
| Coverage medoids only | 148 | 0.1 | 75 | 76 | +1.39 |
| Cost-space forward selection (Werner et al.) | 148 | 1.4 | 82 | 85 | +0.94 |
| Exact criticality only | 148 | 1.0 | 92 | 94 | +0.20 |
| GP criticality only | 148 | 6.1 | 87 | 94 | +0.49 |
| GP criticality only, K = 188 | 188 | 6.1 | 111 | 118 | +0.39 |
| Exact criticality ∪ coverage | 147 | 1.2 | 93 | 95 | +0.67 |
| PI-SASR, no baseload pre-commitment | 148 | 5.6 | 103 | 109 | +0.56 |
| **PI-SASR** | **148** | **4.6** | **71** | **77** | +0.69 |

(a) Features, selection, reduced MILP, and merit-order verification over all S; the offline exact re-evaluation behind the gap column is excluded for all variants. The two reference methods are the fast-forward selection of Heitsch and Römisch (2003) and the problem-driven forward selection in recourse-cost space of Werner et al. (2025).
