# Model and algorithm as implemented

This document states the unit-commitment (UC) model, the evaluation of a fixed commitment, the
merit-order dispatch and the PI-SASR algorithm exactly as they are implemented in `src/pisasr/`.
Section 3 of the paper prints a compact form of the model; Section 10 below lists every
difference between that form and the code. Function names refer to modules of the package
(`uc_model.build_uc_master_model` is `build_uc_master_model` in `src/pisasr/uc_model.py`).

## 1. Notation

**Sets.**

| Symbol | Meaning | Code |
|---|---|---|
| $\mathcal{G}$ | committable units, in the row order of `data/<system>/generator_portfolio.csv` | `m.G` |
| $\mathcal{T}=\{0,\dots,23\}$ | hours of the day | `m.T` |
| $\mathcal{S}$ | scenarios of the model (their identifiers, for example `scenario_1018`) | `m.S` |

**Parameters.** Every unit parameter is a column of the generator portfolio.

| Symbol | Meaning | Unit | Column or constant |
|---|---|---|---|
| $\underline P_g$, $\overline P_g$ | minimum and maximum output when committed | MW | `p_min_mw`, `p_max_mw` |
| $c_g$ | variable (marginal) cost | USD/MWh | `variable_cost` |
| $c^{\mathrm{nl}}_g$ | no-load cost of a committed hour | USD/h | `no_load_cost` |
| $c^{\mathrm{su}}_g$, $c^{\mathrm{sd}}_g$ | startup and shutdown cost | USD | `startup_cost`, `shutdown_cost` (0 for every unit of both portfolios) |
| $RU_g$, $RD_g$ | ramp-up and ramp-down limits | MW/h | `ramp_up_mw`, `ramp_down_mw` |
| $UT_g$, $DT_g$ | minimum up and down times | h | `min_up_hours`, `min_down_hours` |
| $u^0_g$ | status before hour 0 (1 on, 0 off) | | `initial_status` |
| $d_{s,t}$ | net load of scenario $s$ in hour $t$ | MW | scenario files, columns `hour_00` ... `hour_23` |
| $a_{g,s}$ | availability of unit $g$ in scenario $s$ | | Section 4 |
| $\pi_s$ | scenario probability | | Section 3 |
| $c^{\mathrm{voll}}$ | value of lost load | USD/MWh | `constants.VOLL` = 10,000 |
| $\kappa$ | penalty on spilled surplus | USD/MWh | `constants.SPILL_PENALTY` = 1 |
| $q$ | forced-outage rate | | `outages.GEN_FORCED_OUTAGE_RATE` = 0.05 |

**Variables.**

| Symbol | Meaning | Stage | Code |
|---|---|---|---|
| $u_{g,t}\in\{0,1\}$ | commitment (on/off) | first | `m.u` |
| $v_{g,t}\in\{0,1\}$ | startup | first | `m.startup` |
| $w_{g,t}\in\{0,1\}$ | shutdown | first | `m.shutdown` |
| $p_{g,t,s}\ge 0$ | output (MW) | second | `m.p` |
| $\ell_{t,s}\ge 0$ | unserved energy (MWh in the hour) | second | `m.ens` |
| $\rho_{t,s}\ge 0$ | spilled surplus (MWh in the hour) | second | `m.spill` |

## 2. Two-stage stochastic UC (`uc_model.build_uc_master_model`)

One builder serves the deterministic UC, the full (extensive-form) model and the reduced UC of
PI-SASR; they differ only in the scenario set, the probabilities and the availabilities
(Section 3).

**Objective** (`m.objective`): first-stage cost plus the expected recourse cost,

$$
\min\; F(u,v,w) + \sum_{s\in\mathcal{S}} \pi_s \Big[\sum_{g\in\mathcal{G}}\sum_{t\in\mathcal{T}} c_g\, p_{g,t,s}
+ \sum_{t\in\mathcal{T}} \big(c^{\mathrm{voll}}\,\ell_{t,s} + \kappa\,\rho_{t,s}\big)\Big],
\qquad
F(u,v,w)=\sum_{g,t}\big(c^{\mathrm{nl}}_g u_{g,t} + c^{\mathrm{su}}_g v_{g,t} + c^{\mathrm{sd}}_g w_{g,t}\big).
$$

**Constraints.** All hold for every $g\in\mathcal{G}$, $t\in\mathcal{T}$ and $s\in\mathcal{S}$
unless stated otherwise.

1. Commitment logic (`commitment_logic`):
   $u_{g,0}-u^0_g = v_{g,0}-w_{g,0}$ and $u_{g,t}-u_{g,t-1} = v_{g,t}-w_{g,t}$ for $t\ge 1$.
2. Output limits (`gen_min`, `gen_max`):
   $\underline P_g\,u_{g,t}\,a_{g,s} \le p_{g,t,s} \le \overline P_g\,u_{g,t}\,a_{g,s}$.
3. Power balance on net load (`balance`):
   $\sum_{g} p_{g,t,s} + \ell_{t,s} - \rho_{t,s} = d_{s,t}$.
4. Bound on unserved energy (`ens_limit`): $\ell_{t,s} \le \max(d_{s,t},0)$.
5. Ramp limits with startup and shutdown allowances (`ramp_up`, `ramp_down`), for every
   unit-scenario pair with $a_{g,s}\neq 0$:

   $$
   p_{g,t,s}-p_{g,t-1,s} \le RU_g + \overline P_g\,v_{g,t},\qquad
   p_{g,t-1,s}-p_{g,t,s} \le RD_g + \overline P_g\,w_{g,t},
   $$

   with the output before the horizon fixed at $p_{g,-1,s}=\underline P_g\,u^0_g$ (not multiplied
   by the availability). For an outaged unit ($a_{g,s}=0$) both ramp constraints are omitted, so
   the unit drops to zero output at once.
6. Minimum up time (`min_up`), only for units with $UT_g>1$, with the window truncated at the end
   of the day:

   $$
   \sum_{\tau=t}^{e-1} u_{g,\tau} \ge (e-t)\,v_{g,t},\qquad e=\min(t+UT_g,\,24).
   $$

7. Minimum down time (`min_down`), only for units with $DT_g>1$:

   $$
   \sum_{\tau=t}^{e-1} \big(1-u_{g,\tau}\big) \ge (e-t)\,w_{g,t},\qquad e=\min(t+DT_g,\,24).
   $$

8. Domains: $u, v, w$ binary; $p, \ell, \rho$ non-negative.

**Initial conditions.** The only information about the previous day is $u^0_g$ (nuclear and
coal units are on, all others off, in both portfolios) and the implied output
$\underline P_g u^0_g$ in constraint 5. There is no record of how long a unit has been on or off
before hour 0, so the minimum up and down times do not constrain the first hours through the
initial status.

**Remarks.**

* The bound of constraint 4 does not change any optimal value: a solution with
  $\ell_{t,s}>\max(d_{s,t},0)$ must also spill, and reducing $\ell_{t,s}$ and $\rho_{t,s}$ by the
  same amount lowers the cost by $c^{\mathrm{voll}}+\kappa$ for each MWh. The evaluation LPs
  (Sections 5 and 7) omit it.
* There is no constraint $v_{g,t}+w_{g,t}\le 1$. For a unit with $UT_g>1$ and $DT_g>1$,
  constraints 6 and 7 exclude $v_{g,t}=w_{g,t}=1$; for units with one-hour minimum times
  (hydro, and diesel on Texas2k) both may equal one in the same hour, at the price of a startup
  (zero for the TX-123BT hydro units).
  The exact evaluation (Section 5) derives $v$ and $w$ from $u$, so there they equal one only when
  the commitment changes.

**Solution** (`uc_model.solve_pyomo_model`). Pyomo's `gurobi` interface writes the model and
Gurobi solves it with the options of `constants.GUROBI_OPTIONS`: `Method=2` (the barrier method
at the root relaxation, as Section 5.2 of the paper states), `Crossover=0`, `MIPGap=1e-3` (the
0.1% MILP tolerance) and `NodefileStart=8`. No thread count, time limit or random seed is set, so
Gurobi uses all available cores. For these MILPs Gurobi 13.0.2 does not apply `Crossover=0`: its
log prints "Ignoring user setting 'Crossover=0', set 'NodeMethod=2' to disable Crossover", and
the barrier is followed by a short crossover. The option is kept because it is part of the
settings that produced the results. These options apply to every MILP (the deterministic UC, the
full model, the reduced UC and the batch problems of the lower bound); the evaluation and
validation LPs use Gurobi's default settings. The commitment is read as `int(round(value(u)))`
(`uc_model.get_commitment_from_model`).

## 3. Model variants

| Variant | Scenario set $\mathcal{S}$ | $\pi_s$ | $a_{g,s}$ | Code |
|---|---|---|---|---|
| Deterministic (expected-value) UC | one scenario `expected`: the mean profile of all 2,500 training scenarios | 1 | $1-q=0.95$ for every unit | `baselines.run_deterministic` |
| Full (extensive-form) model at S | the first S rows of one random permutation (random seed 42) of the training set | $1/S$ | outage draw of Section 4 | `baselines.run_full`, `baselines.run_full_file_based` |
| Reduced UC of PI-SASR | the active set $\mathcal{R}$, $K=\lvert\mathcal{R}\rvert$ | $1/K$ | outage draw of Section 4 | `pisasr.run_pisasr` |

* In the deterministic UC the factor 0.95 multiplies both output limits of constraint 2, so every
  committed unit has the range $[0.95\,\underline P_g,\,0.95\,\overline P_g]$; the ramp constraints
  apply to every unit because no availability is zero.
* The scenario subsets are nested (`uc_model.select_stochastic_scenarios`): one permutation of the
  2,500 training rows is drawn with `numpy.random.default_rng(42)`, and the full model at S uses
  its first S entries, so the 100 scenarios lie inside the 200, and so on. At S = 2,500 the whole
  training set is used.
* In the reduced UC of PI-SASR the commitment of every baseload unit is fixed,
  $u_{g,t}=1$ for $g\in\mathcal{G}^{\mathrm{base}}$ and all $t$ (Pyomo `fix`).
* The Texas2k full model at S = 2,500 is written to an LP file with symbolic names, the Pyomo model
  is released, and the file is solved with `gurobipy` using the same options
  (`baselines.run_full_file_based`); the model is identical.

## 4. Generator outages (`outages.draw_scenario_availability`)

For a scenario with identifier $s$, a generator `numpy.random.default_rng(seed)` is created with
seed $=\mathrm{crc32}(s) \oplus 12345$ (`zlib.crc32` of the identifier string, XOR the base seed).
The units are visited in portfolio order and unit $g$ is drawn out when `rng.random() < 0.05`. If
more than one unit is drawn, one of them is kept by `rng.choice` and the others are returned to
service. The kept unit has $a_{g,s}=0$; every other unit has $a_{g,s}=1$.

The draw depends only on the scenario identifier, so a scenario has the same outage in every model,
in PI-SASR and in the evaluation, whatever the commitment or the subset size. With 138 or 568
units, almost every scenario has exactly one outage: 2,495 of the 2,500 TX-123BT training
scenarios (five have none) and all 2,500 Texas2k training scenarios.

## 5. Exact evaluation of a fixed commitment (`evaluation.py`)

For a commitment $u$ (a `commitment.csv`), the startup and shutdown indicators are derived as
$v_{g,t}=\max(u_{g,t}-u_{g,t-1},0)$ and $w_{g,t}=\max(u_{g,t-1}-u_{g,t},0)$ with
$u_{g,-1}=u^0_g$ (`evaluation.compute_startup_shutdown_from_commitment`), and the first-stage cost
$F(u)$ is computed from them (`evaluation.compute_fixed_commitment_cost`).

The recourse $Q_s(u)$ of each scenario is the LP of Section 2 with $u$, $v$ and $w$ fixed:
objective $\sum_{g,t}c_g p_{g,t,s}+\sum_t(c^{\mathrm{voll}}\ell_{t,s}+\kappa\rho_{t,s})$,
constraints 2, 3 and 5 (with the startup and shutdown allowances), the outage draw of Section 4,
and no bound on unserved energy (`evaluation.evaluate_commitment_batch`). Fifty scenarios are
solved in one LP; the LP separates across scenarios, so the batch size does not change any value.
The LPs use Gurobi's default settings and must terminate optimal.

* **Evaluation set**: the first 2,000 rows of `data/<system>/net_load_scenarios_test.csv`, which
  are disjoint from the training set (`evaluation.load_net_load_scenarios`). Each row of
  `evaluation.csv` holds $Q_s(u)$ (`dispatch_cost`), the unserved and spilled energy, the number of
  hours with unserved energy above $10^{-4}$ MWh, the total net load and
  `total_cost_with_fixed` $=F(u)+Q_s(u)$.
* **Exact objective of PI-SASR**: the same LP over the S training scenarios of the full model, in
  batches of 100 (`pisasr._exact_recourse_mean`); the objective is $F(u)$ plus the mean recourse.

## 6. Merit-order dispatch (`merit_order.MeritOrderDispatch`)

For a commitment $u$, availabilities $a_{g,s}$ and a net-load profile $d_s$, the merit-order
recourse $\widetilde Q_s(u)$ is computed hour by hour. In hour $t$, let
$\mathcal{O}_t=\{g: u_{g,t}a_{g,s}=1\}$ be the committed and available units.

1. If $\mathcal{O}_t$ is empty, the cost of the hour is $c^{\mathrm{voll}}\max(d_{s,t},0)$.
2. Otherwise every unit in $\mathcal{O}_t$ runs at $\underline P_g$, at cost
   $\sum_{g\in\mathcal{O}_t}c_g\underline P_g$, and the residual is
   $r=d_{s,t}-\sum_{g\in\mathcal{O}_t}\underline P_g$.
3. If $r\le 0$, the surplus $-r$ is spilled at cost $\kappa(-r)$.
4. If $r>0$, the headroom $\overline P_g-\underline P_g$ of the units in $\mathcal{O}_t$ is filled
   in increasing order of $c_g$ until $r$ is met; any remainder above $10^{-6}$ MW is unserved
   and costs $c^{\mathrm{voll}}$ for each MWh.

$\widetilde Q_s(u)$ is the sum over the 24 hours. For a non-negative net load this is the exact
optimum of the dispatch LP of Section 5 without the ramp constraints (the hours decouple), so
$\widetilde Q_s(u)\le Q_s(u)$ for every scenario (Proposition 1(ii) of the paper). All net loads
of both systems are positive (at least 21.5 GW on TX-123BT and 1,000 MW on Texas2k). As Sections
4.2 and 4.4 of the paper state, PI-SASR uses the merit-order dispatch for the design-set recourse
(the GP targets) and for the verification over all S scenarios (`MeritOrderDispatch.recourse_over`);
the component analysis, the criticality-only selection and the GP kernel comparison use it for the
exact criticality ranking, which the paper also computes with the merit-order dispatch (Sections
4.2 and 7.3). The ramp-coupled LP of Section 5 is used only for the exact objective and the
evaluation set.

## 7. Accuracy check of the merit-order dispatch (Section 6.5 of the paper)

`merit_order.validate_against_lp` compares $\widetilde Q_s(u)$ with a single-scenario dispatch LP,
`uc_model.evaluate_fixed_commitment_dispatch_cost`, for the first 70 rows of the training file
(file order) under the stored reference commitment $u^{\mathrm{ref}}$, each scenario with its own
outage (`scripts/validate_merit_order.py`). This validation LP has the same objective and
constraints 2 and 3 as the evaluation LP of Section 5, but its ramp limits carry **no startup or
shutdown allowance**:
$p_{g,t}-p_{g,t-1}\le RU_g$ and $p_{g,t-1}-p_{g,t}\le RD_g$ (again omitted for the outaged unit,
again with $p_{g,-1}=\underline P_g u^0_g$). It returns $10^{15}$ if the LP does not terminate
optimal; this did not occur in the 70 scenarios of either system.

The feasible set of the validation LP is contained in that of the evaluation LP, so for every
scenario $\widetilde Q_s \le Q^{\mathrm{eval}}_s \le Q^{\mathrm{val}}_s$, and therefore
$(Q^{\mathrm{val}}_s-\widetilde Q_s)/Q^{\mathrm{val}}_s \ge (Q^{\mathrm{eval}}_s-\widetilde Q_s)/Q^{\mathrm{eval}}_s$.
The mean and largest relative errors stated in the paper (0.66% and 5.0% on TX-123BT, 0.57%
and 8.2% on Texas2k) are therefore **upper bounds** on the errors of the merit-order dispatch
relative to the evaluation LP that PI-SASR uses for its exact objective. The agreement measure
that the paper reports, the squared correlation, is the square of the Pearson correlation of the
merit-order and LP values over the 70 scenarios (`numpy.corrcoef`): 0.99962 on TX-123BT and
0.99995 on Texas2k.

## 8. PI-SASR (`pisasr.select_active_set`, `pisasr.run_pisasr`)

Settings (`constants.PISASR_DEFAULTS`, identical on both systems):

| Setting | Symbol | Value | Argument |
|---|---|---|---|
| critical scenarios | $K_c$ | 90 | `k_critical` |
| coverage scenarios | $K_r$ | 60 | `k_cover` |
| design set | $\lvert\mathcal{D}\rvert$ | 140 | `design_size` |
| rank of the embedding | $r$ | 5 | `svd_rank` |
| enrichment tolerance | $\epsilon$ | 0.2% (`0.002`) | `cg_tol` |
| enrichment budget | | one round | `cg_rounds` |
| scenarios added in one enrichment round | | 40 | `cg_add` |
| random seed | | 7 | `random_seed` |

**Offline input.** The reference commitment $u^{\mathrm{ref}}$ is the deterministic UC schedule,
read by default from `results/<system>/runs/deterministic/commitment.csv`
(`systems.resolve_reference_commitment`).

**Selection** (`pisasr.select_active_set`, one random generator
`numpy.random.default_rng(7)` used in the order given):

1. *Target scenarios.* The S scenarios of the full model (Section 3).
2. *Features* (`features.build_feature_matrix`). Thirteen physics-informed features of each
   profile (`features.physics_features`): peak, trough, energy, mean, standard deviation, largest
   upward and downward hourly ramp, sum of absolute ramps, load factor (mean over peak),
   capacity-margin deficit $\max_t d_{s,t}-(1-q)\sum_g\overline P_g$, peak utilization
   $\max_t d_{s,t}/\sum_g\overline P_g$, and the numbers of hours above 85% and 95% of
   $(1-q)\sum_g\overline P_g$. They are followed by the rank-5 embedding
   (`features.lowrank_embedding`): the S × 24 matrix of profiles is centered by its column means and
   projected onto its first five right singular vectors (`numpy.linalg.svd`), giving 18 features.
3. *Design set and GP surrogate.* The design set is `rng.choice(S, 140, replace=False)` among
   the target scenarios, and its recourse $y_s=\widetilde Q_s(u^{\mathrm{ref}})$ is computed with
   the merit-order dispatch. A permutation `rng.permutation(140)` sets aside
   $\max(10,\lfloor 0.25\cdot 140\rfloor)=35$ design scenarios; the GP (`surrogate.GPSurrogate`)
   is fitted to the other 105, its coefficient of determination and mean absolute error on the 35
   are recorded (`surrogate_r2`, `surrogate_mae`), and it is then fitted again to all 140. The GP
   standardizes the features (`StandardScaler`) and the costs (mean and population standard deviation),
   uses the kernel `ConstantKernel(1.0, (1e-3, 1e6)) * Matern(length_scale=ones(18),
   length_scale_bounds=(1e-3, 1e6), nu=2.5) + WhiteKernel(1e-2, (1e-10, 1e2))`, `alpha=1e-6` and
   four optimizer restarts with `random_state=7`.
4. *Active set.* The critical scenarios are the $K_c=90$ scenarios with the largest GP posterior
   mean. The coverage scenarios are the medoids of `KMeans(n_clusters=60, n_init=4,
   random_state=7)` on the embedding: for each cluster, the member closest to its centroid. The
   active set is `dict.fromkeys(critical + cover)`: the critical scenarios in ranking order followed
   by the medoids not already included, so $K\le 150$.
5. *Baseload pre-commitment.* $\mathcal{G}^{\mathrm{base}}$ is the set of units on in all 24
   hours of $u^{\mathrm{ref}}$ (64 units on TX-123BT, 58 on Texas2k).

**Reduced UC, verification and enrichment** (`pisasr.run_pisasr`):

6. The reduced UC (Section 3) is solved over the active set with $\pi_s=1/K$ and the baseload
   units fixed on; its objective is $Z_{\mathcal{R}}$.
7. *Merit-order verification.* The verified objective is
   $\widetilde Z = F(u^{\mathcal{R}}) + \frac1S\sum_{s}\widetilde Q_s(u^{\mathcal{R}})$ over all S
   target scenarios, and the under-representation statistic is
   $\widehat\Delta=(\widetilde Z-Z_{\mathcal{R}})/Z_{\mathcal{R}}$ (`under_rep_gap` in
   `pisasr_log.csv`).
8. *Enrichment.* If $\widehat\Delta\le\epsilon$ the loop stops. Otherwise, if a round is left, the
   40 scenarios outside the active set with the largest verified recourse (under the best
   commitment so far) are appended and steps 6 and 7 are repeated. The commitment with the smallest
   verified objective is kept. In every PI-SASR run of the scaling study (Tables 2 and 3, random
   seed 7), $\widehat\Delta\le\epsilon$ after the first reduced UC, so no enrichment round was
   performed; one of the five design-set draws of Section 6.2 needed one round ($K = 181$). The
   component analysis, the criticality-only selection, the embedding comparison, the budget
   sensitivity and the scenario-weight experiments solve one reduced UC for each variant, without
   enrichment.
9. *Exact objective.* The kept commitment is evaluated once with the exact LP of Section 5 over
   the S target scenarios; the result is `objective`, and the benchmark gap is
   $\Delta = 100\,(\text{objective} - Z_{\mathrm{full}})/Z_{\mathrm{full}}$ (`opt_gap_pct_vs_full`),
   where $Z_{\mathrm{full}}$ is the full-model objective at the same S
   (`systems.resolve_full_objective`: a rerun in the output folder if present, otherwise
   `results/`).

**Online time** (`solve_time_seconds`): wall-clock time from the start of `run_pisasr` (data
loading) to the end of step 8. It includes the features, the design-set dispatches, the GP fits,
k-means, the construction and solution of the reduced UC and the merit-order verification. It
excludes the deterministic UC solution that produced $u^{\mathrm{ref}}$ (read from a file) and the exact
evaluation of step 9 (`exact_validation_s`).

## 9. Statistics of the paper (`statistics.py`)

* **Paired cost difference** (Section 6.3; `statistics.paired_difference_interval`): for the
  2,000 evaluation scenarios, $\delta_s$ = total cost of the PI-SASR commitment minus that of the
  full-model commitment; the interval is $\bar\delta \pm t_{0.975,\,N-1}\,\mathrm{sd}(\delta)/\sqrt N$,
  expressed as a percentage of the full model's mean cost.
* **Certified gap** (Sections 4.4 and 6.1; `statistics.certified_gap`): the $M=5$ batch
  incumbents (batch size $S'=200$, and also 500 on TX-123BT; disjoint blocks of a permutation of
  the training set with random seed $1000+S'$) are each multiplied by $1-10^{-3}$; with $\bar L$
  and $\mathrm{sd}_M$ their mean
  and sample standard deviation, $L^-=\bar L - t_{0.975,\,M-1}\mathrm{sd}_M/\sqrt M$. With
  $\bar Z_N$ and $\mathrm{sd}_N$ the mean and sample standard deviation of the total cost of the
  commitment on the $N=2{,}000$ evaluation scenarios, $Z^+=\bar Z_N + t_{0.975,\,N-1}\mathrm{sd}_N/\sqrt N$,
  and the certified gap is $100\,(Z^+-L^-)/L^-$.

## 10. Differences from the compact formulation printed in the paper

1. **Time index and initial conditions.** Hours run from 0 to 23. The paper does not state the
   initial conditions; the code uses the initial status $u^0_g$ in the commitment logic and the
   output $\underline P_g u^0_g$ before hour 0 in the ramp constraints.
2. **Ramp limits.** Equation (ramp) of the paper is printed as
   $-RD_g\le p_{g,t,s}-p_{g,t-1,s}\le RU_g$, and the text states that the limits are widened by
   $\overline P_g$ in startup and shutdown hours. The code implements the widened form
   $RU_g+\overline P_g v_{g,t}$ and $RD_g+\overline P_g w_{g,t}$ in the UC model and in the exact
   evaluation, and omits both limits for an outaged unit.
3. **Minimum up and down times.** The paper refers to minimum up and down time constraints in
   general; the code uses the window form of constraints 6 and 7, truncated at the end of the day,
   without any history before hour 0, and only for units whose minimum time exceeds one hour.
4. **Bound on unserved energy.** The UC model contains $\ell_{t,s}\le\max(d_{s,t},0)$, which is not
   in the paper and does not change the optimal value; the evaluation LPs do not contain it.
5. **Spilled surplus.** The paper describes $\rho_{t,s}$ as curtailed renewable surplus with a small
   penalty $\kappa$; in the code it is any surplus of the committed minimum outputs over the net
   load, priced at 1 USD/MWh.
6. **Deterministic UC.** The paper calls it the expected-value UC. The code solves one scenario, the
   mean of all 2,500 training profiles, with both output limits of every unit multiplied by
   $1-q=0.95$ to account for outages in expectation.
7. **Outage rule.** The paper states independent forced outages with at most one outage for each
   scenario. The code draws every unit independently and, when several are drawn, keeps one at
   random, so nearly every scenario has exactly one outage (Section 4).
8. **Startup and shutdown variables.** There is no constraint $v_{g,t}+w_{g,t}\le 1$ in the UC
   model (Section 2); the exact evaluation and the first-stage costs in the result files derive
   $v$ and $w$ from $u$.
9. **Merit-order accuracy check.** Section 6.5 of the paper compares the merit-order values with
   the exact LP. The check uses the validation LP of Section 7, whose ramp limits lack the
   startup and shutdown allowances of the evaluation LP, so the errors stated in the paper are upper
   bounds relative to the evaluation LP. The check uses the first 70 rows of the training file
   under the deterministic reference commitment, as the paper states.
10. **Physics features.** Section 4.1 of the paper names nine of the 13 physics features (peak,
    energy, peak utilization, capacity-margin deficit, hours above 85% and 95% of the available
    capacity, largest upward and downward hourly ramps, load factor). The code also uses the
    trough, the mean, the standard deviation and the sum of absolute hourly ramps (Section 8,
    step 2).
11. **Cost data.** Shutdown costs are zero in both portfolios. On TX-123BT the no-load cost is the
    coefficient C0 of the source data, whose column header gives \$/MWh, applied as USD/h for each
    committed hour.

Construction details of the input data that the paper does not state (among them the 211
out-of-service units among the Texas2k committable units, the independent copula draws of the
hours of TX-123BT, the Texas2k wind scaling and the net-load floor) are listed in
`docs/REPRODUCIBILITY.md`, Section 5.
