# Results guide

This guide links every table, figure and number in the results of the paper to the files in
`results/` that support it, names the columns used, and gives the commands that regenerate
them. Section 4 describes every CSV file in `results/`, with units.

All paths below are relative to the repository root. `<system>` is `tx123bt` (TX-123BT) or
`texas2k` (Texas2k Series25), and `<S>` is a scenario count of the paper (100, 200, 500, 1,000
or 2,500).

## 1. Rebuilding the tables, figures and numbers (no solver needed)

Three scripts read the CSV files and need only pandas, NumPy, SciPy and matplotlib. Together
they take a few seconds.

```bash
python scripts/summarize_evaluation.py     # results/<system>/summary/evaluation_metrics.csv
python scripts/paired_cost_difference.py   # results/<system>/summary/paired_cost_difference_S1000.csv
python scripts/make_paper_assets.py        # paper_assets/ (Tables 1-4, Figures 2-4, numbers.md)
```

* Without options the scripts read `results/` and rewrite `results/<system>/summary/` and
  `paper_assets/` in place; the content is identical as long as `results/` is unchanged.
* `--check` (first two scripts) recomputes a summary and compares it with the stored file
  (relative tolerance 1e-9) without writing anything.
* `--results-dir outputs` makes any of the three scripts read a rerun (Section 3); for
  `make_paper_assets.py` add `--out outputs/paper_assets` to keep the committed assets unchanged.
* `paper_assets/numbers.md` lists every number printed in the results of the paper with its
  source file and column, the value computed from it, and whether the two agree.
* `python -m pytest tests/test_reporting.py` recomputes the confidence intervals of Section 6.3 and a
  selection of table values from the CSV files (about 2 s).

## 2. Where each result comes from

"Solver runs" names the command that produces the underlying files (Section 3 lists the full
sequence); "Rebuilt by" names the script of Section 1 that turns them into the printed values.

### Table 1 (test systems) and Section 5.1

* Numbers: committable units, committable capacity, maximum peak net load, training and
  evaluation set sizes; the Texas2k fuel mix, hydro units and mean peak net load.
* Files: `data/<system>/generator_portfolio.csv` (row count; sum of `p_max_mw`; `fuel_type`),
  `data/<system>/net_load_scenarios_train.csv` and `net_load_scenarios_test.csv` (the maximum
  peak net load is the largest hourly value over all 7,500 scenarios of the two files; the mean
  peak net load is the mean of the daily maxima), and the row count of any
  `results/<system>/runs/*/evaluation.csv` (2,000 evaluation scenarios, the first rows of the
  test file).
* Solver runs: none. The processed inputs are rebuilt from the raw data by
  `scripts/data/prepare_tx123bt.py` and `scripts/data/prepare_texas2k.py` (see `data/README.md`).
  The wind and solar counts, the 131 units of fuel type "other", the 85.8 GW peak load and the
  92 summer days are properties of the raw data used by those scripts.
* Rebuilt by: `make_paper_assets.py` (`paper_assets/tables/table1_systems.*`).

### Table 2 (primary comparison at S = 1,000)

* Objective, time and scenarios used: `results/<system>/runs/{deterministic,full_S1000,pisasr_S1000}/solve_metrics.csv`
  (`objective`, `solve_time_seconds`, `n_scenarios` or `n_scenarios_used`); the PI-SASR gap is
  `opt_gap_pct_vs_full`.
* Unserved energy: `results/<system>/runs/<run>/evaluation.csv`, sum of `ens_mwh` divided by
  the sum of `total_net_load_mwh`, times 100 (column `unserved_energy_rate` of
  `summary/evaluation_metrics.csv`, a fraction).
* Note c: mean of `total_cost_with_fixed` in `texas2k/runs/deterministic/evaluation.csv`
  (`expected_total_cost` in `summary/evaluation_metrics.csv`), and its ratio to that of
  `full_S1000` and `pisasr_S1000`.
* Solver runs: `run_full_model.py --deterministic --S 1000`, `run_pisasr.py --S 1000`,
  `evaluate.py --runs deterministic full_S1000 pisasr_S1000`.
* Rebuilt by: `make_paper_assets.py` (`table2_primary.*`).

### Table 3 (scenario scaling) and Figures 2 and 3(a)

* Full model: `results/<system>/runs/full_S<S>/solve_metrics.csv` (`objective`,
  `solve_time_seconds`).
* PI-SASR: `results/<system>/runs/pisasr_S<S>/solve_metrics.csv` (`opt_gap_pct_vs_full`,
  `solve_time_seconds`, `n_scenarios_used`); the under-representation statistic is
  `under_rep_gap` (times 100) in the first row of `pisasr_S<S>/pisasr_log.csv`.
* Speedup: full `solve_time_seconds` divided by PI-SASR `solve_time_seconds`.
* The same values are collected in `results/<system>/summary/scaling.csv`; the reporting code
  reads the run folders directly so that it also works on a rerun.
* Solver runs: `run_full_model.py --S 100 200 500 1000 2500` and `run_pisasr.py --S 100 200 500 1000 2500`.
* Rebuilt by: `make_paper_assets.py` (`table3_scaling.*`, `fig2_solution_time.*`,
  `fig3_speedup_and_kmin.*`).

### Table 4 (component analysis) and Section 7.3

* Table 4 and the times and gaps of Section 7.3: `results/tx123bt/experiments/component_timing_S1000.csv`
  (`K`, `t_select_s`, `t_milp_s`, `t_online_s`, `gap_pct`, one row for each `variant`).
* Evaluation-set cost and unserved energy of the criticality-only variants (Section 7.3):
  `results/<system>/experiments/criticality_only_S1000.csv` (`eval_expected_cost` relative to
  the `pisasr` row, `eval_ens_rate_pct`, and `gap_pct` for Texas2k).
* Solver runs: `run_experiments.py --system tx123bt --stage component-timing` and
  `run_experiments.py --system <system> --stage criticality-only`.
* Rebuilt by: `make_paper_assets.py` (`table4_components.*`, `numbers.md`).

### Figure 3(b) and the budget sensitivity of Section 6.2

* File: `results/<system>/experiments/budget_sensitivity.csv`, one row for each `S` and budget
  (`k_critical`, `k_cover`). `K_min(tau)` is the smallest `K` whose `gap_pct` is at most `tau`
  (1% and 0.5%); when no tested budget meets `tau`, the figure shows a hollow marker at the
  largest `K` tested with ">" (for example, Texas2k at S = 2,500: `K` = 198 reaches 1.08%). As in
  the paper, each curve joins the values that meet `tau`; where it passes over such a censored
  value (TX-123BT, `tau` = 0.5%, S = 1,000) the joining segment is drawn lighter.
* Solver runs: `run_experiments.py --system <system> --stage budget-sensitivity --S ...`
  (TX-123BT: S = 200, 500, 1,000, 2,500; Texas2k: S = 500, 1,000, 2,500).
* Rebuilt by: `make_paper_assets.py` (`fig3_speedup_and_kmin.*`, `numbers.md`).

### Figure 4 and Section 6.3 (resilience to generator outages)

* Unserved-energy rates: `results/<system>/runs/{deterministic,full_S<S>,pisasr_S<S>}/evaluation.csv`
  (sum of `ens_mwh` over sum of `total_net_load_mwh`, times 100), also in
  `summary/evaluation_metrics.csv` (`unserved_energy_rate`, a fraction).
* Evaluation-set cost agreement ("within 1.5% at every S"): `expected_total_cost` of
  `pisasr_S<S>` against `full_S<S>` in `summary/evaluation_metrics.csv`.
* Paired confidence intervals: `results/<system>/summary/paired_cost_difference_S1000.csv`
  (`mean_difference_pct`, `ci_low_pct`, `ci_high_pct`; rows `total_cost` and `dispatch_cost`),
  computed from the two `evaluation.csv` files of `pisasr_S1000` and `full_S1000`. The interval
  is the two-sided 95% Student-t interval of the mean paired difference (PI-SASR minus full) over
  the 2,000 evaluation scenarios, divided by the full model's mean cost of the same kind:
  TX-123BT total cost +0.915% [0.650, 1.179]; Texas2k total cost +1.105% [0.896, 1.314];
  Texas2k dispatch cost +0.007% [-0.245, 0.259]. (A normal quantile instead of the Student-t
  quantile would give 0.897 and -0.244 for the Texas2k lower ends.)
* Solver runs: `evaluate.py --system <system> --runs all`.
* Rebuilt by: `summarize_evaluation.py`, `paired_cost_difference.py`, `make_paper_assets.py`
  (`fig4_unserved_energy.*`).

### Section 4.1 (low-rank embedding)

* Variance share of five modes: `results/<system>/experiments/embedding_spectrum_S1000.csv`
  (`cum_explained` at `rank` 5).
* Benchmark gap with medoids clustered in other spaces (TX-123BT): `results/tx123bt/experiments/embedding_space_S1000.csv`
  (`gap_pct` for `space` = `svd5`, `raw24`, `svd2`, `svd10`).
* Solver runs: `run_experiments.py --system <system> --stage embedding`.

### Section 4.2 (GP kernel choice)

* File: `results/<system>/experiments/gp_kernels_S1000.csv`, one row for each `kernel` and
  design-set `seed` (7, 17, 27). The paper quotes the mean over the three seeds of
  `spearman_allS` and of `top90_overlap` (times 100); Section 6.1 quotes the mean `r2_holdout`
  of `matern52_ard` (0.18).
* Solver runs: `run_experiments.py --system <system> --stage gp-kernels` (no MILP; the exact
  criticality comes from the merit-order dispatch).

### Section 4.4 and Section 6.1 (certified gap)

* Lower-bound batches: `results/<system>/experiments/lower_bound_batches.csv`; the time quoted
  in Section 4.4 is the sum of `time_s` over the five batches of size 200 (about 15 min on
  TX-123BT, 2 h on Texas2k).
* Certified gaps: `results/<system>/experiments/certified_gap.csv` (`lb_95`, `gap_cert95_pct`
  for the rows `case` = `pisasr` and `full`, `batch_size` = 200).
* Solver runs: `run_experiments.py --system <system> --stage lower-bound`. With
  `--batches-file results/<system>/experiments/lower_bound_batches.csv` no MILP is solved and
  only `certified_gap.csv` is recomputed from the stored batches (a few seconds;
  `make certified-gap`).

### Section 6.1 (other numbers)

* Benchmark gaps and active-set sizes at S = 1,000 and their largest values over S: as for
  Table 3.
* Range of the under-representation statistic, and "up to 19%" (the largest value of
  1/(1 + statistic) - 1 on Texas2k): `pisasr_S<S>/pisasr_log.csv`, `under_rep_gap`.
* GP coefficient of determination: `surrogate_r2` in `pisasr_S1000/solve_metrics.csv`
  (0.51 on TX-123BT, 0.999 on Texas2k).
* Difference between the stochastic and deterministic objectives (0.9% and 25%):
  `objective` in `full_S1000/solve_metrics.csv` and `deterministic/solve_metrics.csv`.

### Section 6.2 (other numbers)

* Speedups and run times at S = 100 and 2,500: as for Table 3.
* Five random draws of the design set: `results/tx123bt/experiments/design_set_replication_S1000.csv`
  (rows `kind` = `pisasr`: mean and sample standard deviation of `gap_pct` and `time_s`; one
  draw has `K` = 181 > 150 because it needed one enrichment round; rows `kind` = `full`: three
  separate timings of the full model, model construction and solver call). The load of the
  machine during these runs was not recorded (`docs/REPRODUCIBILITY.md`, Section 4). As the paper
  states, the full-model time of 525 ± 11 s includes model construction, whereas the TX-123BT
  full-model time of Tables 2 and 3 at S = 1,000 (434 s) covers the solver call only. Solver run:
  `run_replication.py`.
* Weighting at S = 2,500: `results/<system>/experiments/scenario_weights_S2500.csv`
  (`gap_pct`, `under_rep_pct`, `K` for `scheme` = `uniform` and `voronoi_nn`). Solver run:
  `run_experiments.py --system <system> --stage weights-s2500`.
* The uniform share of the critical scenarios (about 60%): `critical_mass` of the `uniform` row
  of `scenario_weights_S1000.csv`.

### Section 6.4 (scenario weights)

* File: `results/<system>/experiments/scenario_weights_S1000.csv`, one row for each `scheme`
  (`uniform`, `voronoi_nn` for nearest-neighbor redistribution, `cluster_mass`): `critical_mass`,
  `gap_pct`, `under_rep_pct`, `unit_hours_on`, `eval_ens_rate_pct`, and `eval_expected_cost`
  relative to the `uniform` row.
* Solver runs: `run_experiments.py --system <system> --stage scenario-weights`.

### Section 6.5 (merit-order verification)

* Files: `results/<system>/summary/merit_order_validation.csv`, one row for each of the first
  70 training scenarios, evaluated under the deterministic reference commitment. The paper
  quotes the squared correlation of `merit_order` and `lp`, the mean and largest absolute
  `rel_err_pct`, and the sign of `rel_err_pct` (all under-estimates). The LP of this check has no
  startup or shutdown ramp allowance, so its errors are upper bounds on the errors relative to
  the evaluation LP (`docs/MODEL.md`, Section 7).
* The squared correlation is the square of the Pearson correlation of the two columns
  (`numpy.corrcoef`): 0.99962 on TX-123BT and 0.99995 on Texas2k, printed in the paper as 0.9996
  and 0.9999.
* Solver runs: `validate_merit_order.py --system <system>` (one LP for each scenario: about
  12 s on TX-123BT and 46 s on Texas2k); `--check` compares the result with the file in
  `results/` (`make merit-order`).

### Section 7.2 (transmission constraints): not part of this repository

Section 7.2 summarizes a transmission-constrained extension of PI-SASR developed by the authors
in a companion study cited in the paper. Its code, data and results are not part of this
repository.

## 3. The solver runs behind the files

Every rerun writes to `outputs/<system>/...` (or `$PISASR_OUTPUT_DIR`) with the same folder
names as `results/`. By default the scripts that solve or evaluate models never write to
`results/`; they do so only when `--output-dir results` is given, as for the stored merit-order
file of TX-123BT (Section 5). Gurobi with a valid license is needed. Approximate times on the
workstation of the paper (Apple M2 Ultra, 24 cores, 192 GB) are given for orientation; they vary
with hardware and Gurobi version. `docs/REPRODUCIBILITY.md` (Section 6) breaks them down by step
and experiment.

| Step | Command (for each system) | Writes | Time |
|---|---|---|---|
| Deterministic and full models | `python scripts/run_full_model.py --system <system> --deterministic --S 100 200 500 1000 2500` | `runs/deterministic`, `runs/full_S<S>` | about 40 min (TX-123BT), 7 h (Texas2k) |
| PI-SASR | `python scripts/run_pisasr.py --system <system> --S 100 200 500 1000 2500` | `runs/pisasr_S<S>` | about 20 min (TX-123BT), 2 h (Texas2k), including the exact re-evaluation |
| Evaluation set | `python scripts/evaluate.py --system <system> --runs all` | `runs/<run>/evaluation.csv` | about 6 min (TX-123BT) or 25 min (Texas2k) for each of the 11 commitments |
| Experiments | `python scripts/run_experiments.py --system <system> --stage <experiment>` | `experiments/*.csv` | about 5 h (TX-123BT), 15 h (Texas2k) for all experiments |
| Design-set draws | `python scripts/run_replication.py` | `tx123bt/experiments/design_set_replication_S1000.csv` | about 50 min |
| Merit-order validation | `python scripts/validate_merit_order.py --system <system>` | `summary/merit_order_validation.csv` | 12 s (TX-123BT), 46 s (Texas2k) |
| Summaries and assets | Section 1, with `--results-dir outputs` | `summary/*.csv`, `paper_assets/` | seconds |

PI-SASR reads its reference commitment from `results/<system>/runs/deterministic/commitment.csv`
by default (`--reference outputs` uses the one written by `run_full_model.py --deterministic`).
The stored commitment is the MILP incumbent of the original run; a commitment obtained by solving
the deterministic UC again agrees with it only within the 0.1% MILP tolerance
(on TX-123BT it differs in 59 of 3,312 unit-hours), and PI-SASR is deterministic given its
reference commitment. The Texas2k full model at S = 2,500 does not fit in 192 GB of memory, so it
is written to a model file and solved from it (`run_full_model.py` does this by default; see its
`--file-based` option).

## 4. The CSV files in `results/`

Units: costs in dollars (\$); energy in MWh; times in seconds of wall-clock time; columns ending
in `_pct` are percentages; `unserved_energy_rate` and `critical_mass` are fractions. Empty cells
are values that do not apply to that row.

### 4.1 Run folders: `results/<system>/runs/<run>/`

`<run>` is `deterministic` (the deterministic, expected-value UC), `full_S<S>` (the full,
extensive-form two-stage stochastic UC over the first `S` scenarios of a fixed random permutation,
with random seed 42, of the training set; not the first `S` rows of the file) or `pisasr_S<S>`
(PI-SASR at scenario count `S`).

**`commitment.csv`**: the first-stage commitment, one row for each committable unit.

| Column | Meaning |
|---|---|
| `unit` | unit identifier, as in `data/<system>/generator_portfolio.csv` |
| `hour_00` ... `hour_23` | on/off status (1 or 0) in each hour |

**`solve_metrics.csv`** (deterministic and full runs): one row.

| Column | Meaning |
|---|---|
| `status`, `termination` | solver status and termination condition |
| `objective` | objective value (\$): expected total cost over the `S` training scenarios (the single expected-value profile for the deterministic model) |
| `solve_time_seconds` | wall-clock time (s): solver call only on TX-123BT; model construction and solver call on Texas2k; for Texas2k at S = 2,500, model construction, writing the model file and solver call |
| `model` | `deterministic` or `stochastic` |
| `n_scenarios` | number of scenarios in the model (1 for the deterministic model) |
| `gamma` | not used (empty) |
| `lean_solver_only_seconds` | Texas2k S = 2,500 only: solver call alone (s) |

A rerun also records `build_time_seconds` (model construction) and `solver_time_seconds`
(solver call), so that both timing definitions are available.

A rerun of `scripts/run_full_model.py` also writes three files that are not kept in `results/`:
`startup.csv` and `shutdown.csv` (the startup and shutdown variables of the model, in the layout
of `commitment.csv`) and `scenario_operating_summary.csv` (for each scenario of the model, the
total dispatch `total_dispatch_mwh` and unserved energy `total_ens_mwh`, in MWh).

**`solve_metrics.csv`** (PI-SASR runs): one row.

| Column | Meaning |
|---|---|
| `method`, `problem` | `PI-SASR` and `stochastic_<S>` |
| `objective` | exact expected cost (\$) of the PI-SASR commitment over the `S` training scenarios: fixed commitment cost plus the mean exact ramp-coupled LP recourse |
| `opt_gap_pct_vs_full` | benchmark gap (%): `objective` relative to `full_obj` |
| `full_obj` | objective of the full model at the same `S` (\$) |
| `approx_objective` | verified objective (\$): fixed cost plus the mean merit-order recourse over the `S` scenarios |
| `approx_gap_pct` | `approx_objective` relative to `full_obj` (%) |
| `solve_time_seconds` | online time (s), from data loading to the end of the merit-order verification; excludes the deterministic UC solution behind the reference commitment and the exact re-evaluation |
| `exact_validation_s`, `t_exact_cert_s` | time of the exact re-evaluation behind `objective` (s) |
| `n_scenarios_used` | active-set size `K` |
| `n_scenarios_full` | scenario count `S` |
| `n_fixed_baseload_units` | number of units in the baseload pre-commitment |
| `surrogate_r2`, `surrogate_mae` | coefficient of determination and mean absolute error (\$) of the GP surrogate on the quarter of the design set kept aside for validation |
| `t_design_eval_s`, `t_gp_fit_s`, `t_milp_solve_s` | time of the design-set dispatches, of the GP fit, and of the reduced MILP (s) |

A rerun also records `random_seed`, `reference_source` and `full_objective_source`.

**`pisasr_log.csv`** (PI-SASR runs): one row for each solution of the reduced UC (a single row in
every run of the paper, since no enrichment round was needed).

| Column | Meaning |
|---|---|
| `round` | 0 for the first reduced UC, 1 after one enrichment round |
| `n_scenarios` | active-set size `K` in that round |
| `approx_true_obj` | verified objective of that round (\$) |
| `under_rep_gap` | under-representation statistic (a fraction): verified objective relative to the reduced objective |
| `elapsed_s` | time since the start of the run (s) |

**`evaluation.csv`**: exact ramp-coupled LP dispatch of the commitment on each of the 2,000
evaluation scenarios (the first 2,000 rows of `data/<system>/net_load_scenarios_test.csv`),
under the same generator-outage draw for every commitment.

| Column | Meaning |
|---|---|
| `scenario` | scenario identifier |
| `dispatch_cost` | recourse cost (\$): generation cost plus 10,000 \$/MWh of unserved energy plus 1 \$/MWh of spilled energy |
| `ens_mwh` | unserved energy (MWh) |
| `spill_mwh` | spilled surplus (MWh) |
| `ens_hours` | hours with unserved energy above 1e-4 MWh |
| `total_net_load_mwh` | net load of the scenario over 24 hours (MWh) |
| `total_cost_with_fixed` | fixed commitment cost plus `dispatch_cost` (\$) |

### 4.2 Summaries: `results/<system>/summary/`

**`evaluation_metrics.csv`** (written by `scripts/summarize_evaluation.py`): one row for each run.

| Column | Meaning |
|---|---|
| `run`, `model`, `S` | run folder, model (`deterministic`, `full`, `pisasr`) and scenario count (1 for the deterministic model) |
| `fixed_commitment_cost` | no-load, startup and shutdown cost of the commitment (\$) |
| `expected_dispatch_cost`, `expected_total_cost` | means of `dispatch_cost` and `total_cost_with_fixed` over the evaluation set (\$) |
| `p95_total_cost`, `worst_total_cost` | 0.95 quantile (linear interpolation) and maximum of `total_cost_with_fixed` (\$) |
| `expected_ens_mwh`, `total_ens_mwh` | mean and sum of `ens_mwh` (MWh) |
| `ens_probability` | share of evaluation scenarios with unserved energy above 1e-4 MWh |
| `expected_ens_hours` | mean of `ens_hours` |
| `total_spill_mwh` | sum of `spill_mwh` (MWh) |
| `unserved_energy_rate` | sum of `ens_mwh` over sum of `total_net_load_mwh` (a fraction; the paper prints it in percent) |
| `n_evaluation_scenarios` | number of evaluation scenarios (2,000) |
| `no_load_cost`, `startup_cost`, `shutdown_cost` | parts of `fixed_commitment_cost` (\$), recomputed from `commitment.csv` and the generator portfolio |
| `startup_count`, `committed_unit_hours` | startups (counted against each unit's initial status) and committed unit-hours |
| `normalized_expected_cost`, `normalized_worst_cost` | `expected_total_cost` and `worst_total_cost` relative to the deterministic model |

The values agree to within 5e-13 (relative) with the summaries written by the original
evaluation code. The original files also recorded the wall-clock time of each evaluation, which
cannot be recomputed from `evaluation.csv` and is not kept (about 1,500 s for a Texas2k
commitment in the original files; 338 s for a TX-123BT commitment in a rerun).

**`evaluation_summary.csv`** (in each run folder of a rerun; written by `scripts/evaluate.py`, not
kept in `results/`): one row with the columns of `evaluation_metrics.csv` except the two
normalized costs, followed by `evaluation_time_seconds`, the wall-clock time of the evaluation
(s).

**`paired_cost_difference_S1000.csv`** (written by `scripts/paired_cost_difference.py`): rows
`total_cost` (column `total_cost_with_fixed`) and `dispatch_cost`.

| Column | Meaning |
|---|---|
| `system`, `S`, `quantity`, `run_a`, `run_b` | the comparison: `run_a` minus `run_b` (`pisasr_S1000` minus `full_S1000`) |
| `n_pairs` | evaluation scenarios common to both runs (2,000) |
| `confidence`, `distribution` | 0.95 and `t` (two-sided Student-t interval with `n_pairs` - 1 degrees of freedom) |
| `mean_a`, `mean_b` | mean cost of each run (\$) |
| `mean_difference`, `sd_difference`, `standard_error` | mean, sample standard deviation and standard error of the paired differences (\$) |
| `quantile` | Student-t quantile used for the interval |
| `ci_low`, `ci_high` | interval ends (\$) |
| `mean_difference_pct`, `ci_low_pct`, `ci_high_pct` | the same, as a percentage of `mean_b` |

**`scaling.csv`**: the scaling results of Table 3, one row for each `S`, as collected in the
original runs.

| Column | Meaning |
|---|---|
| `S` | scenario count |
| `full_obj`, `full_time_s` | full-model `objective` (\$) and `solve_time_seconds` (s) |
| `pisasr_obj`, `pisasr_gap_pct`, `pisasr_time_s` | PI-SASR `objective` (\$), benchmark gap (%) and online time (s) |
| `pisasr_exact_valid_s` | TX-123BT only: time of the exact re-evaluation (s) |
| `speedup_x` | `full_time_s` / `pisasr_time_s` |
| `K_used`, `fixed_units` | active-set size and number of pre-committed baseload units |
| `surrogate_r2` | GP coefficient of determination |
| `full_oos_cost`, `pisasr_oos_cost` | TX-123BT only: evaluation-set expected total cost (\$); empty for the full model at S = 2,500 |
| `full_oos_ens`, `pisasr_oos_ens` | TX-123BT only: total unserved energy on the evaluation set (MWh) |
| `full_oos_unserved`, `pisasr_oos_unserved` | TX-123BT only: evaluation-set unserved-energy rate (a fraction) |

`summarize_evaluation.py` checks the evaluation columns against the run folders; the full model
at S = 2,500 is covered by `runs/full_S2500/evaluation.csv`.

**`merit_order_validation.csv`**: merit-order recourse against the exact LP, one row for each of
the first 70 training scenarios under the deterministic reference commitment.

| Column | Meaning |
|---|---|
| `scenario` | scenario identifier |
| `merit_order` | merit-order recourse cost (\$) |
| `lp` | recourse cost (\$) of the single-scenario dispatch LP, whose ramp limits have no startup or shutdown allowance (`docs/MODEL.md`, Section 7) |
| `rel_err_pct` | (`merit_order` - `lp`) / `lp`, in percent (negative: under-estimate) |

### 4.3 Experiments: `results/<system>/experiments/`

All experiments use S = 1,000 unless the file name says otherwise, the PI-SASR settings of the
paper, and the reference commitment of `runs/deterministic`. `gap_pct` is always the benchmark
gap (%) of the exact expected cost over the `S` training scenarios against the full model's
objective; `under_rep_pct` is the under-representation statistic in percent; `reduced_obj`,
`approx_obj` and `objective` are the reduced MILP objective, the verified (merit-order) objective
and the exact expected cost (\$). `component_timing_S1000.csv` was run on an otherwise idle
workstation, and the Texas2k budget sensitivity at S = 2,500 and the two
`scenario_weights_S2500.csv` files after all other experiments had finished. The times in the
other files were recorded while other experiments shared the machine, except those of
`design_set_replication_S1000.csv`, for which the load was not recorded
(`docs/REPRODUCIBILITY.md`, Section 4). The paper uses the gaps, costs and sizes of these files,
plus two groups of their times: the lower-bound batch times (Section 4.4, stated as approximate)
and the design-set draw times (Section 6.2).

**`component_timing_S1000.csv`** (TX-123BT; Table 4): one row for each `variant` at equal
active-set size: `pisasr`, `pisasr_nofix` (no baseload pre-commitment), `gp_crit_K148` and
`gp_crit_K188` (the top of the GP ranking only), `exact_crit_K148` (the top of the exact
merit-order ranking only), `exact_crit90_med60` (exact ranking with coverage medoids),
`medoids_K148` (coverage medoids only), `ffs_K148` (fast-forward selection on the net-load
profiles) and `costffs_K148` (forward selection in recourse-cost space).

| Column | Meaning |
|---|---|
| `K` | active-set size |
| `t_features_s`, `t_select_s`, `t_milp_s`, `t_verify_s` | time of the features, of the selection (including design-set dispatches and GP fit, or the exact ranking), of the reduced MILP and of the merit-order verification (s) |
| `t_online_s` | sum of the four (s), the online time of Table 4 |
| `t_exact_eval_s` | time of the exact re-evaluation behind `gap_pct` (s; not part of the online time) |
| `reduced_obj`, `approx_obj`, `under_rep_pct`, `objective`, `gap_pct` | see above |
| `r2_holdout` | GP coefficient of determination (GP variants only) |

**`criticality_only_S1000.csv`** (Section 7.3): one row for each `variant` (TX-123BT: `pisasr`,
`gp_crit_K148`, `exact_crit_K148`, `medoids_K148`, `gp_crit_K188`; Texas2k: the first three).

| Column | Meaning |
|---|---|
| `K`, `gap_pct`, `under_rep_pct` | see above |
| `t_milp_s` | reduced MILP time (s) |
| `unit_hours_on` | committed unit-hours of the resulting commitment |
| `eval_expected_cost`, `eval_cost_sd` | mean and standard deviation of the total cost on the 2,000 evaluation scenarios (\$) |
| `eval_ens_rate_pct`, `eval_ens_mwh` | evaluation-set unserved-energy rate (%) and total unserved energy (MWh) |
| `eval_n` | number of evaluation scenarios |

**`gp_kernels_S1000.csv`** (Section 4.2): one row for each `kernel` (`matern52_ard`,
`matern32_ard`, `matern12_ard`, `rbf_ard`, `matern52_iso`) and design-set `seed` (7, 17, 27).

| Column | Meaning |
|---|---|
| `r2_holdout` | coefficient of determination on the quarter of the design set kept aside |
| `spearman_allS` | Spearman rank correlation of the GP prediction with the exact criticality of all `S` scenarios |
| `top90_overlap`, `top148_overlap` | share of the exact top 90 (top 148) recovered by the GP ranking |
| `t_fit_s` | time of the GP fit and prediction (s) |

**`embedding_spectrum_S1000.csv`** (Section 4.1): one row for each `rank` of the centered
`S` × 24 net-load matrix: `explained` (share of variance of that singular direction) and
`cum_explained` (cumulative share).

**`embedding_space_S1000.csv`** (TX-123BT; Section 4.1): one row for each clustering `space` of
the coverage medoids (`svd5` default, `raw24` raw profiles, `svd2`, `svd10`): `K`, `t_kmeans_s`
(s), `t_milp_s` (s), `under_rep_pct`, `objective`, `gap_pct`.

**`scenario_weights_S1000.csv`** and **`scenario_weights_S2500.csv`** (Sections 6.4 and 6.2): one
row for each `scheme` (`uniform`, `voronoi_nn` for nearest-neighbor redistribution,
`cluster_mass`) on the active set of the PI-SASR run at that `S`. The S = 2,500 file has no
evaluation-set columns (`eval_n` = 0).

| Column | Meaning |
|---|---|
| `K` | active-set size |
| `w_max`, `w_min` | largest and smallest scenario weight |
| `critical_mass` | total weight of the 90 critical scenarios (a fraction) |
| `reduced_obj`, `approx_obj`, `under_rep_pct`, `objective`, `gap_pct` | see above |
| `t_milp_s` | reduced MILP time (s) |
| `unit_hours_on` | committed unit-hours |
| `eval_expected_cost`, `eval_cost_sd`, `eval_ens_rate_pct`, `eval_ens_mwh`, `eval_n` | as in `criticality_only_S1000.csv` |

**`lower_bound_batches.csv`** (Sections 4.4 and 6.1): one row for each batch `m` (0 to 4) of
size `batch_size` (S' = 200; TX-123BT also 500), drawn without overlap from the training set.

| Column | Meaning |
|---|---|
| `objective` | MILP incumbent of the batch problem (\$) |
| `time_s` | model construction and solution time (s) |
| `termination` | solver termination condition |

**`certified_gap.csv`** (Section 6.1): one row for each `case` (`pisasr`, `full`) and
`batch_size`.

| Column | Meaning |
|---|---|
| `S_ref` | scenario count of the two commitments (1,000) |
| `M` | number of batches |
| `lb_mean`, `lb_sd` | mean and standard deviation of the batch incumbents, each reduced by the 0.1% MILP tolerance (\$) |
| `lb_95` | one-sided 97.5% Student-t lower confidence bound on the optimal expected cost (\$) |
| `ub_mean`, `ub_sd`, `ub_95` | mean, standard deviation and one-sided 97.5% upper confidence bound of the commitment's total cost on the evaluation set (\$) |
| `n_eval` | evaluation scenarios in the upper bound (2,000) |
| `insample_obj` | the commitment's objective over the training scenarios (\$) |
| `gap_point_pct` | (`ub_mean` - `lb_mean`) / `lb_mean` (%) |
| `gap_cert95_pct` | certified gap (`ub_95` - `lb_95`) / `lb_95` (%), valid with approximately 95% confidence |
| `gap_insample_vs_lb95_pct` | (`insample_obj` - `lb_95`) / `lb_95` (%) |

**`budget_sensitivity.csv`** (Figure 3(b), Section 6.2): one row for each `S` and budget
(`k_critical`, `k_cover`) = (15, 10) (TX-123BT only), (30, 20), (45, 30), (60, 40), (90, 60),
(120, 80).

| Column | Meaning |
|---|---|
| `K` | active-set size (critical and coverage scenarios can coincide, so `K` can be below `k_critical` + `k_cover`) |
| `objective`, `gap_pct`, `under_rep_pct` | see above |
| `t_milp_s`, `t_online_s`, `t_exact_eval_s` | reduced MILP, online and exact re-evaluation times (s) |

**`design_set_replication_S1000.csv`** (TX-123BT; Section 6.2): rows `kind` = `pisasr` for
five random seeds of the design set (7, 17, 27, 37, 47) and `kind` = `full` for three separate
solutions of the full model at S = 1,000.

| Column | Meaning |
|---|---|
| `seed` | random seed of the design set (for `full`, the repetition number) |
| `gap_pct` | benchmark gap (%; 0 for the full model) |
| `time_s` | PI-SASR online time, or full-model construction and solution time (s) |
| `K` | active-set size (1,000 for the full model) |

## 5. Differences between this repository and the paper

`paper_assets/numbers.md` compares every number of the results with the value computed from the
CSV files. With the files in this repository every computed number agrees with the printed
one at the printed precision. The TX-123BT merit-order validation of Section 6.5 was written by
`scripts/validate_merit_order.py --system tx123bt --output-dir results`; if that file is
missing, its five numbers are shown as "not yet generated".
