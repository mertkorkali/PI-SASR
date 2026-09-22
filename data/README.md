# Processed input data

This folder holds the processed inputs of the two test systems of the paper: the generator
portfolios (the committable units) and the net-load scenario sets. Every result in `results/`
was computed from exactly these files. They are derived from third-party data; see
[Sources, licenses and citations](#sources-licenses-and-citations) and
[`../LICENSE-DATA.md`](../LICENSE-DATA.md).

## Files

| File | Rows × columns | Size (bytes) | Contents |
|---|---|---:|---|
| `tx123bt/generator_portfolio.csv` | 138 × 14 | 10,682 | committable units of TX-123BT (natural gas, coal, hydro, nuclear; 76.4 GW) |
| `tx123bt/net_load_scenarios_train.csv` | 2,500 × 25 (identifier and 24 hours) | 1,123,094 | training set (the full model at S uses the first S entries of a fixed random permutation, random seed 42, of its rows, not the first S rows of the file) |
| `tx123bt/net_load_scenarios_test.csv` | 5,000 × 25 (identifier and 24 hours) | 2,245,509 | test file; its first 2,000 rows form the evaluation set |
| `tx123bt/copula_parameters/` | 8 files | 1,444,669 | Gaussian-copula parameters and the normalized deviations they were fitted to |
| `texas2k/generator_portfolio.csv` | 568 × 14 | 36,098 | committable units of Texas2k Series25 (natural gas, coal, nuclear, hydro, diesel, wood; 77.8 GW) |
| `texas2k/net_load_scenarios_train.csv` | 2,500 × 25 (identifier and 24 hours) | 1,104,393 | training set |
| `texas2k/net_load_scenarios_test.csv` | 5,000 × 25 (identifier and 24 hours) | 2,210,336 | test file; its first 2,000 rows form the evaluation set |
| `SHA256SUMS` | 14 lines | | SHA-256 checksums of the 14 CSV files above |
| `raw/` | | | raw third-party files for the rebuild (not tracked by git; see below) |

The training set and the test file of each system are disjoint and together form a set of
7,500 scenarios.

### Generator portfolios

One row for each committable unit. The row order fixes the order of the units in the UC model and
in the outage draw, so it must not be changed.

| Column | Meaning | Unit |
|---|---|---|
| `unit` | unit identifier (`G<n>` on TX-123BT, the generator number in the source data; `T<n>` on Texas2k, the row of the generator in the case file) | |
| `bus` | bus of the unit in the source network (not used by the model) | |
| `fuel_type` | fuel type as named in the source data | |
| `p_min_mw`, `p_max_mw` | minimum and maximum output when committed | MW |
| `variable_cost` | marginal cost | \$/MWh |
| `startup_cost`, `shutdown_cost` | cost of one startup and one shutdown (shutdown cost 0 for every unit) | \$ |
| `no_load_cost` | cost of each committed hour | \$/h |
| `ramp_up_mw`, `ramp_down_mw` | ramp limits | MW/h |
| `min_up_hours`, `min_down_hours` | minimum up and down times | h |
| `initial_status` | status before the first hour (1 on, 0 off) | |

### Scenario files

One row for each scenario. The first column holds the scenario identifier (`scenario_0001` to
`scenario_7500`, the position in the set of 7,500 scenarios; the column is named `scenario` in the
TX-123BT files and has an empty name in the Texas2k files), followed by `hour_00` to `hour_23`, the
system net load (load minus wind minus solar) in MW. The identifiers seed the generator-outage draw
of each scenario and the row order defines the nested scenario subsets and the evaluation set, so
the files must be used byte for byte as included here.

### TX-123BT copula parameters (`tx123bt/copula_parameters/`)

These files are intermediate results of the TX-123BT scenario construction, included so that the
construction can be inspected; the experiments do not read them.

| File | Contents |
|---|---|
| `summer_normalized_deviations_2017_2021.csv` | hourly system totals of load, wind, solar and net load (MW) for the 460 summer days (June to August) of 2017 to 2021, and the normalized deviations `(x - hourly median) / hourly interquartile range` of load, wind and solar (11,040 rows) |
| `gaussian_copula_pairwise_parameters.csv` | for each block of the day and each pair of variables: sample count, Kendall tau, Gaussian-copula correlation rho = sin(pi tau / 2) and p-value (13 rows) |
| `P1_Overnight_00_05_...csv` to `P5_Evening_20_23_...csv` | Gaussian-copula correlation matrix of each block of the day (00-05, 06-09, 10-14, 15-19 and 20-23 h; solar is left out overnight) |
| `overall_summer_peak_net_load_day_profile.csv` | the 24 hours of 2021-08-30, the day with the largest summer net load (63,561 MW), used as the baseline profile |

## How the scenarios were built

### TX-123BT (`scripts/data/prepare_tx123bt.py`)

1. *Generator portfolio.* The portfolio consists of the 138 units with fuel type natural gas
   (113), coal (13), hydro (10) or nuclear (2) of `Generator_data.xlsx` (sheet "Gen data").
   Variable cost = C1, no-load cost = C0
   (applied to each committed hour), startup cost = Csu, shutdown cost = 0, ramp limits = 60 times
   the ramp rate in MW/min. The source gives no minimum up and down times or initial status; they
   are set by fuel type: 8 h for nuclear, 4 h for coal, 2 h for natural gas and 1 h for hydro, with
   nuclear and coal units on at the start of the day. The 82 wind and 72 solar units enter only
   through the net load.
2. *Hourly totals and copula parameters.* Load, wind and solar are summed over buses and plants
   for every hour of 2017 to 2021 (1,826 days). For the 460 summer days each variable is
   normalized for each hour as (x - hourly median) / hourly interquartile range, and for each of five
   blocks of the day the Kendall tau of every pair of variables is converted to a Gaussian-copula
   correlation.
3. *Scenarios.* For each hour, a draw from the Gaussian copula of that hour's block (random seed 42)
   is mapped through the block's empirical quantiles of the normalized deviations, scaled by the
   hour's interquartile range and added to the profile of the peak day 2021-08-30. Draws are
   independent between hours, so the scenarios carry correlation between load, wind and solar
   within an hour but no correlation between hours. Load is clipped at zero, and wind and solar at
   zero and at their largest observed summer value. Net load = load - wind - solar. The set has
   7,500 scenarios; the largest hourly net load is 77,162 MW.
4. *Split.* A random permutation (random seed 7) of the set gives the 2,500 training scenarios
   (its first 2,500 entries) and the 5,000 test scenarios (the next 5,000).

### Texas2k Series25 (`scripts/data/prepare_texas2k.py`)

1. *Generator portfolio.* The case "Case 1: 2025 Summer Peak" has 1,099 generators. The 568 with
   fuel type natural gas (`ng`, 507), coal (21), nuclear (4), hydro (22), diesel (`dfo`, 10) or wood
   (4) and positive maximum output are committable; the 400 wind and solar units (74.1 GW in the
   case) are represented by system-level profiles in the net load, and the 131 units of fuel type
   "other" are excluded from both the commitment and the net load.
   * Variable cost: the average incremental cost of the unit's cost polynomial over
     [max(P_min, 0.001), P_max] (the marginal cost at P_max when the range is at most 1 MW), or the
     fuel-class default when this is at most \$1/MWh (all hydro and wood units). Because the cost
     coefficients of the case are intended for power-flow studies, each fuel class is then
     rescaled to a target mean (nuclear 17.44, coal 22.0, natural gas 28.0, hydro 12.3, diesel 95.0
     and wood 35.0 \$/MWh), keeping the relative differences within the class and clipping to 0.6 to
     1.8 times the target.
   * A minimum output that is not below the maximum (18 units: 17 natural gas, 1 hydro) is replaced
     by 10% of the maximum (25% for coal and nuclear).
   * Fuel-class defaults: startup cost for each MW of capacity (nuclear 60, coal 40, natural gas
     15, hydro 2, diesel 10, wood 25 \$/MW), no-load cost (0, 800, 500, 0, 100, 200 \$/h), minimum up
     and down times (8, 6, 2, 1, 1, 4 h) and ramp limits as a multiple of P_max in one hour (0.3,
     0.5, 1.0, 3.0, 2.0, 0.5); coal and nuclear units are on at the start of the day.
2. *Scenarios.* Series25 has no multi-year weather profiles, so the 2016 hourly system totals of
   load, wind and solar of its ACTIVSg2000 predecessor are used. The 92 summer days (June to August
   2016) form 72-dimensional daily samples (24 hours of load, wind and solar). A Gaussian copula is
   fitted to their normal scores, with the correlation matrix shrunk toward the identity (weight
   0.2), and sampled 7,500 times (random seed 42); nine constant night-time solar dimensions are
   carried over unchanged, and samples are mapped back through interpolated empirical quantiles.
   Load is scaled by 85,758.89 / 66,275.70 (the Series25 summer peak over the largest 2016 summer
   load), wind by 41,837 / 9,587 and solar by 32,266 / 651 (Series25 over ACTIVSg2000 installed
   capacities, in MW), and the net load is floored at 1,000 MW.
3. *Split.* As for TX-123BT: a random permutation (random seed 7) gives 2,500 training and 5,000
   test scenarios.

The mean daily peak net load of the set is 56,198 MW and its largest value 76,762 MW.
`docs/REPRODUCIBILITY.md` lists construction details that the paper does not state (the share of
scenario-hours in which scaled wind exceeds the installed wind capacity, the share of values raised
to the floor, and the hard-coded capacities).

## Sources, licenses and citations

| System | Source | License of the source |
|---|---|---|
| TX-123BT | J. Lu and X. Li, "Texas Synthetic Power System Test Case (TX-123BT).zip," figshare, dataset, version 6, 2023, doi:[10.6084/m9.figshare.22144616.v6](https://doi.org/10.6084/m9.figshare.22144616.v6) (University of Houston) | CC BY 4.0 |
| Texas2k Series25 | Texas A&M University Electric Grid Test Case Repository, Texas2k Series25 <https://electricgrids.engr.tamu.edu/texas2k-series25/> and ACTIVSg2000 time series <https://electricgrids.engr.tamu.edu/activsg-time-series-data/> | free for commercial or non-commercial use; no license for redistribution |

The processed files in this folder are released by the authors under CC BY 4.0, subject to the
third-party notices in [`../LICENSE-DATA.md`](../LICENSE-DATA.md). If you use them, please cite
the paper of this repository and the sources:

* TX-123BT: J. Lu, X. Li, H. Li, T. Chegini, C. Gamarra, Y. C. E. Yang, M. Cook and
  G. Dillingham, "A synthetic Texas power system with time-series weather-dependent
  spatiotemporal profiles," *Sustainable Energy, Grids and Networks*, vol. 43, art. no. 101774,
  Sep. 2025, doi:10.1016/j.segan.2025.101774.
* Texas2k: A. B. Birchfield, T. Xu, K. M. Gegner, K. S. Shetye and T. J. Overbye, "Grid structural
  characteristics as validation criteria for synthetic networks," *IEEE Transactions on Power
  Systems*, vol. 32, no. 4, pp. 3258–3265, Jul. 2017, doi:10.1109/TPWRS.2016.2616385; H. Li, J. H. Yeo,
  A. L. Bornsheuer and T. J. Overbye, "The creation and validation of load time series for
  synthetic electric power systems," *IEEE Transactions on Power Systems*, vol. 36, no. 2,
  pp. 961–969, Mar. 2021, doi:10.1109/TPWRS.2020.3018936; and the Texas2k Series25 web page above.

## Rebuilding the processed files from the raw sources

The raw files go to `data/raw/` (ignored by git), or to the folder named by the environment
variable `PISASR_RAW_DIR`. The prepare scripts write to `outputs/data/<system>/` (with the
intermediate set of 7,500 scenarios in `intermediate/`) and never overwrite the included files
unless `--out-dir data/<system>` is given; `--check` compares the result with `SHA256SUMS`. All
commands are run from the repository root. The sizes, file counts and checksums of the raw files
given below, and the counts of generators and units in the raw sources given above, were measured
on the raw files used for the paper.

### TX-123BT

The download script fetches the figshare archive (version 6,
<https://ndownloader.figshare.com/files/44942761>, 544,473,438 bytes, MD5
`60db05d9f7f7699380c8bd023b38ba43`), checks its MD5 checksum and extracts `Generator_data.xlsx`,
`Readme.txt` and the folders `Load_5y`, `Wind_5y` and `Solar_5y` (5,480 files, about 80 MB) to
`data/raw/TX-123BT/`:

```bash
python scripts/data/download_tx123bt.py
```

With an archive downloaded by hand from the figshare page:

```bash
python scripts/data/download_tx123bt.py --zip "Texas Synthetic Power System Test Case (TX-123BT) 5year profiles.zip"
```

Then rebuild and compare (23 s on the reference machine):

```bash
python scripts/data/prepare_tx123bt.py --check
```

### Texas2k Series25

The three Texas A&M files are not redistributed. Request them from the Texas2k Series25 page
(the case) and the ACTIVSg time-series page (the two time series), and place them in
`data/raw/texas2k/`, or pass their locations with `--case-file`, `--load-file` and
`--renewable-file`:

| File | Source | Size (bytes) | SHA-256 |
|---|---|---:|---|
| `Texas2k_series25_case1_summerpeak.m` | Texas2k Series25, "Case 1: 2025 Summer Peak", MATPOWER format | 1,079,249 | `ac4184c07fd2abe902a07937af8cc400d7c388defd6440575d454a57def36ea0` |
| `ACTIVISg2000_load_time_series_MW.csv` | ACTIVSg2000 time series, 2016 load | 106,703,681 | `18df01ba3ef990a3b5bad7d0ef7bb0b16b079173c6ac04dd240e7d90c631d02e` |
| `ACTIVISg2000_renewable_time_series_MW.csv` | ACTIVSg2000 time series, 2016 wind and solar | 8,826,625 | `36660442bd1803536519249d06dde28e801156a105f279fec96078bce1941bbf` |

(The time-series files are spelled "ACTIVISg" by the source.) Then rebuild and compare (2 s):

```bash
python scripts/data/prepare_texas2k.py --check
```

On the reference machine (macOS arm64, conda-forge NumPy 2.4.6) both rebuilds reproduce all 14
files byte for byte. Other BLAS/LAPACK libraries can change the last digits of the scenarios; in
that case compare with a numerical tolerance and use the included files to reproduce the paper.

## Checksums

`SHA256SUMS` lists the SHA-256 digest of every CSV file in this folder (paths relative to
`data/`). To check the included files:

```bash
python -c "from pisasr.paths import verify_checksums; verify_checksums()"
```
