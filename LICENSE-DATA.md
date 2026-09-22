# Data license and third-party notices

The source code of this repository (`src/`, `scripts/`, `tests/` and the build files) and its
documentation (`README.md`, `docs/` and `data/README.md`) are licensed under the MIT License (see
[`LICENSE`](LICENSE)). This file covers the data, the results and the paper assets.

## 1. Processed data and results produced by the authors

The following files were produced by the authors and are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
Copyright (c) 2026 Mert Korkali and Davida Andriniaina:

* `data/tx123bt/` and `data/texas2k/`: the generator portfolios, the net-load scenario sets and the
  Gaussian-copula parameters of TX-123BT;
* `results/`: every numerical result of the paper (commitments, solver metrics, evaluation-set
  costs, experiment tables and summaries);
* `paper_assets/`: the tables, figures and list of numbers regenerated from `results/`.

To attribute these files, cite the paper (see [`CITATION.cff`](CITATION.cff) or the README) and
link to this repository. The processed data are derived from the two third-party sources below;
CC BY 4.0 applies only to the authors' contribution, and the notices below apply in addition.
CC BY 4.0 does not permit additional restrictions on the TX-123BT-derived files, and no
restriction beyond the terms below is added to any file.

The raw third-party files are not part of this repository. `data/raw/` is excluded from version
control, and `data/README.md` explains how to obtain the raw files and rebuild the processed data.

## 2. TX-123BT (University of Houston)

The files in `data/tx123bt/` are derived from the Texas Synthetic Power System Test Case
(TX-123BT), version 6, by J. Lu and X. Li (University of Houston), which is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

* Dataset: J. Lu and X. Li, "Texas Synthetic Power System Test Case (TX-123BT).zip," figshare,
  dataset, version 6, 2023, doi:[10.6084/m9.figshare.22144616.v6](https://doi.org/10.6084/m9.figshare.22144616.v6).
* Changes made by the authors: the generator parameters of the 138 natural-gas, coal, hydro and
  nuclear units were reformatted (ramp rates converted to MW/h; minimum up and down times and
  initial status assigned by fuel type); system-level load, wind and solar totals were aggregated
  from the five-year profiles; the net-load scenarios are synthetic Gaussian-copula samples fitted
  to the June to August data of 2017 to 2021.
* The original authors provide the data without warranty.
* Users of TX-123BT are asked to cite:

  J. Lu, X. Li, H. Li, T. Chegini, C. Gamarra, Y. C. E. Yang, M. Cook and G. Dillingham,
  "A synthetic Texas power system with time-series weather-dependent spatiotemporal profiles,"
  *Sustainable Energy, Grids and Networks*, vol. 43, art. no. 101774, Sep. 2025,
  doi:[10.1016/j.segan.2025.101774](https://doi.org/10.1016/j.segan.2025.101774).

## 3. Texas A&M Electric Grid Test Case Repository (Texas2k Series25 and ACTIVSg2000)

The files in `data/texas2k/` are derived from the Texas A&M University Electric Grid Test Case
Repository:

* Texas2k Series25, "Case 1: 2025 Summer Peak" (`Texas2k_series25_case1_summerpeak.m`),
  <https://electricgrids.engr.tamu.edu/texas2k-series25/>;
* the ACTIVSg2000 2016 load and renewable time series (`ACTIVISg2000_load_time_series_MW.csv`,
  `ACTIVISg2000_renewable_time_series_MW.csv`),
  <https://electricgrids.engr.tamu.edu/activsg-time-series-data/>.

These data are provided by Texas A&M University researchers free for commercial or
non-commercial use. They are entirely synthetic, do not represent any actual grid, and contain no
Critical Energy/Electric Infrastructure Information (CEII). The source files carry no license file
and no explicit permission to redistribute them, and the Texas A&M repository asks each user to
request the files through a download form. **The raw Texas A&M files are therefore not redistributed here.**
Download them from the pages above (see `data/README.md` for the expected file names and SHA-256
checksums).

Changes made by the authors: the 568 committable units (natural gas, coal, nuclear, hydro, diesel
and wood) were extracted from the case, their variable costs were derived from the case's cost
polynomials and rescaled to a target mean for each fuel class, and startup costs, no-load
costs, ramp limits and minimum up and down times were assigned by fuel class; the net-load scenarios
are synthetic samples of a shrinkage-regularized Gaussian copula fitted to the 92 summer days of the
2016 system totals and rescaled to the Series25 peak load and renewable capacities.

Users of these data are asked to cite:

* A. B. Birchfield, T. Xu, K. M. Gegner, K. S. Shetye and T. J. Overbye, "Grid structural
  characteristics as validation criteria for synthetic networks," *IEEE Transactions on Power
  Systems*, vol. 32, no. 4, pp. 3258–3265, Jul. 2017,
  doi:[10.1109/TPWRS.2016.2616385](https://doi.org/10.1109/TPWRS.2016.2616385).
* H. Li, J. H. Yeo, A. L. Bornsheuer and T. J. Overbye, "The creation and validation of load time
  series for synthetic electric power systems," *IEEE Transactions on Power Systems*, vol. 36,
  no. 2, pp. 961–969, Mar. 2021,
  doi:[10.1109/TPWRS.2020.3018936](https://doi.org/10.1109/TPWRS.2020.3018936).
* Texas A&M University Electric Grid Test Case Repository, "Texas2k Series25 case," 2025,
  <https://electricgrids.engr.tamu.edu/texas2k-series25/>.

The Texas2k Series25 page also asks users of its dynamic models to cite J. Baek and
A. B. Birchfield, "A tuning method for exciters and governors in realistic synthetic grids with
dynamics," Proc. 2023 North American Power Symposium (NAPS), pp. 1–6,
doi:[10.1109/NAPS58826.2023.10318771](https://doi.org/10.1109/NAPS58826.2023.10318771). This
work does not use the dynamic models.

The authors' CC BY 4.0 license of Section 1 applies only to their own contribution to the
Texas2k-derived files (the transformations and the synthetic samples); the Texas A&M terms above
continue to apply to the underlying data.
