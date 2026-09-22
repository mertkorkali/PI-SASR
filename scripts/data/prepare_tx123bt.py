#!/usr/bin/env python
"""Regenerate the TX-123BT inputs from the raw five-year profiles.

Steps (numbered in the order of the original data preparation):

1. Generator portfolio. The 138 committable units (natural gas, coal, hydro and
   nuclear) are taken from ``Generator_data.xlsx`` (sheet "Gen data"):
   variable cost = C1, no-load cost = C0 (applied for each committed hour),
   startup cost = Csu, shutdown cost = 0, ramp limits = 60 times the ramp rate in
   MW/min, minimum up and down times of 8 h (nuclear), 4 h (coal), 2 h (natural
   gas) and 1 h (hydro), and nuclear and coal units on at the start of the day.
   The 82 wind and 72 solar units enter only through the net load.
2. Copula parameters. Hourly system totals of load, wind and solar are formed
   for every day of 2017 to 2021. For the summer days (June to August,
   460 days) each variable is normalized for each hour as
   (x - hourly median) / hourly interquartile range. For five blocks of the day
   the Kendall tau of every pair of normalized deviations is converted to a
   Gaussian-copula correlation rho = sin(pi tau / 2). The day with the largest
   summer net load (2021-08-30) is kept as the baseline profile.
3. Scenarios. For each hour, a draw from the Gaussian copula of that hour's
   block (random seed 42, 7,500 draws) is mapped through the block's empirical
   quantiles of the normalized deviations, scaled by the hour's interquartile
   range and added to the peak-day profile. Draws are independent between
   hours. Load is clipped at zero, and wind and solar at zero and at their
   largest observed summer value. Net load = load - wind - solar.
4. Split. A random permutation (random seed 7) of the 7,500 scenarios gives a
   2,500-scenario training set and a disjoint 5,000-scenario test file whose
   first 2,000 scenarios form the evaluation set.

Outputs (``--out-dir``, default ``outputs/data/tx123bt``) have the layout of
``data/tx123bt``; ``--check`` compares them with ``data/SHA256SUMS``. The set of
7,500 scenarios, its summary and the hourly system totals are written to
``<out-dir>/intermediate/``. On the platform used for the paper (macOS arm64,
conda-forge NumPy 2.4.6) the regenerated files are byte-identical to the files
included in ``data/``; other linear-algebra libraries may change the last digits.

Example::

    python scripts/data/prepare_tx123bt.py --raw-dir data/raw/TX-123BT --check
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

from pisasr import paths

# ------------------------------------------------------------------
# Settings of the paper
# ------------------------------------------------------------------

YEARS_USED = [2017, 2018, 2019, 2020, 2021]
SUMMER_MONTHS = [6, 7, 8]

TIME_BLOCKS = {
    "P1_Overnight_00_05": list(range(0, 6)),    # 00:00-05:59
    "P2_Morning_06_09": list(range(6, 10)),     # 06:00-09:59
    "P3_Midday_10_14": list(range(10, 15)),     # 10:00-14:59
    "P4_Peak_15_19": list(range(15, 20)),       # 15:00-19:59
    "P5_Evening_20_23": list(range(20, 24)),    # 20:00-23:59
}

N_SCENARIOS = 7500
SCENARIO_SEED = 42
TRAIN_SIZE = 2500
TEST_SIZE = 5000
SPLIT_SEED = 7

DISPATCHABLE_FUELS = ["Natural Gas", "Coal", "Hydro", "Nuclear"]
MIN_UP_DOWN_HOURS = {"Nuclear": 8, "Coal": 4, "Natural Gas": 2, "Hydro": 1}

HOUR_COLUMNS = [f"hour_{h:02d}" for h in range(24)]


# ------------------------------------------------------------------
# Step 1: generator portfolio
# ------------------------------------------------------------------

def build_generator_portfolio(xlsx: Path) -> pd.DataFrame:
    gen = pd.read_excel(xlsx, sheet_name="Gen data")
    print(f"Generator_data.xlsx: {len(gen)} units")

    gen = gen[gen["Fuel type"].isin(DISPATCHABLE_FUELS)].copy()

    portfolio = pd.DataFrame()
    portfolio["unit"] = "G" + gen["Gen Number"].astype(int).astype(str)
    portfolio["bus"] = gen["Bus Number"].astype(int)
    portfolio["fuel_type"] = gen["Fuel type"]
    portfolio["p_min_mw"] = gen["Pmin (MW)"].astype(float)
    portfolio["p_max_mw"] = gen["Pmax (MW)"].astype(float)
    # C1 is the linear production cost, Csu the startup cost and C0 the
    # no-load cost of a committed hour.
    portfolio["variable_cost"] = gen["C1($/MWh)"].astype(float)
    portfolio["startup_cost"] = gen["Csu($)"].astype(float)
    portfolio["shutdown_cost"] = 0.0
    portfolio["no_load_cost"] = gen["C0($/MWh)"].astype(float)
    # Ramp rates are given in MW/min.
    portfolio["ramp_up_mw"] = gen["Ramping Rate(MW/min)"].astype(float) * 60.0
    portfolio["ramp_down_mw"] = gen["Ramping Rate(MW/min)"].astype(float) * 60.0
    # The data set gives no minimum up and down times or initial status.
    portfolio["min_up_hours"] = portfolio["fuel_type"].apply(lambda f: MIN_UP_DOWN_HOURS.get(f, 1))
    portfolio["min_down_hours"] = portfolio["fuel_type"].apply(lambda f: MIN_UP_DOWN_HOURS.get(f, 1))
    portfolio["initial_status"] = 0
    portfolio.loc[portfolio["fuel_type"].isin(["Nuclear", "Coal"]), "initial_status"] = 1
    # A unit without a ramp rate would receive its full capacity (none in the data).
    zero_ramp = portfolio["ramp_up_mw"] <= 0
    portfolio.loc[zero_ramp, "ramp_up_mw"] = portfolio.loc[zero_ramp, "p_max_mw"]
    portfolio.loc[zero_ramp, "ramp_down_mw"] = portfolio.loc[zero_ramp, "p_max_mw"]

    columns = [
        "unit", "bus", "fuel_type", "p_min_mw", "p_max_mw", "variable_cost",
        "startup_cost", "shutdown_cost", "no_load_cost", "ramp_up_mw",
        "ramp_down_mw", "min_up_hours", "min_down_hours", "initial_status",
    ]
    portfolio = portfolio[columns]
    print(f"Committable portfolio: {len(portfolio)} units, "
          f"{portfolio['p_max_mw'].sum():,.1f} MW")
    return portfolio


# ------------------------------------------------------------------
# Step 2: hourly system totals and copula parameters
# ------------------------------------------------------------------

def extract_year_and_day(path: Path) -> tuple[int, int]:
    """Year from the folder name (for example ``Load_annual_2017``) and day from the file name.

    Only the parent folder name is searched for the year, so the location of
    the raw data (for example a folder named after a year) cannot interfere.
    """
    year_match = re.search(r"(20\d{2})", path.parent.name)
    day_match = re.search(r"_D(\d+)", path.stem, flags=re.IGNORECASE)
    if year_match is None or day_match is None:
        raise ValueError(f"Could not extract year/day from file: {path}")
    return int(year_match.group(1)), int(day_match.group(1))


def discover_files(folder: Path) -> dict[tuple[int, int], Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files: dict[tuple[int, int], Path] = {}
    for path in folder.rglob("*.txt"):
        try:
            files[extract_year_and_day(path)] = path
        except ValueError:
            print(f"Skipping unexpected file name: {path}")
    if not files:
        raise FileNotFoundError(f"No .txt profile files found in {folder}")
    return files


def read_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"[\s,\t,]+", engine="python", header=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.isna().any().any():
        raise ValueError(f"Non-numeric values in {path}")
    return df.to_numpy(dtype=float)


def get_hourly_total(matrix: np.ndarray, label: str) -> np.ndarray:
    """System total for each hour (load files: 24 × buses; wind and solar: plants × 24)."""
    rows, cols = matrix.shape
    if rows == 24 and cols != 24:
        hourly = matrix.sum(axis=1)
    elif cols == 24 and rows != 24:
        hourly = matrix.sum(axis=0)
    else:
        raise ValueError(f"Unexpected shape for {label}: {matrix.shape}")
    if len(hourly) != 24:
        raise ValueError(f"{label} hourly total does not have 24 values.")
    return hourly


def build_hourly_system_totals(raw_dir: Path) -> pd.DataFrame:
    load_files = discover_files(raw_dir / "Load_5y")
    wind_files = discover_files(raw_dir / "Wind_5y")
    solar_files = discover_files(raw_dir / "Solar_5y")

    common_keys = sorted(set(load_files) & set(wind_files) & set(solar_files))
    common_keys = [key for key in common_keys if key[0] in YEARS_USED]
    if not common_keys:
        raise ValueError("No day of 2017 to 2021 has load, wind and solar daily files in the raw folders.")
    print(f"Days with load, wind and solar files: {len(common_keys)}")

    rows = []
    for year, day_of_year in common_keys:
        load_hourly = get_hourly_total(read_matrix(load_files[(year, day_of_year)]), "load")
        wind_hourly = get_hourly_total(read_matrix(wind_files[(year, day_of_year)]), "wind")
        solar_hourly = get_hourly_total(read_matrix(solar_files[(year, day_of_year)]), "solar")
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day_of_year - 1)
        for hour in range(24):
            rows.append({
                "date": date,
                "year": year,
                "day_of_year": day_of_year,
                "month": date.month,
                "hour": hour,
                "load_mw": load_hourly[hour],
                "wind_mw": wind_hourly[hour],
                "solar_mw": solar_hourly[hour],
                "net_load_mw": load_hourly[hour] - wind_hourly[hour] - solar_hourly[hour],
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def calculate_normalized_summer_deviations(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Summer deviations from the typical hourly behavior: (x - median) / IQR for each hour."""
    summer_df = hourly_df[
        (hourly_df["year"].isin(YEARS_USED)) & (hourly_df["month"].isin(SUMMER_MONTHS))
    ].copy()
    if summer_df.empty:
        raise ValueError("No summer data were found.")

    for variable in ["load_mw", "wind_mw", "solar_mw"]:
        median_by_hour = summer_df.groupby("hour")[variable].transform("median")
        q25_by_hour = summer_df.groupby("hour")[variable].transform(lambda x: x.quantile(0.25))
        q75_by_hour = summer_df.groupby("hour")[variable].transform(lambda x: x.quantile(0.75))
        iqr_by_hour = (q75_by_hour - q25_by_hour).replace(0, np.nan)
        summer_df[variable.replace("_mw", "_dev")] = (summer_df[variable] - median_by_hour) / iqr_by_hour

    return summer_df


def valid_variables_for_block(block_df: pd.DataFrame) -> list[str]:
    """Deviations with variation in the block (solar is dropped at night)."""
    valid = []
    for variable in ["load_dev", "wind_dev", "solar_dev"]:
        series = block_df[variable].dropna()
        if len(series) >= 10 and series.std() > 1e-10:
            valid.append(variable)
    return valid


def force_positive_semidefinite(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    eigenvalues[eigenvalues < 1e-8] = 1e-8
    adjusted = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(adjusted))
    adjusted = adjusted / np.outer(diagonal, diagonal)
    return pd.DataFrame(adjusted, index=matrix.index, columns=matrix.columns)


def estimate_copula_parameters(summer_dev_df: pd.DataFrame):
    pairwise_rows = []
    matrices = {}
    for block_name, hours in TIME_BLOCKS.items():
        block_df = summer_dev_df[summer_dev_df["hour"].isin(hours)].copy()
        valid_variables = valid_variables_for_block(block_df)
        if len(valid_variables) < 2:
            print(f"Warning: not enough valid variables in {block_name}")
            continue
        matrix = pd.DataFrame(np.eye(len(valid_variables)),
                              index=valid_variables, columns=valid_variables)
        for i, var1 in enumerate(valid_variables):
            for var2 in valid_variables[i + 1:]:
                data = block_df[[var1, var2]].dropna()
                tau, p_value = kendalltau(data[var1], data[var2])
                if pd.isna(tau):
                    continue
                rho = math.sin(math.pi * float(tau) / 2.0)
                matrix.loc[var1, var2] = rho
                matrix.loc[var2, var1] = rho
                pairwise_rows.append({
                    "period": block_name,
                    "hours": f"{min(hours):02d}:00-{max(hours):02d}:59",
                    "variable_1": var1,
                    "variable_2": var2,
                    "sample_count": len(data),
                    "kendall_tau": tau,
                    "gaussian_copula_rho": rho,
                    "p_value": p_value,
                })
        matrices[block_name] = force_positive_semidefinite(matrix)
    return pd.DataFrame(pairwise_rows), matrices


def find_overall_summer_peak_net_load_day(hourly_df: pd.DataFrame) -> pd.DataFrame:
    summer_df = hourly_df[hourly_df["month"].isin(SUMMER_MONTHS)].copy()
    peak_day = summer_df.groupby("date")["net_load_mw"].max().idxmax()
    peak_profile = summer_df[summer_df["date"] == peak_day].copy()
    return peak_profile.sort_values("hour").reset_index(drop=True)


# ------------------------------------------------------------------
# Step 3: Gaussian-copula scenarios
# ------------------------------------------------------------------

def empirical_inverse_cdf(samples: np.ndarray, u: np.ndarray) -> np.ndarray:
    clean = samples[np.isfinite(samples)]
    if clean.size == 0:
        raise ValueError("No valid samples for the empirical inverse CDF.")
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return np.quantile(clean, u, method="linear")


def make_positive_semidefinite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    eigenvalues[eigenvalues < 1e-8] = 1e-8
    adjusted = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(adjusted))
    return adjusted / np.outer(diagonal, diagonal)


def sample_gaussian_copula(matrix: pd.DataFrame, n_samples: int,
                           rng: np.random.Generator) -> pd.DataFrame:
    """Correlated uniforms: Z ~ N(0, R), U = Phi(Z)."""
    variables = list(matrix.index)
    corr = make_positive_semidefinite(matrix.to_numpy(dtype=float))
    z = rng.multivariate_normal(mean=np.zeros(len(variables)), cov=corr, size=n_samples)
    return pd.DataFrame(norm.cdf(z), columns=variables)


def compute_hourly_iqr(summer_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for hour, group in summer_df.groupby("hour"):
        record = {"hour": hour}
        for variable in ["load_mw", "wind_mw", "solar_mw"]:
            iqr = group[variable].quantile(0.75) - group[variable].quantile(0.25)
            if pd.isna(iqr) or iqr <= 1e-9:
                iqr = 0.0
            record[f"{variable}_iqr"] = iqr
        records.append(record)
    return pd.DataFrame(records).sort_values("hour").reset_index(drop=True)


def generate_scenarios(param_dir: Path):
    """Load, wind, solar and net-load scenarios around the summer peak day."""
    rng = np.random.default_rng(SCENARIO_SEED)
    summer_df = pd.read_csv(param_dir / "summer_normalized_deviations_2017_2021.csv",
                            parse_dates=["date"])
    peak_profile = pd.read_csv(param_dir / "overall_summer_peak_net_load_day_profile.csv",
                               parse_dates=["date"])

    hourly_iqr = compute_hourly_iqr(summer_df)
    baseline = peak_profile.sort_values("hour").reset_index(drop=True)
    if set(range(24)) - set(baseline["hour"].astype(int)):
        raise ValueError("Peak profile does not contain all 24 hours.")
    baseline = baseline[["hour", "load_mw", "wind_mw", "solar_mw", "net_load_mw"]]

    observed_max_wind = summer_df["wind_mw"].max()
    observed_max_solar = summer_df["solar_mw"].max()

    load_s = np.zeros((N_SCENARIOS, 24))
    wind_s = np.zeros((N_SCENARIOS, 24))
    solar_s = np.zeros((N_SCENARIOS, 24))

    for block_name, hours in TIME_BLOCKS.items():
        matrix = pd.read_csv(param_dir / f"{block_name}_gaussian_copula_matrix.csv", index_col=0)
        variables = list(matrix.index)
        for hour in hours:
            block_history = summer_df[summer_df["hour"].isin(hours)].copy()
            u_df = sample_gaussian_copula(matrix, N_SCENARIOS, rng)
            devs = {v: empirical_inverse_cdf(block_history[v].dropna().to_numpy(),
                                             u_df[v].to_numpy())
                    for v in variables}

            base_row = baseline[baseline["hour"] == hour].iloc[0]
            iqr_row = hourly_iqr[hourly_iqr["hour"] == hour].iloc[0]

            load_values = np.full(N_SCENARIOS, base_row["load_mw"], dtype=float)
            wind_values = np.full(N_SCENARIOS, base_row["wind_mw"], dtype=float)
            solar_values = np.full(N_SCENARIOS, base_row["solar_mw"], dtype=float)
            if "load_dev" in devs:
                load_values = base_row["load_mw"] + devs["load_dev"] * iqr_row["load_mw_iqr"]
            if "wind_dev" in devs:
                wind_values = base_row["wind_mw"] + devs["wind_dev"] * iqr_row["wind_mw_iqr"]
            if "solar_dev" in devs:
                solar_values = base_row["solar_mw"] + devs["solar_dev"] * iqr_row["solar_mw_iqr"]

            load_values = np.maximum(load_values, 0.0)
            wind_values = np.minimum(np.maximum(wind_values, 0.0), observed_max_wind)
            solar_values = np.minimum(np.maximum(solar_values, 0.0), observed_max_solar)

            load_s[:, hour] = load_values
            wind_s[:, hour] = wind_values
            solar_s[:, hour] = solar_values

    index = [f"scenario_{i + 1:04d}" for i in range(N_SCENARIOS)]
    frames = []
    for values in (load_s, wind_s, solar_s, load_s - wind_s - solar_s):
        df = pd.DataFrame(values, index=index, columns=HOUR_COLUMNS)
        df.index.name = "scenario"
        frames.append(df)
    return tuple(frames)


def summarize_scenarios(load_df, wind_df, solar_df, net_load_df) -> pd.DataFrame:
    summary = pd.DataFrame(index=load_df.index)
    summary["total_load_mwh"] = load_df.sum(axis=1)
    summary["total_wind_mwh"] = wind_df.sum(axis=1)
    summary["total_solar_mwh"] = solar_df.sum(axis=1)
    summary["total_net_load_mwh"] = net_load_df.sum(axis=1)
    summary["peak_load_mw"] = load_df.max(axis=1)
    summary["peak_net_load_mw"] = net_load_df.max(axis=1)
    summary["min_net_load_mw"] = net_load_df.min(axis=1)
    ramps = np.diff(net_load_df.to_numpy(), axis=1)
    summary["max_upward_net_load_ramp_mw"] = ramps.max(axis=1)
    summary["max_downward_net_load_ramp_mw"] = ramps.min(axis=1)
    summary["probability"] = 1.0 / N_SCENARIOS
    summary.index.name = "scenario"
    return summary.reset_index()


# ------------------------------------------------------------------
# Step 4: training set and test file
# ------------------------------------------------------------------

def split_pool(pool_file: Path, train_file: Path, test_file: Path) -> None:
    pool = pd.read_csv(pool_file, index_col=0)
    if len(pool) < TRAIN_SIZE + TEST_SIZE:
        raise ValueError(f"Pool has {len(pool)} scenarios; {TRAIN_SIZE + TEST_SIZE} needed.")
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(len(pool))
    train_df = pool.iloc[shuffled[:TRAIN_SIZE]].copy()
    test_df = pool.iloc[shuffled[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]].copy()
    overlap = set(train_df.index) & set(test_df.index)
    if overlap:
        raise RuntimeError(f"Training and test scenarios overlap: {sorted(overlap)[:5]}")
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    print(f"Training set: {len(train_df)} scenarios; test file: {len(test_df)} scenarios "
          "(disjoint)")


# ------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the TX-123BT generator portfolio, Gaussian-copula "
                    "parameters and net-load scenarios from the raw five-year profiles.",
    )
    parser.add_argument("--raw-dir", type=Path, default=paths.RAW_DIR / "TX-123BT",
                        help="folder with Generator_data.xlsx, Load_5y, Wind_5y and Solar_5y "
                             "(default: data/raw/TX-123BT, or $PISASR_RAW_DIR/TX-123BT; "
                             "see scripts/data/download_tx123bt.py)")
    parser.add_argument("--out-dir", type=Path, default=paths.OUTPUT_DIR / "data" / "tx123bt",
                        help="output folder (default: outputs/data/tx123bt, or $PISASR_OUTPUT_DIR/data/tx123bt); "
                             "pass data/tx123bt to "
                             "replace the included files")
    parser.add_argument("--check", action="store_true",
                        help="compare the outputs with data/SHA256SUMS")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if the output folder already holds the files")
    return parser.parse_args(argv)


def _check(out: Path) -> int:
    print("\nComparing with data/SHA256SUMS:")
    return 0 if paths.verify_checksums(out, prefix="tx123bt") else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    raw, out = args.raw_dir, args.out_dir
    if (out / "net_load_scenarios_test.csv").exists() and not args.force:
        print(f"{out} already holds the processed files; skipping (use --force to regenerate).")
        return _check(out) if args.check else 0
    if not (raw / "Generator_data.xlsx").exists():
        print(f"Missing {raw / 'Generator_data.xlsx'}; run scripts/data/download_tx123bt.py first.")
        return 1
    params = out / "copula_parameters"
    inter = out / "intermediate"
    for folder in (out, params, inter):
        folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Reading raw data from: {raw}")
    print(f"Writing processed data to: {out}")

    print("\n[1/4] Generator portfolio")
    portfolio = build_generator_portfolio(raw / "Generator_data.xlsx")
    portfolio.to_csv(out / "generator_portfolio.csv", index=False)

    print("\n[2/4] Hourly system totals and Gaussian-copula parameters")
    hourly_df = build_hourly_system_totals(raw)
    hourly_df.to_csv(inter / "tx123bt_hourly_system_totals_2017_2021.csv", index=False)
    summer_dev_df = calculate_normalized_summer_deviations(hourly_df)
    summer_dev_df.to_csv(params / "summer_normalized_deviations_2017_2021.csv", index=False)
    pairwise_df, matrices = estimate_copula_parameters(summer_dev_df)
    pairwise_df.to_csv(params / "gaussian_copula_pairwise_parameters.csv", index=False)
    for block_name, matrix in matrices.items():
        matrix.to_csv(params / f"{block_name}_gaussian_copula_matrix.csv")
    peak_profile = find_overall_summer_peak_net_load_day(hourly_df)
    peak_profile.to_csv(params / "overall_summer_peak_net_load_day_profile.csv", index=False)
    print(f"Summer peak net-load day: {peak_profile['date'].iloc[0].date()} "
          f"({peak_profile['net_load_mw'].max():,.2f} MW)")

    print("\n[3/4] Gaussian-copula scenarios")
    load_df, wind_df, solar_df, net_load_df = generate_scenarios(params)
    pool_file = inter / "net_load_scenarios_24h.csv"
    net_load_df.to_csv(pool_file)
    summarize_scenarios(load_df, wind_df, solar_df, net_load_df).to_csv(
        inter / "scenario_summary.csv", index=False)
    print(f"{len(net_load_df)} scenarios; largest peak net load "
          f"{net_load_df.max(axis=1).max():,.1f} MW")

    print("\n[4/4] Training set and test file")
    split_pool(pool_file, out / "net_load_scenarios_train.csv", out / "net_load_scenarios_test.csv")
    print(f"\nDone in {time.time() - t0:.0f} s.")
    return _check(out) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
