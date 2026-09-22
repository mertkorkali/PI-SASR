#!/usr/bin/env python
"""Regenerate the Texas2k Series25 inputs from the Texas A&M files.

Inputs (not redistributed; download them from the Texas A&M Electric Grid Test
Case Repository, see data/README.md):

* ``--case-file``: ``Texas2k_series25_case1_summerpeak.m`` (Texas2k Series25,
  "Case 1: 2025 Summer Peak", MATPOWER format);
* ``--load-file``: ``ACTIVISg2000_load_time_series_MW.csv`` (ACTIVSg2000 2016
  load time series);
* ``--renewable-file``: ``ACTIVISg2000_renewable_time_series_MW.csv``
  (ACTIVSg2000 2016 wind and solar time series).

Generator portfolio. The committable units are those with fuel type natural gas,
coal, nuclear, hydro, diesel or wood (568 units, 77.8 GW); wind and solar
(400 units) are netted into demand, and the 131 units of fuel type "other" are
excluded. The variable cost of a unit is the average incremental cost of its
cost polynomial over [max(P_min, 0.001), P_max] (a fuel-class default when this
is at most $1/MWh); because the case's costs are intended for power-flow studies,
each fuel class is then rescaled to a target mean marginal cost while keeping the
within-class variation (clipped to 0.6 to 1.8 times the class mean). Startup
costs, no-load costs, ramp limits and minimum up and down times are fuel-class
defaults; coal and nuclear units are on at the start of the day.

Scenarios. The hourly system totals of load, wind and solar of the 92 summer
days (June to August 2016) form 72-dimensional daily samples. A Gaussian copula
with correlation shrunk toward the identity (weight 0.2) is fitted to their
normal scores and sampled 7,500 times (random seed 42); samples are mapped back
through interpolated empirical quantiles. Load is scaled to the 85.76-GW summer
peak of the Series25 case and wind and solar to its installed capacities; net
load is floored at 1,000 MW. A random permutation (random seed 7) gives a
2,500-scenario training set and a disjoint 5,000-scenario test file whose first
2,000 scenarios form the evaluation set.

Outputs (``--out-dir``, default ``outputs/data/texas2k``) have the layout of
``data/texas2k``; the set of 7,500 scenarios and its summary go to
``<out-dir>/intermediate/``. ``--check`` compares the outputs with
``data/SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pisasr import paths

# ------------------------------------------------------------------
# Generator portfolio settings
# ------------------------------------------------------------------

COMMITTABLE_FUELS = ["ng", "coal", "nuclear", "hydro", "dfo", "wood"]

DEFAULT_VARIABLE_COST = {   # $/MWh; fallback and target mean of each fuel class
    "nuclear": 17.44, "coal": 22.0, "ng": 28.0, "hydro": 12.3,
    "dfo": 95.0, "wood": 35.0,
}
STARTUP_COST_PER_MW = {     # $ for each MW of capacity
    "nuclear": 60.0, "coal": 40.0, "ng": 15.0, "hydro": 2.0,
    "dfo": 10.0, "wood": 25.0,
}
NO_LOAD_COST = {            # $/h while committed
    "nuclear": 0.0, "coal": 800.0, "ng": 500.0, "hydro": 0.0,
    "dfo": 100.0, "wood": 200.0,
}
MIN_UP = {"nuclear": 8, "coal": 6, "ng": 2, "hydro": 1, "dfo": 1, "wood": 4}
MIN_DOWN = {"nuclear": 8, "coal": 6, "ng": 2, "hydro": 1, "dfo": 1, "wood": 4}
RAMP_PU_PER_HOUR = {        # fraction of P_max in one hour
    "nuclear": 0.3, "coal": 0.5, "ng": 1.0, "hydro": 3.0, "dfo": 2.0, "wood": 0.5,
}

# ------------------------------------------------------------------
# Scenario settings
# ------------------------------------------------------------------

LOAD_PEAK_25 = 85758.89          # MW, total load of the Series25 summer-peak case
WIND_CAP_16, WIND_CAP_25 = 9587.0, 41837.0     # MW, ACTIVSg2000 and Series25 wind
SOLAR_CAP_16, SOLAR_CAP_25 = 651.0, 32266.0    # MW, ACTIVSg2000 and Series25 solar

POOL_SIZE = 7500
TRAIN_SIZE = 2500
TEST_SIZE = 5000
GEN_SEED = 42
SPLIT_SEED = 7
SHRINKAGE = 0.2                 # weight of the identity in the shrunk correlation
NET_LOAD_FLOOR = 1000.0         # MW

HOUR_COLS = [f"hour_{h:02d}" for h in range(24)]


# ------------------------------------------------------------------
# Generator portfolio
# ------------------------------------------------------------------

def _block(txt: str, name: str, brace: str = "[]") -> str:
    o, c = brace
    return re.search(rf"mpc\.{name} = {re.escape(o)}(.*?){re.escape(c)};", txt, re.S).group(1)


def build_generator_portfolio(case_file: Path) -> pd.DataFrame:
    txt = case_file.read_text()
    gen_rows = [r.split() for r in _block(txt, "gen").strip().split("\n")]
    cost_rows = [r.split() for r in _block(txt, "gencost").strip().split("\n")]
    fuels = re.findall(r"'([^']+)'", _block(txt, "genfuel", "{}"))
    if not (len(gen_rows) == len(cost_rows) == len(fuels)):
        raise ValueError("gen, gencost and genfuel tables have different lengths")
    print(f"Parsed {len(gen_rows)} generators from {case_file.name}")

    rows = []
    for i, (g, c, fuel) in enumerate(zip(gen_rows, cost_rows, fuels)):
        if fuel not in COMMITTABLE_FUELS:
            continue
        bus = int(float(g[0]))
        pmax = float(g[8])
        pmin = float(g[9])
        if pmax <= 0:
            continue
        # MATPOWER polynomial cost: model, startup, shutdown, n, then n
        # coefficients from the highest power down to the constant.
        coeffs = [float(x) for x in c[4:]]
        if any(abs(x) > 1e-9 for x in coeffs[:-1]):
            p_lo, p_hi = max(pmin, 1e-3), pmax
            n = len(coeffs)

            def total_cost(p):
                return sum(coeffs[k] * p ** (n - 1 - k) for k in range(n))

            if p_hi - p_lo > 1.0:
                var_cost = (total_cost(p_hi) - total_cost(p_lo)) / (p_hi - p_lo)
            else:
                # unit with (almost) fixed output: marginal cost at P_max
                var_cost = sum(
                    (n - 1 - k) * coeffs[k] * p_hi ** (n - 2 - k)
                    for k in range(n - 1)
                )
        else:
            var_cost = 0.0
        if var_cost <= 1.0:           # missing cost data: fuel-class default
            var_cost = DEFAULT_VARIABLE_COST[fuel]

        if pmin >= pmax or pmin < 0:
            pmin = 0.25 * pmax if fuel in ("nuclear", "coal") else 0.1 * pmax

        rows.append({
            "unit": f"T{i + 1}",
            "bus": bus,
            "fuel_type": fuel,
            "p_min_mw": round(pmin, 2),
            "p_max_mw": round(pmax, 2),
            "variable_cost": round(var_cost, 2),
            "startup_cost": round(STARTUP_COST_PER_MW[fuel] * pmax, 0),
            "shutdown_cost": 0.0,
            "no_load_cost": NO_LOAD_COST[fuel],
            "ramp_up_mw": round(RAMP_PU_PER_HOUR[fuel] * pmax, 1),
            "ramp_down_mw": round(RAMP_PU_PER_HOUR[fuel] * pmax, 1),
            "min_up_hours": MIN_UP[fuel],
            "min_down_hours": MIN_DOWN[fuel],
            "initial_status": 1 if fuel in ("nuclear", "coal") else 0,
        })

    df = pd.DataFrame(rows)

    # Rescale each fuel class to its target mean marginal cost, keeping the
    # relative differences within the class.
    for fuel, target_mean in DEFAULT_VARIABLE_COST.items():
        m = df["fuel_type"] == fuel
        if not m.any():
            continue
        raw = df.loc[m, "variable_cost"]
        if raw.mean() > 0 and raw.std() > 1e-9:
            scaled = target_mean * raw / raw.mean()
            scaled = scaled.clip(0.6 * target_mean, 1.8 * target_mean)
        else:
            scaled = pd.Series(target_mean, index=raw.index)
        df.loc[m, "variable_cost"] = scaled.round(2)

    print(f"Committable portfolio: {len(df)} units, {df.p_max_mw.sum() / 1000:.1f} GW")
    print(df.groupby("fuel_type").agg(
        n=("unit", "count"), GW=("p_max_mw", lambda s: s.sum() / 1000),
        mean_cost=("variable_cost", "mean")).round(2).to_string())
    return df


# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------

def load_summer_days(load_file: Path, renewable_file: Path) -> np.ndarray:
    """(n_days, 72) array: 24 h load | 24 h wind | 24 h solar of the 2016 summer days."""
    load = pd.read_csv(load_file, skiprows=1, usecols=["Date", "Time", "Total MW Load"])
    ren = pd.read_csv(renewable_file, skiprows=1,
                      usecols=["Date", "Time", "Total solar Gen", "Total wind Gen"])
    df = load.merge(ren, on=["Date", "Time"])
    df["month"] = pd.to_datetime(df["Date"]).dt.month
    df = df[df["month"].isin([6, 7, 8])].reset_index(drop=True)

    n_days = len(df) // 24
    days = []
    for d in range(n_days):
        block = df.iloc[d * 24:(d + 1) * 24]
        days.append(np.concatenate([
            block["Total MW Load"].to_numpy(float),
            block["Total wind Gen"].to_numpy(float),
            block["Total solar Gen"].to_numpy(float),
        ]))
    X = np.array(days)
    print(f"Summer days: {X.shape[0]} (72 values each)")
    return X


def gaussian_copula_sample(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sample n rows that keep the marginals and the copula dependence of X."""
    rng = np.random.default_rng(seed)
    m, d = X.shape

    # Constant dimensions (solar at night) carry no dependence; the copula is
    # fitted to the varying dimensions and the constants are reinserted.
    std = X.std(axis=0)
    varying = np.where(std > 1e-9)[0]
    constant = np.where(std <= 1e-9)[0]
    Xv = X[:, varying]
    dv = len(varying)

    Z = np.empty_like(Xv)
    for j in range(dv):
        ranks = stats.rankdata(Xv[:, j]) / (m + 1)
        Z[:, j] = stats.norm.ppf(ranks)

    corr = np.corrcoef(Z, rowvar=False)
    corr = (1 - SHRINKAGE) * corr + SHRINKAGE * np.eye(dv)
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(dv))

    z_new = rng.standard_normal((n, dv)) @ L.T
    u_new = stats.norm.cdf(z_new)

    out = np.empty((n, d))
    probs = np.arange(1, m + 1) / (m + 1)
    for k, j in enumerate(varying):
        srt = np.sort(X[:, j])
        out[:, j] = np.interp(u_new[:, k], probs, srt, left=srt[0], right=srt[-1])
    for j in constant:
        out[:, j] = X[0, j]
    print(f"Copula dimensions: {dv} varying, {len(constant)} constant (night solar)")
    return out


def build_scenarios(load_file: Path, renewable_file: Path, out: Path, inter: Path) -> None:
    X = load_summer_days(load_file, renewable_file)
    sampled = gaussian_copula_sample(X, POOL_SIZE, GEN_SEED)
    load_s, wind_s, solar_s = sampled[:, :24], sampled[:, 24:48], sampled[:, 48:]

    load_scale = LOAD_PEAK_25 / X[:, :24].max()
    wind_scale = WIND_CAP_25 / WIND_CAP_16
    solar_scale = SOLAR_CAP_25 / SOLAR_CAP_16
    print(f"Scale factors: load {load_scale:.3f}, wind {wind_scale:.2f}, solar {solar_scale:.2f}")

    net = load_s * load_scale - wind_s * wind_scale - solar_s * solar_scale
    clipped = (net < NET_LOAD_FLOOR).sum()
    print(f"Net-load values raised to the {NET_LOAD_FLOOR:.0f}-MW floor: "
          f"{clipped} of {net.size} ({100 * clipped / net.size:.2f}%)")
    net = np.maximum(net, NET_LOAD_FLOOR)

    idx = [f"scenario_{i + 1:04d}" for i in range(POOL_SIZE)]
    pool = pd.DataFrame(net, index=idx, columns=HOUR_COLS)
    pool.to_csv(inter / "net_load_scenarios_24h.csv")

    order = np.random.default_rng(SPLIT_SEED).permutation(POOL_SIZE)
    train = pool.iloc[order[:TRAIN_SIZE]]
    test = pool.iloc[order[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]]
    if set(train.index) & set(test.index):
        raise RuntimeError("Training and test scenarios overlap.")
    train.to_csv(out / "net_load_scenarios_train.csv")
    test.to_csv(out / "net_load_scenarios_test.csv")

    summary = pd.DataFrame({
        "peak_net_load_mw": pool.max(axis=1),
        "min_net_load_mw": pool.min(axis=1),
        "energy_mwh": pool.sum(axis=1),
        "max_ramp_up_mw": pool.diff(axis=1).max(axis=1),
    })
    summary.to_csv(inter / "scenario_summary.csv")
    print(f"Pool: mean peak net load {summary['peak_net_load_mw'].mean():,.0f} MW, "
          f"largest {summary['peak_net_load_mw'].max():,.0f} MW")
    print(f"Training set: {len(train)} scenarios; test file: {len(test)} scenarios (disjoint)")


# ------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    raw = paths.RAW_DIR / "texas2k"
    parser = argparse.ArgumentParser(
        description="Regenerate the Texas2k Series25 generator portfolio and net-load "
                    "scenarios from the Texas A&M case and time-series files.",
    )
    parser.add_argument("--case-file", type=Path,
                        default=raw / "Texas2k_series25_case1_summerpeak.m",
                        help="Series25 summer-peak case in MATPOWER format "
                             "(default: data/raw/texas2k/, or $PISASR_RAW_DIR/texas2k/)")
    parser.add_argument("--load-file", type=Path,
                        default=raw / "ACTIVISg2000_load_time_series_MW.csv",
                        help="ACTIVSg2000 load time series (default: in data/raw/texas2k/)")
    parser.add_argument("--renewable-file", type=Path,
                        default=raw / "ACTIVISg2000_renewable_time_series_MW.csv",
                        help="ACTIVSg2000 renewable time series (default: in data/raw/texas2k/)")
    parser.add_argument("--out-dir", type=Path, default=paths.OUTPUT_DIR / "data" / "texas2k",
                        help="output folder (default: outputs/data/texas2k, or $PISASR_OUTPUT_DIR/data/texas2k); "
                             "pass data/texas2k to "
                             "replace the included files")
    parser.add_argument("--check", action="store_true",
                        help="compare the outputs with data/SHA256SUMS")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if the output folder already holds the files")
    return parser.parse_args(argv)


def _check(out: Path) -> int:
    print("\nComparing with data/SHA256SUMS:")
    return 0 if paths.verify_checksums(out, prefix="texas2k") else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out_dir
    if (out / "net_load_scenarios_test.csv").exists() and not args.force:
        print(f"{out} already holds the processed files; skipping (use --force to regenerate).")
        return _check(out) if args.check else 0
    for f in (args.case_file, args.load_file, args.renewable_file):
        if not f.exists():
            print(f"Missing input file: {f}\nSee data/README.md for the download locations.")
            return 1
    inter = out / "intermediate"
    inter.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Reading: {args.case_file}\n         {args.load_file}\n         {args.renewable_file}")
    print(f"Writing processed data to: {out}")

    print("\n[1/2] Generator portfolio")
    build_generator_portfolio(args.case_file).to_csv(out / "generator_portfolio.csv", index=False)

    print("\n[2/2] Gaussian-copula scenarios, training set and test file")
    build_scenarios(args.load_file, args.renewable_file, out, inter)
    print(f"\nDone in {time.time() - t0:.1f} s.")
    return _check(out) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
