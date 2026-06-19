"""Multi-seed sweep over circuit configs -> sweep.parquet + a median/IQR table.

RESULTS.md found a 37x MSE spread across parameter-init seeds within one config,
so a single-seed comparison measures seed luck, not the circuit. This script
runs each config across several seeds and reports the distribution (median and
inter-quartile range) plus the best-of-N restart selected by training loss --
the comparisons that actually clear the seed noise.

Edit CONFIGS / SEEDS below. Cost ~ len(CONFIGS) x len(SEEDS) runs; a 100-epoch
run is ~12-17 min, so screen at fewer epochs first.

    uv run python sweep.py --epochs 40   # quick screen
    uv run python sweep.py               # full 100-epoch sweep
"""

import argparse

import pandas as pd

from src.experiment import result_row, run_config

# Each dict overrides run_config's defaults (n_qubits, n_layers, scale, ...).
CONFIGS = [
    {"n_qubits": 2, "n_layers": 3, "scale": 2.0},
    {"n_qubits": 3, "n_layers": 3, "scale": 2.0},
]
SEEDS = [22, 23, 42, 7, 101]
OUT_FILE = "sweep.parquet"


def label(cfg):
    return f"{cfg['n_qubits']}q/{cfg['n_layers']}l s{cfg['scale']}"


def fmt4(v):
    return f"{v:.4f}"


def iqr(s):
    return s.quantile(0.75) - s.quantile(0.25)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed circuit-config sweep.")
    parser.add_argument("--epochs", type=int, default=100, help="epochs per run")
    args = parser.parse_args()

    n_runs = len(CONFIGS) * len(SEEDS)
    print(
        f"Sweeping {len(CONFIGS)} configs x {len(SEEDS)} seeds = {n_runs} runs "
        f"at {args.epochs} epochs each.\n"
    )

    rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            print(f"--- {label(cfg)}  seed {seed} ---")
            result = run_config(epochs=args.epochs, seed=seed, **cfg)
            row = result_row(result)
            row["config"] = label(cfg)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_FILE}.\n")

    # Distribution per config: median = typical run, IQR = seed spread. One config
    # only beats another if their medians differ by more than this spread.
    agg = df.groupby("config").agg(
        n=("mse", "size"),
        mse_median=("mse", "median"),
        mse_iqr=("mse", iqr),
        mse_best=("mse", "min"),
        r2_median=("r2", "median"),
        nlpd_median=("nlpd", "median"),
    )
    print("Per-config distribution across seeds:")
    print(agg.to_string(float_format=fmt4))

    # Best-of-N restart selected by final training loss (no test peeking) -- the
    # model-selection protocol RESULTS.md found reliable.
    best = df.loc[df.groupby("config")["final_loss"].idxmin()]
    print("\nBest restart per config by final training loss (no test peeking):")
    print(
        best[["config", "seed", "final_loss", "mse", "r2"]].to_string(
            index=False, float_format=fmt4
        )
    )


if __name__ == "__main__":
    main()
