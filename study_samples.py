"""Training-set-size study: a learning curve over the number of samples.

Sweeps n_train and repeats over parameter-init seeds, tracing how the fit and
its uncertainty improve as data grows. This is NOT the paper's Fig. 3(b) (which
varies measurement shots -- see study_shots.py) but the companion data-size
question. Every run is a full retrain, so cost ~ len(N_TRAIN) x len(SEEDS) runs;
screen at fewer epochs first.

    uv run python study_samples.py --epochs 40   # quick screen
    uv run python study_samples.py --qubits 3 --layers 3
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless backend; drop this line for interactive plots

import pandas as pd

from src.experiment import result_row, run_config
from src.plotting import plot_study_curve, set_plot_style

N_TRAIN = [5, 10, 20, 30, 40]
SEEDS = [22, 23, 42]
OUT_FILE = "study_samples.parquet"
RBF_BASELINE = 0.0027  # classical RBF GP, same protocol (RESULTS.md yardstick)


def main():
    parser = argparse.ArgumentParser(description="Training-set-size learning curve.")
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    n_runs = len(N_TRAIN) * len(SEEDS)
    print(
        f"Learning curve: {len(N_TRAIN)} sizes x {len(SEEDS)} seeds = {n_runs} runs.\n"
    )

    rows = []
    for n_train in N_TRAIN:
        for seed in SEEDS:
            print(f"--- n_train={n_train}  seed {seed} ---")
            result = run_config(
                n_qubits=args.qubits,
                n_layers=args.layers,
                scale=args.scale,
                epochs=args.epochs,
                seed=seed,
                n_train=n_train,
            )
            rows.append(result_row(result))
            print(f"    MSE {result.metrics['mse']:.4f}  R2 {result.metrics['r2']:.4f}")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_FILE}.")

    # MSE vs training size: mean +/- std over seeds, with the classical-RBF line.
    xs = sorted(df["N_train"].unique())
    grouped = df.groupby("N_train")
    mse_by_n = [grouped.get_group(n)["mse"].to_numpy() for n in xs]
    set_plot_style()
    plot_study_curve(
        xs,
        mse_by_n,
        xlabel="training-set size",
        ylabel="Test MSE",
        save_path="study_samples.png",
        reference=RBF_BASELINE,
        reference_label="classical RBF",
    )
    print("Saved study_samples.png")


if __name__ == "__main__":
    main()
