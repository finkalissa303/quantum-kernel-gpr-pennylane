"""Sample-based (finite-shot) study -- the paper's Fig. 3(b), as a sweep.

Trains the kernel analytically, then FREEZES the optimal parameters and re-runs
the GP with the kernel estimated from a finite number of measurement shots, as
on real hardware. Fig. 3(b) is the single point N = 10,000; here we sweep N and
repeat over shot seeds to trace how the fit and its calibrated variance degrade
as the measurement budget shrinks.

Each finite-shot evaluation rebuilds a fresh QGP so the *whole* Gram matrix
(train-train included) is re-estimated under shots, and turns on eig_cutoff (the
paper's Sec. 1.1 PSD projection) so the noisy kernel stays usable at low N.
Total cost is one analytic training run plus many cheap evaluations.

    uv run python study_shots.py
    uv run python study_shots.py --epochs 100 --qubits 3 --layers 3 --seed 23
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless backend; drop this line for interactive plots

import gpytorch
import pandas as pd

from src.experiment import run_config
from src.model import QGP
from src.plotting import plot_study_curve, set_plot_style
from src.training import evaluate_model

SHOTS = [100, 300, 1000, 3000, 10000, 30000]
SHOT_SEEDS = [0, 1, 2, 3, 4]
OUT_FILE = "study_shots.parquet"


def main():
    parser = argparse.ArgumentParser(description="Finite-shot (Fig 3b) study.")
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=23, help="param-init seed for the analytic train"
    )
    args = parser.parse_args()

    # 1. Train analytically and keep the optimal parameters.
    print("Training analytically (exact kernel)...")
    result = run_config(
        n_qubits=args.qubits,
        n_layers=args.layers,
        scale=args.scale,
        epochs=args.epochs,
        seed=args.seed,
    )
    kernel = result.kernel
    likelihood = result.likelihood
    analytic_mse = result.metrics["mse"]
    print(
        f"Analytic reference: MSE {analytic_mse:.4f}  R2 {result.metrics['r2']:.4f}\n"
    )

    # 2. Freeze params; PSD-project the now-noisy Gram matrices; sweep shots.
    kernel.eig_cutoff = True
    rows = []
    for shots in SHOTS:
        for shot_seed in SHOT_SEEDS:
            kernel.set_shots(shots=shots, seed=shot_seed)
            # Fresh QGP so the cached train-train block is recomputed under shots.
            model = QGP(result.X_train, result.y_train, likelihood, kernel)
            with gpytorch.settings.cholesky_jitter(1e-3):
                _, _, m = evaluate_model(
                    model, likelihood, result.X_test, result.y_test
                )
            rows.append({"shots": shots, "shot_seed": shot_seed, **m})
            print(
                f"shots={shots:>6} seed={shot_seed}  MSE {m['mse']:.4f}  "
                f"R2 {m['r2']:.4f}  cov95_err {m['cov95_err']:.3f}"
            )
    kernel.set_shots(None)  # restore exact evaluation

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_FILE}.")

    # 3. MSE vs shots: mean +/- std over shot seeds, with the analytic reference.
    xs = sorted(df["shots"].unique())
    grouped = df.groupby("shots")
    mse_by_shot = [grouped.get_group(n)["mse"].to_numpy() for n in xs]
    set_plot_style()
    plot_study_curve(
        xs,
        mse_by_shot,
        xlabel="measurement shots",
        ylabel="Test MSE",
        save_path="study_shots.png",
        logx=True,
        reference=analytic_mse,
        reference_label="analytic (exact)",
    )
    print("Saved study_shots.png")


if __name__ == "__main__":
    main()
