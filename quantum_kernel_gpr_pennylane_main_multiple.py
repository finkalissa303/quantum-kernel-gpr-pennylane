#!/usr/bin/env python3

from pathlib import Path
import time

import torch
import gpytorch
import matplotlib.pyplot as plt

from src.data import make_sine_data, func
from src.quantum_kernel import QuantumKernel
from src.model import QGP
from src.training import train_model, evaluate_model
from src.plotting import plot_regression
from src.logger import build_experiment_row, save_experiment


def main():
    # =========================
    # CONFIG HYPERPARAMETERS
    # =========================

    SEED = 22
    torch.manual_seed(SEED)

    lr = 0.01
    epochs = 50
    eps = 0.1
    phi = torch.arccos

    QUBITS_LIST = [2,3]
    LAYERS_LIST = [2, 3]
    SCALES_LIST = [0.5,1.0,2.0,3.0]

    print("Torch:", torch.__version__)
    print("GPyTorch imported OK")

    print("Seed set:", SEED)
    print(torch.rand(3))
    print("Config loaded:")
    print(f"lr={lr}, epochs={epochs}, eps={eps}")

    # =========================
    # DATA
    # =========================

    X_train, y_train = make_sine_data(
        n_points=30,
        noise=eps,
    )

    X_test = torch.linspace(-1.0, 1.0, 50)
    y_test = func(X_test)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("X_train sample:", X_train[:5])
    print("y_train sample:", y_train[:5])

    plt.scatter(X_train, y_train)
    plt.title("Training Data Sanity Check")
    plt.show()

    print(X_test[:10])

    # =========================
    # PATHS
    # =========================

    PROJECT_DIR = Path(__file__).resolve().parent

    SAVE_MODE = "scratch"

    EXPERIMENT_DIR = PROJECT_DIR / "experiments"
    PLOT_DIR = PROJECT_DIR / "plots"

    SCRATCH_FILE = EXPERIMENT_DIR / "scratch.parquet"
    FINAL_FILE = EXPERIMENT_DIR / "final.parquet"

    EXPERIMENT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    # =========================
    # EXPERIMENT LOOP
    # =========================

    for n_qubits in QUBITS_LIST:

        for n_layers in LAYERS_LIST:

            for encoding_scale in SCALES_LIST:

                print("\n==========================")
                print(
                    f"Q={n_qubits}, "
                    f"L={n_layers}, "
                    f"S={encoding_scale}"
                )
                print("==========================")

                # -------------------------
                # MODEL
                # -------------------------

                kernel = QuantumKernel(
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    phi=phi,
                    encoding_scale=encoding_scale,
                )

                likelihood = gpytorch.likelihoods.GaussianLikelihood()

                model = QGP(
                    X_train,
                    y_train,
                    likelihood,
                    kernel,
                )

                # -------------------------
                # TRAINING
                # -------------------------

                start_time = time.time()

                losses, initial_params = train_model(
                    model=model,
                    likelihood=likelihood,
                    kernel=kernel,
                    X=X_train,
                    y=y_train,
                    lr=lr,
                    epochs=epochs,
                )

                runtime_sec = time.time() - start_time

                # -------------------------
                # EVALUATION
                # -------------------------

                pred_mean, pred_var, mse = evaluate_model(
                    model=model,
                    likelihood=likelihood,
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                )

                print(f"MSE = {mse:.5f}")

                # -------------------------
                # SAVE REGRESSION PLOT
                # -------------------------

                plot_file = (
                    PLOT_DIR
                    / f"q{n_qubits}_l{n_layers}_s{encoding_scale}.png"
                )

                plot_regression(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    pred_mean,
                    pred_var,
                    save_path=str(plot_file),
                )

                # -------------------------
                # LOGGING
                # -------------------------

                row = build_experiment_row(
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    encoding_scale=encoding_scale,
                    kernel=kernel,
                    lr=lr,
                    epochs=epochs,
                    N_train=X_train.shape[0],
                    N_test=X_test.shape[0],
                    noise_eps=eps,
                    phi=phi,
                    mse=mse,
                    runtime_sec=runtime_sec,
                    likelihood=likelihood,
                    losses=losses,
                    initial_params=initial_params,
                    seed=SEED,
                )

                save_experiment(
                    row=row,
                    save_mode=SAVE_MODE,
                    scratch_file=str(SCRATCH_FILE),
                    final_file=str(FINAL_FILE),
                )

                print("Saved.")


if __name__ == "__main__":
    main()