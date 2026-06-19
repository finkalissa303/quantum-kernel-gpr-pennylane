"""Append experiment results to a parquet log."""

import os
from datetime import datetime

import pandas as pd
import torch

# ======================================================
# PARQUET APPENDER
# ======================================================


def append_parquet(row, parquet_file):
    new_df = pd.DataFrame([row])

    if os.path.exists(parquet_file):
        old_df = pd.read_parquet(parquet_file)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_parquet(parquet_file, index=False)


# ======================================================
# BUILD ROW
# ======================================================


def build_experiment_row(
    n_qubits,
    n_layers,
    kernel,
    lr,
    epochs,
    N_train,
    N_test,
    noise_eps,
    phi,
    scale,
    metrics,
    runtime_sec,
    likelihood,
    losses,
    initial_params,
    seed=22,
):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": kernel.n_params,
        "learning_rate": lr,
        "epochs": epochs,
        "N_train": N_train,
        "N_test": N_test,
        "noise_eps": noise_eps,
        "phi": str(phi.__name__) if hasattr(phi, "__name__") else str(phi),
        "scale": float(scale),
        "mse": float(metrics["mse"]),
        "r2": float(metrics["r2"]),
        "nlpd": float(metrics["nlpd"]),
        "msll": float(metrics["msll"]),
        "cov95_err": float(metrics["cov95_err"]),
        "runtime_sec": runtime_sec,
        "likelihood_noise": float(likelihood.noise.item()),
        "final_loss": float(losses[-1]),
        "parameter_movement": float(
            torch.norm(kernel.params.detach() - initial_params).item()
        ),
        "seed": seed,
    }


# ======================================================
# SAVE RESULT
# ======================================================


def save_experiment(row, save_mode, scratch_file, final_file):
    if save_mode == "scratch":
        append_parquet(row, scratch_file)
        print("Saved to scratch database")

    elif save_mode == "final":
        append_parquet(row, final_file)
        print("Saved to final database")

    else:
        raise ValueError(f"Unknown SAVE_MODE: {save_mode}")
