"""End-to-end demo: train the quantum-kernel GP on sin(pi x) and plot results.

Run from the repository root:

    python run.py

Hyperparameters are CLI flags (see --help); the defaults reproduce the README
run. Writes loss.png, regression.png, and an experiments_scratch.parquet log.
For multi-seed sweeps and the sample-based studies see sweep.py / study_*.py.
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless backend; drop this line for interactive plots

import torch

from src.config import FINAL_FILE, SAVE_MODE, SCRATCH_FILE
from src.experiment import result_row, run_config
from src.logger import save_experiment
from src.plotting import plot_loss, plot_regression, set_plot_style

# ---- configuration -------------------------------------------------------
parser = argparse.ArgumentParser(description="Train the quantum-kernel GP demo.")
parser.add_argument("--qubits", type=int, default=2, help="circuit width (minimum 2)")
parser.add_argument("--layers", type=int, default=3, help="entangling layers")
parser.add_argument(
    "--lr",
    type=float,
    default=0.1,
    help="Adam learning rate (the original 0.001 is ~100x too small to learn)",
)
parser.add_argument("--epochs", type=int, default=40, help="training epochs")
parser.add_argument(
    "--scale",
    type=float,
    default=2.0,
    help="feature-map angle multiplier; lower means a smoother kernel",
)
parser.add_argument(
    "--seed",
    type=int,
    default=22,
    help="parameter-init seed; the data draw always uses seed 22, so "
    "restarts with different seeds train on identical data",
)
args = parser.parse_args()

NOISE = 0.1
PHI = torch.arccos

# ---- build, train, evaluate ----------------------------------------------
result = run_config(
    n_qubits=args.qubits,
    n_layers=args.layers,
    scale=args.scale,
    lr=args.lr,
    epochs=args.epochs,
    seed=args.seed,
    noise=NOISE,
    phi=PHI,
)
m = result.metrics
print(
    f"\nTest MSE {m['mse']:.4f} | R2 {m['r2']:.4f} | NLPD {m['nlpd']:.3f} | "
    f"MSLL {m['msll']:.3f} | 95% coverage err {m['cov95_err']:.3f}"
)
print("(MSE yardsticks: constant-mean baseline ~0.49, classical RBF ~0.003)")

# ---- plot ----------------------------------------------------------------
set_plot_style()
plot_loss(result.losses, save_path="loss.png")
plot_regression(
    result.X_train,
    result.y_train,
    result.X_test,
    result.y_test,
    result.pred_mean,
    result.pred_var,
    save_path="regression.png",
)
print("Saved loss.png and regression.png")

# ---- log -----------------------------------------------------------------
save_experiment(result_row(result), SAVE_MODE, SCRATCH_FILE, FINAL_FILE)
