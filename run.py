"""End-to-end demo: train the quantum-kernel GP on sin(pi x) and plot results.

Run from the repository root:

    python run.py

Hyperparameters are CLI flags (see --help); the defaults reproduce the README
run. Writes loss.png, regression.png, and an experiments_scratch.parquet log.
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")  # headless backend; drop this line for interactive plots

import gpytorch
import torch

from src.config import FINAL_FILE, SAVE_MODE, SCRATCH_FILE
from src.data import func, make_sine_data
from src.feature_maps import ChebyshevFeatureMap
from src.logger import build_experiment_row, save_experiment
from src.model import QGP
from src.plotting import plot_loss, plot_regression, set_plot_style
from src.quantum_kernel import QuantumKernel
from src.training import evaluate_model, train_model

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

DATA_SEED = 22
NOISE = 0.1
PHI = torch.arccos

# ---- data ----------------------------------------------------------------
torch.manual_seed(DATA_SEED)
X_train, y_train = make_sine_data(n_points=30, noise=NOISE)
X_test = torch.linspace(-1.0, 1.0, 50)
y_test = func(X_test)

# Re-seed only for non-default init seeds: the default run keeps consuming the
# data-draw RNG stream and so reproduces the README numbers bit for bit.
if args.seed != DATA_SEED:
    torch.manual_seed(args.seed)

# ---- model ---------------------------------------------------------------
feature_map = ChebyshevFeatureMap(
    n_qubits=args.qubits, n_layers=args.layers, phi=PHI, scale=args.scale
)
kernel = QuantumKernel(feature_map)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = QGP(X_train, y_train, likelihood, kernel)

# ---- train ---------------------------------------------------------------
t0 = time.time()
losses, initial_params = train_model(
    model, likelihood, kernel, X_train, y_train, lr=args.lr, epochs=args.epochs
)
runtime = time.time() - t0

# ---- evaluate ------------------------------------------------------------
pred_mean, pred_var, mse = evaluate_model(model, likelihood, X_test, y_test)
print(f"\nTest MSE: {mse:.4f}  (constant-mean baseline ~0.49; trained model ~0.1)")

# ---- plot ----------------------------------------------------------------
set_plot_style()
plot_loss(losses, save_path="loss.png")
plot_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    pred_mean,
    pred_var,
    save_path="regression.png",
)
print("Saved loss.png and regression.png")

# ---- log -----------------------------------------------------------------
row = build_experiment_row(
    n_qubits=args.qubits,
    n_layers=args.layers,
    kernel=kernel,
    lr=args.lr,
    epochs=args.epochs,
    N_train=X_train.shape[0],
    N_test=X_test.shape[0],
    noise_eps=NOISE,
    phi=PHI,
    scale=args.scale,
    mse=mse,
    runtime_sec=runtime,
    likelihood=likelihood,
    losses=losses,
    initial_params=initial_params,
    seed=args.seed,
)
save_experiment(row, SAVE_MODE, SCRATCH_FILE, FINAL_FILE)
