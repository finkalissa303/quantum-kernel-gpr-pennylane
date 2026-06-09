"""End-to-end demo: train the quantum-kernel GP on sin(pi x) and plot results.

Run from the repository root:

    python run.py

Writes loss.png, regression.png, and an experiments_scratch.parquet log.
"""

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
SEED = 22
N_QUBITS = 2
N_LAYERS = 3
LR = 0.1  # the original 0.001 is ~100x too small and fails to learn
EPOCHS = 40
NOISE = 0.1
PHI = torch.arccos

torch.manual_seed(SEED)

# ---- data ----------------------------------------------------------------
X_train, y_train = make_sine_data(n_points=30, noise=NOISE)
X_test = torch.linspace(-1.0, 1.0, 50)
y_test = func(X_test)

# ---- model ---------------------------------------------------------------
feature_map = ChebyshevFeatureMap(n_qubits=N_QUBITS, n_layers=N_LAYERS, phi=PHI)
kernel = QuantumKernel(feature_map)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = QGP(X_train, y_train, likelihood, kernel)

# ---- train ---------------------------------------------------------------
t0 = time.time()
losses, initial_params = train_model(
    model, likelihood, kernel, X_train, y_train, lr=LR, epochs=EPOCHS
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
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    kernel=kernel,
    lr=LR,
    epochs=EPOCHS,
    N_train=X_train.shape[0],
    N_test=X_test.shape[0],
    noise_eps=NOISE,
    phi=PHI,
    mse=mse,
    runtime_sec=runtime,
    likelihood=likelihood,
    losses=losses,
    initial_params=initial_params,
    seed=SEED,
)
save_experiment(row, SAVE_MODE, SCRATCH_FILE, FINAL_FILE)
