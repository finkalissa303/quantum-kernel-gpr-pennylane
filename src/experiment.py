"""Shared experiment runner: build the quantum-kernel GP, train it, evaluate it.

run.py and the study scripts (sweep.py, study_shots.py, study_samples.py) all go
through `run_config`, so the data draw, parameter init, and seeding logic live in
exactly one place. The seeding matches the original run.py: the data is always
drawn from `data_seed`, and the parameter init reuses that same RNG stream when
seed == data_seed (so the default run stays bit-for-bit reproducible) or a fresh
stream seeded by `seed` otherwise.
"""

import time
from dataclasses import dataclass

import gpytorch
import torch

from src.data import func, make_sine_data
from src.feature_maps import ChebyshevFeatureMap
from src.logger import build_experiment_row
from src.model import QGP
from src.quantum_kernel import QuantumKernel
from src.training import evaluate_model, train_model


@dataclass
class ExperimentResult:
    """Everything one training run produces: the trained objects, the data and
    settings it used, and the predictions/metrics on the test grid."""

    model: QGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    kernel: QuantumKernel
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    pred_mean: torch.Tensor
    pred_var: torch.Tensor
    metrics: dict[str, float]
    losses: list[float]
    initial_params: torch.Tensor
    runtime_sec: float
    lr: float
    epochs: int
    seed: int
    noise: float


def run_config(
    *,
    n_qubits=2,
    n_layers=3,
    scale=2.0,
    lr=0.1,
    epochs=40,
    seed=22,
    data_seed=22,
    noise=0.1,
    n_train=30,
    n_test=50,
    phi=torch.arccos,
    eig_cutoff=False,
):
    """Build, train (analytically), and evaluate one quantum-kernel GP.

    Returns an ExperimentResult. For a finite-shot evaluation, call
    `result.kernel.set_shots(...)` afterwards and re-evaluate on a fresh QGP
    (see study_shots.py) -- training itself always uses the exact kernel.
    """
    # Data is always drawn from data_seed so every config/seed trains on the
    # same points; only the parameter init below varies with `seed`.
    torch.manual_seed(data_seed)
    X_train, y_train = make_sine_data(n_points=n_train, noise=noise)
    X_test = torch.linspace(-1.0, 1.0, n_test)
    y_test = func(X_test)

    # Re-seed only for non-default inits: seed == data_seed keeps consuming the
    # data-draw RNG stream, reproducing the README numbers bit for bit.
    if seed != data_seed:
        torch.manual_seed(seed)

    feature_map = ChebyshevFeatureMap(
        n_qubits=n_qubits, n_layers=n_layers, phi=phi, scale=scale
    )
    kernel = QuantumKernel(feature_map, eig_cutoff=eig_cutoff)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = QGP(X_train, y_train, likelihood, kernel)

    t0 = time.time()
    losses, initial_params = train_model(
        model, likelihood, kernel, X_train, y_train, lr=lr, epochs=epochs
    )
    runtime = time.time() - t0

    pred_mean, pred_var, metrics = evaluate_model(model, likelihood, X_test, y_test)

    return ExperimentResult(
        model=model,
        likelihood=likelihood,
        kernel=kernel,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pred_mean=pred_mean,
        pred_var=pred_var,
        metrics=metrics,
        losses=losses,
        initial_params=initial_params,
        runtime_sec=runtime,
        lr=lr,
        epochs=epochs,
        seed=seed,
        noise=noise,
    )


def result_row(result):
    """Build a parquet log row from a finished run, derived entirely from the
    ExperimentResult (the feature map carries qubits/layers/scale/phi). Shared
    by run.py, sweep.py, and study_samples.py."""
    fm = result.kernel.feature_map
    return build_experiment_row(
        n_qubits=fm.n_qubits,
        n_layers=fm.n_layers,
        kernel=result.kernel,
        lr=result.lr,
        epochs=result.epochs,
        N_train=result.X_train.shape[0],
        N_test=result.X_test.shape[0],
        noise_eps=result.noise,
        phi=fm.phi,
        scale=fm.scale,
        metrics=result.metrics,
        runtime_sec=result.runtime_sec,
        likelihood=result.likelihood,
        losses=result.losses,
        initial_params=result.initial_params,
        seed=result.seed,
    )
