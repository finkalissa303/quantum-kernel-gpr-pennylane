"""Training and evaluation loops for the quantum-kernel GP."""

import gpytorch
import torch
from gpytorch.metrics import (
    mean_standardized_log_loss,
    negative_log_predictive_density,
    quantile_coverage_error,
)


def train_model(model, likelihood, kernel, X, y, lr, epochs):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    initial_params = kernel.params.detach().clone()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        nll = mll(output, y)
        # ExactMarginalLogLikelihood returns a scalar Tensor (the gpytorch
        # annotation is a wider union, so narrow it for the type checker).
        assert isinstance(nll, torch.Tensor)
        loss = -nll
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch:3d} - Loss {loss.item():.4f}")

    return losses, initial_params


def evaluate_model(model, likelihood, X_test, y_test):
    """Predict on X_test and score the fit with point and probabilistic metrics.

    Returns (mean, var, metrics). All metrics are computed on the *predictive*
    distribution likelihood(model(X_test)), i.e. with observation noise folded
    in, matching the band drawn by plot_regression. The probabilistic metrics
    (nlpd, msll, cov95_err) reward calibrated uncertainty, not just a good mean
    -- the reason to run a GP at all. Coverage against the noiseless truth is
    therefore slightly conservative; for function-space calibration, score the
    latent model(X_test) instead of the noise-inflated predictive.

    metrics keys:
      mse       mean squared error of the posterior mean (lower better)
      r2        coefficient of determination, 1 - SS_res/SS_tot (1.0 best)
      nlpd      negative log predictive density (proper scoring, lower better)
      msll      mean standardized log loss vs a trivial Gaussian (<0 beats it)
      cov95_err |empirical - nominal| coverage of the 95% interval (0 = ideal)
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Wrap the model output in the likelihood so the predictive variance
        # includes observation noise, not just the latent posterior variance.
        pred = likelihood(model(X_test))

    mean = pred.mean
    var = pred.variance
    ss_res = torch.sum((mean - y_test) ** 2)
    ss_tot = torch.sum((y_test - y_test.mean()) ** 2)
    metrics = {
        "mse": torch.mean((mean - y_test) ** 2).item(),
        "r2": (1.0 - ss_res / ss_tot).item(),
        "nlpd": negative_log_predictive_density(pred, y_test).item(),
        "msll": mean_standardized_log_loss(pred, y_test).item(),
        "cov95_err": quantile_coverage_error(pred, y_test, quantile=95).item(),
    }

    return mean, var, metrics
