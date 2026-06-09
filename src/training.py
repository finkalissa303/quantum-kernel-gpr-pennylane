# -*- coding: utf-8 -*-
"""Training and evaluation loops for the quantum-kernel GP."""

import torch
import gpytorch


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
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch:3d} - Loss {loss.item():.4f}")

    return losses, initial_params


def evaluate_model(model, likelihood, X_train, X_test, y_test):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Wrap the model output in the likelihood so the predictive variance
        # includes observation noise, not just the latent posterior variance.
        pred = likelihood(model(X_test))

    mean = pred.mean
    var = pred.variance
    mse = torch.mean((mean - y_test) ** 2).item()

    return mean, var, mse
