# -*- coding: utf-8 -*-
"""Exact GP model with a constant mean and a (quantum) covariance kernel."""

import gpytorch


class QGP(gpytorch.models.ExactGP):
    """Exact Gaussian Process whose covariance is supplied by `kernel`."""

    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        # Describes the GP prior/posterior distribution for inputs x.
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
