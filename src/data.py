"""Toy 1-D regression dataset: f(x) = sin(pi * x) on [-1, 1] with noise."""

import torch


def func(x):
    return torch.sin(torch.pi * x)


def make_sine_data(n_points=30, noise=0.1, seed=None):
    """Sample `n_points` noisy observations of f(x) = sin(pi * x) on [-1, 1].

    Pass `seed` to make the draw reproducible without relying on the caller
    having set the global RNG state beforehand."""
    if seed is not None:
        torch.manual_seed(seed)
    X = torch.rand(n_points) * 2 - 1
    y = func(X) + noise * torch.randn_like(X)
    return X, y
