"""Plotting helpers: training loss, kernel matrix, and regression results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from cycler import cycler

# ======================================================
# GLOBAL STYLE (optional but recommended)
# ======================================================


def set_plot_style(use_latex=False):
    # use_latex=True requires a working LaTeX install (texlive + dvipng);
    # it is off by default so plots render in a bare environment.
    plt.rcParams.update(
        {
            "axes.prop_cycle": cycler("color", sns.color_palette("viridis", 5)),
            "figure.figsize": (5.0, 4.0),
            "font.family": "serif",
            "font.size": 15,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "text.usetex": use_latex,
            "mathtext.fontset": "dejavuserif",
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
            "axes.linewidth": 0.6,
            "lines.linewidth": 2.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.dpi": 300,
        }
    )


# ======================================================
# 1. LOSS PLOT
# ======================================================


def plot_loss(losses, save_path=None):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# ======================================================
# 2. KERNEL MATRIX VISUALIZATION
# ======================================================


def plot_kernel_matrix(K, title="Kernel Matrix", save_path=None):
    plt.figure()
    plt.imshow(K.detach().cpu().numpy(), aspect="auto")
    plt.colorbar()
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# ======================================================
# 3. REGRESSION RESULT (MAIN PLOT)
# ======================================================


def plot_regression(
    X_train, y_train, X_test, y_test, pred_mean, pred_var, save_path=None
):
    X_train = X_train.detach().cpu()
    y_train = y_train.detach().cpu()

    X_test = X_test.detach().cpu()
    y_test = y_test.detach().cpu()

    mean = pred_mean.detach().cpu()
    std = torch.sqrt(pred_var.detach().cpu())

    lower = mean - std
    upper = mean + std

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    ax.plot(X_train.numpy(), y_train.numpy(), "k.", label="Samples", markersize=8.0)

    ax.plot(X_test.numpy(), mean.numpy(), label="Mean")

    ax.plot(X_test.numpy(), y_test.numpy(), color="k", label="True")

    ax.fill_between(
        X_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label="Variance"
    )

    ax.plot(X_test.numpy(), lower.numpy(), "k--", linewidth=1.5)

    ax.plot(X_test.numpy(), upper.numpy(), "k--", linewidth=1.5)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.5, 1.5)

    ax.set_xlabel(r"$\mathbf{x}$")
    ax.set_ylabel(r"$f(\mathbf{x})$")

    ax.legend(loc="best")
    ax.grid()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# ======================================================
# 4. STUDY CURVE (metric vs a swept axis, band over seeds)
# ======================================================


def plot_study_curve(
    x,
    values,
    xlabel,
    ylabel,
    save_path=None,
    logx=False,
    reference=None,
    reference_label="reference",
):
    """Plot a metric against a swept axis with a mean +/- std band over repeats.

    Parameters
    ----------
    x : sequence
        The swept values (e.g. shot counts, or training-set sizes).
    values : array-like, shape (len(x),) or (len(x), n_repeats)
        The metric at each x; the second axis (if present) holds repeats over
        seeds, drawn as a +/-1 std band around the mean.
    logx : bool
        Log-scale the x axis -- natural for a shot-count sweep.
    reference : float, optional
        A horizontal baseline (e.g. the analytic or classical-RBF value).
    """
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    mean = values.mean(axis=1)
    std = values.std(axis=1)

    plt.figure()
    plt.plot(x, mean, marker="o", label="mean over seeds")
    plt.fill_between(x, mean - std, mean + std, alpha=0.3, label=r"$\pm$1 std")
    if reference is not None:
        plt.axhline(
            reference, color="k", linestyle="--", linewidth=1.5, label=reference_label
        )
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# ======================================================
# 5. PREDICTION UNCERTAINTY ONLY (OPTIONAL DEBUG TOOL)
# ======================================================


def plot_uncertainty(X, mean, var):
    X = X.detach().cpu()
    mean = mean.detach().cpu()
    std = torch.sqrt(var.detach().cpu())

    plt.figure()
    plt.plot(X, mean, label="Mean")
    plt.fill_between(X.numpy(), (mean - std).numpy(), (mean + std).numpy(), alpha=0.3)
    plt.legend()
    plt.title("Uncertainty")
    plt.show()


# ======================================================
# 6. PARAMETER / DIAGNOSTIC PLOTS (OPTIONAL)
# ======================================================


def plot_losses_and_gradients(losses, grad_norms=None):
    plt.figure()
    plt.plot(losses, label="Loss")

    if grad_norms is not None:
        plt.plot(grad_norms, label="Grad norm")

    plt.legend()
    plt.title("Training Diagnostics")
    plt.show()
