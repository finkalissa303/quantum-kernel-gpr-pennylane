# Hyperparameter sweep results

Date: 2026-06-09. All runs use the CLI version of `run.py` (flags listed per
run below), on the same task: learn `f(x) = sin(πx)` from 30 noisy samples
(data seed 22, noise std 0.1), Adam with lr 0.1, test MSE measured against the
*noiseless* true function on a 50-point grid.

## Yardsticks

| reference | Test MSE | final loss (−MLL/n) |
|---|---|---|
| Constant-mean baseline (predict 0) | ~0.49 | — |
| Classical RBF GP, same data and protocol, 40 epochs | 0.0027 | −0.407 |
| Theoretical noise floor (perfect fit, σ² = 0.01) | 0 | ≈ −0.884 |

The RBF control (GPyTorch `ScaleKernel(RBFKernel)`) trains in under a second
and learns lengthscale 0.50, outputscale 0.74, noise variance 0.018.

## Quantum kernel runs

All sweep runs: 100 epochs, single-threaded. The original baseline is the
40-epoch README run. One run per configuration (single seed, 22).

| config | command | params | Test MSE | final loss | learned noise | ‖Δθ‖ | runtime |
|---|---|---|---|---|---|---|---|
| 2q / 3l, scale 2.0, 40 ep (baseline) | `python run.py` | 8 | 0.0997 | +0.261 | 0.0153 | 3.8 | 3.5 min |
| 2q / 3l, scale 2.0 | `python run.py --epochs 100` | 8 | 0.0835 | −0.197 | 0.0027 | 4.2 | 12 min |
| 2q / 3l, scale 1.0 | `python run.py --epochs 100 --scale 1.0` | 8 | 0.0323 | −0.299 | 0.0038 | 4.5 | 12 min |
| 3q / 2l, scale 2.0 | `python run.py --epochs 100 --qubits 3 --layers 2` | 9 | 0.0923 | −0.410 | 0.0040 | 2.0 | 13 min |
| **3q / 3l, scale 2.0** | `python run.py --epochs 100 --qubits 3 --layers 3` | 12 | **0.0092** | **−0.682** | 0.0037 | 6.9 | 17 min |

Each command regenerates the corresponding `loss.png` / `regression.png` and
appends a row to the parquet log.

## Findings

1. **3 qubits × 3 layers is the clear winner: MSE 0.0092** — 11× better than
   the 40-epoch baseline and within 3.4× of the classical RBF. Its final
   marginal likelihood (−0.682) *beats* the RBF control (−0.407) and reaches
   two thirds of the way to the noise floor. The loss curve plateaus from
   epoch ~60, so the run is converged, and the regression plot shows the mean
   tracking the true sine closely with a tight variance band.
2. **Depth and width only pay together.** Adding a qubit alone (3q/2l) did not
   improve MSE over 2q/3l, and its parameters moved only half as far in norm —
   it settled into a good-likelihood / mediocre-mean optimum. Qubits and
   layers together (3q/3l) transformed the fit.
3. **Lower `scale` smooths the kernel, but capacity supersedes it.** At
   2 qubits, halving the feature-map scale to 1.0 cut MSE 3× (0.0835 → 0.0323)
   by taming the high-frequency ringing in the posterior mean. The 3q/3l run
   beat it anyway at scale 2.0: with enough capacity the optimizer controls
   the frequency content on its own.
4. **More epochs alone helped likelihood much more than fit.** Extending the
   baseline from 40 to 100 epochs moved the loss from +0.26 to −0.20 but MSE
   only from 0.0997 to 0.0835 — undertraining was real but was not the main
   bottleneck.

## Caveats and open questions

- Every configuration was trained from a single parameter init (seed 22).
  These landscapes are multimodal, so the 3q/3l result needs a seed sweep
  (e.g. `--seed 7 / 23 / 42`, identical data by construction) before it can
  be called robust.
- Learned noise variances (0.0027–0.0040) fall below the true 0.01 in all
  100-epoch runs — a mild sign of the kernel absorbing noise as signal.
- Untested: 3q/3l combined with scale 1.0, and wrapping the kernel in
  `gpytorch.kernels.ScaleKernel` to free the prior amplitude (the RBF control
  chose outputscale 0.74, suggesting unit amplitude is not optimal).
