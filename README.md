# Quantum-Kernel Gaussian Process Regression

A small PennyLane + GPyTorch framework for **Gaussian Process Regression with a
trainable quantum fidelity kernel**. The kernel entries are quantum state
fidelities `k(x, x') = |⟨φ(x')|φ(x)⟩|²`, and the feature-map parameters are
learned end-to-end by maximizing the GP marginal log-likelihood.

It is **inspired by** Rapp & Roth, *"Quantum Gaussian Process Regression for
Bayesian Optimization"* ([arXiv:2304.12923](https://arxiv.org/abs/2304.12923))
— the fidelity kernel (Eqs. 6–8), the Chebyshev-style `arccos` feature map
(Fig. 2), and MLL training (Eq. 9). It is **not** a reproduction of the paper's
experiments: this repo regresses the simpler `f(x) = sin(πx)` on `[-1, 1]`
rather than the paper's `f(x) = x·sin(x)`, and the paper's eigenvalue-cutoff
Gram-matrix regularization is provided as an *option* (`eig_cutoff`) rather than
on by default.

## Install

This project uses [uv](https://docs.astral.sh/uv/); Python is pinned to 3.12.

```bash
uv sync
```

That creates `.venv/` with the exact locked dependencies, including the CPU
build of PyTorch (`torch`, `gpytorch 1.15.2`, `pennylane 0.45`).

## Run

```bash
uv run python run.py
```

Trains the GP, prints the test MSE, and writes `loss.png` + `regression.png`.
Expect **MSE ≈ 0.1** (vs a constant-mean baseline of ≈ 0.49).

> ⚠️ **Learning rate matters.** The original notebooks shipped `lr = 0.001`,
> which is ~100× too small — the loss barely moves and the model scores *worse*
> than predicting the mean (MSE ≈ 0.59). `run.py` uses `lr = 0.1`.

## Development

Lint, format, and type-check with the Astral toolchain (installed by `uv sync`):

```bash
uv run ruff format .     # format
uv run ruff check .      # lint (add --fix to autofix)
uv run ty check          # type-check
```

## Project layout

```
run.py                         # recommended entry point (headless)
pyproject.toml                 # project metadata, deps, ruff + ty config
uv.lock                        # pinned dependency lockfile
src/
  data.py                      # sin(πx) toy dataset
  feature_maps.py              # FeatureMap interface + ChebyshevFeatureMap (the circuit)
  quantum_kernel.py            # QuantumKernel: fidelity-kernel machinery (circuit-agnostic)
  model.py                     # QGP: ExactGP with constant mean
  training.py                  # train_model / evaluate_model
  plotting.py                  # loss, kernel-matrix, regression plots
  logger.py                    # append experiment rows to parquet
  config.py                    # experiment-log paths (override via $PROJECT_DIR)
quantum_kernel_gpr_pennylane.ipynb        # from-scratch tutorial notebook
quantum_kernel_gpr_pennylane_main.ipynb   # notebook driver (run.py is the cleaner path)
2304.12923v1-2.pdf             # reference paper
```

## Key hyperparameters (`run.py`)

| name        | value          | notes                                           |
|-------------|----------------|-------------------------------------------------|
| `n_qubits`  | 2              | more qubits → more expressive (and slower)      |
| `n_layers`  | 3              | repeated RX-encoding + CRZ-entangler blocks     |
| `lr`        | 0.1            | **do not lower to 0.001** (won't learn)         |
| `epochs`    | 40             | ~converged on this toy task                     |
| `phi`       | `torch.arccos` | domain `[-1, 1]`; inputs must be scaled         |

## Notes & limitations

- **Cost:** the kernel is evaluated pair-by-pair (`O(N²)` circuit calls), so
  training is slow even on this toy problem.
- **Expressivity:** 2 qubits is minimal; quantum fidelity kernels can suffer
  from *exponential concentration* as qubit count grows (paper, ref. [44]).
- **`eig_cutoff`:** enable `QuantumKernel(..., eig_cutoff=True)` to project the
  Gram matrix onto the PSD cone (paper's regularization); mainly useful for
  (near-)noiseless targets.
- **Pluggable circuits:** the feature map is a separate component. Subclass
  `FeatureMap` (or reparameterize `ChebyshevFeatureMap`) and pass it to
  `QuantumKernel(feature_map)`; the kernel reads `n_params` from the feature map,
  so a new circuit needs no kernel changes.
