# Quantum-Kernel Gaussian Process Regression

Gaussian Process Regression with a **trainable quantum kernel**, built with
PennyLane and GPyTorch. A parameterized quantum circuit encodes each data point
into a quantum state, and the kernel value of two points is the fidelity
(overlap) of their states: `k(x, x') = |⟨φ(x')|φ(x)⟩|²`. The circuit parameters
are trained by maximizing the GP marginal log-likelihood. As a demo, it learns
the toy function `f(x) = sin(πx)` from 30 noisy samples.

Inspired by Rapp & Roth, [arXiv:2304.12923](https://arxiv.org/abs/2304.12923)
(the paper is included in the repo).

## Requirements

- [uv](https://docs.astral.sh/uv/) — handles Python (3.12) and all
  dependencies for you
- No GPU needed: everything runs on CPU, on Linux, Windows, or macOS

## Run it

```bash
uv sync               # one-time: create the environment
uv run python run.py  # train + evaluate (~3 min on a laptop)
```

You should see the training loss fall from ~1.3 to ~0.26 and end with:

```
Test MSE: 0.0997
Saved loss.png and regression.png
```

`regression.png` shows the GP fit against the true function. Hyperparameters
are CLI flags — `--qubits --layers --epochs --lr --scale --seed` (defaults
reproduce the run above; keep `--lr` near `0.1`, the original `0.001` does
not learn). Best configuration so far ([RESULTS.md](RESULTS.md)):

```bash
uv run python run.py --qubits 3 --layers 3 --epochs 100  # Test MSE 0.0092
```

## Code overview

| file | what it does |
|---|---|
| `run.py` | end-to-end demo: data → train → evaluate → plots |
| `src/feature_maps.py` | the quantum circuit (swap in your own here) |
| `src/quantum_kernel.py` | turns a feature map into a GPyTorch kernel |
| `src/model.py`, `src/training.py` | GP model, train/eval loops |
| `src/data.py`, `src/plotting.py`, `src/logger.py` | toy data, plots, parquet log |

## Development

```bash
uv run ruff format .   # format
uv run ruff check .    # lint
uv run ty check        # type-check
```
