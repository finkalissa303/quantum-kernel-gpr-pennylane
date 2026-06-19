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
uv run python run.py --qubits 3 --layers 3 --epochs 100 --seed 23  # Test MSE 0.0023
```

Every run also reports R², NLPD, MSLL, and 95%-interval coverage alongside MSE —
a GP is only as good as its calibrated uncertainty, not just its mean.

## Sweeps and studies

The parameter-init seed alone swings test MSE ~37× ([RESULTS.md](RESULTS.md)), so
single-run comparisons measure seed luck. Three helpers compare *distributions*
and reproduce the paper's sample-based experiment:

```bash
uv run python sweep.py --epochs 40   # configs × seeds → median/IQR + best-of-N
uv run python study_shots.py         # freeze params, sweep measurement shots (Fig 3b)
uv run python study_samples.py       # learning curve over training-set size
```

Each writes a `*.parquet` log (and the studies a PNG). Edit the config/seed/shot
grids at the top of each script; pass `--help` for the per-run flags.

## Code overview

| file | what it does |
|---|---|
| `run.py` | end-to-end demo: data → train → evaluate → plots |
| `sweep.py` | multi-seed sweep over circuit configs → median/IQR table |
| `study_shots.py` | finite-shot (sample-based) study — the paper's Fig. 3(b) |
| `study_samples.py` | training-set-size learning curve |
| `src/feature_maps.py` | the quantum circuit (swap in your own here) |
| `src/quantum_kernel.py` | turns a feature map into a GPyTorch kernel |
| `src/experiment.py` | shared build → train → evaluate (`run_config`) |
| `src/model.py`, `src/training.py` | GP model, train/eval loops + metrics |
| `src/data.py`, `src/plotting.py`, `src/logger.py` | toy data, plots, parquet log |

## Development

```bash
uv run ruff format .   # format
uv run ruff check .    # lint
uv run ty check        # type-check
```
