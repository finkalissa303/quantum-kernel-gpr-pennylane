# Studies: exploring circuit components and sample-based evaluation

This document answers two questions about the quantum-kernel GP and documents the
tooling added to support them:

1. How can I test the best combinations of circuit components more rigorously,
   and should I look at metrics other than MSE?
2. How can I run a sample-based study like Figure 3(b) of Rapp & Roth
   ([arXiv:2304.12923](https://arxiv.org/abs/2304.12923))?

---

## Question 1 — testing circuit combinations, and looking beyond MSE

### The blocker: seed variance dominates single-run comparisons

[RESULTS.md](RESULTS.md) found a **37× MSE spread (0.0023–0.0847)** across four
parameter-init seeds of a *single* configuration. Any comparison between configs
run at one seed each is therefore measuring seed luck, not the circuit. The first
fix is methodological, not more configs:

- **Run every config across several seeds** and compare the *distribution*
  (median + inter-quartile range), never a single number.
- **Select the production model with best-of-N restarts ranked by training loss**
  (marginal likelihood), with no test-set peeking. RESULTS.md found the final
  loss ranks restarts almost perfectly, so this is the protocol this model needs.

`sweep.py` implements both: it loops a config grid × seeds, logs every run, and
prints a per-config median/IQR table plus the best restart selected by training
loss.

```bash
uv run python sweep.py --epochs 40   # quick screen
uv run python sweep.py               # full 100-epoch sweep
```

Edit the `CONFIGS` and `SEEDS` lists at the top. **Cost matters**: the fidelity
kernel is O(N²) circuit evaluations per epoch, so a 100-epoch run is 12–17 min.
Screen at fewer epochs, then confirm the top configs at full budget × seeds.

### Which circuit components to vary

The sweeps so far covered qubits, layers, and `scale`. The untested axes — and
`src/feature_maps.py` is built to be swapped here — are where new signal lives:

| Component | Currently | Worth trying |
|---|---|---|
| Entangler | CRZ ring | CNOT ring, all-to-all, none (ablation) |
| Encoding `phi` | `arccos` (Chebyshev) | identity, scaled variants |
| Parameter sharing | RX reuses RY angles | independent RX params (more capacity) |
| Prior amplitude | fixed (fidelity ∈ [0, 1]) | wrap in `gpytorch.kernels.ScaleKernel` |

### Beyond MSE: yes — measure calibrated uncertainty

MSE only scores the posterior **mean**. But the whole point of a *Gaussian
Process* is calibrated **uncertainty**, and the paper's Fig. 3 foregrounds that
"the variance information can be retained." A model that nails the mean but
reports nonsense error bars looks perfect under MSE and is still a bad GP.

Every run now reports four metrics alongside MSE (computed on the predictive
distribution via `gpytorch.metrics`, logged to the parquet):

| Metric | What it measures | Good value |
|---|---|---|
| **MSE** | squared error of the posterior mean | lower (RBF control ~0.003) |
| **R²** | fraction of variance explained — comparable to the paper | → 1 (paper 0.996) |
| **NLPD** | negative log predictive density — a proper scoring rule | lower |
| **MSLL** | NLPD standardized against a trivial Gaussian baseline | < 0 beats the baseline |
| **95% coverage err** | \|empirical − nominal\| coverage of the 95% interval | → 0 |

These use the noise-inflated predictive distribution (matching the band
`plot_regression` draws), so coverage against the noiseless truth is slightly
conservative; for function-space calibration, score the latent `model(X_test)`
instead of the predictive.

**Worked example:** the default 40-epoch run scores MSE 0.0997 (looks acceptable)
but **95% coverage error 0.170** — its uncertainty is meaningfully miscalibrated,
which plain MSE never reveals. That gap is exactly why these metrics were added.

---

## Question 2 — a sample-based study like Figure 3(b)

### What Fig. 3(b) actually is

"Sample-based" is overloaded, so to be precise: Fig. 3(b) is a **measurement-shot
study, not a training-set-size curve.** It takes the *optimal parameters from the
ideal statevector run* (3a), **freezes them — no retraining** — and re-evaluates
the kernel from a finite number of measurements (N = 10,000 per point). The
kernel becomes a noisy estimate k̃ of k; the result is only slightly worse
(MSE 0.024 vs 0.022, R² 0.996) *provided the Gram matrix is regularized*.

### The protocol: `study_shots.py`

```bash
uv run python study_shots.py
uv run python study_shots.py --epochs 100 --qubits 3 --layers 3 --seed 23
```

1. Train **analytically** (exact kernel) and keep the optimal parameters.
2. Freeze them; switch the kernel to finite shots via `kernel.set_shots(N, seed)`,
   which rebuilds the PennyLane device so the trained params are untouched but the
   shot sampler is reseeded. (PennyLane seeds its sampler at *device construction*,
   so a global torch/numpy seed does **not** make draws reproducible — the device
   seed does.)
3. For each (N, shot-seed), build a **fresh `QGP`** so the entire Gram matrix
   (train–train included) is re-estimated under shots, then score it.
4. Sweep N ∈ {100 … 30,000} × several shot-seeds and plot MSE vs N with a
   mean ± std band and the analytic reference line.

This extends Fig. 3(b)'s single point into a curve showing how the fit and its
calibrated variance degrade as the measurement budget shrinks.

### Regularization

Finite shots make the off-diagonal kernel entries noisy (the diagonal stays
exactly 1, since U(x)U†(x) = I), which can push the Gram matrix off the PSD cone
and break the Cholesky factorization. `study_shots.py` turns on `eig_cutoff` —
the kernel's PSD projection, which **is** the paper's Sec. 1.1 regularization —
plus a little jitter, so low-N runs stay stable.

**Honest caveat:** at the default noise = 0.1, shot noise barely moves MSE even at
200 shots, because the GP's likelihood-noise term already dominates the kernel's
sampling noise. To actually see degradation (and stress the regularization, as
the paper emphasizes), push shots low and/or lower the likelihood noise. The grid
goes down to 100 shots for this reason.

### Companion study: `study_samples.py`

The training-set-size question (a learning curve) is a *different* study from
Fig. 3(b), but a natural companion. `study_samples.py` sweeps `n_train` × seeds
and plots MSE vs training size against the classical-RBF baseline.

```bash
uv run python study_samples.py --epochs 40   # quick screen
```

---

## What was added

### New files

- **`sweep.py`** — multi-seed config sweep → `sweep.parquet` + median/IQR and
  best-of-N tables.
- **`study_shots.py`** — the Fig. 3(b) finite-shot study → `study_shots.parquet`
  + `study_shots.png`.
- **`study_samples.py`** — training-set-size learning curve →
  `study_samples.parquet` + `study_samples.png`.
- **`src/experiment.py`** — shared `run_config()` (build → train → evaluate,
  faithful to `run.py`'s seeding) and `result_row()`, used by all of the above so
  results stay consistent.

### Modified files

- **`src/training.py`** — `evaluate_model` now returns a metrics dict
  (MSE, R², NLPD, MSLL, coverage).
- **`src/quantum_kernel.py`** — added `shots` + `set_shots(shots, seed)` for
  finite-shot evaluation.
- **`src/logger.py`** — logs all five metrics to the parquet row.
- **`src/plotting.py`** — added `plot_study_curve()` (mean ± std band vs a swept
  axis, optional log-x and reference line).
- **`run.py`** — now a thin driver over `run_config`; prints the richer metrics.
- **`README.md`**, **`.gitignore`**, **`pyproject.toml`** — docs, ignore study
  PNGs, ruff E402 ignores for the new scripts.

### Verification

- `run.py` at defaults still prints **Test MSE 0.0997**, so the refactor preserved
  the seeding bit-for-bit.
- The finite-shot path is reproducible (same seed → same result) and faithful
  (the whole Gram matrix is re-estimated under shots).
- All three scripts run end-to-end; `ruff format`, `ruff check`, and `ty check`
  pass.
