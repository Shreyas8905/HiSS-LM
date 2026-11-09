# HiSS-LM — Phase 1: Diagonal Kalman State-Space Cell

This is the first executable research primitive for HiSS-LM. It implements a
learned, input-conditioned state transition and a differentiable diagonal
Gaussian Kalman correction using PyTorch autograd.

## Requirements

- Python 3.10+
- PyTorch 2.x
- pytest (for tests)

Install PyTorch using the command appropriate for your CPU/CUDA environment:
https://pytorch.org/get-started/locally/

Then install pytest:

```bash
python -m pip install pytest
```

## Run tests

From this directory:

```bash
python -m pytest -q
```

## Run the smoke test

```bash
python examples/smoke_test.py
```

The smoke test unrolls the cell over a short synthetic sequence and performs
one optimizer step. It is a differentiability/stability test, not language
model training and not evidence of model quality.

## Implemented equations

Prediction:

\[
m_t^- = a_t \odot m_{t-1} + b_t
\]

\[
P_t^- = a_t^2 \odot P_{t-1} + Q_t
\]

Correction with adaptive confidence:

\[
R_{t,\mathrm{eff}} = R_t / (\rho_t + \epsilon)
\]

\[
K_t = P_t^- / (P_t^- + R_{t,\mathrm{eff}})
\]

\[
m_t = m_t^- + K_t \odot (y_t-m_t^-)
\]

\[
P_t = P_t^- \odot R_{t,\mathrm{eff}} /
      (P_t^-+R_{t,\mathrm{eff}})
\]

All covariance quantities are diagonal variances. This version does not yet
implement low-rank covariance, cross-level coupling, a tokenizer, a language
model head, or a training corpus.

## Important research limitations

- The observation matrix is currently the identity in latent coordinates:
  the observation projection maps the input directly to the state dimension.
- The fast-level cell has no medium/slow context.
- The confidence gate is learned from the current input only.
- The diagonal covariance approximation ignores correlations between latent
  dimensions.
- The current implementation uses ordinary PyTorch autograd. Custom backward
  kernels are deferred until numerical validation and profiling justify them.
- The cell is a sequential reference implementation. It is not yet optimized
  for throughput.

## Suggested next step

Validate the cell, then implement a dense small-dimensional Kalman reference
and a hybrid diagonal-plus-low-rank covariance version. Compare the structured
implementation against the dense reference before integrating it into the
hierarchical model.
