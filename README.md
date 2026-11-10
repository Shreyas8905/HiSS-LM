# HiSS-LM Phase 2 — Hybrid Covariance Kalman Filter

This phase implements the covariance representation

\[
P = \operatorname{diag}(D) + UU^\top
\]

and validates it against a dense full-covariance reference.

## Scope

- Dense prediction and Joseph-form correction for small dimensions.
- Structured hybrid prediction with diagonal transition and process noise.
- Structured Woodbury correction for identity observation operator and diagonal observation noise.
- PyTorch autograd and float64 `gradcheck`.
- A learned input-conditioned hybrid Kalman cell.
- No hierarchy, tokenizer, LM head, or language-model training yet.

## Install and test (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python examples/smoke_test.py
```

## Core API

```python
from hiss.kalman import (
    HybridGaussianState,
    hybrid_predict,
    hybrid_correct,
    dense_predict,
    dense_correct,
    HybridKalmanCell,
)
```

`HybridGaussianState` stores:
- `mean`: `[..., d]`
- `variance`: `[..., d]`, positive diagonal component
- `factor`: `[..., d, r]`

`state.covariance()` materializes a dense matrix for diagnostics only. Do not call it in the production recurrent hot path.

## Mathematical constraints

Exact closure relies on:
1. Diagonal state transition \(A=\operatorname{diag}(a)\).
2. Diagonal process covariance \(Q=\operatorname{diag}(q)\).
3. Identity observation operator \(H=I\).
4. Diagonal observation covariance \(R\).

Under these assumptions, prediction and correction preserve rank at most `r` exactly. A non-diagonal observation operator, full process covariance, or cross-level joint covariance may not preserve this representation.

## Numerical policy

- Covariance operations default to FP32 in `HybridKalmanCell`.
- Dense reference tests use float64.
- Cholesky factorization is applied only to the small `r x r` Woodbury matrix in the hybrid correction.
- No explicit matrix inverse is used.
- The implementation does not claim calibrated uncertainty or language-model quality.

## Phase 2 acceptance criteria

- Structured prediction matches dense prediction.
- Structured correction matches dense Joseph-form correction.
- Posterior covariance is positive definite for positive diagonal components.
- Gradients are finite and correction passes float64 `gradcheck`.
- A short sequence unroll completes backward and an optimizer step.
