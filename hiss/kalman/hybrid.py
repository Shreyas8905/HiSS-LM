"""Diagonal-plus-low-rank covariance Kalman operations.

Covariance representation:
    P = diag(D) + U @ U.T

The correction is exact for H=I and diagonal observation covariance R.
No dense [d, d] covariance is materialized in the hybrid path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class HybridGaussianState:
    """Gaussian state with P = diag(variance) + factor @ factor.T."""
    mean: Tensor       # [..., d]
    variance: Tensor   # [..., d], strictly positive diagonal component
    factor: Tensor     # [..., d, r], low-rank covariance factor

    def covariance(self) -> Tensor:
        """Materialize covariance; intended for tests/debugging, not hot path."""
        return torch.diag_embed(self.variance) + self.factor @ self.factor.transpose(-1, -2)


def _check_state(state: HybridGaussianState) -> None:
    if state.mean.ndim < 1 or state.variance.shape != state.mean.shape:
        raise ValueError("mean and variance must have the same [..., d] shape")
    if state.factor.ndim != state.mean.ndim + 1:
        raise ValueError("factor must have shape [..., d, r]")
    if state.factor.shape[:-2] != state.mean.shape[:-1]:
        raise ValueError("factor batch dimensions must match mean")
    if state.factor.shape[-2] != state.mean.shape[-1]:
        raise ValueError("factor latent dimension must match mean")
    if state.factor.shape[-1] < 1:
        raise ValueError("rank must be at least 1")


def hybrid_predict(
    state: HybridGaussianState,
    transition: Tensor,
    process_variance: Tensor,
    mean_drive: Tensor | None = None,
) -> HybridGaussianState:
    """Predict with diagonal A and diagonal Q, preserving the hybrid form.

    A = diag(transition)
    D' = transition**2 * D + Q
    U' = diag(transition) @ U
    """
    _check_state(state)
    if transition.shape != state.mean.shape:
        raise ValueError("transition must have shape [..., d]")
    if process_variance.shape != state.mean.shape:
        raise ValueError("process_variance must have shape [..., d]")
    if mean_drive is None:
        mean_drive = torch.zeros_like(state.mean)
    if mean_drive.shape != state.mean.shape:
        raise ValueError("mean_drive must have shape [..., d]")
    if torch.any(process_variance <= 0):
        raise ValueError("process_variance must be strictly positive")

    mean = transition * state.mean + mean_drive
    variance = transition.square() * state.variance + process_variance
    factor = transition.unsqueeze(-1) * state.factor
    return HybridGaussianState(mean, variance, factor)


def _woodbury_solve(
    diagonal: Tensor,
    factor: Tensor,
    rhs: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute (diag(diagonal) + factor factor.T)^-1 rhs.

    Returns the solve result and Cholesky factor of the small rank-space matrix.
    """
    inv_diag = diagonal.reciprocal()
    weighted_rhs = inv_diag * rhs
    weighted_factor = inv_diag.unsqueeze(-1) * factor
    rank_matrix = torch.eye(
        factor.shape[-1], dtype=factor.dtype, device=factor.device
    ) + factor.transpose(-1, -2) @ weighted_factor
    chol = torch.linalg.cholesky(rank_matrix)
    projected = factor.transpose(-1, -2) @ weighted_rhs.unsqueeze(-1)
    solved = torch.cholesky_solve(projected, chol).squeeze(-1)
    result = weighted_rhs - (weighted_factor @ solved.unsqueeze(-1)).squeeze(-1)
    return result, chol


def hybrid_correct(
    state: HybridGaussianState,
    observation: Tensor,
    observation_variance: Tensor,
    eps: float = 1e-8,
) -> HybridGaussianState:
    """Exact H=I correction with diagonal observation covariance.

    Uses Woodbury for the innovation solve and returns an exact posterior
    representation D_post + U_post U_post.T of rank no greater than input rank.
    """
    _check_state(state)
    if observation.shape != state.mean.shape:
        raise ValueError("observation must have shape [..., d]")
    if observation_variance.shape != state.mean.shape:
        raise ValueError("observation_variance must have shape [..., d]")
    if torch.any(state.variance <= 0):
        raise ValueError("state variance diagonal must be strictly positive")
    if torch.any(observation_variance <= 0):
        raise ValueError("observation_variance must be strictly positive")

    # S = P + R = diag(D + R) + U U^T.
    c = state.variance + observation_variance
    innovation = observation - state.mean
    s_inv_innovation, chol = _woodbury_solve(c, state.factor, innovation)

    # Since P = S - R, K e = P S^-1 e = e - R S^-1 e.
    mean = state.mean + innovation - observation_variance * s_inv_innovation

    # Exact posterior:
    # D_post = D R / (D + R)
    # U_post = diag(R/(D+R)) U G^(-1/2),
    # G = I + U.T diag((D+R)^-1) U.
    posterior_diagonal = state.variance * observation_variance / c
    scaled_factor = (
        observation_variance / c
    ).unsqueeze(-1) * state.factor

    # If G = L L.T, then G^-1 = L^-T L^-1 and
    # U_post = scaled_factor @ L^-T.
    posterior_factor = torch.linalg.solve_triangular(
        chol,
        scaled_factor.transpose(-1, -2),
        upper=False,
    ).transpose(-1, -2)

    # Guard against tiny roundoff only; this is not a rank truncation.
    posterior_diagonal = posterior_diagonal.clamp_min(eps)
    return HybridGaussianState(mean, posterior_diagonal, posterior_factor)


def dense_predict(
    mean: Tensor,
    covariance: Tensor,
    transition: Tensor,
    process_variance: Tensor,
    mean_drive: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Dense reference prediction. Intended only for small dimensions."""
    d = mean.shape[-1]
    if mean_drive is None:
        mean_drive = torch.zeros_like(mean)
    a = torch.diag_embed(transition)
    q = torch.diag_embed(process_variance)
    next_mean = transition * mean + mean_drive
    next_cov = a @ covariance @ a.transpose(-1, -2) + q
    return next_mean, 0.5 * (next_cov + next_cov.transpose(-1, -2))


def dense_correct(
    mean: Tensor,
    covariance: Tensor,
    observation: Tensor,
    observation_variance: Tensor,
) -> tuple[Tensor, Tensor]:
    """Dense H=I Kalman correction using Cholesky solves, not explicit inverse."""
    r = torch.diag_embed(observation_variance)
    innovation_cov = covariance + r
    innovation = (observation - mean).unsqueeze(-1)
    chol = torch.linalg.cholesky(innovation_cov)
    solved_innovation = torch.cholesky_solve(innovation, chol)
    # K e = P S^-1 e
    next_mean = mean + (covariance @ solved_innovation).squeeze(-1)

    # Joseph form: (I-K)P(I-K)^T + K R K^T
    # K = P S^-1; obtain it with a Cholesky solve.
    gain = torch.cholesky_solve(covariance.transpose(-1, -2), chol).transpose(-1, -2)
    identity = torch.eye(mean.shape[-1], dtype=mean.dtype, device=mean.device)
    residual_map = identity - gain
    next_cov = (
        residual_map @ covariance @ residual_map.transpose(-1, -2)
        + gain @ r @ gain.transpose(-1, -2)
    )
    next_cov = 0.5 * (next_cov + next_cov.transpose(-1, -2))
    return next_mean, next_cov


class HybridKalmanCell(nn.Module):
    """Learned input-conditioned hybrid covariance state-space cell.

    This is a state-estimation primitive, not yet a hierarchical block or LM.
    The low-rank factor is initialized near zero, while diagonal uncertainty
    starts positive. Covariance calculations are promoted to float32 by default.
    """
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        rank: int = 16,
        delta_min: float = 0.05,
        delta_max: float = 1.0,
        variance_floor: float = 1e-4,
        covariance_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or state_dim <= 0 or rank <= 0:
            raise ValueError("input_dim, state_dim and rank must be positive")
        if not 0 < delta_min < delta_max:
            raise ValueError("require 0 < delta_min < delta_max")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.rank = min(rank, state_dim)
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.variance_floor = variance_floor
        self.covariance_dtype = covariance_dtype

        self.rate = nn.Linear(input_dim, state_dim)
        self.delta = nn.Linear(input_dim, state_dim)
        self.drive = nn.Linear(input_dim, state_dim)
        self.drive_nonlinear = nn.Linear(input_dim, state_dim)
        self.process_noise = nn.Linear(input_dim, state_dim)
        self.observation = nn.Linear(input_dim, state_dim)
        self.observation_noise = nn.Linear(input_dim, state_dim)
        self.confidence = nn.Linear(input_dim, state_dim)
        self.low_rank = nn.Linear(input_dim, state_dim * self.rank)

        nn.init.zeros_(self.low_rank.weight)
        nn.init.zeros_(self.low_rank.bias)
        nn.init.constant_(self.confidence.bias, 0.0)

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        device: torch.device | None = None,
    ) -> HybridGaussianState:
        shape = (*batch_shape, self.state_dim)
        mean = torch.zeros(shape, device=device, dtype=self.covariance_dtype)
        variance = torch.ones(shape, device=device, dtype=self.covariance_dtype)
        factor = torch.zeros(
            *batch_shape, self.state_dim, self.rank,
            device=device, dtype=self.covariance_dtype,
        )
        return HybridGaussianState(mean, variance, factor)

    def forward(
        self,
        inputs: Tensor,
        state: HybridGaussianState,
    ) -> tuple[Tensor, HybridGaussianState, Tensor]:
        if inputs.shape[:-1] != state.mean.shape[:-1]:
            raise ValueError("input and state batch dimensions must match")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError("unexpected input dimension")

        # Neural projections may run in AMP dtype; sensitive covariance math
        # is explicitly promoted to the configured covariance dtype.
        x = inputs
        a = torch.exp(
            -F.softplus(self.rate(x))
            * (
                self.delta_min
                + (self.delta_max - self.delta_min) * torch.sigmoid(self.delta(x))
            )
        )
        drive = self.drive(x) + F.silu(self.drive_nonlinear(x))
        q = F.softplus(self.process_noise(x)) + self.variance_floor
        y = self.observation(x)
        r = F.softplus(self.observation_noise(x)) + self.variance_floor
        confidence = torch.sigmoid(self.confidence(x))
        r_eff = r / (confidence + 1e-6)

        dtype = self.covariance_dtype
        state_fp = HybridGaussianState(
            state.mean.to(dtype), state.variance.to(dtype), state.factor.to(dtype)
        )
        predicted = hybrid_predict(
            state_fp, a.to(dtype), q.to(dtype), drive.to(dtype)
        )
        corrected = hybrid_correct(predicted, y.to(dtype), r_eff.to(dtype))
        return corrected.mean, corrected, confidence


__all__ = [
    "HybridGaussianState", "hybrid_predict", "hybrid_correct",
    "HybridKalmanCell", "dense_predict", "dense_correct",
]
