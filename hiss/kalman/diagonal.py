"""Differentiable diagonal-covariance Kalman state-space cell for HiSS-LM.

This is a research reference implementation, not an optimized production kernel.
All covariance quantities are variances (diagonal entries), not standard deviations.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DiagonalGaussianState(NamedTuple):
    """Factorized Gaussian belief q(z) = N(mean, diag(variance))."""

    mean: Tensor
    variance: Tensor


class KalmanDiagnostics(NamedTuple):
    """Intermediate quantities useful for debugging and research analysis."""

    decay: Tensor
    process_variance: Tensor
    observation: Tensor
    observation_variance: Tensor
    confidence: Tensor
    effective_observation_variance: Tensor
    kalman_gain: Tensor
    innovation: Tensor


def diagonal_kalman_predict(
    state: DiagonalGaussianState,
    decay: Tensor,
    input_drive: Tensor,
    process_variance: Tensor,
    variance_floor: float = 1e-8,
) -> DiagonalGaussianState:
    """Perform diagonal-covariance state prediction.

    Equations:
        m^-_t = a_t * m_{t-1} + b_t
        p^-_t = a_t^2 * p_{t-1} + q_t

    Shapes are broadcast according to PyTorch rules. `decay`, `input_drive`,
    and `process_variance` should be broadcastable to `state.mean`.
    """
    mean_prior = decay * state.mean + input_drive
    variance_prior = decay.square() * state.variance + process_variance
    variance_prior = variance_prior.clamp_min(variance_floor)
    return DiagonalGaussianState(mean_prior, variance_prior)


def diagonal_kalman_correct(
    prior: DiagonalGaussianState,
    observation: Tensor,
    observation_variance: Tensor,
    confidence: Optional[Tensor] = None,
    eps: float = 1e-8,
    variance_floor: float = 1e-8,
):
    """Correct a diagonal Gaussian prior using a noisy observation.

    With confidence rho in (0, 1], observation variance is inflated as
        R_eff = R / (rho + eps).

    The diagonal H=I Kalman update is:
        S = P^- + R_eff
        K = P^- / S
        m = m^- + K * (y - m^-)
        P = P^- * R_eff / S

    The final variance expression is algebraically equivalent to the scalar
    Joseph-form update for H=I. It avoids explicitly constructing matrices.
    """
    if confidence is None:
        confidence = torch.ones_like(prior.mean)

    effective_r = observation_variance / (confidence + eps)
    innovation_variance = (prior.variance + effective_r).clamp_min(eps)
    gain = prior.variance / innovation_variance
    innovation = observation - prior.mean

    posterior_mean = prior.mean + gain * innovation
    posterior_variance = (
        prior.variance * effective_r / innovation_variance
    ).clamp_min(variance_floor)

    posterior = DiagonalGaussianState(posterior_mean, posterior_variance)
    diagnostics = {
        "effective_observation_variance": effective_r,
        "innovation_variance": innovation_variance,
        "kalman_gain": gain,
        "innovation": innovation,
    }
    return posterior, diagnostics


class DiagonalKalmanCell(nn.Module):
    """Learned input-conditioned diagonal Kalman state-space cell.

    The cell implements a first, deliberately simplified HiSS-LM fast-level
    primitive. It has no cross-level context and no low-rank covariance yet.

    State transition:
        lambda_t = softplus(W_lambda u_t + b_lambda)
        a_t = exp(-lambda_t * delta_t)
        q_t = softplus(W_q u_t + b_q) + q_floor
        b_t = W_b u_t + SiLU(W_f u_t)

    Observation model:
        y_t = W_y u_t
        R_t = softplus(W_R u_t + b_R) + r_floor
        rho_t = sigmoid(W_rho u_t + b_rho)
        R_eff = R_t / (rho_t + eps)

    The prior and observation are combined with an elementwise Kalman update.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        delta_min: float = 0.05,
        delta_max: float = 1.0,
        process_noise_floor: float = 1e-5,
        observation_noise_floor: float = 1e-5,
        variance_floor: float = 1e-8,
        eps: float = 1e-8,
    ):
        super().__init__()
        if input_dim <= 0 or state_dim <= 0:
            raise ValueError("input_dim and state_dim must be positive.")
        if not (0.0 < delta_min < delta_max):
            raise ValueError("Require 0 < delta_min < delta_max.")

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.process_noise_floor = float(process_noise_floor)
        self.observation_noise_floor = float(observation_noise_floor)
        self.variance_floor = float(variance_floor)
        self.eps = float(eps)

        self.decay_rate = nn.Linear(input_dim, state_dim)
        self.delta_gate = nn.Linear(input_dim, state_dim)
        self.input_drive = nn.Linear(input_dim, state_dim)
        self.nonlinear_drive = nn.Linear(input_dim, state_dim)

        self.process_noise = nn.Linear(input_dim, state_dim)
        self.observation = nn.Linear(input_dim, state_dim)
        self.observation_noise = nn.Linear(input_dim, state_dim)
        self.confidence = nn.Linear(input_dim, state_dim)

        # Start with moderate decay rates and non-extreme noise/confidence.
        nn.init.zeros_(self.decay_rate.weight)
        nn.init.constant_(self.decay_rate.bias, -1.5)
        nn.init.zeros_(self.delta_gate.weight)
        nn.init.zeros_(self.delta_gate.bias)
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)

    def initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> DiagonalGaussianState:
        """Create the default prior m_0=0, P_0=I for a batch."""
        parameter = next(self.parameters())
        device = device if device is not None else parameter.device
        dtype = dtype if dtype is not None else parameter.dtype
        mean = torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)
        variance = torch.ones(batch_size, self.state_dim, device=device, dtype=dtype)
        return DiagonalGaussianState(mean, variance)

    def forward(
        self,
        inputs: Tensor,
        state: DiagonalGaussianState,
        return_diagnostics: bool = False,
    ):
        """Process one time step.

        Args:
            inputs: [batch, input_dim]
            state: previous Gaussian state, each tensor [batch, state_dim]
            return_diagnostics: if True, also return intermediate filter values.

        Returns:
            updated_state, or (updated_state, diagnostics).
        """
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [batch, input_dim].")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {inputs.shape[-1]}."
            )
        expected = (inputs.shape[0], self.state_dim)
        if tuple(state.mean.shape) != expected or tuple(state.variance.shape) != expected:
            raise ValueError(f"State mean and variance must both have shape {expected}.")

        # Positive, bounded time interval and stable diagonal transition.
        delta = self.delta_min + (
            self.delta_max - self.delta_min
        ) * torch.sigmoid(self.delta_gate(inputs))
        rate = F.softplus(self.decay_rate(inputs))
        decay = torch.exp(-rate * delta)

        drive = self.input_drive(inputs) + F.silu(self.nonlinear_drive(inputs))
        q = F.softplus(self.process_noise(inputs)) + self.process_noise_floor

        prior = diagonal_kalman_predict(
            state, decay, drive, q, variance_floor=self.variance_floor
        )

        y = self.observation(inputs)
        r = F.softplus(self.observation_noise(inputs)) + self.observation_noise_floor
        rho = torch.sigmoid(self.confidence(inputs))

        posterior, correction = diagonal_kalman_correct(
            prior,
            y,
            r,
            confidence=rho,
            eps=self.eps,
            variance_floor=self.variance_floor,
        )

        if not return_diagnostics:
            return posterior

        diagnostics = {
            "decay": decay,
            "delta": delta,
            "decay_rate": rate,
            "process_variance": q,
            "observation": y,
            "observation_variance": r,
            "confidence": rho,
            **correction,
        }
        return posterior, diagnostics
