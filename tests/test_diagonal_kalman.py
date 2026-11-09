import torch

from hiss.kalman.diagonal import (
    DiagonalGaussianState,
    DiagonalKalmanCell,
    diagonal_kalman_correct,
    diagonal_kalman_predict,
)


def test_prediction_equations():
    mean = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    var = torch.tensor([[0.5, 2.0]], dtype=torch.float64)
    decay = torch.tensor([[0.8, 0.5]], dtype=torch.float64)
    drive = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    q = torch.tensor([[0.01, 0.02]], dtype=torch.float64)

    out = diagonal_kalman_predict(
        DiagonalGaussianState(mean, var), decay, drive, q
    )
    assert torch.allclose(out.mean, decay * mean + drive)
    assert torch.allclose(out.variance, decay.square() * var + q)


def test_correction_equations_and_positive_variance():
    prior = DiagonalGaussianState(
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    )
    y = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    r = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    rho = torch.ones_like(y)

    post, diag = diagonal_kalman_correct(prior, y, r, rho)
    expected_gain = prior.variance / (prior.variance + r)
    assert torch.allclose(diag["kalman_gain"], expected_gain)
    assert torch.allclose(post.mean, prior.mean + expected_gain * (y - prior.mean))
    assert torch.all(post.variance > 0)
    assert torch.all(post.variance <= prior.variance)


def test_low_confidence_reduces_gain():
    prior = DiagonalGaussianState(
        torch.zeros(2, 3, dtype=torch.float64),
        torch.ones(2, 3, dtype=torch.float64),
    )
    y = torch.ones(2, 3, dtype=torch.float64)
    r = torch.ones(2, 3, dtype=torch.float64)

    _, high = diagonal_kalman_correct(prior, y, r, torch.full_like(y, 0.9))
    _, low = diagonal_kalman_correct(prior, y, r, torch.full_like(y, 0.1))
    assert torch.all(low["kalman_gain"] < high["kalman_gain"])


def test_cell_shapes_and_finite_outputs():
    torch.manual_seed(7)
    cell = DiagonalKalmanCell(input_dim=6, state_dim=8).double()
    x = torch.randn(4, 6, dtype=torch.float64)
    state = cell.initial_state(4, dtype=torch.float64)
    out, diagnostics = cell(x, state, return_diagnostics=True)

    assert out.mean.shape == (4, 8)
    assert out.variance.shape == (4, 8)
    assert torch.isfinite(out.mean).all()
    assert torch.isfinite(out.variance).all()
    assert torch.all(out.variance > 0)
    assert torch.all((diagnostics["confidence"] > 0) & (diagnostics["confidence"] < 1))
    assert torch.all((diagnostics["decay"] > 0) & (diagnostics["decay"] <= 1))


def test_gradients_are_finite():
    torch.manual_seed(11)
    cell = DiagonalKalmanCell(input_dim=5, state_dim=7).double()
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    state = cell.initial_state(3, dtype=torch.float64)
    out = cell(x, state)
    loss = out.mean.square().mean() + out.variance.log().mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in cell.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_gradcheck_diagonal_correction():
    # Finite-difference check of the primitive in double precision.
    torch.manual_seed(3)
    m = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    p = (torch.rand(2, 3, dtype=torch.float64) + 0.5).requires_grad_()
    y = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    r = (torch.rand(2, 3, dtype=torch.float64) + 0.5).requires_grad_()
    rho = (torch.rand(2, 3, dtype=torch.float64) * 0.8 + 0.1).requires_grad_()

    def fn(m_, p_, y_, r_, rho_):
        out, _ = diagonal_kalman_correct(
            DiagonalGaussianState(m_, p_), y_, r_, rho_
        )
        return out.mean, out.variance

    assert torch.autograd.gradcheck(
        fn, (m, p, y, r, rho), eps=1e-6, atol=1e-5, rtol=1e-4
    )
