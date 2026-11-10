import torch

from hiss.kalman import (
    HybridGaussianState, hybrid_predict, hybrid_correct,
    dense_predict, dense_correct, HybridKalmanCell,
)


def make_state(batch=2, dim=7, rank=3, dtype=torch.float64):
    torch.manual_seed(7)
    mean = torch.randn(batch, dim, dtype=dtype)
    variance = torch.rand(batch, dim, dtype=dtype) + 0.3
    factor = 0.2 * torch.randn(batch, dim, rank, dtype=dtype)
    return HybridGaussianState(mean, variance, factor)


def test_hybrid_prediction_matches_dense():
    state = make_state()
    a = torch.rand_like(state.mean) * 0.8 + 0.1
    q = torch.rand_like(state.mean) + 0.05
    drive = torch.randn_like(state.mean)
    got = hybrid_predict(state, a, q, drive)
    ref_m, ref_p = dense_predict(
        state.mean, state.covariance(), a, q, drive
    )
    torch.testing.assert_close(got.mean, ref_m, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(got.covariance(), ref_p, rtol=1e-10, atol=1e-10)


def test_hybrid_correction_matches_dense_joseph():
    state = make_state()
    y = torch.randn_like(state.mean)
    r = torch.rand_like(state.mean) + 0.1
    got = hybrid_correct(state, y, r)
    ref_m, ref_p = dense_correct(state.mean, state.covariance(), y, r)
    torch.testing.assert_close(got.mean, ref_m, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(got.covariance(), ref_p, rtol=1e-9, atol=1e-9)


def test_batched_and_unbatched_shapes():
    state = make_state(batch=4, dim=5, rank=2)
    y = torch.randn_like(state.mean)
    r = torch.ones_like(state.mean)
    out = hybrid_correct(state, y, r)
    assert out.mean.shape == (4, 5)
    assert out.factor.shape == (4, 5, 2)
    single = HybridGaussianState(
        state.mean[0], state.variance[0], state.factor[0]
    )
    one = hybrid_correct(single, y[0], r[0])
    assert one.mean.shape == (5,)
    assert one.factor.shape == (5, 2)


def test_posterior_is_positive_definite_for_positive_diagonal():
    state = make_state()
    out = hybrid_correct(state, torch.randn_like(state.mean),
                         torch.rand_like(state.mean) + 0.05)
    eig = torch.linalg.eigvalsh(out.covariance())
    assert torch.all(eig > 0)


def test_gradients_are_finite():
    state = make_state(dtype=torch.float32)
    state = HybridGaussianState(
        state.mean.requires_grad_(),
        state.variance.requires_grad_(),
        state.factor.requires_grad_(),
    )
    y = torch.randn_like(state.mean, requires_grad=True)
    r = (torch.rand_like(state.mean) + 0.2).requires_grad_()
    out = hybrid_correct(state, y, r)
    loss = out.mean.square().mean() + out.variance.mean() + out.factor.square().mean()
    loss.backward()
    for x in (state.mean, state.variance, state.factor, y, r):
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_gradcheck_hybrid_correction():
    torch.manual_seed(11)
    mean = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    variance = (torch.rand(1, 3, dtype=torch.float64) + 0.5).requires_grad_()
    factor = (0.1 * torch.randn(1, 3, 2, dtype=torch.float64)).requires_grad_()
    y = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    r = (torch.rand(1, 3, dtype=torch.float64) + 0.4).requires_grad_()

    def fn(m, d, u, obs, obs_var):
        out = hybrid_correct(HybridGaussianState(m, d, u), obs, obs_var)
        return out.mean, out.variance, out.factor

    assert torch.autograd.gradcheck(
        fn, (mean, variance, factor, y, r), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_cell_unroll_backward_and_optimizer_step():
    torch.manual_seed(4)
    cell = HybridKalmanCell(input_dim=6, state_dim=8, rank=3)
    opt = torch.optim.AdamW(cell.parameters(), lr=1e-3)
    state = cell.initial_state((2,))
    total = 0.0
    for _ in range(5):
        x = torch.randn(2, 6)
        out, state, confidence = cell(x, state)
        total = total + out.square().mean() + 0.01 * state.variance.mean()
    opt.zero_grad()
    total.backward()
    opt.step()
    assert torch.isfinite(total)
    assert torch.isfinite(state.covariance()).all()
    assert confidence.shape == (2, 8)
