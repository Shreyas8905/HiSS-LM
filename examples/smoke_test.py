"""Run a short differentiability and state-evolution smoke test."""

import sys
from pathlib import Path

import torch

# Allow running this file directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiss.kalman import DiagonalKalmanCell


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size, input_dim, state_dim, sequence_length = 8, 32, 64, 40
    cell = DiagonalKalmanCell(input_dim, state_dim).to(device)
    optimizer = torch.optim.AdamW(cell.parameters(), lr=3e-4)

    state = cell.initial_state(batch_size, device=device)
    inputs = torch.randn(batch_size, sequence_length, input_dim, device=device)

    optimizer.zero_grad(set_to_none=True)
    mean_history = []
    variance_history = []

    for t in range(sequence_length):
        state, diag = cell(inputs[:, t, :], state, return_diagnostics=True)
        mean_history.append(state.mean)
        variance_history.append(state.variance)

    means = torch.stack(mean_history, dim=1)
    variances = torch.stack(variance_history, dim=1)

    # A synthetic stability objective only. This is NOT an LM training loss.
    loss = means.square().mean() + 0.01 * variances.log().square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(cell.parameters(), max_norm=1.0)
    optimizer.step()

    print(f"device: {device}")
    print(f"synthetic objective: {loss.item():.6f}")
    print(f"mean tensor shape: {tuple(means.shape)}")
    print(f"variance range: [{variances.min().item():.6g}, {variances.max().item():.6g}]")
    print(f"mean confidence: {diag['confidence'].mean().item():.4f}")
    print("Smoke test completed. This does not establish language-model quality.")


if __name__ == "__main__":
    main()
