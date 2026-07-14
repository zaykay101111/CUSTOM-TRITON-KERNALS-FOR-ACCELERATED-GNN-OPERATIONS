# Test that it produces the correct output

# test_baseline.py

import torch
from baseline import (
    linear_projection,
    compute_attention_logits,
    compute_softmax_per_node,
    aggregate_neighbors,
    gat_forward_baseline,
    train_loop
)
from torch_geometric.utils import scatter


def test_linear_projection():
    X = torch.randn(10, 128)
    W = torch.randn(128, 64)
    H = linear_projection(X, W)
    assert H.shape == (10, 64), f"Expected (10, 64), got {H.shape}"
    print("✓ linear_projection")


def test_attention_logits():
    H = torch.randn(10, 64)
    a_L = torch.randn(64)
    a_R = torch.randn(64)
    source = torch.tensor([0, 0, 1, 1, 2])
    dest = torch.tensor([1, 2, 0, 2, 0])
    e = compute_attention_logits(H, a_L, a_R, source, dest)
    assert e.shape == (5,), f"Expected (5,), got {e.shape}"
    print("✓ compute_attention_logits")


def test_softmax():
    logits = torch.tensor([0.5, 0.3, 0.8, 0.2, 0.6])
    source = torch.tensor([0, 0, 0, 1, 1])
    alpha = compute_softmax_per_node(logits, source, num_nodes=2)
    assert alpha.shape == (5,), f"Expected (5,), got {alpha.shape}"
    # Node 0 has 3 edges, node 1 has 2 edges
    assert torch.allclose(alpha[0:3].sum(), torch.tensor(1.0), atol=1e-5), "Node 0 weights should sum to 1"
    assert torch.allclose(alpha[3:5].sum(), torch.tensor(1.0), atol=1e-5), "Node 1 weights should sum to 1"
    assert torch.all(alpha >= 0), "Weights should be non-negative"
    print("✓ compute_softmax_per_node")


def test_aggregate():
    H = torch.randn(5, 64)
    alpha = torch.tensor([0.6, 0.4, 0.3, 0.7, 1.0])
    source = torch.tensor([0, 0, 1, 1, 2])
    dest = torch.tensor([1, 2, 0, 2, 0])
    out = aggregate_neighbors(H, alpha, source, dest, num_nodes=3)
    assert out.shape == (3, 64), f"Expected (3, 64), got {out.shape}"
    print("✓ aggregate_neighbors")


def test_forward():
    N, F_in, F_out = 100, 128, 64
    X = torch.randn(N, F_in)
    W = torch.randn(F_in, F_out, requires_grad=True)
    a_L = torch.randn(F_out, requires_grad=True)
    a_R = torch.randn(F_out, requires_grad=True)
    row_ptr = torch.tensor([i * 3 for i in range(N + 1)])  # each node has 3 neighbors
    col_idx = torch.randint(0, N, (N * 3,))
    out = gat_forward_baseline(X, W, a_L, a_R, row_ptr, col_idx)
    assert out.shape == (N, F_out), f"Expected ({N}, {F_out}), got {out.shape}"

    # Test backward works
    loss = out.sum()
    loss.backward()
    assert W.grad is not None, "W should have gradients"
    assert a_L.grad is not None, "a_L should have gradients"
    assert not torch.any(torch.isnan(W.grad)), "Gradients should not be NaN"
    print("✓ gat_forward_baseline (forward + backward)")


if __name__ == "__main__":
    test_linear_projection()
    test_attention_logits()
    test_softmax()
    test_aggregate()
    test_forward()
    print("\nAll tests passed! Run train_loop() when ready.")