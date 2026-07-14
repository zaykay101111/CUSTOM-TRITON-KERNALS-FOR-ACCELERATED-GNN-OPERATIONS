# CUSTOM TRITON KERNALS FOR ACCELERATED GNN OPERATIONS #

## Project Structure ##



### Baseline ###

The `Baseline/` folder is a plain-PyTorch (no custom kernels) implementation
of a multi-head Graph Attention Network (GAT), trained and evaluated on the
Cora citation graph. It exists to (1) establish a numerically-correct
reference implementation and (2) provide the "known-good" output that the
fused Triton kernels in the main project must match.

Pipeline, at a glance: load Cora via `torch_geometric.datasets.Planetoid` →
add self-loops and convert the edge list to CSR (`row_ptr`/`col_idx`) →
for each of `K` attention heads, project node features (`linear_projection`),
score each edge (`compute_attention_logits`), normalize per-node
(`compute_softmax_per_node`), and aggregate neighbors (`aggregate_neighbors`)
→ concatenate all heads' outputs (`gat_multihead_forward`) → classify with a
linear layer → train with cross-entropy on `train_mask`, tracking accuracy on
`val_mask`/`test_mask`. A full line-by-line walkthrough of this pipeline,
with diagrams and analogies, lives in `Baseline/GAT_pipeline_walkthrough.md`.

```text
project/
├── baseline.py           # The slow PyTorch implementation
├── test_baseline.py      # Tests that it produces correct output
└── data/
    └── synthetic_graphs.py  # Generate fake graphs for testing
```

### Main Project ###

**Status: scaffolded, not yet implemented.** The `Project/` folder mirrors the
same GAT computation as `Baseline/baseline.py`, but with the per-node
projection/attention/softmax/aggregation steps fused into custom Triton
kernels instead of separate PyTorch tensor ops, to cut down on the
intermediate `[E, F_out]` tensors the baseline materializes in memory.

- `kernals/forward.py` — the `@triton.jit` kernel(s) computing one attention
  head's forward pass (project → attention logits → per-node softmax →
  aggregate), fused into as few launches as possible.
- `kernals/backward.py` — the `@triton.jit` kernel(s) computing gradients
  w.r.t. `X`, `W`, `a_L`, `a_R` by hand, since autograd can't see inside a
  raw Triton kernel the way it can for ordinary PyTorch ops.
- `kernals/launcher.py` — Python-side wrappers that allocate outputs,
  compute the launch grid, and call the kernels above.
- `layer.py` — a `torch.autograd.Function` wrapping the forward/backward
  launchers, plus a `GATLayer` `nn.Module` (multi-head, concatenated) that
  plugs into a normal PyTorch model the same way `nn.Linear` would.
- `test_fusion.py` — checks the fused kernels' output (and eventually
  gradients) numerically match `Baseline/baseline.py`'s unfused reference.
- `benchmark.py` — speed/memory comparison between the baseline and the
  fused kernels across graph sizes.

```text
project/
├── kernals/
│   ├── forward.py        # @triton.jit fused forward kernel
│   ├── backward.py       # @triton.jit backward kernel
│   └── launcher.py       # Python functions that call the kernels
├── layer.py              # torch.autograd.Function wrapper + nn.Module
├── test_fusion.py        # Verify it matches baseline output
└── benchmark.py          # Speed/memory comparison
```

## References ##

The baseline implementation trains a Graph Attention Network on the Cora citation
dataset, using PyTorch Geometric. Relevant citations:

- Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., & Eliassi-Rad, T. (2008). Collective classification in network data. *AI Magazine*, 29(3), 93-106. (Original Cora dataset.)

- Yang, Z., Cohen, W., & Salakhutdinov, R. (2016). Revisiting semi-supervised learning with graph embeddings. *International Conference on Machine Learning (ICML)*. (Standard train/val/test split used via `torch_geometric.datasets.Planetoid`.)

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations (ICLR)*.

- Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*.
