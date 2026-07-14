# The slow pytorch implementation of GAT

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter, add_self_loops
from torch_geometric.datasets import Planetoid

def xavier_uniform_param(*shape):
    """
    Creates a learnable parameter (requires_grad=True) initialized with
    Xavier/Glorot uniform init: values drawn uniformly from
    [-bound, bound] where bound = sqrt(6 / (fan_in + fan_out)).

    This keeps the variance of a layer's output roughly the same as its
    input's variance, regardless of how large fan_in/fan_out are — unlike
    plain torch.randn (std=1), which lets outputs blow up in scale as the
    number of input features grows (1433 -> 64 in our case).

    shape: for a weight matrix, pass (fan_in, fan_out) e.g. (F_in, F_out).
           for a 1D attention vector, pass (fan_in,) — fan_out is treated as 1.
    """
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else 1
    bound = (6.0 / (fan_in + fan_out)) ** 0.5

    param = torch.empty(*shape).uniform_(-bound, bound)
    param.requires_grad_()
    return param


def linear_projection(X, W):
    """
    X: N x F_in — node features, each row is features per node
    W: F_in x F_out — learned weight matrix
    Output: [N, F_out] — projected features
    """
    # Multiply each node's features by W to get shorter, filtered version
    return X @ W


def compute_attention_logits(H, a_L, a_R, source_nodes, dest_nodes):
    """
    H: [N, F_out] — projected features for all nodes
    a_L, a_R: [F_out] — learned attention vectors
    source_nodes: [E] — which node each edge comes from
    dest_nodes: [E] — which node each edge goes to
    Output: [E] — raw attention score per edge
    """
    # Get source and dest node features
    source_features = H[source_nodes]  # [E, F_out]
    dest_features = H[dest_nodes]  # [E, F_out]


    # Compute LeakyReLU(a_L · h_source + a_R · h_dest)
    L = source_features @ a_L.T  # [E]
    R = dest_features @ a_R.T  # [E]

    logits = F.leaky_relu(L + R, negative_slope=0.2)  # [E]


    # Return one score per edge
    return logits


def compute_softmax_per_node(logits, source_nodes, num_nodes):
    """
    logits: [E] — raw attention scores
    source_nodes: [E] — which source node each edge belongs to
    num_nodes: N — total number of nodes
    Output: [E] — attention weights (percentages)
    """
    # Find max logit per source node
    logits_max = scatter(logits, source_nodes, dim=0, dim_size=num_nodes, reduce='max')  # [N], max logit per node

    # Subtract max from all logits (for numerical stability)
    logits_stable = logits - logits_max[source_nodes]  # [E]

    # Exponentiate each logit
    exp_logits = torch.exp(logits_stable)  # [E]
    # Sum exponentials per source node
    exp_sum = scatter(exp_logits, source_nodes, dim=0, dim_size=num_nodes, reduce='sum')  # [N]
    # Divide each logit by sum (now they're percentages)
    attention_weights = exp_logits / exp_sum[source_nodes]  # [E]

    return attention_weights


def aggregate_neighbors(H, attention_weights, source_nodes, dest_nodes, num_nodes):
    """
    H: [N, F_out] — projected features
    attention_weights: [E] — softmax weights
    source_nodes: [E] — which source node
    dest_nodes: [E] — which neighbor
    num_nodes: N
    Output: [N, F_out] — aggregated features per node
    """
    # Multiply each neighbor's features by its attention weight
    weighted_neightbors = attention_weights.unsqueeze(1) * H[dest_nodes]  # [E, F_out]

    # Sum all weighted neighbors for each source node
    aggregated_features = scatter(weighted_neightbors, source_nodes, dim=0, dim_size=num_nodes, reduce='sum')  # [N, F_out]
    # Return the aggregated result
    return F.elu(aggregated_features)  # [N, F_out]


def gat_forward_baseline(X, W, a_L, a_R, row_ptr, col_idx, training=True, dropout_p=0.6):
    """
    X: [N, F_in] — input node features
    W: [F_in, F_out] — weight matrix
    a_L, a_R: [F_out] — attention vectors
    row_ptr, col_idx: CSR graph format
    training: whether to apply dropout (True during training, False during eval)
    dropout_p: dropout probability, applied to input features and attention weights
    Output: [N, F_out] — updated node features
    """

    N = X.shape[0]  # number of nodes

    # Dropout on the raw input features. Randomly zeroes out a fraction of each
    # node's feature entries during training only (no-op when training=False).
    # This stops the model from leaning on any single input feature too heavily,
    # which is one of the two regularizers fighting the overfitting we saw.
    X = F.dropout(X, p=dropout_p, training=training)

    # Project features
    H = linear_projection(X, W)  # [N, F_out]
    # Convert CSR to source/dest node lists
    dest_nodes = col_idx  # [E]

    source_nodes = torch.repeat_interleave(torch.arange(N), row_ptr[1:] - row_ptr[:-1])  # [E]


    # Compute attention logits for each edge
    e = compute_attention_logits(H, a_L, a_R, source_nodes, dest_nodes)  # [E]
    # Softmax each node's logits
    a = compute_softmax_per_node(e, source_nodes, N)  # [E]

    # Dropout on the normalized attention weights themselves. Randomly drops
    # some edges' contribution to the aggregation step each forward pass, so
    # the model can't over-rely on any one neighbor being present every time.
    a = F.dropout(a, p=dropout_p, training=training)

    # Aggregate neighbors weighted by attention
    h_new = aggregate_neighbors(H, a, source_nodes, dest_nodes, N)  # [N, F_out]
    return h_new


def gat_multihead_forward(X, W_list, a_L_list, a_R_list, row_ptr, col_idx, training=True, dropout_p=0.6):
    """
    Runs K independent attention heads and concatenates their outputs,
    implementing:

        h_i' = Concat_{k=1..K}( sigma( sum_j alpha_ij^k * W^k h_j ) )

    X: [N, F_in] — input node features (shared by every head)
    W_list: list of K weight matrices, each [F_in, F_out]
    a_L_list, a_R_list: list of K attention vectors, each [F_out]
    row_ptr, col_idx: CSR graph format (shared by every head)
    training, dropout_p: forwarded to each head's gat_forward_baseline call
    Output: [N, K * F_out] — every head's output glued side-by-side
    """
    # Run the existing single-head computation once per head. Each head has
    # its own W^k, a_L^k, a_R^k, so each one learns to focus on a different
    # notion of "which neighbors matter," rather than all sharing one view.
    head_outputs = [
        gat_forward_baseline(X, W_k, a_L_k, a_R_k, row_ptr, col_idx,
                              training=training, dropout_p=dropout_p)
        for W_k, a_L_k, a_R_k in zip(W_list, a_L_list, a_R_list)
    ]  # K tensors, each [N, F_out]

    # Concatenate along the feature dimension (this is the "||" in the
    # equation) so a node's final representation is K * F_out wide instead
    # of just F_out — each head's opinion sits side-by-side in the vector.
    return torch.cat(head_outputs, dim=1)  # [N, K * F_out]


def train_step(X, W_list, a_L_list, a_R_list, row_ptr, col_idx, y_true, train_mask, classifier, optimizer, loss_fn):
    """
    X: input features
    W_list, a_L_list, a_R_list: learnable parameters, one set per attention head
    row_ptr, col_idx: graph
    y_true: correct labels
    train_mask: [N] bool — which nodes to compute the loss on
    classifier: final linear layer mapping GAT output to class logits
    optimizer: Adam or SGD
    loss_fn: CrossEntropyLoss
    Output: loss value
    """
    # Forward pass: run data through every attention head and concatenate
    output = gat_multihead_forward(X, W_list, a_L_list, a_R_list, row_ptr, col_idx)  # [N, K * F_out]

    # Convert GAT output to class predictions
    logits = classifier(output)  # [N, num_classes]

    # Measure how wrong the predictions are, only on training nodes
    loss = loss_fn(logits[train_mask], y_true[train_mask])  # single number

    # Clear old gradients
    optimizer.zero_grad()

    # Compute new gradients for every parameter
    loss.backward()

    # Update every parameter using its gradient
    optimizer.step()

    return loss.item()  # .item() converts single-number tensor to plain Python float


@torch.no_grad()
def evaluate(X, W_list, a_L_list, a_R_list, row_ptr, col_idx, y_true, mask, classifier):
    """
    Runs a forward pass with no gradient tracking and reports accuracy
    on whichever nodes `mask` selects (train_mask, val_mask, or test_mask).

    X, W_list, a_L_list, a_R_list, row_ptr, col_idx: same as train_step
    y_true: [N] — true labels for every node
    mask: [N] bool — which nodes to score accuracy on
    classifier: final linear layer mapping GAT output to class logits
    Output: accuracy as a plain Python float in [0, 1]
    """
    # training=False turns off both dropout calls inside gat_forward_baseline,
    # so evaluation always sees the full, deterministic feature/attention set.
    output = gat_multihead_forward(X, W_list, a_L_list, a_R_list, row_ptr, col_idx, training=False)  # [N, K * F_out]
    logits = classifier(output)  # [N, num_classes]

    preds = logits.argmax(dim=1)  # [N] — predicted class per node
    correct = (preds[mask] == y_true[mask]).sum()
    accuracy = correct.float() / mask.sum()

    return accuracy.item()


def train_loop(num_epochs=200, num_heads=8):
    """
    num_epochs: how many times to loop through training
    num_heads: K, the number of independent attention heads to run in
               parallel and concatenate (8 matches the original GAT paper)
    """
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    # Initialize X, W, a_L, a_R randomly
    N= data.num_nodes  # number of nodes
    f_in = data.num_features
    f_out = 64
    num_classes = dataset.num_classes

    X = data.x
    y_true = data.y

    # Add a self-loop (i, i) for every node before building the graph. Without
    # this, a node's aggregated output is purely a weighted average of its
    # neighbors — its own features are discarded entirely. Self-loops let each
    # node attend to itself as one of its own "neighbors," so its own signal
    # survives the aggregation step alongside its neighbors' signal.
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=N)

    # Convert edge_index [2, E] to CSR (row_ptr, col_idx) using plain torch,
    # avoiding the torch_sparse dependency (same ABI-crash risk as torch_scatter)
    src, dst = edge_index
    sort_idx = torch.argsort(src)
    sorted_src = src[sort_idx]
    col_idx = dst[sort_idx]  # [E]
    counts = torch.bincount(sorted_src, minlength=N)
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, dim=0)])  # [N + 1]

    # Learnable parameters, Xavier-initialized so H = X @ W (and the
    # attention dot products that follow) don't blow up in scale — see
    # xavier_uniform_param's docstring above for why this matters.
    # One independent W/a_L/a_R triple per head, each initialized separately
    # so the heads start out looking at the graph differently.
    W_list = [xavier_uniform_param(f_in, f_out) for _ in range(num_heads)]
    a_L_list = [xavier_uniform_param(f_out) for _ in range(num_heads)]
    a_R_list = [xavier_uniform_param(f_out) for _ in range(num_heads)]

    # Classifier and optimizer. Input width is num_heads * f_out because
    # gat_multihead_forward concatenates every head's [N, f_out] output.
    classifier = torch.nn.Linear(num_heads * f_out, num_classes)
    # weight_decay adds L2 penalty (shrinks weights toward 0 each step), which
    # discourages any single weight from growing large enough to memorize
    # individual training examples — the other half of fighting overfitting,
    # alongside the dropout added above. 5e-4 matches the original GAT paper.
    optimizer = torch.optim.Adam(
        W_list + a_L_list + a_R_list + list(classifier.parameters()),
        lr=0.005,
        weight_decay=5e-4
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train
    for epoch in range(num_epochs):
        loss = train_step(X, W_list, a_L_list, a_R_list, row_ptr, col_idx,
                          y_true, data.train_mask, classifier, optimizer, loss_fn)

        if epoch % 20 == 0:
            val_acc = evaluate(X, W_list, a_L_list, a_R_list, row_ptr, col_idx,
                               y_true, data.val_mask, classifier)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

    test_acc = evaluate(X, W_list, a_L_list, a_R_list, row_ptr, col_idx,
                        y_true, data.test_mask, classifier)
    print(f"Final Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    train_loop()