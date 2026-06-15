#!/usr/bin/env python3
"""
GNN Timing Prediction (DAC 2022 Style)
=======================================
Graph Neural Network for gate-level timing prediction.
- Nodes = standard cells with features (gate type, fanin/fanout, x/y, capacitance)
- Edges = net connections (wires between pins)
- Target = arrival time at each node (regression)

Uses GCN and GraphSAGE backbones on synthetic circuit graphs.
"""

import os, math, random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------
# Gate library (synthetic)
# ------------------------------------------------------------------
GATE_TYPES = ["INV", "BUF", "NAND2", "NAND3", "NOR2", "NOR3", "AND2", "OR2", "XOR2", "XNOR2", "MUX2", "DFF"]
NUM_GATE_TYPES = len(GATE_TYPES)
GATE_DELAY = {  # intrinsic delay (ps)
    "INV": 5, "BUF": 3, "NAND2": 8, "NAND3": 10, "NOR2": 9, "NOR3": 12,
    "AND2": 10, "OR2": 10, "XOR2": 12, "XNOR2": 12, "MUX2": 14, "DFF": 0
}
GATE_CAP = {  # input capacitance (fF)
    "INV": 1.0, "BUF": 1.0, "NAND2": 1.2, "NAND3": 1.5, "NOR2": 1.3, "NOR3": 1.6,
    "AND2": 1.4, "OR2": 1.4, "XOR2": 1.6, "XNOR2": 1.6, "MUX2": 1.8, "DFF": 2.0
}
WIRE_CAP_PER_UM = 0.2   # fF/um
WIRE_RES_PER_UM = 0.5   # Ohm/um

# ------------------------------------------------------------------
# Synthetic circuit graph generation
# ------------------------------------------------------------------

def _random_fanin_fanout(gate_name):
    if gate_name in ("INV", "BUF"):
        return 1, random.randint(1, 4)
    if gate_name == "DFF":
        return 1, random.randint(1, 3)
    return 2, random.randint(1, 4)


def generate_circuit_graph(num_nodes_range=(80, 250)):
    """Generate one synthetic circuit DAG and compute arrival times."""
    num_nodes = random.randint(*num_nodes_range)
    # --- build a random DAG by topological layering ---
    # assign each node to a layer (more layers = deeper logic)
    num_layers = random.randint(4, max(5, num_nodes // 10))
    node_layer = np.random.randint(0, num_layers, size=num_nodes)
    node_layer = np.sort(node_layer)  # sort so early layers have lower indices

    # node features
    gate_ids = np.random.randint(0, NUM_GATE_TYPES, size=num_nodes)
    gate_names = [GATE_TYPES[g] for g in gate_ids]
    x_pos = np.random.uniform(0, 500, size=num_nodes).astype(np.float32)
    y_pos = np.random.uniform(0, 500, size=num_nodes).astype(np.float32)
    cap_load = np.array([GATE_CAP[g] for g in gate_names], dtype=np.float32)
    drive_strength = np.random.uniform(0.5, 2.0, size=num_nodes).astype(np.float32)

    # one-hot gate type
    gate_onehot = np.zeros((num_nodes, NUM_GATE_TYPES), dtype=np.float32)
    for i, g in enumerate(gate_ids):
        gate_onehot[i, g] = 1.0

    # fanin / fanout per node
    fanin_count = np.zeros(num_nodes, dtype=np.float32)
    fanout_count = np.zeros(num_nodes, dtype=np.float32)

    # build edges: for each node, pick random fanins from earlier layers
    src_list, dst_list = [], []
    wire_lengths = []
    for i in range(num_nodes):
        fi, _ = _random_fanin_fanout(gate_names[i])
        fi = min(fi, max(1, node_layer[:i].sum()))  # ensure possible
        candidates = np.where(node_layer < node_layer[i])[0]
        if len(candidates) == 0 and i > 0:
            candidates = np.array([i - 1])
        if i == 0:
            continue
        k = min(fi, len(candidates))
        fanins = np.random.choice(candidates, size=k, replace=False)
        for f in fanins:
            src_list.append(int(f))
            dst_list.append(int(i))
            dist = math.sqrt((x_pos[f] - x_pos[i])**2 + (y_pos[f] - y_pos[i])**2)
            wire_lengths.append(dist)
            fanin_count[i] += 1
            fanout_count[f] += 1

    if len(src_list) == 0:
        return generate_circuit_graph(num_nodes_range)  # retry

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    wire_lengths = np.array(wire_lengths, dtype=np.float32)

    # --- compute arrival times (STA-like, deterministic given graph) ---
    arrival = np.zeros(num_nodes, dtype=np.float32)
    wire_delay = wire_lengths * WIRE_RES_PER_UM * WIRE_CAP_PER_UM * 0.5  # simplified RC
    # process in topological order (already partly sorted by layer; use BFS)
    in_degree = np.zeros(num_nodes, dtype=int)
    for s, d in zip(src_list, dst_list):
        in_degree[d] += 1
    queue = [i for i in range(num_nodes) if in_degree[i] == 0]
    processed = 0
    while queue:
        n = queue.pop(0)
        processed += 1
        gate_d = GATE_DELAY[gate_names[n]]
        # find all outgoing edges from n
        for idx, (s, d) in enumerate(zip(src_list, dst_list)):
            if s == n:
                edge_arr = arrival[n] + gate_d + wire_delay[idx]
                arrival[d] = max(arrival[d], edge_arr)
            if d == n:
                in_degree[d_post := d]  # already decremented via queue logic
        # decrement in-degree of successors
        for idx, (s, d) in enumerate(zip(src_list, dst_list)):
            if s == n:
                in_degree[d] -= 1
                if in_degree[d] == 0:
                    queue.append(d)

    # normalize arrival times to [0, 1] range for training stability
    at_max = arrival.max()
    if at_max > 0:
        arrival = arrival / at_max
    arrival = arrival.astype(np.float32)

    # slack = 1 - arrival (simplified)
    slack = (1.0 - arrival).astype(np.float32)

    # build node feature matrix: [gate_onehot | fanin | fanout | cap_load | drive_strength | x_norm | y_norm]
    x_feat = np.concatenate([
        gate_onehot,
        (fanin_count / max(fanin_count.max(), 1))[:, None],
        (fanout_count / max(fanout_count.max(), 1))[:, None],
        cap_load[:, None] / 3.0,
        drive_strength[:, None] / 2.0,
        (x_pos / 500.0)[:, None],
        (y_pos / 500.0)[:, None],
    ], axis=1).astype(np.float32)

    data = Data(
        x=torch.from_numpy(x_feat),
        edge_index=edge_index,
        y=torch.from_numpy(arrival[:, None]),   # predict arrival time
        slack=torch.from_numpy(slack[:, None]),
    )
    return data


def generate_dataset(num_graphs=2000):
    graphs = [generate_circuit_graph() for _ in range(num_graphs)]
    return graphs

# ------------------------------------------------------------------
# GNN models
# ------------------------------------------------------------------

class TimingGCN(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        self.norms.append(nn.LayerNorm(hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, ei)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class TimingGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        self.norms.append(nn.LayerNorm(hidden))
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, ei)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
        n += batch.num_nodes
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        total_loss += loss.item() * batch.num_nodes
        n += batch.num_nodes
        preds.append(pred.cpu())
        trues.append(batch.y.cpu())
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    mae = (preds - trues).abs().mean().item()
    rmse = ((preds - trues)**2).mean().sqrt().item()
    # R^2
    ss_res = ((preds - trues)**2).sum()
    ss_tot = ((trues - trues.mean())**2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return total_loss / n, mae, rmse, r2.item()


def run_experiment():
    print("=" * 60)
    print("GNN Timing Prediction  (DAC 2022 style)")
    print("=" * 60)

    # --- generate data ---
    print("\n[1/5] Generating synthetic circuit graphs ...")
    graphs = generate_dataset(num_graphs=1500)
    random.shuffle(graphs)
    n_train = 1200
    train_data = graphs[:n_train]
    test_data = graphs[n_train:]
    in_dim = train_data[0].x.shape[1]
    print(f"  {len(train_data)} train / {len(test_data)} test graphs,  node feat dim = {in_dim}")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # --- build models ---
    print("\n[2/5] Building GCN and GraphSAGE models ...")
    gcn = TimingGCN(in_dim, hidden=128, layers=4).to(DEVICE)
    sage = TimingGraphSAGE(in_dim, hidden=128, layers=4).to(DEVICE)
    print(f"  GCN params:  {sum(p.numel() for p in gcn.parameters()):,}")
    print(f"  SAGE params: {sum(p.numel() for p in sage.parameters()):,}")

    # --- train ---
    print("\n[3/5] Training (60 epochs each) ...")
    EPOCHS = 60
    criterion = nn.MSELoss()

    history = {"gcn_train": [], "gcn_test": [], "sage_train": [], "sage_test": [],
               "gcn_mae": [], "gcn_rmse": [], "gcn_r2": [],
               "sage_mae": [], "sage_rmse": [], "sage_r2": []}

    for name, model in [("GCN", gcn), ("GraphSAGE", sage)]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        key = name.lower().replace("graph", "")  # gcn / sage
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            te_loss, mae, rmse, r2 = evaluate(model, test_loader, criterion)
            scheduler.step()
            history[f"{key}_train"].append(tr_loss)
            history[f"{key}_test"].append(te_loss)
            history[f"{key}_mae"].append(mae)
            history[f"{key}_rmse"].append(rmse)
            history[f"{key}_r2"].append(r2)
            if epoch % 15 == 0 or epoch == 1:
                print(f"  [{name}] epoch {epoch:3d}  train_loss={tr_loss:.5f}  "
                      f"test_loss={te_loss:.5f}  MAE={mae:.4f}  R²={r2:.4f}")

    # --- final metrics ---
    print("\n[4/5] Final metrics:")
    for key, name in [("gcn", "GCN"), ("sage", "GraphSAGE")]:
        print(f"  {name:12s}  MSE={history[f'{key}_test'][-1]:.5f}  "
              f"MAE={history[f'{key}_mae'][-1]:.4f}  "
              f"RMSE={history[f'{key}_rmse'][-1]:.4f}  "
              f"R²={history[f'{key}_r2'][-1]:.4f}")

    # --- plot ---
    print("\n[5/5] Plotting ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GNN Timing Prediction (DAC 2022 style)", fontsize=14, fontweight="bold")
    epochs = range(1, EPOCHS + 1)

    # (a) training loss
    ax = axes[0, 0]
    ax.plot(epochs, history["gcn_train"], label="GCN train", color="#2563eb")
    ax.plot(epochs, history["gcn_test"], "--", label="GCN test", color="#2563eb", alpha=0.6)
    ax.plot(epochs, history["sage_train"], label="GraphSAGE train", color="#dc2626")
    ax.plot(epochs, history["sage_test"], "--", label="GraphSAGE test", color="#dc2626", alpha=0.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.set_title("(a) Training & Test Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (b) MAE
    ax = axes[0, 1]
    ax.plot(epochs, history["gcn_mae"], label="GCN", color="#2563eb")
    ax.plot(epochs, history["sage_mae"], label="GraphSAGE", color="#dc2626")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE"); ax.set_title("(b) Mean Absolute Error")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (c) R²
    ax = axes[1, 0]
    ax.plot(epochs, history["gcn_r2"], label="GCN", color="#2563eb")
    ax.plot(epochs, history["sage_r2"], label="GraphSAGE", color="#dc2626")
    ax.set_xlabel("Epoch"); ax.set_ylabel("R²"); ax.set_title("(c) R² Score")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (d) scatter: pred vs true for best model
    ax = axes[1, 1]
    best_key = "gcn" if history["gcn_r2"][-1] >= history["sage_r2"][-1] else "sage"
    best_model = gcn if best_key == "gcn" else sage
    best_model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            p = best_model(batch).cpu().numpy().flatten()
            t = batch.y.cpu().numpy().flatten()
            all_pred.extend(p); all_true.extend(t)
    ax.scatter(all_true, all_pred, s=6, alpha=0.4, color="#2563eb")
    mn, mx = min(min(all_true), min(all_pred)), max(max(all_true), max(all_pred))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="ideal")
    ax.set_xlabel("True Arrival Time"); ax.set_ylabel("Predicted Arrival Time")
    best_name = "GCN" if best_key == "gcn" else "GraphSAGE"
    ax.set_title(f"(d) Pred vs True ({best_name})"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "gnn_timing.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    print("\nDone. ✓")


if __name__ == "__main__":
    run_experiment()
