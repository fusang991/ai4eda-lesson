#!/usr/bin/env python3
"""
Simplified PyTorch reproduction of Google's AlphaChip (Nature 2021).

Core idea:
  1. Encode circuit netlist with a GNN (Graph Attention)
  2. Autoregressive policy sequentially places cells on a grid
  3. Train with REINFORCE + value baseline + entropy bonus
  4. Reward = negative HPWL (Half-Perimeter Wirelength)
  5. Synthetic benchmark: 30 cells, 45 nets
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

# ─── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')

# ─── Hyperparameters ──────────────────────────────────────────────────────
NUM_CELLS       = 30
GRID_ROWS       = 6
GRID_COLS       = 5
NUM_NETS        = 45
CELL_FEAT_DIM   = 4
GNN_HIDDEN      = 64
GNN_HEADS       = 4
EMBED_DIM       = 64
LR              = 5e-4
NUM_EPISODES    = 1200
EVAL_INTERVAL   = 50
BASELINE_DECAY  = 0.95
ENTROPY_COEF    = 0.02
VALUE_COEF      = 0.5
GRAD_CLIP       = 1.0

# ─── Synthetic Circuit Generator ──────────────────────────────────────────

def generate_synthetic_circuit(num_cells, num_nets, seed=SEED):
    rng = np.random.RandomState(seed)
    nets = []
    for _ in range(num_nets):
        net_size = rng.choice([2, 3, 4], p=[0.5, 0.35, 0.15])
        cells = rng.choice(num_cells, size=min(net_size, num_cells), replace=False).tolist()
        nets.append(cells)
    degrees = defaultdict(int)
    for net in nets:
        for c in net:
            degrees[c] += 1
    max_deg = max(degrees.values()) if degrees else 1
    cell_features = []
    for i in range(num_cells):
        w = 0.8 + 0.4 * rng.random()
        h = 0.8 + 0.4 * rng.random()
        d = degrees.get(i, 0) / max_deg
        cell_features.append([w, h, d, 0.0])
    return nets, np.array(cell_features, dtype=np.float32)


def build_graph_data(nets, cell_features):
    edge_set = set()
    for net in nets:
        for i in range(len(net)):
            for j in range(i + 1, len(net)):
                edge_set.add((net[i], net[j]))
                edge_set.add((net[j], net[i]))
    if len(edge_set) == 0:
        edge_index = torch.zeros((2, 1), dtype=torch.long)
    else:
        src, dst = zip(*edge_set)
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    x = torch.tensor(cell_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def compute_hpwl(placements, nets, grid_rows, grid_cols):
    total_hpwl = 0.0
    for net in nets:
        rows = [placements[c][0] for c in net if c in placements]
        cols = [placements[c][1] for c in net if c in placements]
        if len(rows) < 2:
            continue
        hpwl = (max(rows) - min(rows)) + (max(cols) - min(cols))
        total_hpwl += hpwl
    return total_hpwl


# ─── Model: GNN Encoder + Autoregressive Policy + Value Network ──────────

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        x, edge_index = data.x.to(DEVICE), data.edge_index.to(DEVICE)
        x = F.elu(self.norm1(self.gat1(x, edge_index)))
        x = F.elu(self.norm2(self.gat2(x, edge_index)))
        x = self.gat3(x, edge_index)
        return x


class PlacementPolicy(nn.Module):
    def __init__(self, gnn_hidden, embed_dim, grid_rows, grid_cols):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_slots = grid_rows * grid_cols
        self.gnn_hidden = gnn_hidden

        # Shared context encoder
        self.context_proj = nn.Linear(gnn_hidden, embed_dim)

        # Cell selector
        self.cell_mlp = nn.Sequential(
            nn.Linear(gnn_hidden + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.cell_head = nn.Linear(embed_dim, 1)

        # Slot selector
        self.slot_mlp = nn.Sequential(
            nn.Linear(gnn_hidden + embed_dim + 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.slot_head = nn.Linear(embed_dim, 1)

        self.slot_embeddings = nn.Embedding(self.num_slots, embed_dim)

        # Value network (baseline)
        self.value_net = nn.Sequential(
            nn.Linear(gnn_hidden + 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def get_global_context(self, cell_embeddings, placed_indices):
        if len(placed_indices) > 0:
            ctx = cell_embeddings[placed_indices].mean(dim=0, keepdim=True)
        else:
            ctx = torch.zeros(1, self.gnn_hidden, device=cell_embeddings.device)
        return self.context_proj(ctx)

    def select_cell(self, cell_embeddings, placed_indices):
        num_cells = cell_embeddings.size(0)
        ctx = self.get_global_context(cell_embeddings, placed_indices)  # [1, E]
        ctx_exp = ctx.expand(num_cells, -1)

        h = self.cell_mlp(torch.cat([cell_embeddings.detach(), ctx_exp.detach()], dim=-1))
        cell_logits = self.cell_head(h).squeeze(-1)

        cell_mask = torch.zeros(num_cells, dtype=torch.bool, device=cell_embeddings.device)
        for idx in placed_indices:
            cell_mask[idx] = True
        cell_logits = cell_logits.masked_fill(cell_mask, float('-inf'))

        cell_probs = F.softmax(cell_logits, dim=0)
        # Add small epsilon for numerical stability
        cell_probs = cell_probs + 1e-8
        cell_probs = cell_probs / cell_probs.sum()
        cell_dist = torch.distributions.Categorical(probs=cell_probs)
        cell_idx = cell_dist.sample()
        return cell_idx, cell_dist.log_prob(cell_idx), cell_dist.entropy()

    def select_slot(self, cell_embedding, occupied_slots):
        device = cell_embedding.device
        slot_ids = torch.arange(self.num_slots, device=device)
        slot_embs = self.slot_embeddings(slot_ids)
        coords = torch.stack([slot_ids // self.grid_cols, slot_ids % self.grid_cols], dim=1).float()

        chosen_exp = cell_embedding.detach().unsqueeze(0).expand(self.num_slots, -1)
        h = self.slot_mlp(torch.cat([chosen_exp, slot_embs, coords], dim=-1))
        slot_logits = self.slot_head(h).squeeze(-1)

        slot_mask = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        for idx in occupied_slots:
            slot_mask[idx] = True
        slot_logits = slot_logits.masked_fill(slot_mask, float('-inf'))

        slot_probs = F.softmax(slot_logits, dim=0)
        slot_probs = slot_probs + 1e-8
        slot_probs = slot_probs / slot_probs.sum()
        slot_dist = torch.distributions.Categorical(probs=slot_probs)
        slot_idx = slot_dist.sample()
        return slot_idx, slot_dist.log_prob(slot_idx), slot_dist.entropy()

    def compute_state_value(self, cell_embeddings, placed_indices, occupied_slots):
        """Estimate value of current partial placement state."""
        device = cell_embeddings.device
        if len(placed_indices) == 0:
            return torch.tensor(0.0, device=device)

        # Mean embedding of placed cells
        placed_emb = cell_embeddings[placed_indices].mean(dim=0)  # [H]

        # Average slot position
        rows = torch.tensor([s // self.grid_cols for s in occupied_slots], dtype=torch.float, device=device)
        cols = torch.tensor([s % self.grid_cols for s in occupied_slots], dtype=torch.float, device=device)
        avg_pos = torch.tensor([rows.mean(), cols.mean()], device=device)

        state = torch.cat([placed_emb, avg_pos])
        return self.value_net(state).squeeze()


# ─── Greedy Placement ────────────────────────────────────────────────────

def greedy_place(gnn, policy, graph_data, num_cells, grid_rows, grid_cols):
    """Run greedy (argmax) placement."""
    gnn.eval()
    policy.eval()
    with torch.no_grad():
        cell_embeddings = gnn(graph_data)
        placements = {}
        placed_indices = []
        occupied_slots = []

        for step in range(num_cells):
            num_c = cell_embeddings.size(0)
            ctx = policy.get_global_context(cell_embeddings, placed_indices)
            ctx_exp = ctx.expand(num_c, -1)

            h = policy.cell_mlp(torch.cat([cell_embeddings, ctx_exp], dim=-1))
            cell_logits = policy.cell_head(h).squeeze(-1)
            cell_mask = torch.zeros(num_c, dtype=torch.bool, device=DEVICE)
            for idx in placed_indices:
                cell_mask[idx] = True
            cell_logits = cell_logits.masked_fill(cell_mask, float('-inf'))
            cell_idx = torch.argmax(cell_logits).item()

            chosen_emb = cell_embeddings[cell_idx]
            slot_ids = torch.arange(grid_rows * grid_cols, device=DEVICE)
            slot_embs = policy.slot_embeddings(slot_ids)
            coords = torch.stack([slot_ids // grid_cols, slot_ids % grid_cols], dim=1).float()
            chosen_exp = chosen_emb.unsqueeze(0).expand(grid_rows * grid_cols, -1)
            h_s = policy.slot_mlp(torch.cat([chosen_exp, slot_embs, coords], dim=-1))
            slot_logits = policy.slot_head(h_s).squeeze(-1)
            slot_mask = torch.zeros(grid_rows * grid_cols, dtype=torch.bool, device=DEVICE)
            for idx in occupied_slots:
                slot_mask[idx] = True
            slot_logits = slot_logits.masked_fill(slot_mask, float('-inf'))
            slot_idx = torch.argmax(slot_logits).item()

            row = slot_idx // grid_cols
            col = slot_idx % grid_cols
            placements[cell_idx] = (row, col)
            placed_indices.append(cell_idx)
            occupied_slots.append(slot_idx)

    return placements


# ─── Training Loop ────────────────────────────────────────────────────────

def train():
    print("=" * 70)
    print("  AlphaChip Simplified Reproduction (PyTorch + PyG)")
    print("  GNN + REINFORCE + Value Baseline for Circuit Placement")
    print("=" * 70)

    nets, cell_features = generate_synthetic_circuit(NUM_CELLS, NUM_NETS)
    graph_data = build_graph_data(nets, cell_features)
    print(f"\nSynthetic circuit: {NUM_CELLS} cells, {NUM_NETS} nets")
    print(f"Grid: {GRID_ROWS}x{GRID_COLS} = {GRID_ROWS * GRID_COLS} slots")
    print(f"Edges in cell graph: {graph_data.edge_index.size(1)}")

    gnn = GNNEncoder(CELL_FEAT_DIM, GNN_HIDDEN, GNN_HEADS).to(DEVICE)
    policy = PlacementPolicy(GNN_HIDDEN, EMBED_DIM, GRID_ROWS, GRID_COLS).to(DEVICE)
    all_params = list(gnn.parameters()) + list(policy.parameters())
    optimizer = torch.optim.Adam(all_params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPISODES, eta_min=1e-5)

    episode_rewards = []
    episode_hpwls = []
    running_mean = None
    running_var = None
    reward_buffer = []

    print(f"\nTraining for {NUM_EPISODES} episodes...\n")

    for ep in range(1, NUM_EPISODES + 1):
        gnn.train()
        policy.train()

        cell_embeddings = gnn(graph_data)

        placements = {}
        placed_indices = []
        occupied_slots = []
        log_probs = []
        entropies = []
        values = []

        for step in range(NUM_CELLS):
            # State value before this step
            val = policy.compute_state_value(cell_embeddings, placed_indices, occupied_slots)
            values.append(val)

            cell_idx, cell_lp, cell_ent = policy.select_cell(cell_embeddings, placed_indices)
            slot_idx, slot_lp, slot_ent = policy.select_slot(
                cell_embeddings[cell_idx.item()], occupied_slots
            )

            log_probs.append(cell_lp + slot_lp)
            entropies.append(cell_ent + slot_ent)

            row = slot_idx.item() // GRID_COLS
            col = slot_idx.item() % GRID_COLS
            placements[cell_idx.item()] = (row, col)
            placed_indices.append(cell_idx.item())
            occupied_slots.append(slot_idx.item())

        hpwl = compute_hpwl(placements, nets, GRID_ROWS, GRID_COLS)
        reward = -hpwl / 100.0  # Scale reward

        # Update running statistics for reward normalization
        reward_buffer.append(reward)
        if len(reward_buffer) > 100:
            reward_buffer.pop(0)

        if len(reward_buffer) >= 10:
            r_mean = np.mean(reward_buffer)
            r_std = max(np.std(reward_buffer), 1e-6)
        else:
            r_mean = 0.0
            r_std = 1.0

        # REINFORCE loss with learned value baseline
        policy_loss = 0
        value_loss = 0
        total_entropy = 0

        for i in range(NUM_CELLS):
            advantage = (reward - r_mean) / r_std
            # Subtract value baseline
            advantage_detached = advantage - values[i].detach()
            policy_loss += -log_probs[i] * advantage_detached
            value_loss += F.mse_loss(values[i], torch.tensor(reward, dtype=torch.float, device=DEVICE))
            total_entropy += entropies[i]

        policy_loss = policy_loss / NUM_CELLS
        value_loss = value_loss / NUM_CELLS
        total_entropy = total_entropy / NUM_CELLS

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * total_entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        episode_rewards.append(reward)
        episode_hpwls.append(hpwl)

        if ep % EVAL_INTERVAL == 0 or ep == 1:
            avg_hpwl = np.mean(episode_hpwls[-EVAL_INTERVAL:])
            min_hpwl = np.min(episode_hpwls[-EVAL_INTERVAL:])
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Ep {ep:4d} | HPWL: {hpwl:.0f} | Avg: {avg_hpwl:.0f} | "
                  f"Best({EVAL_INTERVAL}): {min_hpwl:.0f} | LR: {lr_now:.1e}")

    # ── Greedy evaluation ──────────────────────────────────────────────
    print("\nGreedy evaluation...")
    placements_greedy = greedy_place(gnn, policy, graph_data, NUM_CELLS, GRID_ROWS, GRID_COLS)
    greedy_hpwl = compute_hpwl(placements_greedy, nets, GRID_ROWS, GRID_COLS)
    print(f"Greedy HPWL: {greedy_hpwl:.1f}")

    # ── Random baseline ────────────────────────────────────────────────
    random_hpwls = []
    for _ in range(500):
        cells = list(range(NUM_CELLS))
        random.shuffle(cells)
        slots = list(range(GRID_ROWS * GRID_COLS))
        random.shuffle(slots)
        rp = {cells[i]: (slots[i] // GRID_COLS, slots[i] % GRID_COLS)
              for i in range(NUM_CELLS)}
        random_hpwls.append(compute_hpwl(rp, nets, GRID_ROWS, GRID_COLS))
    random_avg = np.mean(random_hpwls)
    random_min = np.min(random_hpwls)
    print(f"Random HPWL: avg={random_avg:.1f}, min={random_min:.1f} (500 samples)")

    # Best training episode
    best_train_hpwl = min(episode_hpwls)
    best_train_ep = episode_hpwls.index(best_train_hpwl) + 1

    improvement_vs_avg = (random_avg - greedy_hpwl) / random_avg * 100
    improvement_vs_best = (random_min - best_train_hpwl) / random_min * 100

    # ── Visualization ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (a) Training curve: HPWL
    ax = axes[0, 0]
    window = 30
    if len(episode_hpwls) >= window:
        smoothed = np.convolve(episode_hpwls, np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(episode_hpwls) + 1), smoothed, 'b-', linewidth=1.5,
                label=f'Smoothed ({window}-ep)')
    ax.plot(episode_hpwls, alpha=0.2, color='blue', linewidth=0.5)
    ax.axhline(random_avg, color='red', linestyle='--', linewidth=1.5,
               label=f'Random avg ({random_avg:.0f})')
    ax.axhline(greedy_hpwl, color='green', linestyle='-', linewidth=2,
               label=f'Greedy ({greedy_hpwl:.0f})')
    ax.axhline(best_train_hpwl, color='orange', linestyle=':', linewidth=2,
               label=f'Best train ({best_train_hpwl:.0f}, ep {best_train_ep})')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('HPWL', fontsize=12)
    ax.set_title('(a) Training: HPWL over Episodes', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) Reward curve
    ax = axes[0, 1]
    if len(episode_rewards) >= window:
        smoothed_r = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(episode_rewards) + 1), smoothed_r, 'g-', linewidth=1.5,
                label='Smoothed reward')
    ax.plot(episode_rewards, alpha=0.15, color='green', linewidth=0.5)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (-HPWL/100)', fontsize=12)
    ax.set_title('(b) Training: Reward Curve', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (c) Greedy placement visualization
    ax = axes[1, 0]
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'(c) Greedy Placement (HPWL={greedy_hpwl:.0f})', fontsize=13)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            rect = Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                             fill=True, facecolor='#f8f8f8', edgecolor='#cccccc', linewidth=0.5)
            ax.add_patch(rect)

    cmap = plt.cm.Set3
    cell_colors = {}
    for ni, net in enumerate(nets):
        for c in net:
            if c not in cell_colors:
                cell_colors[c] = cmap(ni % 12)

    for cell_id, (row, col) in placements_greedy.items():
        color = cell_colors.get(cell_id, 'steelblue')
        rect = Rectangle((col - 0.35, row - 0.35), 0.7, 0.7,
                         fill=True, facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(col, row, str(cell_id), ha='center', va='center', fontsize=7, fontweight='bold')

    # Draw net bounding boxes
    for ni, net in enumerate(nets[:20]):
        pts = [placements_greedy[c] for c in net if c in placements_greedy]
        if len(pts) >= 2:
            rows_pts = [p[0] for p in pts]
            cols_pts = [p[1] for p in pts]
            rmin, rmax = min(rows_pts), max(rows_pts)
            cmin, cmax = min(cols_pts), max(cols_pts)
            rect = Rectangle((cmin - 0.35, rmin - 0.35),
                             cmax - cmin + 0.7, rmax - rmin + 0.7,
                             fill=False, edgecolor='red', linewidth=0.4, linestyle='--', alpha=0.35)
            ax.add_patch(rect)

    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.invert_yaxis()

    # (d) HPWL distribution
    ax = axes[1, 1]
    ax.hist(random_hpwls, bins=30, alpha=0.6, color='salmon', label='Random (500 samples)', density=True)
    ax.axvline(greedy_hpwl, color='green', linewidth=2.5, linestyle='-',
               label=f'Greedy ({greedy_hpwl:.0f})')
    ax.axvline(best_train_hpwl, color='orange', linewidth=2.5, linestyle=':',
               label=f'Best train ({best_train_hpwl:.0f})')
    ax.axvline(random_avg, color='red', linewidth=2, linestyle='--',
               label=f'Random avg ({random_avg:.0f})')
    ax.set_xlabel('HPWL', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('(d) HPWL Distribution: Learned vs Random', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('AlphaChip Simplified Reproduction\n'
                 f'GNN Encoder + REINFORCE Policy | {NUM_CELLS} cells, {NUM_NETS} nets, '
                 f'{GRID_ROWS}×{GRID_COLS} grid',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alphachip_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {out_path}")

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Architecture:    3-layer GAT + autoregressive policy + value baseline")
    print(f"  Circuit:         {NUM_CELLS} cells, {NUM_NETS} nets")
    print(f"  Grid:            {GRID_ROWS}×{GRID_COLS}")
    print(f"  Episodes:        {NUM_EPISODES}")
    print(f"  Random avg HPWL: {random_avg:.1f}")
    print(f"  Random min HPWL: {random_min:.1f}")
    print(f"  Best train HPWL: {best_train_hpwl:.1f} (ep {best_train_ep})")
    print(f"  Greedy HPWL:     {greedy_hpwl:.1f}")
    print(f"  vs random avg:   {improvement_vs_avg:+.1f}%")
    print("=" * 70)

    return {
        'greedy_hpwl': greedy_hpwl,
        'best_train_hpwl': best_train_hpwl,
        'random_avg_hpwl': random_avg,
        'random_min_hpwl': random_min,
        'improvement_vs_avg': improvement_vs_avg,
    }


if __name__ == '__main__':
    results = train()
