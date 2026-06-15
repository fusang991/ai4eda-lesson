#!/usr/bin/env python3
"""
RouteNet Experiment — ICCAD 2018 Paper Reproduction
=====================================================
CNN-based routability prediction: predict 2D congestion maps from placement features.

Reference: RouteNet: Routability Prediction for Mixed-Size Designs Using
           Convolutional Neural Network (ICCAD 2018)

This script:
1. Generates synthetic placement feature maps (cell density, pin density, net density)
2. Generates synthetic congestion maps as ground truth labels
3. Builds a U-Net style CNN to predict congestion from placement features
4. Trains the model and evaluates (MAE, Pearson correlation)
5. Plots predicted vs actual congestion maps → routenet_results.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── hyperparameters ──────────────────────────────────────────────────────────
GRID_SIZE = 64          # 64×64 placement grid
N_CHANNELS = 3          # cell density, pin density, net density
N_SAMPLES = 500         # synthetic design samples
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── 1. Synthetic data generation ─────────────────────────────────────────────

def generate_sample(grid_size=GRID_SIZE):
    """Generate one synthetic placement + congestion pair.

    Features (3 channels):
        Ch0 – cell density:  clusters of cells placed at certain locations
        Ch1 – pin density:   correlated with cell density + boundary pins
        Ch2 – net density:   wires connecting cell clusters (spatial spread)

    Label:
        Congestion map proportional to local net density × (1 + cell density noise).
    """
    g = grid_size
    # base random field
    base = np.random.rand(g, g).astype(np.float32)

    # cell density — spatially correlated (Gaussian blobs)
    cell_density = np.zeros((g, g), dtype=np.float32)
    n_clusters = np.random.randint(3, 10)
    for _ in range(n_clusters):
        cx, cy = np.random.randint(0, g, 2)
        sx, sy = np.random.uniform(5, 15, 2)
        yy, xx = np.ogrid[:g, :g]
        blob = np.exp(-((xx - cx)**2 / (2*sx**2) + (yy - cy)**2 / (2*sy**2)))
        cell_density += blob * np.random.uniform(0.3, 1.0)
    cell_density = np.clip(cell_density, 0, 1)

    # pin density — correlated with cells + uniform baseline
    pin_density = 0.3 * cell_density + 0.15 * base
    # add boundary emphasis
    border = np.zeros((g, g), dtype=np.float32)
    border[:3, :] += 0.2; border[-3:, :] += 0.2
    border[:, :3] += 0.2; border[:, -3:] += 0.2
    pin_density = np.clip(pin_density + border * 0.15, 0, 1)

    # net density — spread version of cell density (convolution-like)
    net_density = np.zeros((g, g), dtype=np.float32)
    n_nets = np.random.randint(5, 20)
    for _ in range(n_nets):
        x1, y1 = np.random.randint(0, g, 2)
        x2, y2 = np.random.randint(0, g, 2)
        # line between two points
        length = max(abs(x2 - x1), abs(y2 - y1), 1)
        xs = np.linspace(x1, x2, length * 3).astype(int).clip(0, g - 1)
        ys = np.linspace(y1, y2, length * 3).astype(int).clip(0, g - 1)
        net_density[ys, xs] += 1.0
    # blur slightly
    from scipy.ndimage import gaussian_filter
    net_density = gaussian_filter(net_density, sigma=2.0)
    net_density = net_density / (net_density.max() + 1e-8)

    # congestion label — driven primarily by net density modulated by cell density
    congestion = 0.6 * net_density + 0.3 * cell_density + 0.1 * np.random.rand(g, g).astype(np.float32)
    congestion = gaussian_filter(congestion, sigma=1.5)
    congestion = congestion / (congestion.max() + 1e-8)

    features = np.stack([cell_density, pin_density, net_density], axis=0)  # (3, H, W)
    return features, congestion[np.newaxis, ...]  # (1, H, W)


class RouteNetDataset(Dataset):
    def __init__(self, n_samples, grid_size=GRID_SIZE):
        self.features = []
        self.labels = []
        for _ in range(n_samples):
            f, l = generate_sample(grid_size)
            self.features.append(f)
            self.labels.append(l)
        self.features = torch.from_numpy(np.array(self.features))
        self.labels = torch.from_numpy(np.array(self.labels))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── 2. CNN Model (lightweight U-Net style) ───────────────────────────────────

class RouteNet(nn.Module):
    """Encoder-decoder CNN for congestion map prediction."""

    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # encoder
        self.enc1 = self._block(in_ch, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = self._block(128, 256)
        # decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)
        self.out_conv = nn.Conv2d(32, out_ch, 1)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)              # (B,32,64,64)
        e2 = self.enc2(self.pool(e1))  # (B,64,32,32)
        e3 = self.enc3(self.pool(e2))  # (B,128,16,16)
        b  = self.bottleneck(self.pool(e3))  # (B,256,8,8)
        d3 = self.up3(b)                      # (B,128,16,16)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3)                     # (B,64,32,32)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)                     # (B,32,64,64)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        return torch.sigmoid(self.out_conv(d1))


# ── 3. Training & Evaluation ─────────────────────────────────────────────────

def train_model(model, train_loader, val_loader):
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                val_loss += criterion(pred, y).item() * X.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  train_MAE={train_loss:.4f}  val_MAE={val_loss:.4f}")

    return train_losses, val_losses


def evaluate(model, loader):
    """Return MAE and per-sample Pearson correlation."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            pred = model(X).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
    all_pred = np.concatenate(all_pred, 0)
    all_true = np.concatenate(all_true, 0)

    mae = np.mean(np.abs(all_pred - all_true))
    # per-sample correlation
    corrs = []
    for i in range(len(all_pred)):
        p = all_pred[i].flatten()
        t = all_true[i].flatten()
        if p.std() > 1e-8 and t.std() > 1e-8:
            corrs.append(pearsonr(p, t)[0])
    mean_corr = np.mean(corrs)
    return mae, mean_corr, all_pred, all_true


# ── 4. Plotting ──────────────────────────────────────────────────────────────

def plot_results(train_losses, val_losses, mae, corr, preds, trues, save_path="routenet_results.png"):
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle("RouteNet — CNN Routability Prediction (ICCAD 2018 Reproduction)", fontsize=15, fontweight="bold")

    # row 0: training curves + metrics
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train MAE", linewidth=1.5)
    ax.plot(val_losses, label="Val MAE", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE Loss"); ax.set_title("Training Curve"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.axis("off")
    ax.text(0.5, 0.5, f"Test MAE: {mae:.4f}\nPearson r: {corr:.4f}\nSamples: {len(preds)}\nGrid: {GRID_SIZE}×{GRID_SIZE}",
            transform=ax.transAxes, fontsize=14, va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))
    ax.set_title("Metrics")

    # scatter of pixel values (sample 0)
    ax = axes[0, 2]
    ax.scatter(trues[0].flatten()[::8], preds[0].flatten()[::8], s=2, alpha=0.3)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Pixel Scatter (sample 0)"); ax.grid(True, alpha=0.3)

    # error distribution
    ax = axes[0, 3]
    errors = (preds - trues).flatten()
    ax.hist(errors, bins=80, color="steelblue", edgecolor="none")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Prediction Error"); ax.set_title("Error Distribution"); ax.grid(True, alpha=0.3)

    # rows 1-2: show 4 samples — input channels, true congestion, predicted congestion
    for i in range(4):
        col = i
        sample_idx = i * (len(preds) // 4)

        # row 1: true congestion
        ax = axes[1, col]
        im = ax.imshow(trues[sample_idx, 0], cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"True Congestion #{sample_idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # row 2: predicted congestion
        ax = axes[2, col]
        im = ax.imshow(preds[sample_idx, 0], cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"Predicted #{sample_idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Results saved to {save_path}")


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RouteNet — CNN Routability Prediction (ICCAD 2018)")
    print("=" * 60)

    # generate data
    print(f"\n[1] Generating {N_SAMPLES} synthetic placement samples ({GRID_SIZE}×{GRID_SIZE}, {N_CHANNELS} channels)...")
    dataset = RouteNetDataset(N_SAMPLES, GRID_SIZE)
    # 70/15/15 split
    n_train = int(0.7 * N_SAMPLES)
    n_val = int(0.15 * N_SAMPLES)
    n_test = N_SAMPLES - n_train - n_val
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test],
                                                                generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"    Train={n_train}  Val={n_val}  Test={n_test}")

    # build model
    print(f"\n[2] Building RouteNet CNN on {DEVICE}...")
    model = RouteNet(in_ch=N_CHANNELS, out_ch=1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {n_params:,}")

    # train
    print(f"\n[3] Training for {EPOCHS} epochs...")
    train_losses, val_losses = train_model(model, train_loader, val_loader)

    # evaluate
    print("\n[4] Evaluating on test set...")
    mae, corr, preds, trues = evaluate(model, test_loader)
    print(f"    Test MAE:         {mae:.4f}")
    print(f"    Pearson corr:     {corr:.4f}")

    # plot
    print("\n[5] Plotting results...")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(out_dir, "routenet_results.png")
    plot_results(train_losses, val_losses, mae, corr, preds, trues, save_path)

    print("\nDone!")
    return mae, corr


if __name__ == "__main__":
    main()
