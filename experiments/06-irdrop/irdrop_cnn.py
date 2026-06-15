#!/usr/bin/env python
"""
IR Drop Prediction using Inception-style CNN
=============================================
Predicts IR drop maps from chip physical features using a multi-scale
convolutional neural network inspired by GoogLeNet/Inception architecture.

Channels (input features):
  0 - Switching activity density
  1 - Cell density
  2 - Distance to nearest power pad
  3 - Clock network density
  4 - Metal layer density

Output: IR drop voltage map (same spatial resolution)
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── reproducibility ──────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── synthetic data generation ───────────────────────────────────────
GRID       = 64        # spatial resolution 64x64
N_SAMPLES  = 800       # total samples
PAD_POSITIONS = [(8,8),(8,56),(56,8),(56,56)]  # 4 power pads at corners

def _smooth(arr, k=5):
    """Box-blur smoothing using numpy convolution."""
    kernel = np.ones((k, k), dtype=np.float32) / (k * k)
    # Pad edges with reflect mode for clean boundaries
    pad = k // 2
    padded = np.pad(arr, pad, mode='reflect')
    # Sliding-window via stride tricks
    shape = arr.shape + (k, k)
    strides = padded.strides * 2
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return (windows * kernel).sum(axis=(-2, -1))

def generate_sample():
    """Return (5, GRID, GRID) features, (1, GRID, GRID) IR drop."""
    # Switching activity – hot-spots in random regions
    switch = np.random.rand(GRID, GRID) * 0.3
    for _ in range(np.random.randint(3, 8)):
        cx, cy = np.random.randint(0, GRID, 2)
        r = np.random.uniform(5, 15)
        yy, xx = np.ogrid[:GRID, :GRID]
        blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r**2))
        switch += blob * np.random.uniform(0.4, 1.0)
    switch = np.clip(switch, 0, 1)

    # Cell density – correlated with switching but with noise
    cell_den = _smooth(switch * 0.6 + np.random.rand(GRID, GRID) * 0.4, k=4)
    cell_den = (cell_den - cell_den.min()) / (cell_den.max() - cell_den.min() + 1e-8)

    # Distance to nearest power pad (normalised 0-1)
    yy, xx = np.mgrid[:GRID, :GRID]
    dist = np.full((GRID, GRID), 1e9, dtype=np.float32)
    for (py, px) in PAD_POSITIONS:
        d = np.sqrt((yy - py)**2 + (xx - px)**2)
        dist = np.minimum(dist, d)
    dist = dist / dist.max()

    # Clock density – higher along central stripe + noise
    clock = np.zeros((GRID, GRID), dtype=np.float32)
    clock[:, GRID//2-3:GRID//2+3] = 0.7
    clock += np.random.rand(GRID, GRID) * 0.3
    clock = _smooth(clock, k=3)
    clock = (clock - clock.min()) / (clock.max() - clock.min() + 1e-8)

    # Metal density – random per-region
    metal = _smooth(np.random.rand(GRID, GRID), k=6)
    metal = (metal - metal.min()) / (metal.max() - metal.min() + 1e-8)

    features = np.stack([switch, cell_den, dist, clock, metal], axis=0).astype(np.float32)

    # IR drop: primarily driven by switching × distance, with some noise
    ir = 0.55 * switch + 0.30 * dist + 0.10 * cell_den + 0.05 * np.random.rand(GRID, GRID).astype(np.float32)
    ir = _smooth(ir, k=3).astype(np.float32)
    ir = (ir - ir.min()) / (ir.max() - ir.min() + 1e-8)  # normalise to [0,1]
    ir = ir[np.newaxis, ...]  # (1, H, W)

    return features, ir

class IRDropDataset(Dataset):
    def __init__(self, n=N_SAMPLES):
        self.X, self.Y = [], []
        for _ in range(n):
            x, y = generate_sample()
            self.X.append(x)
            self.Y.append(y)
        self.X = torch.from_numpy(np.stack(self.X))
        self.Y = torch.from_numpy(np.stack(self.Y))

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.Y[i]

# ── Inception-style CNN ─────────────────────────────────────────────

class InceptionBlock(nn.Module):
    """Four-branch Inception module: 1×1 | 3×3 | 5×5 | maxpool→1×1."""
    def __init__(self, in_ch, ch1, ch3_reduce, ch3, ch5_reduce, ch5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1, 1), nn.BatchNorm2d(ch1), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, ch3_reduce, 1), nn.BatchNorm2d(ch3_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(ch3_reduce, ch3, 3, padding=1), nn.BatchNorm2d(ch3), nn.ReLU(inplace=True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, ch5_reduce, 1), nn.BatchNorm2d(ch5_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(ch5_reduce, ch5, 5, padding=2), nn.BatchNorm2d(ch5), nn.ReLU(inplace=True))
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, 1), nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch3(x),
                          self.branch5(x), self.branch_pool(x)], dim=1)

class IRDropInceptionCNN(nn.Module):
    """
    Encoder:  stem conv → 3 Inception blocks → decoder upsamples back.
    Pixel-wise regression (U-Net-lite with Inception blocks).
    """
    def __init__(self):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        # encoder inception blocks
        self.inc1 = InceptionBlock(64,  16, 24, 32,  4,  8,  8)   # out = 64
        self.pool1 = nn.MaxPool2d(2)
        self.inc2 = InceptionBlock(64,  32, 32, 48,  8, 16, 16)   # out = 112
        self.pool2 = nn.MaxPool2d(2)
        self.inc3 = InceptionBlock(112, 48, 48, 64, 12, 24, 24)   # out = 160

        # decoder (transposed convolutions)
        self.up2 = nn.ConvTranspose2d(160, 112, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(112 + 112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(112, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.head = nn.Conv2d(64, 1, 1)

    def _pad_to(self, x, target):
        """Pad spatial dims of x to match target tensor."""
        diff_y = target.size(2) - x.size(2)
        diff_x = target.size(3) - x.size(3)
        return F.pad(x, [0, diff_x, 0, diff_y])

    def forward(self, x):
        s = self.stem(x)               # (B,64,H,W)
        e1 = self.inc1(s)               # (B,64,H,W)
        p1 = self.pool1(e1)             # (B,64,H/2,W/2)
        e2 = self.inc2(p1)              # (B,112,H/2,W/2)
        p2 = self.pool2(e2)             # (B,112,H/4,W/4)
        e3 = self.inc3(p2)              # (B,160,H/4,W/4)

        d2 = self.up2(e3)               # (B,112,H/2,W/2)
        d2 = self._pad_to(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)               # (B,64,H,W)
        d1 = self._pad_to(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.head(d1))

# ── training ─────────────────────────────────────────────────────────

def train():
    print("\n=== Generating synthetic dataset ===")
    ds = IRDropDataset(N_SAMPLES)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train],
                                     generator=torch.Generator().manual_seed(SEED))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} samples  |  Grid {GRID}x{GRID}")

    model = IRDropInceptionCNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.MSELoss()

    EPOCHS = 30
    best_val = 1e9
    history = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_ds)

        # ── validate ──
        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                running += criterion(model(xb), yb).item() * xb.size(0)
        val_loss = running / len(val_ds)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "best_model.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train MSE={train_loss:.5f}  val MSE={val_loss:.5f}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  |  best val MSE={best_val:.5f}")

    # load best
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "best_model.pt"),
                                     map_location=DEVICE, weights_only=True))
    return model, val_dl, history

# ── evaluation ───────────────────────────────────────────────────────

def spatial_correlation(pred, true):
    """Pearson correlation per sample, averaged."""
    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = p.norm(dim=1) * t.norm(dim=1) + 1e-8
    return (num / den).mean().item()

def evaluate(model, val_dl):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())
            gts.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    gts   = np.concatenate(gts,   axis=0)

    mae  = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts)**2))
    sp_corr = spatial_correlation(torch.from_numpy(preds), torch.from_numpy(gts))

    print(f"\n=== Evaluation (val set) ===")
    print(f"  MAE              : {mae:.5f}")
    print(f"  RMSE             : {rmse:.5f}")
    print(f"  Spatial corr (r) : {sp_corr:.4f}")
    return preds, gts, {"mae": mae, "rmse": rmse, "spatial_corr": sp_corr}

# ── visualisation ────────────────────────────────────────────────────

def visualize(preds, gts, metrics, history):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("IR Drop Prediction – Inception-style CNN", fontsize=16, fontweight="bold")
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.30)

    # ── row 0: training curves ──
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.plot(history["train_loss"], label="Train", linewidth=2)
    ax0.plot(history["val_loss"],   label="Val",   linewidth=2)
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("MSE Loss")
    ax0.set_title("Training Curves"); ax0.legend(); ax0.grid(True, alpha=0.3)

    # metrics text box
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax1.axis("off")
    txt = (f"Validation Metrics\n{'─'*30}\n"
           f"MAE              = {metrics['mae']:.5f}\n"
           f"RMSE             = {metrics['rmse']:.5f}\n"
           f"Spatial corr (r) = {metrics['spatial_corr']:.4f}\n\n"
           f"Grid size   : {GRID}×{GRID}\n"
           f"Parameters  : see model\n"
           f"Device      : {DEVICE}")
    ax1.text(0.1, 0.5, txt, fontsize=13, fontfamily="monospace",
             verticalalignment="center", bbox=dict(boxstyle="round", facecolor="lightyellow"))

    # ── rows 1-2: sample predictions ──
    sample_ids = [0, 1, 2, 3]
    for col, idx in enumerate(sample_ids):
        gt   = gts[idx, 0]
        pred = preds[idx, 0]
        diff = np.abs(gt - pred)

        # ground truth
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(gt, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"GT #{idx}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # prediction
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(pred, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"Pred #{idx}  MAE={np.mean(diff):.4f}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "irdrop_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualisation saved → {out_path}")

# ── main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IR Drop Prediction – Inception-style CNN")
    print("=" * 60)
    model, val_dl, history = train()
    preds, gts, metrics = evaluate(model, val_dl)
    visualize(preds, gts, metrics, history)
    print("\nDone.")

if __name__ == "__main__":
    main()
