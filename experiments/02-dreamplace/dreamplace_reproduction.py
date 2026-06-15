"""
DREAMPlace Simplified PyTorch Reproduction
==========================================
Reproduces the core concept of DREAMPlace (Lin et al., DAC 2019):
  - Treats VLSI placement as a neural network training problem
  - Uses differentiable wirelength (Log-Sum-Exp HPWL approximation)
  - Applies density penalty via bell-shaped potential
  - Optimizes cell positions with gradient descent (Adam)

This implements the essential algorithms from DREAMPlace without the C++/CUDA
custom ops. It demonstrates global placement on a synthetic benchmark circuit.

Reference: Yibo Lin et al., "DREAMPlace: Deep Learning Toolkit-Enabled GPU
Acceleration for Modern VLSI Placement", DAC 2019
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import OrderedDict
import time
import json
import os


# ============================================================
# 1. Synthetic Benchmark Generator (ISPD-style)
# ============================================================

class Circuit:
    """Synthetic benchmark circuit for placement."""
    
    def __init__(self, num_cells, num_nets, die_width, die_height,
                 num_macros=0, macro_size_range=(8, 20), cell_size_range=(1, 3),
                 seed=42):
        np.random.seed(seed)
        self.die_width = die_width
        self.die_height = die_height
        
        self.num_cells = num_cells
        self.num_macros = num_macros
        self.total_nodes = num_cells + num_macros
        
        # Fixed pins at the die boundary (IO pads)
        self.num_fixed = 4
        self.fixed_positions = np.array([
            [0, die_height / 2],        # left
            [die_width, die_height / 2], # right
            [die_width / 2, 0],          # bottom
            [die_width / 2, die_height], # top
        ], dtype=np.float64)
        
        # Cell sizes
        self.node_widths = np.zeros(self.total_nodes, dtype=np.float64)
        self.node_heights = np.zeros(self.total_nodes, dtype=np.float64)
        
        for i in range(num_cells):
            w = np.random.uniform(*cell_size_range)
            h = np.random.uniform(*cell_size_range)
            self.node_widths[i] = w
            self.node_heights[i] = h
        
        for i in range(num_cells, self.total_nodes):
            w = np.random.uniform(*macro_size_range)
            h = np.random.uniform(*macro_size_range)
            self.node_widths[i] = w
            self.node_heights[i] = h
        
        # Generate nets (connectivity)
        # Use a mix of 2-pin and multi-pin nets
        self.nets = []
        self.net_weights = []
        
        # Random nets
        for _ in range(num_nets):
            # Fanout between 2 and 6
            fanout = np.random.randint(2, min(7, self.total_nodes // 2))
            # Pick cells with higher probability for cells (more connected)
            probs = np.ones(self.total_nodes + self.num_fixed)
            probs[:num_cells] *= 2.0  # standard cells more likely to be connected
            probs /= probs.sum()
            
            pins = np.random.choice(
                self.total_nodes + self.num_fixed,
                size=fanout, replace=False, p=probs
            )
            self.nets.append(pins.tolist())
            self.net_weights.append(1.0)
        
        # Add clock-like nets (connecting many cells)
        for _ in range(max(1, num_cells // 20)):
            fanout = min(np.random.randint(10, max(11, num_cells // 3)), num_cells)
            pins = np.random.choice(num_cells, size=fanout, replace=False)
            self.nets.append(pins.tolist())
            self.net_weights.append(0.5)  # lower weight for clock nets
        
        self.num_nets = len(self.nets)
        
        # Area utilization
        total_cell_area = sum(self.node_widths[i] * self.node_heights[i]
                              for i in range(self.total_nodes))
        die_area = die_width * die_height
        self.utilization = total_cell_area / die_area
        
        print(f"Circuit Summary:")
        print(f"  Cells: {num_cells} standard + {num_macros} macros = {self.total_nodes} movable")
        print(f"  Fixed pins: {self.num_fixed}")
        print(f"  Nets: {self.num_nets}")
        print(f"  Die: {die_width:.0f} x {die_height:.0f}")
        print(f"  Utilization: {self.utilization:.1%}")


# ============================================================
# 2. DREAMPlace-style Differentiable Placement Engine
# ============================================================

class DreamPlaceEngine(nn.Module):
    """
    Differentiable placement engine implementing DREAMPlace's core approach.
    
    Key components (matching DREAMPlace's architecture):
    1. Log-Sum-Exp wirelength model (differentiable HPWL approximation)
    2. Bell-shaped density model (electrostatic-based)
    3. Adam optimizer with learning rate scheduling
    """
    
    def __init__(self, circuit: Circuit, device='cuda'):
        super().__init__()
        self.circuit = circuit
        self.device = device
        
        N = circuit.total_nodes
        
        # Learnable cell positions (center coordinates)
        # Initialize with random placement within die
        init_x = np.random.uniform(0, circuit.die_width - circuit.node_widths[:N], N).astype(np.float64)
        init_y = np.random.uniform(0, circuit.die_height - circuit.node_heights[:N], N).astype(np.float64)
        
        # Add some structure to initialization (spread evenly)
        grid_n = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            row = i // grid_n
            col = i % grid_n
            init_x[i] = (col + 0.5) * circuit.die_width / grid_n
            init_y[i] = (row + 0.5) * circuit.die_height / grid_n
        
        self.pos_x = nn.Parameter(torch.tensor(init_x, dtype=torch.float64, device=device))
        self.pos_y = nn.Parameter(torch.tensor(init_y, dtype=torch.float64, device=device))
        
        # Fixed pin positions
        self.fixed_x = torch.tensor(circuit.fixed_positions[:, 0], dtype=torch.float64, device=device)
        self.fixed_y = torch.tensor(circuit.fixed_positions[:, 1], dtype=torch.float64, device=device)
        
        # Cell dimensions
        self.cell_w = torch.tensor(circuit.node_widths[:N], dtype=torch.float64, device=device)
        self.cell_h = torch.tensor(circuit.node_heights[:N], dtype=torch.float64, device=device)
        
        # Build net connectivity tensors for efficient computation
        self._build_net_tensors()
        
        # Density grid parameters (matching DREAMPlace's approach)
        self.num_bins_x = max(32, int(np.sqrt(N) * 2))
        self.num_bins_y = max(32, int(np.sqrt(N) * 2))
        self.bin_w = circuit.die_width / self.num_bins_x
        self.bin_h = circuit.die_height / self.num_bins_y
        
        # LSE wirelength hyperparameter (controls smoothness)
        # DREAMPlace uses a gradually increasing gamma
        self.gamma = 0.5  # initial gamma (will be updated during optimization)
        
        # Density weight (trade-off with wirelength)
        self.density_weight = 1.0
        
        print(f"Placement Engine:")
        print(f"  Density bins: {self.num_bins_x} x {self.num_bins_y}")
        print(f"  Bin size: {self.bin_w:.2f} x {self.bin_h:.2f}")
    
    def _build_net_tensors(self):
        """Pre-compute net connectivity as tensors for GPU efficiency."""
        flat_indices_x = []
        flat_indices_y = []
        flat_net_ids = []
        net_pin_counts = []
        
        total_pins_x = []
        total_pins_y = []
        pin_net_ids = []
        
        N = self.circuit.total_nodes
        num_nets = self.circuit.num_nets
        
        # For each net, store which nodes participate
        # We'll use scatter operations for efficiency
        for net_id, (pins, weight) in enumerate(
            zip(self.circuit.nets, self.circuit.net_weights)
        ):
            pin_positions_x = []
            pin_positions_y = []
            
            for pin in pins:
                if pin < N:  # movable cell pin (pin at cell center)
                    pin_positions_x.append(pin)  # index into pos_x
                    pin_positions_y.append(pin)
                    flat_indices_x.append(pin)
                    flat_indices_y.append(pin)
                    flat_net_ids.append(net_id)
                else:  # fixed pin
                    fixed_idx = pin - N
                    # Use a large negative index to mark as fixed
                    pin_positions_x.append(-(fixed_idx + 1))
                    pin_positions_y.append(-(fixed_idx + 1))
            
            net_pin_counts.append(len(pins))
        
        # Store as lists for processing (nets have variable pin counts)
        self.net_info = []
        for net_id, (pins, weight) in enumerate(
            zip(self.circuit.nets, self.circuit.net_weights)
        ):
            movable = [p for p in pins if p < N]
            fixed = [p - N for p in pins if p >= N]
            self.net_info.append({
                'movable': movable,
                'fixed': fixed,
                'weight': weight,
            })
    
    def compute_wirelength_lse(self, pos_x, pos_y, gamma):
        """
        Compute differentiable wirelength using Log-Sum-Exp (LSE) approximation.
        
        For each net, HPWL = (max_x - min_x) + (max_y - min_y)
        
        LSE approximation:
          max(x) ≈ (1/gamma) * log(sum(exp(gamma * x)))
          min(x) ≈ -(1/gamma) * log(sum(exp(-gamma * x)))
          
        So: max(x) - min(x) ≈ (1/gamma) * [log(sum(exp(gamma*x))) + log(sum(exp(-gamma*x)))]
        
        This is the exact formulation from DREAMPlace (Lin et al., DAC 2019).
        """
        total_wl = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        for net_info in self.net_info:
            movable = net_info['movable']
            fixed = net_info['fixed']
            weight = net_info['weight']
            
            if len(movable) == 0:
                continue
            
            # Collect x coordinates
            x_coords = []
            y_coords = []
            
            if len(movable) > 0:
                x_coords.append(pos_x[movable])
                y_coords.append(pos_y[movable])
            
            for fi in fixed:
                x_coords.append(self.fixed_x[fi].unsqueeze(0))
                y_coords.append(self.fixed_y[fi].unsqueeze(0))
            
            x_all = torch.cat(x_coords)
            y_all = torch.cat(y_coords)
            
            # LSE wirelength for x and y dimensions
            # Use the numerically stable version
            exp_gx = torch.exp(gamma * x_all)
            exp_neg_gx = torch.exp(-gamma * x_all)
            exp_gy = torch.exp(gamma * y_all)
            exp_neg_gy = torch.exp(-gamma * y_all)
            
            wl_x = (torch.log(exp_gx.sum()) + torch.log(exp_neg_gx.sum())) / gamma
            wl_y = (torch.log(exp_gy.sum()) + torch.log(exp_neg_gy.sum())) / gamma
            
            total_wl = total_wl + weight * (wl_x + wl_y)
        
        return total_wl
    
    def compute_wirelength_wa(self, pos_x, pos_y, gamma):
        """
        Compute differentiable wirelength using Weighted-Average (WA) model.
        This is DREAMPlace's preferred model - smoother gradient landscape.
        
        max(x) ≈ sum(x * exp(gamma*x)) / sum(exp(gamma*x))
        min(x) ≈ sum(x * exp(-gamma*x)) / sum(exp(-gamma*x))
        """
        total_wl = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        for net_info in self.net_info:
            movable = net_info['movable']
            fixed = net_info['fixed']
            weight = net_info['weight']
            
            if len(movable) == 0:
                continue
            
            x_coords = []
            y_coords = []
            
            if len(movable) > 0:
                x_coords.append(pos_x[movable])
                y_coords.append(pos_y[movable])
            
            for fi in fixed:
                x_coords.append(self.fixed_x[fi].unsqueeze(0))
                y_coords.append(self.fixed_y[fi].unsqueeze(0))
            
            x_all = torch.cat(x_coords)
            y_all = torch.cat(y_coords)
            
            # WA max/min approximation
            exp_gx = torch.exp(gamma * x_all)
            exp_neg_gx = torch.exp(-gamma * x_all)
            
            max_x = (x_all * exp_gx).sum() / exp_gx.sum()
            min_x = (x_all * exp_neg_gx).sum() / exp_neg_gx.sum()
            
            exp_gy = torch.exp(gamma * y_all)
            exp_neg_gy = torch.exp(-gamma * y_all)
            
            max_y = (y_all * exp_gy).sum() / exp_gy.sum()
            min_y = (y_all * exp_neg_gy).sum() / exp_neg_gy.sum()
            
            wl = weight * ((max_x - min_x) + (max_y - min_y))
            total_wl = total_wl + wl
        
        return total_wl
    
    def compute_density(self, pos_x, pos_y):
        """
        Compute density penalty using bell-shaped (Gaussian) distribution.
        
        Following DREAMPlace, each cell contributes a bell-shaped density
        to overlapping bins. The density penalty encourages uniform distribution:
          density_penalty = sum over bins of (bin_density - target_density)^2
        
        This is analogous to the electrostatic potential in DREAMPlace's
        ePlace approach (Lu et al., ICCAD 2015).
        """
        N = self.circuit.total_nodes
        nbx = self.num_bins_x
        nby = self.num_bins_y
        
        # Target density
        total_cell_area = (self.cell_w * self.cell_h).sum()
        die_area = self.circuit.die_width * self.circuit.die_height
        target_density = total_cell_area / die_area
        
        # Compute bin density using bell-shaped function
        # For each cell, its density contribution to nearby bins follows
        # a quadratic B-spline (as in DREAMPlace)
        density_map = torch.zeros(nbx, nby, dtype=torch.float64, device=self.device)
        
        # Cell centers
        cx = pos_x + self.cell_w / 2
        cy = pos_y + self.cell_h / 2
        
        # For efficiency, compute density by direct binning with smooth spreading
        for i in range(N):
            # Cell center in grid coordinates
            gx = cx[i] / self.bin_w
            gy = cy[i] / self.bin_h
            
            # Spread cell density to neighboring bins using triangle distribution
            cx_bin = int(torch.clamp(gx, 0, nbx - 1).item())
            cy_bin = int(torch.clamp(gy, 0, nby - 1).item())
            
            # Spread radius (in bins) proportional to cell size
            rx = max(1, int(np.ceil(self.circuit.node_widths[i] / self.bin_w)) + 1)
            ry = max(1, int(np.ceil(self.circuit.node_heights[i] / self.bin_h)) + 1)
            
            cell_area = self.cell_w[i] * self.cell_h[i]
            
            for bx in range(max(0, cx_bin - rx), min(nbx, cx_bin + rx + 1)):
                for by in range(max(0, cy_bin - ry), min(nby, cy_bin + ry + 1)):
                    # Bell-shaped weight: triangle kernel
                    dx = abs(bx - gx) / (rx + 0.5)
                    dy = abs(by - gy) / (ry + 0.5)
                    
                    wx = torch.clamp(1.0 - dx, min=0.0)
                    wy = torch.clamp(1.0 - dy, min=0.0)
                    
                    density_map[bx, by] += cell_area * wx * wy / (self.bin_w * self.bin_h)
        
        # Density penalty: L2 norm of excess density
        penalty = ((density_map - target_density).clamp(min=0) ** 2).sum()
        
        return penalty, density_map, target_density
    
    def compute_density_fast(self, pos_x, pos_y):
        """
        Vectorized density computation for better GPU utilization.
        Uses a simplified bin-based approach.
        """
        N = self.circuit.total_nodes
        nbx = self.num_bins_x
        nby = self.num_bins_y
        
        total_cell_area = (self.cell_w * self.cell_h).sum()
        die_area = self.circuit.die_width * self.circuit.die_height
        target_density = total_cell_area / die_area
        
        # Cell centers
        cx = pos_x + self.cell_w / 2
        cy = pos_y + self.cell_h / 2
        
        # Bin assignment
        bin_x = torch.clamp(cx / self.bin_w, 0, nbx - 1)
        bin_y = torch.clamp(cy / self.bin_h, 0, nby - 1)
        
        bx_int = bin_x.long()
        by_int = bin_y.long()
        
        # Create density map using scatter add
        density_map = torch.zeros(nbx, nby, dtype=torch.float64, device=self.device)
        
        cell_areas = self.cell_w * self.cell_h
        bin_areas = self.bin_w * self.bin_h
        
        # Smooth spreading: add to current bin and 4 neighbors
        offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        weights = [0.4, 0.15, 0.15, 0.15, 0.15]
        
        for (dx, dy), w in zip(offsets, weights):
            nx = torch.clamp(bx_int + dx, 0, nbx - 1)
            ny = torch.clamp(by_int + dy, 0, nby - 1)
            
            indices = nx * nby + ny
            flat_density = density_map.view(-1)
            contribution = (cell_areas * w / bin_areas)
            flat_density.scatter_add_(0, indices, contribution)
        
        # Penalty
        excess = (density_map - target_density).clamp(min=0)
        penalty = (excess ** 2).sum()
        
        return penalty, density_map, target_density
    
    def forward(self, use_wa=True):
        """Compute combined objective: wirelength + density_weight * density_penalty."""
        pos_x = self.pos_x
        pos_y = self.pos_y
        
        # Wirelength
        if use_wa:
            wl = self.compute_wirelength_wa(pos_x, pos_y, self.gamma)
        else:
            wl = self.compute_wirelength_lse(pos_x, pos_y, self.gamma)
        
        # Density
        dp, density_map, target_density = self.compute_density_fast(pos_x, pos_y)
        
        # Combined loss
        total_loss = wl + self.density_weight * dp
        
        return total_loss, wl, dp, density_map
    
    def get_hpwl(self):
        """Compute exact HPWL (non-differentiable) for evaluation."""
        pos_x = self.pos_x.detach()
        pos_y = self.pos_y.detach()
        
        total_hpwl = 0.0
        for net_info in self.net_info:
            movable = net_info['movable']
            fixed = net_info['fixed']
            
            x_coords = []
            y_coords = []
            
            if len(movable) > 0:
                x_coords.extend(pos_x[movable].cpu().numpy().tolist())
                y_coords.extend(pos_y[movable].cpu().numpy().tolist())
            
            for fi in fixed:
                x_coords.append(self.fixed_x[fi].item())
                y_coords.append(self.fixed_y[fi].item())
            
            if len(x_coords) > 0:
                hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
                total_hpwl += hpwl
        
        return total_hpwl


# ============================================================
# 3. Optimization Loop (matching DREAMPlace's adaptive schedule)
# ============================================================

def run_placement(circuit, device='cuda', num_iterations=500, lr=1.0):
    """
    Run DREAMPlace-style global placement optimization.
    
    DREAMPlace uses:
    1. Adam optimizer with initial learning rate
    2. Gradually increasing gamma for wirelength (coarse-to-fine)
    3. Adaptive density weight adjustment (Lagrangian-style)
    4. Optional Nesterov momentum
    """
    print("\n" + "=" * 60)
    print("DREAMPlace-style Global Placement")
    print("=" * 60)
    
    engine = DreamPlaceEngine(circuit, device=device)
    engine = engine.to(device)
    
    # DREAMPlace uses Adam with specific parameters
    optimizer = optim.Adam([engine.pos_x, engine.pos_y], lr=lr)
    
    # Learning rate schedule (DREAMPlace reduces LR over iterations)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=0.01)
    
    # History for plotting
    history = {
        'iteration': [],
        'wirelength': [],
        'density': [],
        'total_loss': [],
        'hpwl': [],
        'gamma': [],
        'lr': [],
        'density_weight': [],
    }
    
    # Snapshots for animation frames
    snapshots = {}
    snapshot_iters = sorted(set(
        list(range(0, min(50, num_iterations), 5)) +
        list(range(50, min(200, num_iterations), 20)) +
        list(range(200, num_iterations, 50))
    ))
    
    t_start = time.time()
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # DREAMPlace's gamma schedule: gradually increase (coarse-to-fine)
        # This controls the smoothness of the wirelength approximation
        engine.gamma = 0.5 + 4.5 * (iteration / num_iterations) ** 0.8
        
        # DREAMPlace's density weight schedule (Lagrangian relaxation)
        # Increase density weight as optimization progresses
        if iteration > 0 and iteration % 50 == 0:
            engine.density_weight = min(engine.density_weight * 1.5, 100.0)
        
        # Forward pass
        total_loss, wl_loss, dp_loss, density_map = engine()
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping (DREAMPlace clips gradients)
        torch.nn.utils.clip_grad_norm_([engine.pos_x, engine.pos_y], max_norm=10.0)
        
        optimizer.step()
        scheduler.step()
        
        # Boundary enforcement (DREAMPlace applies boundary constraints)
        with torch.no_grad():
            N = circuit.total_nodes
            half_w = engine.cell_w / 2
            half_h = engine.cell_h / 2
            engine.pos_x.data.clamp_(half_w, circuit.die_width - half_w)
            engine.pos_y.data.clamp_(half_h, circuit.die_height - half_h)
        
        # Compute exact HPWL for evaluation (every 10 iterations)
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            hpwl = engine.get_hpwl()
            
            current_lr = optimizer.param_groups[0]['lr']
            history['iteration'].append(iteration)
            history['wirelength'].append(wl_loss.item())
            history['density'].append(dp_loss.item())
            history['total_loss'].append(total_loss.item())
            history['hpwl'].append(hpwl)
            history['gamma'].append(engine.gamma)
            history['lr'].append(current_lr)
            history['density_weight'].append(engine.density_weight)
            
            if iteration % 50 == 0 or iteration == num_iterations - 1:
                elapsed = time.time() - t_start
                print(f"  Iter {iteration:4d} | WL: {wl_loss.item():12.1f} | "
                      f"Density: {dp_loss.item():10.4f} | HPWL: {hpwl:12.1f} | "
                      f"gamma: {engine.gamma:.2f} | DW: {engine.density_weight:.1f} | "
                      f"Time: {elapsed:.1f}s")
        
        # Save snapshot
        if iteration in snapshot_iters:
            snapshots[iteration] = {
                'pos_x': engine.pos_x.detach().cpu().numpy().copy(),
                'pos_y': engine.pos_y.detach().cpu().numpy().copy(),
            }
    
    total_time = time.time() - t_start
    print(f"\nPlacement completed in {total_time:.1f}s")
    print(f"Final HPWL: {history['hpwl'][-1]:.1f}")
    
    return engine, history, snapshots


# ============================================================
# 4. Visualization
# ============================================================

def plot_results(engine, circuit, history, snapshots, output_path):
    """Create comprehensive visualization of placement results."""
    
    fig = plt.figure(figsize=(24, 20))
    
    pos_x = engine.pos_x.detach().cpu().numpy()
    pos_y = engine.pos_y.detach().cpu().numpy()
    N = circuit.total_nodes
    
    # ---- Plot 1: Final Placement ----
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Draw die boundary
    ax1.add_patch(Rectangle((0, 0), circuit.die_width, circuit.die_height,
                             fill=False, edgecolor='black', linewidth=2))
    
    # Draw standard cells
    for i in range(circuit.num_cells):
        w = circuit.node_widths[i]
        h = circuit.node_heights[i]
        ax1.add_patch(Rectangle((pos_x[i], pos_y[i]), w, h,
                                 facecolor='steelblue', alpha=0.6, edgecolor='navy',
                                 linewidth=0.3))
    
    # Draw macros
    for i in range(circuit.num_cells, N):
        w = circuit.node_widths[i]
        h = circuit.node_heights[i]
        ax1.add_patch(Rectangle((pos_x[i], pos_y[i]), w, h,
                                 facecolor='salmon', alpha=0.7, edgecolor='darkred',
                                 linewidth=0.8))
    
    # Draw fixed pins
    ax1.scatter(engine.fixed_x.cpu(), engine.fixed_y.cpu(),
                c='red', s=100, marker='D', zorder=5, label='IO Pads')
    
    # Draw a subset of nets (first 50)
    max_nets_to_draw = min(50, len(circuit.nets))
    for net_id in range(max_nets_to_draw):
        pins = circuit.nets[net_id]
        xs, ys = [], []
        for pin in pins:
            if pin < N:
                xs.append(pos_x[pin] + circuit.node_widths[pin] / 2)
                ys.append(pos_y[pin] + circuit.node_heights[pin] / 2)
            else:
                fi = pin - N
                xs.append(engine.fixed_x[fi].item())
                ys.append(engine.fixed_y[fi].item())
        if len(xs) > 1:
            # Star topology: connect all pins to bounding box center
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            for x, y in zip(xs, ys):
                ax1.plot([cx, x], [cy, y], 'g-', alpha=0.15, linewidth=0.5)
    
    ax1.set_xlim(-5, circuit.die_width + 5)
    ax1.set_ylim(-5, circuit.die_height + 5)
    ax1.set_aspect('equal')
    ax1.set_title(f'Final Placement\nHPWL = {history["hpwl"][-1]:.0f}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.2)
    
    # ---- Plot 2: Placement Evolution (early) ----
    ax2 = fig.add_subplot(2, 3, 2)
    early_snapshots = sorted(snapshots.keys())[:6]
    colors = plt.cm.viridis(np.linspace(0, 1, len(early_snapshots)))
    
    for idx, snap_iter in enumerate(early_snapshots):
        sx = snapshots[snap_iter]['pos_x']
        sy = snapshots[snap_iter]['pos_y']
        alpha = 0.3 + 0.7 * (idx / max(1, len(early_snapshots) - 1))
        ax2.scatter(sx[:circuit.num_cells], sy[:circuit.num_cells],
                    s=3, c=[colors[idx]], alpha=alpha, label=f'Iter {snap_iter}')
    
    ax2.add_patch(Rectangle((0, 0), circuit.die_width, circuit.die_height,
                              fill=False, edgecolor='black', linewidth=1.5))
    ax2.set_xlim(-5, circuit.die_width + 5)
    ax2.set_ylim(-5, circuit.die_height + 5)
    ax2.set_aspect('equal')
    ax2.set_title('Cell Movement (Early Phase)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.2)
    
    # ---- Plot 3: Density Map ----
    ax3 = fig.add_subplot(2, 3, 3)
    _, density_map, target_density = engine.compute_density_fast(engine.pos_x, engine.pos_y)
    dm = density_map.detach().cpu().numpy()
    im = ax3.imshow(dm.T, origin='lower', cmap='YlOrRd',
                    extent=[0, circuit.die_width, 0, circuit.die_height],
                    aspect='equal', interpolation='bilinear')
    plt.colorbar(im, ax=ax3, label='Density', shrink=0.8)
    ax3.set_title(f'Density Map (target={target_density:.3f})', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    # ---- Plot 4: HPWL Convergence ----
    ax4 = fig.add_subplot(2, 3, 4)
    iters = history['iteration']
    hpwl = history['hpwl']
    ax4.semilogy(iters, hpwl, 'b-', linewidth=2, label='HPWL')
    ax4.semilogy(iters, history['wirelength'], 'r--', linewidth=1.5, alpha=0.7, label='Wirelength Loss')
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Wirelength (log scale)', fontsize=12)
    ax4.set_title('Wirelength Convergence', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max(iters))
    
    # ---- Plot 5: Loss Components ----
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(iters, history['total_loss'], 'k-', linewidth=2, label='Total Loss')
    ax5_twin = ax5.twinx()
    ax5_twin.plot(iters, history['density'], 'g-', linewidth=1.5, alpha=0.7, label='Density Penalty')
    ax5_twin.set_ylabel('Density Penalty', fontsize=12, color='green')
    ax5.set_xlabel('Iteration', fontsize=12)
    ax5.set_ylabel('Total Loss', fontsize=12)
    ax5.set_title('Loss Components', fontsize=14, fontweight='bold')
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # ---- Plot 6: Hyperparameters ----
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(iters, history['gamma'], 'b-', linewidth=2, label='gamma (LSE smoothness)')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(iters, history['density_weight'], 'r-', linewidth=2, label='Density Weight')
    ax6_twin.set_ylabel('Density Weight', fontsize=12, color='red')
    ax6.set_xlabel('Iteration', fontsize=12)
    ax6.set_ylabel('Gamma', fontsize=12, color='blue')
    ax6.set_title('Optimization Schedule', fontsize=14, fontweight='bold')
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # ---- Suptitle ----
    fig.suptitle(
        'DREAMPlace Simplified Reproduction: Gradient-Descent VLSI Placement\n'
        'Using PyTorch with Differentiable Wirelength (WA/LSE) and Density Penalty',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to {output_path}")
    plt.close()


# ============================================================
# 5. Main
# ============================================================

def main():
    print("=" * 60)
    print("DREAMPlace Simplified PyTorch Reproduction")
    print("=" * 60)
    print()
    print("This reproduces the core concept of DREAMPlace (Lin et al., DAC 2019):")
    print("  - Treats VLSI placement as a neural network training problem")
    print("  - Uses differentiable wirelength (Weighted-Average / LSE models)")
    print("  - Applies bell-shaped density penalty (electrostatic model)")
    print("  - Optimizes cell positions with Adam optimizer")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print()
    
    # Create synthetic benchmark (moderate size)
    circuit = Circuit(
        num_cells=200,
        num_nets=600,
        die_width=200.0,
        die_height=200.0,
        num_macros=10,
        macro_size_range=(12, 25),
        cell_size_range=(1.5, 4.0),
        seed=42,
    )
    print()
    
    # Run placement
    engine, history, snapshots = run_placement(
        circuit, device=device,
        num_iterations=500,
        lr=1.0,
    )
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'dreamplace_results.png')
    
    plot_results(engine, circuit, history, snapshots, output_path)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Benchmark: {circuit.num_cells} cells + {circuit.num_macros} macros, {circuit.num_nets} nets")
    print(f"  Die size: {circuit.die_width:.0f} x {circuit.die_height:.0f}")
    print(f"  Utilization: {circuit.utilization:.1%}")
    print(f"  Final HPWL: {history['hpwl'][-1]:.1f}")
    print(f"  Initial HPWL: {history['hpwl'][0]:.1f}")
    print(f"  HPWL reduction: {(1 - history['hpwl'][-1] / history['hpwl'][0]) * 100:.1f}%")
    print(f"  Results: {output_path}")
    
    # Save numerical results
    results = {
        'benchmark': {
            'num_cells': circuit.num_cells,
            'num_macros': circuit.num_macros,
            'num_nets': circuit.num_nets,
            'die_width': circuit.die_width,
            'die_height': circuit.die_height,
            'utilization': circuit.utilization,
        },
        'placement': {
            'final_hpwl': history['hpwl'][-1],
            'initial_hpwl': history['hpwl'][0],
            'hpwl_reduction_pct': (1 - history['hpwl'][-1] / history['hpwl'][0]) * 100,
            'final_wirelength_loss': history['wirelength'][-1],
            'final_density_penalty': history['density'][-1],
            'num_iterations': 500,
        },
        'convergence': {
            'iterations': history['iteration'],
            'hpwl': history['hpwl'],
        }
    }
    
    results_path = os.path.join(output_dir, 'dreamplace_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON results: {results_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
