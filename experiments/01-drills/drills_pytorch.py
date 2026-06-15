"""
DRiLLS Reproduction: Deep Reinforcement Learning for Logic Synthesis (ASP-DAC 2020)
PyTorch reimplementation of the A2C agent for ABC logic synthesis optimization.

Workflow: Verilog -> Yosys(synth) -> BLIF -> ABC(optimization) -> metrics
"""

import os
import re
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ABCEnvironment:
    """Logic synthesis environment using Yosys + ABC."""
    
    OPTIMIZATIONS = [
        'rewrite',
        'rewrite -z',
        'refactor',
        'refactor -z',
        'resub',
        'resub -z',
        'balance',
    ]
    
    def __init__(self, design_file, abc_binary='yosys-abc', yosys_binary='yosys'):
        self.design_file = os.path.abspath(design_file)
        self.abc_binary = abc_binary
        self.yosys_binary = yosys_binary
        self.n_actions = len(self.OPTIMIZATIONS)
        self.n_features = 6
        
        self.iteration = 0
        self.episode = 0
        self.area = float('inf')  # AND gate count
        self.levels = float('inf')  # logic levels
        self.playground_dir = '/tmp/drills_playground'
        
        self.best_area = float('inf')
        self.best_levels = float('inf')
        
        # Pre-synthesize design to BLIF
        self.blif_file = '/tmp/drills_playground/design.blif'
        os.makedirs('/tmp/drills_playground', exist_ok=True)
        self._presynthesize()
    
    def _presynthesize(self):
        cmd = f"read_verilog {self.design_file}; synth; write_blif {self.blif_file}"
        subprocess.run([self.yosys_binary, '-QT', '-p', cmd], capture_output=True, timeout=30)
    
    def reset(self):
        self.iteration = 0
        self.episode += 1
        self.area = float('inf')
        self.levels = float('inf')
        self.episode_dir = os.path.join(self.playground_dir, str(self.episode))
        os.makedirs(self.episode_dir, exist_ok=True)
        self.history = []
        state, _, _ = self._run_abc([])
        return state
    
    def step(self, action_idx):
        opt = self.OPTIMIZATIONS[action_idx]
        old_area, old_levels = self.area, self.levels
        self.iteration += 1
        
        new_state, area, levels = self._run_abc([opt])
        
        # Reward: area improvement is primary, levels improvement secondary
        area_imp = 1 if area < old_area else (-1 if area > old_area else 0)
        lev_imp = 1 if levels < old_levels else (-1 if levels > old_levels else 0)
        reward = 2.0 * area_imp + 1.0 * lev_imp
        
        self.area = area
        self.levels = levels
        
        if area < self.best_area:
            self.best_area = area
        if levels < self.best_levels:
            self.best_levels = levels
        
        self.history.append({'iter': self.iteration, 'action': opt, 'area': area, 'levels': levels, 'reward': reward})
        
        done = (self.iteration >= 20)
        return new_state, reward, done
    
    def _run_abc(self, optimizations):
        """Run ABC and parse print_stats output."""
        opt_str = '; '.join(optimizations) if optimizations else ''
        cmd = f"read {self.blif_file}; strash; {opt_str}; print_stats" if opt_str else f"read {self.blif_file}; strash; print_stats"
        
        try:
            proc = subprocess.check_output([self.abc_binary, '-c', cmd], stderr=subprocess.STDOUT, timeout=30)
            output = proc.decode('utf-8')
            inp, out, ands, edges, levels, latches = 0, 0, 0, 0, 0, 0
            for line in output.split('\n'):
                if 'i/o' in line:
                    m = re.search(r'i/o\s*=\s*(\d+)\s*/\s*(\d+)', line)
                    if m: inp, out = int(m.group(1)), int(m.group(2))
                    m = re.search(r'and\s*=\s*(\d+)', line)
                    if m: ands = int(m.group(1))
                    m = re.search(r'edge\s*=\s*(\d+)', line)
                    if m: edges = int(m.group(1))
                    m = re.search(r'lev\s*=\s*(\d+)', line)
                    if m: levels = int(m.group(1))
                    m = re.search(r'lat\s*=\s*(\d+)', line)
                    if m: latches = int(m.group(1))
            
            features = np.array([inp, out, ands, edges, levels, latches], dtype=np.float32)
            return features, float(ands), float(levels)
        except Exception as e:
            return np.zeros(self.n_features, dtype=np.float32), self.area, self.levels


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_dim))
        self.critic = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, state):
        features = self.shared(state)
        return self.actor(features), self.critic(features)


class RunningNormalizer:
    def __init__(self, n):
        self.n = 0
        self.mean = np.zeros(n)
        self.var = np.ones(n)
        self.mean_diff = np.zeros(n)
    
    def observe(self, x):
        self.n += 1
        old_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - old_mean) * (x - self.mean)
        self.var = np.clip(self.mean_diff / max(self.n, 1), 1e-2, 1e9)
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
    
    def reset(self):
        self.n = 0
        self.mean = np.zeros_like(self.mean)
        self.mean_diff = np.zeros_like(self.mean_diff)


def train_drills(design_file, episodes=30, gamma=0.99, lr=0.005):
    env = ABCEnvironment(design_file)
    model = ActorCritic(env.n_features, env.n_actions)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    normalizer = RunningNormalizer(env.n_features)
    
    episode_rewards = []
    best_areas = []
    
    print(f"\n{'='*60}")
    print(f"DRiLLS: Deep RL for Logic Synthesis (ASP-DAC 2020)")
    print(f"Design: {os.path.basename(design_file)}")
    print(f"Optimizations: {env.OPTIMIZATIONS}")
    print(f"{'='*60}\n")
    
    for ep in range(episodes):
        state = env.reset()
        normalizer.reset()
        normalizer.observe(state)
        state_norm = normalizer.normalize(state)
        
        log_probs, values, rewards = [], [], []
        
        done = False
        while not done:
            state_t = torch.FloatTensor(state_norm).unsqueeze(0)
            action_logits, value = model(state_t)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            new_state, reward, done = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())
            rewards.append(reward)
            
            normalizer.observe(new_state)
            state_norm = normalizer.normalize(new_state)
        
        # Discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        advantages = returns - values_t.detach()
        
        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        best_areas.append(env.best_area)
        
        print(f"Ep {ep+1:2d}/{episodes} | Reward: {total_reward:5.1f} | "
              f"Area: {env.area:6.0f} | Levels: {env.levels:3.0f} | "
              f"Best Area: {env.best_area:6.0f}")
    
    return episode_rewards, best_areas, env


def plot_results(rewards, best_areas, output_dir='.'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(rewards, 'b-', alpha=0.5)
    if len(rewards) >= 5:
        ma = np.convolve(rewards, np.ones(5)/5, mode='valid')
        axes[0].plot(range(4, len(rewards)), ma, 'r-', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('DRiLLS Training Reward (A2C)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(best_areas, 'g-')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Best Area (AND gates)')
    axes[1].set_title('Best Area Over Training')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drills_training.png'), dpi=150, bbox_inches='tight')
    print(f"Plot: {output_dir}/drills_training.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', default='multiplier.v')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()
    
    rewards, best, env = train_drills(args.design, episodes=args.episodes, lr=args.lr)
    plot_results(rewards, best)
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Best Area: {env.best_area:.0f} AND gates")
    print(f"  Best Levels: {env.best_levels:.0f}")
    print(f"  Last Area: {env.area:.0f}")
    print(f"{'='*60}")
