# 3.7 上机实验

## 实验一：基于模拟退火的Macro Placement

### 实验目标

使用Python实现一个基本的模拟退火布局算法，放置Macro（大模块如存储器）。

### 完整代码

```python
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class MacroPlacementSA:
    def __init__(self, macros, die_w, die_h):
        self.macros = macros  # [(name, w, h), ...]
        self.die_w = die_w
        self.die_h = die_h
        self.n = len(macros)
        
        # 初始随机布局
        self.positions = []
        for name, w, h in macros:
            x = random.uniform(0, die_w - w)
            y = random.uniform(0, die_h - h)
            self.positions.append((x, y))
    
    def compute_hpwl(self, nets):
        """计算总HPWL"""
        total = 0
        for net in nets:
            x_coords = [self.positions[i][0] + self.macros[i][1]/2 for i in net]
            y_coords = [self.positions[i][1] + self.macros[i][2]/2 for i in net]
            total += (max(x_coords) - min(x_coords)) + \
                     (max(y_coords) - min(y_coords))
        return total
    
    def has_overlap(self, i, j):
        """检测两个Macro是否重叠"""
        x1, y1 = self.positions[i]
        w1, h1 = self.macros[i][1], self.macros[i][2]
        x2, y2 = self.positions[j]
        w2, h2 = self.macros[j][1], self.macros[j][2]
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def optimize(self, nets, T_start=1000, T_end=0.1, alpha=0.995, steps_per_T=100):
        """模拟退火优化"""
        cost = self.compute_hpwl(nets)
        best_cost = cost
        best_pos = self.positions[:]
        
        T = T_start
        history = []
        
        while T > T_end:
            for _ in range(steps_per_T):
                # 随机选择一个Macro并移动
                idx = random.randint(0, self.n - 1)
                old_pos = self.positions[idx]
                
                new_x = old_pos[0] + random.uniform(-50, 50)
                new_y = old_pos[1] + random.uniform(-50, 50)
                new_x = max(0, min(self.die_w - self.macros[idx][1], new_x))
                new_y = max(0, min(self.die_h - self.macros[idx][2], new_y))
                
                self.positions[idx] = (new_x, new_y)
                new_cost = self.compute_hpwl(nets)
                delta = new_cost - cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    cost = new_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = self.positions[:]
                else:
                    self.positions[idx] = old_pos
            
            history.append(cost)
            T *= alpha
        
        self.positions = best_pos
        return best_cost, history
    
    def plot(self, nets=None):
        """可视化布局结果"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlim(0, self.die_w)
        ax.set_ylim(0, self.die_h)
        
        for i, (name, w, h) in enumerate(self.macros):
            x, y = self.positions[i]
            rect = plt.Rectangle((x, y), w, h, 
                                  fill=True, facecolor='steelblue', alpha=0.6,
                                  edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, name, ha='center', va='center', fontsize=8)
        
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_title('Macro Placement Result')
        ax.set_aspect('equal')
        plt.savefig('macro_placement.png', dpi=150, bbox_inches='tight')
        plt.show()

# 运行示例
macros = [(f'M{i}', random.randint(50, 150), random.randint(50, 150)) 
          for i in range(10)]
nets = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8, 9], [0, 9]]

placer = MacroPlacementSA(macros, die_w=500, die_h=500)
best_cost, history = placer.optimize(nets)
print(f"Best HPWL: {best_cost:.0f}")
placer.plot()
```

## 实验二：强化学习DQN——倒立摆CartPole-v1

### 实验目标

使用DQN算法解决OpenAI Gym的CartPole-v1任务。

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # 计算Q值
        current_q = self.policy_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(dim=1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 更新网络
        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# 训练
env = gym.make('CartPole-v1')
agent = DQNAgent(state_dim=4, action_dim=2)

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.memory.append((state, action, reward, next_state, float(done)))
        agent.train_step()
        state = next_state
        total_reward += reward
        if done or truncated:
            break
    
    if episode % 10 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.0f}, Epsilon: {agent.epsilon:.3f}")
```

## 实验三：Google AlphaChip

### 安装和运行

```bash
# 克隆仓库
git clone https://github.com/google-research/circuit_training.git
cd circuit_training

# 安装依赖
pip install -r requirements.txt

# 运行训练
python -m circuit_training.environment.placement_env
```

### 源码关键模块

- `circuit_training/environment/`：强化学习环境
- `circuit_training/gnn/`：图神经网络编码器
- `circuit_training/agent/`：RL Agent

## 实验四：DREAMPlace

### 安装和部署

```bash
# 克隆仓库
git clone https://github.com/limbo018/DREAMPlace.git
cd DREAMPlace

# 安装
pip install -r requirements.txt
python setup.py install

# 运行示例
cd benchmarks/ispd2005/adaptec1
python ../../dreamplace/Placer.py --aux adaptec1.aux
```

### DREAMPlace使用

```bash
# 配置文件示例 (json)
{
    "aux_input": "benchmarks/ispd2005/adaptec1/adaptec1.aux",
    "gpu": 0,
    "num_bins_x": 64,
    "num_bins_y": 64,
    "global_place_stages": [
        {"num_bins_x": 64, "num_bins_y": 64}
    ],
    "target_density": 1.0
}
```
