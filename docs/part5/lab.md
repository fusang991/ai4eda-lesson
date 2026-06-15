# 5.7 上机实验

## 实验一：Python编程实现Maze Routing算法

### 实验目标

实现迷宫布线算法，并可视化布线结果。

### 完整代码

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class MazeRouter:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.routes = []
    
    def add_obstacle(self, r, c, h=1, w=1):
        """添加障碍物"""
        self.grid[r:r+h, c:c+w] = 1
    
    def route(self, start, end):
        """BFS迷宫布线"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        
        while queue:
            r, c = queue.popleft()
            
            if (r, c) == end:
                path = []
                current = end
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                self.routes.append(path)
                # 标记路径为已占用
                for pr, pc in path:
                    if self.grid[pr][pc] == 0:
                        self.grid[pr][pc] = 2
                return path
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols and
                    (nr, nc) not in visited and self.grid[nr][nc] != 1):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        
        return None  # 无可行路径
    
    def plot(self, show_grid=True):
        """可视化布线结果"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 绘制网格
        if show_grid:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r][c] == 1:
                        rect = plt.Rectangle((c, self.rows-1-r), 1, 1,
                                           facecolor='gray', edgecolor='lightgray')
                        ax.add_patch(rect)
                    elif self.grid[r][c] == 2:
                        rect = plt.Rectangle((c, self.rows-1-r), 1, 1,
                                           facecolor='lightblue', edgecolor='lightgray')
                        ax.add_patch(rect)
        
        # 绘制路径
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, route in enumerate(self.routes):
            y = [self.rows - 1 - p[0] for p in route]
            x = [p[1] for p in route]
            ax.plot(x, y, color=colors[i % len(colors)], linewidth=3, alpha=0.7,
                   label=f'Net {i+1}')
        
        ax.set_xlim(-0.5, self.cols + 0.5)
        ax.set_ylim(-0.5, self.rows + 0.5)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Maze Routing Result')
        ax.grid(True, alpha=0.3)
        plt.savefig('maze_routing.png', dpi=150, bbox_inches='tight')
        plt.show()

# 运行示例
router = MazeRouter(20, 20)

# 添加障碍物
router.add_obstacle(5, 5, h=3, w=10)
router.add_obstacle(10, 2, h=8, w=3)
router.add_obstacle(3, 12, h=5, w=5)

# 布线多个网络
router.route((0, 0), (19, 19))
router.route((0, 19), (19, 0))
router.route((10, 0), (10, 19))

router.plot()
```

## 实验二：IR Drop预测——基于Inception的CNN模型

### 实验目标

使用PyTorch搭建基于Inception模块的CNN模型，预测芯片的IR Drop。

### 完整代码

```python
import torch
import torch.nn as nn
import numpy as np

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, ch1x1, ch3x3, ch5x5, pool_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1x1, 1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3x3, 3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5x5, 5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_ch, 1),
            nn.BatchNorm2d(pool_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], dim=1)

class IRDropCNN(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.inception1 = InceptionBlock(32, 16, 24, 8, 8)   # -> 56
        self.inception2 = InceptionBlock(56, 32, 48, 16, 16)  # -> 112
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.regressor = nn.Sequential(
            nn.Linear(112 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

# 生成模拟数据
def generate_synthetic_data(n_samples=1000, grid_size=64):
    """生成模拟的IR Drop数据"""
    X = np.random.randn(n_samples, 5, grid_size, grid_size).astype(np.float32)
    # 模拟IR Drop: 与开关活动和距离相关
    y = (X[:, 1].mean(axis=(1, 2)) * 0.3 + 
         np.random.randn(n_samples) * 0.1).astype(np.float32)
    return torch.FloatTensor(X), torch.FloatTensor(y)

# 训练
model = IRDropCNN(in_channels=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

X_train, y_train = generate_synthetic_data(800)
X_test, y_test = generate_synthetic_data(200)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(50):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        pred = model(batch_x).squeeze()
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).squeeze()
        test_mae = torch.abs(test_pred - y_test).mean().item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, "
              f"Test MAE={test_mae:.4f}")
```

## 实验三：论文复现——DRiLLS深度强化学习用于逻辑综合

### DRiLLS框架讲解

DRiLLS将逻辑综合过程建模为强化学习问题：

```
环境: ABC逻辑综合工具
状态: 电路特征（面积、延迟、门数等）
动作: 逻辑优化操作（rewrite, refactor, resub等）
奖励: 面积和延迟的改善
```

### 源码解读和运行

```bash
# 安装DRiLLS
git clone https://github.com/kanhelou/DRiLLS.git
cd DRiLLS
pip install -r requirements.txt

# 需要安装ABC逻辑综合工具
# ABC: https://github.com/berkeley-abc/abc
```

### 关键代码

```python
# drills/environment.py
class LogicSynthesisEnv:
    """逻辑综合环境"""
    
    ACTIONS = [
        'balance',
        'rewrite', 
        'rewrite -z',
        'refactor',
        'refactor -z',
        'resubstitute',
        'resubstitute -z',
        'resubstitute -N 2',
    ]
    
    def __init__(self, circuit_file):
        self.circuit_file = circuit_file
        self.abc = ABC()  # ABC工具接口
        
    def reset(self):
        """重置环境"""
        self.abc.read(self.circuit_file)
        self.abc.balance()
        return self.get_state()
    
    def step(self, action_idx):
        """执行一步优化"""
        action = self.ACTIONS[action_idx]
        
        # 记录优化前的状态
        old_state = self.get_state()
        
        # 执行优化
        self.abc.run(action)
        
        # 获取新状态
        new_state = self.get_state()
        
        # 计算奖励
        reward = self.compute_reward(old_state, new_state)
        
        return new_state, reward, False, {}
    
    def get_state(self):
        """提取电路特征"""
        stats = self.abc.get_stats()
        return [
            stats['gates'],          # 门数
            stats['levels'],         # 逻辑深度
            stats['area'],           # 面积
            stats['delay'],          # 延迟
            stats['wires'],          # 连线数
        ]
```

### 训练和测试

```python
# 训练Agent
agent = DRiLLSAgent(state_dim=5, n_actions=8)
env = LogicSynthesisEnv('benchmark.blif')

for episode in range(200):
    state = env.reset()
    total_reward = 0
    
    for step in range(20):
        action = agent.select_action(state)
        next_state, reward, _, _ = env.step(action)
        agent.store(state, action, reward, next_state)
        agent.train()
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode}: Reward={total_reward:.4f}")
```
