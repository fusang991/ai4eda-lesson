# 3.5 强化学习基础

## 强化学习的基本思想

强化学习（Reinforcement Learning, RL）是一种通过与环境交互来学习最优策略的机器学习方法。

### 核心要素

| 要素 | 含义 | EDA中的对应 |
|------|------|------------|
| 智能体（Agent） | 做决策的主体 | 布局优化算法 |
| 环境（Environment） | 智能体所处的外部世界 | 芯片设计状态 |
| 状态（State） | 环境的当前描述 | 当前布局方案 |
| 动作（Action） | 智能体可以执行的操作 | 移动/交换单元 |
| 奖励（Reward） | 动作的好坏反馈 | 线长减少量 |

### 交互循环

```
在每个时间步t:
  1. 智能体观察状态 s_t
  2. 智能体选择动作 a_t
  3. 环境返回奖励 r_t 和新状态 s_{t+1}
  4. 智能体根据经验更新策略
```

### 价值函数和策略

**状态价值函数 V(s)**：从状态s出发，遵循策略pi，期望获得的累积奖励

```
V(s) = E[r_0 + gamma*r_1 + gamma^2*r_2 + ... | s_0 = s]
```

**动作价值函数 Q(s,a)**：在状态s执行动作a后，遵循策略pi，期望获得的累积奖励

```
Q(s,a) = E[r_0 + gamma*r_1 + gamma^2*r_2 + ... | s_0 = s, a_0 = a]
```

其中 gamma (折扣因子) 控制对未来奖励的重视程度。

## 强化学习的机制和常用算法

### Q-Learning

基于值函数的离策略算法：

```
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

### DQN（Deep Q-Network）

用神经网络近似Q函数：

```
Q(s,a; theta) ≈ Q*(s,a)

其中theta是神经网络参数
```

DQN的关键技巧：
1. **经验回放（Experience Replay）**：存储历史经验，随机采样训练
2. **目标网络（Target Network）**：使用固定的目标网络稳定训练
3. **epsilon-贪心策略**：平衡探索和利用

```python
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
```

### 策略梯度（Policy Gradient）

直接优化策略函数：

```
J(theta) = E[sum(gamma^t * r_t)]
梯度: nabla J(theta) = E[nabla log pi(a|s; theta) * G_t]
```

### PPO（Proximal Policy Optimization）

当前最常用的策略梯度算法，通过限制策略更新幅度来稳定训练。

## 强化学习的应用场景

### 典型应用场景

- **游戏**：AlphaGo、Atari游戏
- **机器人控制**：机械臂操作、行走控制
- **自动驾驶**：路径规划
- **推荐系统**：个性化推荐
- **资源调度**：任务分配、网络路由

### 在EDA领域的应用

| 应用 | 算法 | 论文 |
|------|------|------|
| 芯片布局 | Policy Gradient | AlphaChip (Nature 2021) |
| 布线顺序 | 异步RL | (DATE 2021) |
| 逻辑综合 | DQN | DRiLLS (ASP-DAC 2020) |
| 板级布线 | DQN | RL_PCB (DATE 2024) |
