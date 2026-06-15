# 3.6 深度学习用于Placement优化

## 基于扩散模型的Macro Placement方案（PMLR 2025）

### 核心思想

利用扩散模型（Diffusion Model）生成高质量的Macro布局方案：

```
噪声布局 -> 扩散模型去噪 -> 优化后的Macro布局
```

### 方法

1. 将Macro布局表示为2D图像
2. 使用扩散模型学习布局的分布
3. 在推理时从随机噪声生成布局
4. 结合强化学习微调优化

## Google AlphaChip（Nature 2021）

### 背景

Google在2021年发表于Nature的工作，是第一个将深度强化学习应用于芯片布局并取得超越人类专家效果的方法。

### 核心方法

```
1. 将芯片布局建模为强化学习问题:
   - 状态: 当前已放置的单元位置
   - 动作: 选择下一个要放置的单元及其位置
   - 奖励: 负的线长（线长越短奖励越大）

2. 使用Policy Gradient训练策略网络:
   - 输入: 芯片的图表示（网表拓扑）
   - 网络: GNN + MLP
   - 输出: 每个位置的放置概率

3. 训练流程:
   a. 从多个芯片设计中收集布局经验
   b. 使用RL优化策略网络
   c. 迁移到新设计上快速生成布局
```

### 架构

```
输入图（网表）
    |
    v
GNN编码器（提取节点特征）
    |
    v
MLP解码器（输出放置概率）
    |
    v
采样动作 -> 放置单元 -> 更新状态
    |
    v
计算奖励（-HPWL）-> 更新策略
```

### 关键创新

1. **图表示学习**：用GNN处理电路网表的图结构
2. **迁移学习**：在多个设计上训练，迁移到新设计
3. **高效的奖励设计**：中间奖励（部分放置的线长）+ 最终奖励

## DREAMPlace（ICCAD 2023）

### 核心思想

将布局问题转化为类似于深度学习训练的优化问题，利用GPU加速求解。

### 方法

```
1. 将布局目标函数表示为可微分的形式:
   Loss = Wirelength + Density + Boundary
   
2. 使用PyTorch的自动微分计算梯度

3. 使用Adam优化器迭代更新单元位置
   （类似于训练神经网络）
```

### DREAMPlace的创新

- **可微HPWL近似**：用Log-Sum-Exp函数平滑HPWL
- **GPU加速**：利用GPU并行计算密度和线长
- **混合尺度优化**：同时处理标准单元和Macro

### 代码结构（简化）

```python
import torch

class DREAMPlace:
    def __init__(self, cells, nets, die_area):
        # 可学习参数：单元位置
        self.pos_x = torch.tensor(initial_x, requires_grad=True)
        self.pos_y = torch.tensor(initial_y, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.pos_x, self.pos_y], lr=0.01)
    
    def wirelength_loss(self):
        """可微分的HPWL近似"""
        total = 0
        for net in self.nets:
            x = self.pos_x[net.pins]
            y = self.pos_y[net.pins]
            # Log-Sum-Exp平滑近似
            wl_x = torch.logsumexp(x / self.gamma, 0) * self.gamma - \
                   torch.logsumexp(-x / self.gamma, 0) * self.gamma
            wl_y = torch.logsumexp(y / self.gamma, 0) * self.gamma - \
                   torch.logsumexp(-y / self.gamma, 0) * self.gamma
            total += wl_x + wl_y
        return total
    
    def density_loss(self):
        """密度均匀性惩罚"""
        # 将芯片划分为网格，计算每个格子的单元密度
        # 目标：均匀分布
        pass
    
    def optimize(self, n_steps=1000):
        for step in range(n_steps):
            loss = self.wirelength_loss() + self.density_weight * self.density_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

## RL_PCB（DATE 2024）

### 核心思想

将强化学习应用于板级（PCB）布线：

```
状态: 当前布线状态（已布线、待布线、障碍物）
动作: 选择下一步布线方向
奖励: 线长减少 + DRC违规惩罚
```

## 各方法对比

| 方法 | 类型 | 年份 | 会议 | 特点 |
|------|------|------|------|------|
| AlphaChip | 强化学习 | 2021 | Nature | 迁移学习、GNN |
| DREAMPlace | 深度学习优化 | 2023 | ICCAD | GPU加速、可微分 |
| 扩散模型 | 生成模型 | 2025 | PMLR | 生成式布局 |
| RL_PCB | 强化学习 | 2024 | DATE | 板级布线 |
