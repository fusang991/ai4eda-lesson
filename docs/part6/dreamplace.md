# 6.2 DREAMPlace: 深度学习VLSI布局

## 论文信息

| 项目 | 内容 |
|------|------|
| 标题 | DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement |
| 作者 | Yibo Lin et al. |
| 会议 | DAC 2019 (ICCAD 2023扩展版) |
| 代码 | https://github.com/limbo018/DREAMPlace |

## 问题背景

VLSI布局是物理设计中最耗时的步骤之一。传统方法（如模拟退火、力导向布局）在大规模设计上运行缓慢。

## 核心思想

**关键洞察**：布局问题的数学形式与深度学习训练非常相似——都是优化一个目标函数。

```
深度学习:  最小化 Loss(theta)    -> 梯度下降更新theta
DREAMPlace: 最小化 WL(x,y) + lambda*D(x,y) -> 梯度下降更新(x,y)
```

## 核心算法

### 1. 可微分HPWL（Log-Sum-Exp近似）

HPWL不可微（max/min函数在相同值处不可微）。DREAMPlace使用Log-Sum-Exp平滑：

```
近似max(x1,...,xn) ≈ gamma * log(sum(exp(xi/gamma)))

WL_x = gamma * log(sum(exp(xi/gamma))) - gamma * log(sum(exp(-xi/gamma)))
```

gamma从小到大逐渐调整：初期粗略搜索，后期精细优化。

### 2. 密度惩罚（Bell-Shaped模型）

将单元密度分布到网格中，计算每个格子的密度，施加惩罚：

```
对于每个格子(i,j):
  density(i,j) = sum(单元k在格子(i,j)中的重叠面积)
  penalty += (density(i,j) - target_density)^2
```

### 3. 优化流程

```
for 每次迭代:
    loss = wirelength_loss + density_weight * density_loss
    loss.backward()          # PyTorch自动微分
    optimizer.step()         # Adam更新单元位置
    adjust_gamma()           # 逐渐增大gamma
    adjust_density_weight()  # 逐渐增大密度权重
```

## 复现实现

```python
class DREAMPlace:
    def __init__(self, cells, nets, die_area):
        # 可学习参数：单元位置
        self.pos_x = torch.tensor(initial_x, requires_grad=True)
        self.pos_y = torch.tensor(initial_y, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.pos_x, self.pos_y], lr=1.0)
    
    def wirelength_loss(self):
        """Log-Sum-Exp可微HPWL"""
        for net in nets:
            x = self.pos_x[net.pins]
            wl_x = gamma * logsumexp(x/gamma) - gamma * logsumexp(-x/gamma)
        return total_wl
    
    def optimize(self, n_steps=200):
        for step in range(n_steps):
            loss = self.wirelength_loss() + density_weight * self.density_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

## 实验结果

```
基准: 50 cells + 3 macros, 152 nets, Die 100x100
初始HPWL: 15,715
最终HPWL: 1,416 (-91.0%)
迭代次数: 200
运行时间: 67s (GPU)
```

![DREAMPlace布局结果](/part6/dreamplace_results.png)

## 关键发现

1. **"布局即训练"**：把布局问题转化为PyTorch优化问题，自动获得GPU加速和自动微分
2. **gamma调度策略**：从小gamma到大gamma的渐进策略类似模拟退火的降温
3. **密度权重自适应**：初期关注线长优化，后期逐渐加入密度约束消除重叠
4. **GPU并行**：所有单元的位置同时更新，比逐个移动的传统方法快得多

## 与原版的差异

原版DREAMPlace使用C++/CUDA自定义算子加速密度计算和线长计算。我们的复现使用纯PyTorch实现，验证了核心算法的正确性，但运行速度较慢。
