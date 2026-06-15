# 6.3 AlphaChip: GNN+RL芯片布局

## 论文信息

| 项目 | 内容 |
|------|------|
| 标题 | A graph placement methodology for fast chip design |
| 作者 | A. Mirhoseini et al. |
| 期刊 | Nature 2021 |
| 机构 | Google Research |
| 代码 | https://github.com/google-research/circuit_training |

## 问题背景

芯片布局（特别是Macro Placement）是一个NP-hard问题。传统方法依赖人类专家的经验和大量迭代。

## 核心方法

### 整体流程

```
电路网表 -> GNN编码器 -> 策略网络 -> 依次放置单元 -> 计算HPWL -> 更新策略
```

### 1. 图表示学习（GNN编码器）

将电路网表建模为二部图：
- 节点：标准单元/Macro
- 边：网线连接

使用Graph Attention Network (GAT)编码：
```python
# 每层GAT
h_v' = sigma(sum(a_{vu} * W * h_u))  对于邻居u
a_{vu} = softmax(LeakyReLU(a^T [Wh_v || Wh_u]))
```

### 2. 自回归策略网络

依次放置每个单元：
```
for 每个待放置的单元v:
    1. GNN编码当前状态（已放置+未放置的图）
    2. 获取单元v的嵌入 h_v
    3. 计算每个网格位置的得分: score(pos) = MLP([h_v, h_pos])
    4. Softmax得到概率分布
    5. 采样选择位置
```

### 3. REINFORCE训练

```
策略: pi(a|s; theta)
目标: max E[sum(-HPWL)]

梯度更新:
  nabla J = E[nabla log pi(a|s) * (R - baseline)]
  
  R = -最终HPWL
  baseline = 指数移动平均的历史HPWL
```

## 复现实现

### 简化设置

```python
class AlphaChipRepro:
    def __init__(self):
        self.gnn = GAT(in_features, hidden_dim, n_heads=4)  # 3层GAT
        self.policy = MLP(hidden_dim, grid_size)  # 策略头
    
    def place(self, circuit_graph):
        placements = []
        for cell in circuit_graph.cells:
            h = self.gnn(circuit_graph)  # 编码当前图状态
            probs = softmax(self.policy(h[cell]))
            pos = Categorical(probs).sample()
            placements.append(pos)
        return placements
```

### 训练

```python
for episode in range(1200):
    placement = agent.place(circuit)
    hpwl = compute_hpwl(placement, nets)
    reward = -hpwl
    
    # REINFORCE
    loss = -log_prob * (reward - baseline)
    loss.backward()
    optimizer.step()
    baseline = 0.99 * baseline + 0.01 * reward
```

## 实验结果

```
电路: 30 cells, 45 nets
网格: 6x5
训练回合: 1200

随机放置平均HPWL: 218.8
随机放置最优HPWL: 185.0 (500次采样)
RL Agent最佳HPWL: 184.0 (-15.9% vs 平均)
```

![AlphaChip训练结果](/part6/alphachip_results.png)

## 关键发现

1. **GNN是电路的理想编码器**：电路天然具有图结构，GNN的消息传递机制能捕获拓扑信息
2. **自回归放置**：依次放置比一次性输出所有位置更灵活，能考虑已放置单元的影响
3. **迁移学习潜力**：在多个设计上训练的Agent可以迁移到新设计（原论文的核心贡献）
4. **训练不稳定**：REINFORCE方差大，baseline和学习率调参很关键

## 与原版的差异

原版AlphaChip在Google TPU设计上训练，使用了大规模分布式RL训练。我们的复现在小规模合成电路上验证了核心方法的有效性。
