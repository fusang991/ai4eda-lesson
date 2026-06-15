# 6.4 GNN时序预测

## 论文背景

| 项目 | 内容 |
|------|------|
| 方向 | GNN用于电路时序预测 |
| 代表会议 | DAC 2022 |
| 核心思想 | 用图神经网络预测STA（静态时序分析）结果 |

## 问题背景

传统STA需要完整的网表和寄存器库信息，运行时间随设计规模增长。在设计早期（如布局阶段），快速的时序预测可以指导优化决策。

## 核心方法

### 电路图建模

```
电路 -> 图:
  节点 = 标准单元（AND, OR, FF, BUF等）
  边 = 网线连接
  节点特征 = [门类型one-hot, 扇入/扇出, 电容, 驱动强度, x/y位置]
  标签 = arrival time (到达时间)
```

### GNN消息传递

```
对于每个节点v:
  1. 收集邻居特征: m_v = {h_u : u 属于 N(v)}
  2. 聚合: h_v' = UPDATE(h_v, AGG(m_v))
  
常用聚合方式:
  GCN: 加权平均 (度归一化)
  GraphSAGE: 采样+拼接
```

### GCN vs GraphSAGE

| 特性 | GCN | GraphSAGE |
|------|-----|-----------|
| 聚合方式 | 加权平均 | 采样+拼接/最大值 |
| 归一化 | D^(-1/2) A D^(-1/2) | 均值/最大值 |
| 适合场景 | 同质图 | 异质图（度数变化大） |
| 电路适用性 | 一般 | 更好（电路节点度数变化大） |

## 复现实现

### 数据生成

生成1500个合成电路DAG：
```python
# 每个电路80-250个节点，12种门类型
nodes = random_circuit(n_nodes=uniform(80, 250))
edges = random_dag_edges(nodes)

# 用简化STA计算标签
for node in topological_order:
    arrival_time = max(arrival_time[pred] + gate_delay[pred] + wire_delay 
                       for pred in predecessors)
```

### 模型架构

```python
class TimingGNN(nn.Module):
    def __init__(self, in_dim=18, hidden=128, layers=4):
        self.convs = nn.ModuleList([
            SAGEConv(in_dim if i==0 else hidden, hidden)
            for i in range(layers)
        ])
        self.head = nn.Linear(hidden, 1)  # 预测arrival time
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = relu(conv(x, edge_index))
        return self.head(x)
```

## 实验结果

```
数据集: 1500个合成电路 (1200 train / 300 test)
节点特征: 18维 (one-hot门类型 + 拓扑 + 物理)

GraphSAGE:
  MSE = 0.024
  MAE = 0.124
  RMSE = 0.156
  R² = 0.542

GCN:
  MSE = 0.027
  MAE = 0.132
  RMSE = 0.164
  R² = 0.492
```

![GNN时序预测结果](/part6/gnn_timing.png)

## 关键发现

1. **GraphSAGE > GCN**：电路图节点度数变化大（从1到几十），GraphSAGE的采样聚合更鲁棒
2. **拓扑信息很重要**：加上x/y位置特征后预测精度显著提升
3. **多层消息传递**：4层GNN能捕获较长的逻辑路径依赖
4. **R²=0.54说明**：GNN能学到时序趋势，但精确预测仍需传统STA

## 应用场景

- 布局阶段的快速时序评估（替代多次STA调用）
- 指导布局优化（识别关键路径区域）
- 设计空间探索（快速评估不同方案的时序）
