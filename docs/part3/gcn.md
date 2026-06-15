# 3.4 图卷积神经网络

## 图数据结构和图卷积神经网络简介

### 图数据结构

图（Graph）由节点（Node）和边（Edge）组成，非常适合表示电路：

```
电路 -> 图:
  标准单元 -> 节点
  网线(Net) -> 边
```

在EDA中，电路网表天然是图结构，因此图神经网络（GNN）是处理电路数据的理想选择。

### 图的数学表示

```
图 G = (V, E)

邻接矩阵 A: A[i][j] = 1 如果节点i和j之间有边
度矩阵 D: D[i][i] = 节点i的度数
特征矩阵 X: X[i] = 节点i的特征向量
```

## 图神经网络应用于电路和网表建模

### 节点特征设计

在电路图中，每个节点（单元）的特征可以包括：

| 特征 | 说明 |
|------|------|
| 单元类型 | AND, OR, FF, BUF等 |
| 输入/输出引脚数 | 网络拓扑信息 |
| 扇入/扇出度 | 连接关系 |
| 面积 | 物理属性 |
| 时序余量 | 时序信息 |
| 位置坐标 | 物理位置 |

### 边特征设计

| 特征 | 说明 |
|------|------|
| 网线权重 | 连接的重要性 |
| 线长 | 物理距离 |
| 连接类型 | 信号/时钟/电源 |

## 深入理解图卷积——消息传递过程

### 消息传递范式（Message Passing）

图神经网络的核心是消息传递机制：

```
对于每个节点 v:

1. 消息生成（Message）:
   m_uv = MSG(h_u)  对于每个邻居u

2. 消息聚合（Aggregate）:
   m_v = AGG({m_uv : u 属于 N(v)})
   
   常用聚合函数: sum, mean, max

3. 节点更新（Update）:
   h_v' = UPDATE(h_v, m_v)
```

### 图卷积网络（GCN）的数学公式

GCN是图神经网络的一种经典形式：

```
H^(l+1) = sigma(D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l) * W^(l))

其中:
  A_hat = A + I  (添加自连接)
  D_hat = 度矩阵（基于A_hat）
  H^(l) = 第l层的节点特征矩阵
  W^(l) = 第l层的可学习权重矩阵
  sigma = 激活函数（如ReLU）
```

直观理解：
1. `D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l)`: 聚合邻居特征（归一化）
2. `* W^(l)`: 线性变换
3. `sigma()`: 非线性激活

### PyTorch Geometric实现

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class CircuitGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch=None):
        # 第1层图卷积
        x = F.relu(self.conv1(x, edge_index))
        # 第2层图卷积
        x = F.relu(self.conv2(x, edge_index))
        # 第3层图卷积
        x = self.conv3(x, edge_index)
        
        # 图级别的输出（可选）
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x
```

## 图卷积和图像卷积的区别

| 特性 | 图像卷积（CNN） | 图卷积（GCN） |
|------|-----------------|---------------|
| 数据结构 | 规则网格（2D/3D） | 不规则图结构 |
| 卷积核 | 固定大小（3x3, 5x5） | 基于邻域动态确定 |
| 参数共享 | 空间共享 | 通道维度共享 |
| 邻居数量 | 固定 | 每个节点不同 |
| 位置敏感 | 有固定的空间位置 | 无固定位置概念 |
| 应用场景 | 图像、语音 | 分子、社交网络、电路 |

### 关键区别

1. **不规则结构**：图没有固定的"左上角"或"右下角"概念
2. **邻居数量可变**：每个节点的邻居数不同，无法使用固定大小的卷积核
3. **置换不变性**：节点的编号顺序不应影响结果
