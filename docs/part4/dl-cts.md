# 4.4 深度学习与CTS

## 生成对抗网络（GAN）应用于时钟树预测和优化（TCAD 2022）

### GAN基本原理

GAN由两个网络组成：

```
生成器G: 生成假样本（如时钟树结构）
判别器D: 判断样本是真实还是生成的

训练目标:
  G: 最小化 log(1 - D(G(z)))
  D: 最大化 log(D(x)) + log(1 - D(G(z)))
```

### GAN用于时钟树预测

```
输入: 芯片布局信息（寄存器位置、Macro位置等）
    |
    v
生成器G: 预测时钟树拓扑和Buffer位置
    |
    v
判别器D: 判断预测的时钟树是否合理
    |
    v
输出: 优化的时钟树结构
```

### 优势

- 能够生成多种候选方案
- 学习时钟树的分布特征
- 快速生成初始时钟树，指导后续优化

## 图神经网络应用于时序预测（DAC 2022）

### 问题定义

预测芯片中每个节点的时序信息（到达时间、松弛时间等），用于指导CTS和后续优化。

### 方法

将电路表示为图，使用GNN进行时序预测：

```
电路图:
  节点 = 标准单元/寄存器
  边 = 网线连接

GNN模型:
  输入: 节点特征（单元类型、位置、驱动强度等）
  消息传递: 多层GCN/GraphSAGE
  输出: 每个节点的时序信息
```

### 模型架构

```python
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class TimingGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)  # 预测slack
        
    def forward(self, x, edge_index, batch=None):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        return self.head(x)
```

### 训练数据

```
特征:
  - 单元类型 (one-hot编码)
  - 位置坐标 (x, y)
  - 驱动强度
  - 负载电容
  - 扇入/扇出数
  
标签:
  - STA计算的到达时间（Arrival Time）
  - 松弛时间（Slack）
```

## iCTS框架（TCAD 2025）

### 核心思想

iCTS是一个集成的时钟树综合框架，将传统算法与机器学习相结合：

```
1. ML预测: 使用机器学习预测最优的聚类参数
2. 自适应聚类: 根据预测结果进行寄存器聚类
3. DME平衡: 使用改进的DME算法构建时钟树
4. 迭代优化: 根据时序反馈迭代调整
```

### 创新点

- 自动调整聚类粒度
- 考虑时序约束的自适应CTS
- 与后端流程的紧密集成

## TritonCTS（TCAD 2018）

### 简介

TritonCTS是一个先进的开源CTS算法，是OpenROAD项目的一部分。

### 核心特点

```
1. 多目标优化: 同时优化Skew、Latency、DRC
2. 灵活的拓扑: 支持多种时钟树结构
3. 可扩展性: 处理大规模设计
4. 开源可用: 集成在OpenROAD中
```

### TritonCTS的流程

```
输入: 网表 + 布局结果
    |
    v
1. 时钟网络分析
    |
    v
2. 寄存器聚类（基于物理距离）
    |
    v
3. 时钟树构建（Buffer插入）
    |
    v
4. 延迟平衡（DME）
    |
    v
5. DRC修复
    |
    v
输出: 时钟树网表
```

### 与传统CTS的区别

| 特性 | 传统CTS | TritonCTS |
|------|---------|-----------|
| 聚类方法 | 固定参数 | 自适应参数 |
| 优化目标 | 单一（Skew） | 多目标 |
| 开源 | 否 | 是 |
| 可定制性 | 有限 | 高度可定制 |
