# 5.3 深度学习与Routing

## RouteNet——AI应用芯片可布线性预测（ICCAD 2018）

### 问题定义

在详细布线之前，预测哪些区域可能存在布线困难（不可布线），从而在布局阶段就进行优化。

### 方法

使用CNN分析布局特征，预测拥塞度图：

```
输入: 布局后的特征图（单元密度、引脚密度、走线需求等）
    |
    v
CNN模型: 多层卷积 + 池化 + 全连接
    |
    v
输出: 拥塞度图（每个区域的拥塞程度）
```

### 模型架构

```python
import torch.nn as nn

class RouteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),   # 输入: 5个特征通道
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),  # 输出: 1通道拥塞图
        )
    
    def forward(self, x):
        features = self.encoder(x)
        congestion_map = self.decoder(features)
        return congestion_map
```

## 异步强化学习框架应用于详细布线（DATE 2021）

### 核心思想

使用强化学习优化详细布线中线网的布线顺序：

```
问题: N个线网的布线顺序会影响最终结果质量
方法: 用RL学习最优的布线顺序策略

状态: 当前布线状态（已布线、部分拥塞）
动作: 选择下一个要布线的线网
奖励: -DRC违规数 - 线长增量
```

### 异步RL框架

```
多个Agent并行工作:
  Agent 1: 探索不同的布线顺序
  Agent 2: 探索不同的布线顺序
  ...
  Agent N: 探索不同的布线顺序

汇总经验 -> 更新共享策略网络
```

## DR-Guide——应用于详细布线的生成式AI框架（MLCAD 2025）

### 核心思想

使用生成式AI（如扩散模型）直接生成高质量的布线方案：

```
输入: 布局信息 + 网表拓扑
    |
    v
生成式模型: 生成候选布线方案
    |
    v
评估和优化: 选择最优方案并修复DRC
    |
    v
输出: 详细布线结果
```

## TritonRoute——OpenROAD的Route组件

### 简介

TritonRoute是OpenROAD项目中的详细布线工具，是当前最先进的开源布线器之一。

### 核心特点

```
1. 并行布线: 多线程并行处理
2. DRC驱动: 以DRC为中心的布线策略
3. 2-Pass方法:
   - Pass 1: 快速初始布线
   - Pass 2: DRC感知的精细优化
4. 迭代rip-up和reroute
```

### TritonRoute流程

```
1. 读入全局布线结果
2. Track分配
3. 详细布线（多轮迭代）:
   a. 为每个2-pin pin-pair分配路径
   b. 检测DRC违规
   c. Rip-up违规的走线
   d. 重新布线
4. 后处理和优化
```

## 各方法对比

| 方法 | 类型 | 年份 | 会议 | 应用 |
|------|------|------|------|------|
| RouteNet | CNN | 2018 | ICCAD | 可布线性预测 |
| 异步RL | 强化学习 | 2021 | DATE | 布线顺序优化 |
| DR-Guide | 生成式AI | 2025 | MLCAD | 详细布线 |
| TritonRoute | 传统算法 | 2018 | TCAD | 开源详细布线器 |
