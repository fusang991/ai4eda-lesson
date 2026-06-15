# 第三部分：AI4EDA——APR Placement布局算法

## 课程概览

本部分是课程的核心模块，深入介绍芯片物理设计中的布局（Placement）过程和相关算法，以及图神经网络、强化学习在布局优化中的应用。

## 学习目标

- 理解后端物理设计的完整流程
- 掌握Placement从Global到Detailed的全过程
- 理解布局优化的目标：线长、拥塞度、PPA
- 掌握图卷积神经网络（GCN）的原理和应用
- 理解强化学习的基本思想和DQN算法
- 了解AlphaChip、DREAMPlace等经典AI4EDA工作

## 内容结构

| 章节 | 内容 | 类型 |
|------|------|------|
| [3.1 后端物理设计流程](./pd-flow) | Netlist到GDSII的完整流程 | 理论 |
| [3.2 Placement过程详解](./placement-detail) | Global/Detailed Placement | 理论 |
| [3.3 Placement相关算法](./algorithms) | Min-Cut、SA、解析法 | 理论 |
| [3.4 图卷积神经网络](./gcn) | GNN原理、消息传递 | 理论 |
| [3.5 强化学习基础](./rl) | DQN、策略梯度 | 理论 |
| [3.6 深度学习用于Placement优化](./dl-placement) | AlphaChip、DREAMPlace | 理论 |
| [3.7 上机实验](./lab) | SA布局、DQN、AlphaChip、DREAMPlace | 实验 |
