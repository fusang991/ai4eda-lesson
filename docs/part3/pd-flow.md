# 3.1 后端物理设计流程

## 物理设计的整体流程

后端物理设计（Physical Design）是将门级网表转换为可制造版图的过程。完整的物理设计流程包括：

```
门级网表(Netlist)
    |
    v
Netlist Partition（网表切分）
    |
    v
Chip Planning（芯片规划）
    |
    v
Pin Assignment（引脚分配）
    |
    v
Powerplan（电源规划）
    |
    v
Placement（布局）
    |
    v
CTS（时钟树综合）
    |
    v
Route（布线）
    |
    v
GDSII（版图输出）
```

## Netlist Partition（网表切分）

将大规模电路网表切分为可管理的子模块，便于并行处理。使用第一部分介绍的超图分割算法（KL/FM算法）。

## Chip Planning（芯片规划）

- 确定芯片的面积和形状（Aspect Ratio）
- 规划Macro（存储器、IP）的位置
- 划分电源域（Power Domain）
- 确定I/O Pad的位置

## Pin Assignment（引脚分配）

- 为每个模块分配引脚位置
- 优化目标：减少模块间的连线交叉
- 考虑布线通道的可用性

## Powerplan（电源规划）

- 设计电源网络（Power Grid）
- 确保IR Drop在可接受范围内
- 通常使用多层金属网格结构

## 形式验证和物理验证

### 形式验证（Formal Verification, FM）

验证综合后的网表与原始RTL的功能等价性：
- 使用SAT求解器或BDD（二叉决策图）
- 确保逻辑变换没有引入功能错误

### 版图一致性验证（LVS, Layout vs. Schematic）

验证物理版图与电路网表的一致性：
- 从版图中提取电路
- 与原始网表进行比较
- 确保所有连接关系正确

### 静态时序分析（STA, Static Timing Analysis）

不运行仿真，通过分析电路拓扑来计算时序：
- 建立时间（Setup Time）约束
- 保持时间（Hold Time）约束
- 时钟偏斜（Clock Skew）分析

### 时序的基本概念

**建立时间（Setup Time）**：
- 数据必须在时钟沿到来之前稳定的最小时间
- Setup Slack = Clock Period - Data Path Delay - Setup Time
- Setup Slack > 0 表示满足时序

**保持时间（Hold Time）**：
- 数据在时钟沿到来之后必须保持稳定的最小时间
- Hold Slack = Data Path Delay - Clock Skew - Hold Time
- Hold Slack > 0 表示满足时序

### 电源完整性（Power Integrity）

- IR Drop分析：确保电源网络提供稳定的电压
- EM（Electromigration）分析：确保电流密度在安全范围内
- SSN（Simultaneous Switching Noise）分析
