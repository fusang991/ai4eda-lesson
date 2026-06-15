# 1.2 芯片设计流程与EDA应用

## 芯片设计的完整流程——从Spec到Tape Out

芯片设计是一个从需求规格到最终制造的复杂工程，通常分为以下几个主要阶段：

### 1. 系统规格定义（Spec）

- 确定芯片的功能需求、性能指标、功耗预算
- 定义接口标准和协议
- 划分硬件/软件边界

### 2. 架构设计（Architecture）

- 设计芯片的整体架构
- 确定模块划分和互连方式
- 评估PPA（Power, Performance, Area）

### 3. RTL设计（Register Transfer Level）

- 使用硬件描述语言（Verilog/VHDL）编写功能代码
- 进行功能仿真和验证
- 编写测试平台（Testbench）

### 4. 逻辑综合（Logic Synthesis）

- 将RTL代码转换为门级网表（Gate-level Netlist）
- 进行时序约束优化
- 生成综合报告

### 5. 物理设计（Physical Design / APR）

这是本课程的重点，后续章节将详细介绍：

```
RTL -> 逻辑综合 -> 门级网表 -> Floorplan -> Placement -> CTS -> Route -> GDSII
```

### 6. 签核验证（Sign-off）

- 静态时序分析（STA）
- 形式验证（Formal Verification）
- 版图一致性检查（LVS）
- 设计规则检查（DRC）
- 电源完整性分析（PI/IR Drop）

### 7. Tape Out

- 生成最终GDSII文件
- 送交晶圆厂制造

## EDA工具在后端物理设计中的应用——从RTL到GDSII

物理设计（APR）是将门级网表转换为可制造版图的过程，包含以下关键步骤：

### Floorplan（平面规划）

- 确定芯片面积和形状
- 放置I/O Pad和Macro（存储器、IP等）
- 规划电源网络（Power Grid）
- 划分供电区域

### Placement（布局）

- **Global Placement**：将标准单元粗略放置到芯片上
- **Detailed Placement**：精细调整单元位置
- 优化目标：线长最小化、拥塞度控制、时序优化

### CTS（Clock Tree Synthesis，时钟树综合）

- 构建时钟分配网络
- 优化目标：时钟偏斜（Skew）最小化、时钟延迟控制
- 插入时钟缓冲器（Buffer）

### Routing（布线）

- **Global Routing**：规划布线区域和通道
- **Detailed Routing**：完成实际金属连线
- 遵守设计规则（DRC），优化线长和信号完整性

### 物理验证

- **DRC**（Design Rule Check）：检查版图是否满足制造规则
- **LVS**（Layout vs. Schematic）：验证版图与电路图一致性
- **ERC**（Electrical Rule Check）：检查电气规则

## AI4EDA的发展历程

| 年份 | 里程碑 | 来源 |
|------|--------|------|
| 2018 | RouteNet：CNN用于可布线性预测 | ICCAD |
| 2019 | GCN用于电路可测性提升 | DAC |
| 2020 | DRiLLS：强化学习用于逻辑综合 | ASP-DAC |
| 2021 | Google AlphaChip：强化学习用于芯片布局 | Nature |
| 2021 | 异步强化学习用于详细布线 | DATE |
| 2022 | GNN用于时序预测 | DAC |
| 2022 | GAN用于时钟树预测 | TCAD |
| 2023 | DREAMPlace：深度学习VLSI布局工具 | ICCAD |
| 2025 | iCTS框架 | TCAD |
| 2025 | DR-Guide：生成式AI用于详细布线 | MLCAD |
