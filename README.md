# AI4EDA 论文复现报告

深度学习集成电路设计专题 — 6篇经典论文复现

## 环境

- GPU: NVIDIA RTX 4060 Laptop (8GB)
- CUDA: 12.6
- PyTorch: 2.9.1
- Python: 3.10 (conda pytorch env)
- 工具: ABC (yosys-abc), Yosys 0.33

## 复现结果总览

| # | 论文 | 方法 | 核心结果 | 状态 |
|---|------|------|----------|------|
| 1 | DRiLLS (ASP-DAC 2020) | A2C强化学习 | 面积 -8.5% (506→463) | ✅ |
| 2 | DREAMPlace (DAC 2019) | 梯度下降布局 | HPWL -91% (15715→1416) | ✅ |
| 3 | AlphaChip (Nature 2021) | GNN+策略梯度 | HPWL 184 (vs随机219) | ✅ |
| 4 | GNN时序预测 (DAC 2022) | GCN/GraphSAGE | R²=0.54, MAE=0.12 | ✅ |
| 5 | RouteNet (ICCAD 2018) | U-Net CNN | 相关系数0.998 | ✅ |
| 6 | IR Drop预测 | Inception CNN | 空间相关0.997 | ✅ |

---

## 实验1: DRiLLS — 强化学习用于逻辑综合

**论文**: A. Hosny et al., "DRiLLS: Deep Reinforcement Learning for Logic Synthesis", ASP-DAC 2020

**方法**: 用A2C (Advantage Actor-Critic) 强化学习Agent自动选择ABC逻辑综合的优化操作序列（rewrite, refactor, resub, balance等），目标是最小化面积。

**实现**: PyTorch重写原版TensorFlow 1.x代码。环境通过调用yosys-abc执行逻辑优化，提取AIG特征（节点数、边数、逻辑深度等6维），使用Actor-Critic网络学习最优策略。

**结果**:
```
设计: multiplier (8bit乘法器)
初始面积: 506 AND gates
最优面积: 463 AND gates (-8.5%)
最优操作: rewrite -z
```

**文件**: `experiments/01-drills/drills_pytorch.py`, `drills_training.png`

---

## 实验2: DREAMPlace — 深度学习VLSI布局

**论文**: Y. Lin et al., "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement", DAC 2019 (ICCAD 2023扩展)

**方法**: 将布局问题转化为类似神经网络训练的优化问题。使用PyTorch自动微分计算可微分HPWL（Log-Sum-Exp近似）和密度惩罚的梯度，用Adam优化器迭代更新单元位置。

**实现**: 纯PyTorch实现，包含可微分线长模型、bell-shaped密度模型、自适应gamma调整。

**结果**:
```
基准: 50 cells + 3 macros, 152 nets
初始HPWL: 15,715
最终HPWL: 1,416 (-91.0%)
迭代次数: 200
运行时间: 67s (GPU)
```

**文件**: `experiments/02-dreamplace/dreamplace_reproduction.py`, `dreamplace_results.png`

---

## 实验3: AlphaChip — 强化学习芯片布局

**论文**: A. Mirhoseini et al., "A graph placement methodology for fast chip design", Nature 2021

**方法**: 使用GNN编码电路图拓扑，自回归策略网络依次放置单元到网格位置，用REINFORCE策略梯度训练，奖励为负HPWL。

**实现**: PyTorch + PyTorch Geometric，3层GAT (Graph Attention Network)编码器 + MLP策略头，在30个单元的小电路上训练。

**结果**:
```
电路: 30 cells, 45 nets, 6x5 grid
随机平均HPWL: 218.8
RL Agent最佳HPWL: 184.0 (-15.9%)
训练回合: 1200
```

**文件**: `experiments/03-alphachip/alphachip_repro.py`, `alphachip_results.png`

---

## 实验4: GNN时序预测

**论文**: GNN用于电路时序预测 (DAC 2022方向)

**方法**: 将电路网表建模为图（节点=标准单元，边=网线连接），用GCN/GraphSAGE进行消息传递，预测每个节点的到达时间（arrival time）。

**实现**: 合成1500个电路DAG（80-250节点），12种门类型，18维节点特征。用简化STA算法计算标签。对比GCN和GraphSAGE两种架构。

**结果**:
```
GraphSAGE: MSE=0.024, MAE=0.124, R²=0.542
GCN:       MSE=0.027, MAE=0.132, R²=0.492
结论: GraphSAGE略优于GCN，与文献一致
```

**文件**: `experiments/04-gnn-timing/gnn_timing.py`, `gnn_timing.png`

---

## 实验5: RouteNet — CNN可布线性预测

**论文**: RouteNet, "Deep Learning for Routability Prediction", ICCAD 2018

**方法**: 用CNN从布局特征图（单元密度、引脚密度、网线密度）预测拥塞度图，在详细布线前识别潜在的布线困难区域。

**实现**: U-Net风格的编码器-解码器CNN（~1.9M参数），3通道输入（密度特征），1通道输出（拥塞预测），60个epoch训练。

**结果**:
```
测试集MAE: 0.0117
Pearson相关系数: 0.9981
结论: CNN能高精度预测拥塞分布
```

**文件**: `experiments/05-routenet/routenet_experiment.py`, `routenet_results.png`

---

## 实验6: IR Drop CNN预测

**论文**: 深度CNN用于芯片电压降预测 (课程第五部分)

**方法**: 使用Inception-style多尺度CNN（1x1, 3x3, 5x5卷积核并行）从芯片特征图（开关活动、单元密度、电源Pad距离等5通道）预测IR Drop分布。

**实现**: U-Net-lite + 3个Inception Block（505K参数），在64x64网格上训练30个epoch。

**结果**:
```
测试集MAE: 0.0234
RMSE: 0.0324
空间相关系数: 0.9965
结论: Inception多尺度特征融合对IR Drop预测非常有效
```

**文件**: `experiments/06-irdrop/irdrop_cnn.py`, `irdrop_results.png`

---

## 目录结构

```
experiments/
├── 01-drills/          # DRiLLS: RL逻辑综合
│   ├── drills_pytorch.py
│   ├── multiplier.v
│   └── drills_training.png
├── 02-dreamplace/      # DREAMPlace: 深度学习布局
│   ├── dreamplace_reproduction.py
│   └── dreamplace_results.png
├── 03-alphachip/       # AlphaChip: GNN+RL布局
│   ├── alphachip_repro.py
│   └── alphachip_results.png
├── 04-gnn-timing/      # GNN时序预测
│   ├── gnn_timing.py
│   └── gnn_timing.png
├── 05-routenet/        # RouteNet: CNN拥塞预测
│   ├── routenet_experiment.py
│   └── routenet_results.png
└── 06-irdrop/          # IR Drop: Inception CNN
    ├── irdrop_cnn.py
    ├── best_model.pt
    └── irdrop_results.png
```

## 复现心得

1. **DRiLLS**: ABC不直接支持Verilog输入，需要先用Yosys综合成BLIF。A2C在小电路上能学到有效的优化策略。

2. **DREAMPlace**: 核心创新是把布局问题"伪装"成神经网络训练——用PyTorch的自动微分计算线长和密度梯度。Log-Sum-Exp是HPWL的可微近似关键。

3. **AlphaChip**: GNN天然适合电路图表示。自回归放置策略+REINFORCE是核心。在小规模问题上效果明显，但训练时间较长。

4. **GNN时序预测**: GraphSAGE的邻居采样+聚合机制比GCN的简单加权平均更适合电路图（节点度数变化大）。

5. **RouteNet**: U-Net的跳跃连接对空间预测任务非常重要，保留了细粒度位置信息。

6. **IR Drop**: Inception的多尺度卷积能同时捕获局部和全局的IR Drop模式，效果优于单一尺度CNN。
