# 第六部分：Paper阅读与复现

## 课程概览

本部分对课程中涉及的6篇AI4EDA经典论文进行阅读理解和代码复现。每篇论文都提供了：论文解读、核心方法分析、PyTorch代码实现和实验结果。

## 复现环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4060 Laptop (8GB) |
| CUDA | 12.6 |
| PyTorch | 2.9.1 |
| Python | 3.10 |
| 工具 | ABC (yosys-abc), Yosys 0.33 |

## 复现结果总览

| 论文 | 方法 | 核心结果 |
|------|------|----------|
| [DRiLLS (ASP-DAC 2020)](./drills) | A2C强化学习 | 面积 -8.5% (506→463 AND gates) |
| [DREAMPlace (DAC 2019)](./dreamplace) | 梯度下降布局 | HPWL -91% (15715→1416) |
| [AlphaChip (Nature 2021)](./alphachip) | GNN+策略梯度 | HPWL 184 (vs 随机 219, -16%) |
| [GNN时序预测 (DAC 2022)](./gnn-timing) | GraphSAGE | R²=0.54, MAE=0.12 |
| [RouteNet (ICCAD 2018)](./routenet) | U-Net CNN | 相关系数 0.998 |
| [IR Drop预测](./irdrop) | Inception CNN | 空间相关 0.997 |

## 代码仓库

所有复现代码位于GitHub仓库的 `reproduce` 分支：
```
https://github.com/fusang991/ai4eda-lesson/tree/reproduce
```

## 运行方式

每个实验都是自包含的Python脚本，使用以下命令运行：

```bash
conda activate pytorch
python experiments/01-drills/drills_pytorch.py --design multiplier.v --episodes 20
python experiments/02-dreamplace/dreamplace_reproduction.py
python experiments/03-alphachip/alphachip_repro.py
python experiments/04-gnn-timing/gnn_timing.py
python experiments/05-routenet/routenet_experiment.py
python experiments/06-irdrop/irdrop_cnn.py
```
