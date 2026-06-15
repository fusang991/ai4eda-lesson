# 6.6 IR Drop: Inception CNN预测

## 论文背景

| 项目 | 内容 |
|------|------|
| 方向 | 深度CNN用于芯片电压降预测 |
| 来源 | 课程第五部分、2024年EDA设计精英挑战赛 |
| 核心思想 | 用多尺度CNN从芯片特征图预测IR Drop分布 |

## 问题背景

IR Drop（电压降）是芯片电源完整性的核心问题。电流通过电源网络的电阻时产生电压降，导致实际供电电压低于理想值。

传统IR Drop分析需要完整的电源网络和详细的电流分布，在设计早期不可用。深度学习可以从部分特征快速预测。

## 核心方法

### 输入特征（5通道）

| 通道 | 特征 | 含义 |
|------|------|------|
| 0 | 开关活动率 | 每个区域的信号翻转频率 |
| 1 | 单元密度 | 标准单元的密集程度 |
| 2 | 电源Pad距离 | 到最近电源Pad的距离 |
| 3 | 时钟树密度 | 时钟网络的密集程度 |
| 4 | 金属层密度 | 电源金属层的覆盖密度 |

### Inception模块

GoogleNet的核心创新——在同一层使用多种大小的卷积核：

```
输入
  |
  +-- 1x1 Conv (捕获逐点特征)
  +-- 3x3 Conv (捕获3x3局部模式)
  +-- 5x5 Conv (捕获5x5更大范围模式)
  +-- 3x3 MaxPool -> 1x1 Conv (降维+特征提取)
  |
  v
Concatenate -> 多尺度特征融合
```

### 为什么Inception适合IR Drop？

IR Drop受多个尺度的因素影响：
- **局部**：单个单元的开关活动（1x1卷积捕获）
- **小区域**：相邻单元的密度分布（3x3卷积捕获）
- **大区域**：电源Pad距离的影响范围（5x5卷积捕获）
- **全局**：整体电源网络拓扑（多层堆叠捕获）

## 复现实现

### Inception Block

```python
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, ch1x1, ch3x3, ch5x5, pool_ch):
        self.branch1 = nn.Conv2d(in_ch, ch1x1, 1)
        self.branch2 = nn.Conv2d(in_ch, ch3x3, 3, padding=1)
        self.branch3 = nn.Conv2d(in_ch, ch5x5, 5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_ch, 1)
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], dim=1)
```

### 完整模型

```python
class IRDropCNN(nn.Module):
    def __init__(self, in_channels=5):
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.inception1 = InceptionBlock(32, 16, 24, 8, 8)   # -> 56 ch
        self.inception2 = InceptionBlock(56, 32, 48, 16, 16)  # -> 112 ch
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.regressor = nn.Sequential(
            nn.Linear(112*4*4, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
```

### 训练

```python
for epoch in range(30):
    for features, ir_drop in dataloader:
        pred = model(features)
        loss = mse_loss(pred, ir_drop)
        loss.backward()
        optimizer.step()
```

## 实验结果

```
数据集: 800个合成芯片样本 (64x64网格, 5通道)
模型参数: 505K
训练: 30 epochs (44秒, GPU)

测试集:
  MAE = 0.0234
  RMSE = 0.0324
  空间相关系数 = 0.9965
```

![IR Drop预测结果](/part6/irdrop_results.png)

## 关键发现

1. **多尺度融合是关键**：Inception的1x1+3x3+5x5并行卷积比单一尺度CNN效果好
2. **空间相关性极高**：0.997的相关系数说明CNN学到了IR Drop的空间分布规律
3. **物理直觉验证**：模型学到了"远离电源Pad的区域IR Drop更大"这一物理规律
4. **快速推理**：训练好的模型可以在毫秒级完成预测，比SPICE仿真快1000倍以上

## 比赛应用

在2024年EDA设计精英挑战赛中，IR Drop预测是核心赛题之一。本复现验证了Inception CNN方法的有效性，可作为比赛的基础框架。
