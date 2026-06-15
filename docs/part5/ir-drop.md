# 5.4 IR Drop预测

## 芯片电源完整性和IR Drop简介

### 什么是IR Drop？

IR Drop是指电流通过电源网络的电阻时产生的电压降：

```
V_actual = V_ideal - I * R

其中:
  V_ideal = 理想供电电压 (如 0.8V)
  I = 电流
  R = 电源网络电阻
```

### IR Drop的影响

- **时序违例**：电压降低导致门延迟增加
- **功能错误**：严重IR Drop可能导致逻辑错误
- **可靠性问题**：长期在低电压下工作影响芯片寿命

### IR Drop的类型

```
1. 静态IR Drop (Static IR Drop):
   - 由平均电流引起的持续电压降
   - 主要影响电源网络设计

2. 动态IR Drop (Dynamic IR Drop):
   - 由瞬时大量开关活动引起的电压波动
   - 更难预测和修复
```

### 电源网络结构

```
VDD Pad ─────┬─────┬─────┬─────┬───── VDD Pad
              |     |     |     |
            ──┴── ──┴── ──┴── ──┴──  Metal 4 (垂直)
              |     |     |     |
            ────────────────────────  Metal 3 (水平)
              |     |     |     |
              FF    FF    FF    FF   寄存器/标准单元
              |     |     |     |
            ────────────────────────  Metal 3 (水平)
              |     |     |     |
            ──┴── ──┴── ──┴── ──┴──  Metal 4 (垂直)
              |     |     |     |
VSS Pad ─────┴─────┴─────┴─────┴───── VSS Pad
```

## IR Drop预测相关的深度学习方法

### 问题定义

给定芯片版图特征，预测每个位置的IR Drop值，以便在物理设计早期发现问题。

### 为什么需要深度学习？

传统IR Drop分析需要：
- 完整的电源网络
- 详细的电流分布
- SPICE级别的仿真

这些信息在设计早期不可用，且仿真耗时。深度学习可以从部分特征快速预测IR Drop。

## 深度卷积神经网络应用于电压降预测

### 方法

将芯片版图划分为网格，每个网格单元作为一个"像素"，使用CNN预测每个网格的IR Drop：

```
输入特征图 (每个网格单元):
  - 单元密度
  - 切换活动率 (Switching Activity)
  - 距离最近电源Pad的距离
  - 时钟树密度
  - 金属层密度

输出:
  - 该区域的IR Drop值
```

### GoogleNet和Inception模块

#### Inception模块的核心思想

在同一个层中使用多种不同大小的卷积核，然后将结果拼接：

```
输入
  |
  +-- 1x1 Conv ──────────────────────┐
  +-- 1x1 Conv -> 3x3 Conv ─────────┤
  +-- 1x1 Conv -> 5x5 Conv ─────────┤ -> Concat -> 输出
  +-- 3x3 MaxPool -> 1x1 Conv ──────┘
```

#### 为什么Inception有效？

- **多尺度特征**：不同大小的卷积核捕获不同尺度的特征
- **稀疏连接**：1x1卷积降低通道数，减少计算量
- **信息融合**：拼接多种特征表示

### IR Drop预测模型

```python
import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, ch1x1, 1)
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3x3_reduce, 1),
            nn.Conv2d(ch3x3_reduce, ch3x3, 3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5x5_reduce, 1),
            nn.Conv2d(ch5x5_reduce, ch5x5, 5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, 1)
        )
    
    def forward(self, x):
        b1 = torch.relu(self.branch1(x))
        b2 = torch.relu(self.branch2(x))
        b3 = torch.relu(self.branch3(x))
        b4 = torch.relu(self.branch4(x))
        return torch.cat([b1, b2, b3, b4], dim=1)


class IRDropPredictor(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            InceptionBlock(64, 32, 48, 64, 8, 16, 16),
            InceptionBlock(128, 64, 64, 96, 16, 32, 32),
            nn.MaxPool2d(2),
            InceptionBlock(256, 96, 80, 128, 16, 48, 48),
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.regressor(features)

# 使用示例
model = IRDropPredictor(in_channels=5)
# 输入: [batch, 5, H, W] (5个特征通道)
# 输出: [batch, 1] (IR Drop预测值)
dummy_input = torch.randn(4, 5, 64, 64)
output = model(dummy_input)
print(f"Predicted IR Drop: {output.shape}")
```
