# 6.5 RouteNet: CNN拥塞预测

## 论文信息

| 项目 | 内容 |
|------|------|
| 标题 | Deep Learning for Routability Prediction |
| 会议 | ICCAD 2018 |
| 核心思想 | 用CNN从布局特征预测布线拥塞图 |

## 问题背景

详细布线是物理设计中最耗时的步骤之一。如果在布局阶段就能预测哪些区域会出现拥塞，就可以提前优化布局，避免布线失败。

## 核心方法

### 输入特征

将芯片版图划分为2D网格，每个格子提取3个特征通道：

| 通道 | 特征 | 含义 |
|------|------|------|
| 0 | 单元密度 | 该区域标准单元的密集程度 |
| 1 | 引脚密度 | 该区域引脚的数量 |
| 2 | 网线密度 | 该区域经过的网线数量 |

### 网络架构

使用U-Net风格的编码器-解码器CNN：

```
输入(3, 64, 64)
  |
  v
编码器: Conv32 -> Conv64 -> Conv128 (每层MaxPool)
  |
  v
瓶颈层: Conv256
  |
  v
解码器: UpConv128 -> UpConv64 -> UpConv32 (跳跃连接)
  |
  v
输出(1, 64, 64) = 拥塞图
```

### 跳跃连接的作用

U-Net的跳跃连接将编码器的特征直接传递给解码器，保留了细粒度的空间位置信息。这对拥塞预测非常重要——拥塞是高度位置敏感的。

## 复现实现

### 数据生成

```python
def generate_sample():
    # 通道0: 单元密度 (高斯团簇)
    cell_density = gaussian_blob_clusters(n_clusters=5)
    
    # 通道1: 引脚密度 (与单元密度相关 + 边界)
    pin_density = cell_density * 0.7 + boundary_emphasis * 0.3
    
    # 通道2: 网线密度 (线模式)
    net_density = line_patterns(n_lines=20)
    
    # 标签: 拥塞 = 网线密度 * (1 + cell_density的梯度)
    congestion = net_density * (1 + gradient(cell_density))
    
    return features, congestion
```

### 模型

```python
class RouteNet(nn.Module):
    """U-Net for congestion prediction"""
    
    def __init__(self):
        # 编码器
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        
        # 瓶颈
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        
        # 解码器 (带跳跃连接)
        self.dec3 = nn.Sequential(nn.Conv2d(256+128, 128, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(128+64, 64, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(64+32, 32, 3, padding=1), nn.ReLU())
        
        self.output = nn.Conv2d(32, 1, 1)
```

## 实验结果

```
数据集: 500个合成布局样本 (64x64网格)
模型参数: ~1.9M
训练: 60 epochs, Adam, L1Loss

测试集:
  MAE = 0.0117
  Pearson相关系数 = 0.9981
```

![RouteNet拥塞预测结果](/part6/routenet_results.png)

## 关键发现

1. **U-Net效果极好**：0.998的相关系数说明CNN几乎完美学到了拥塞分布模式
2. **跳跃连接是关键**：没有跳跃连接的简单CNN效果差很多（空间信息丢失）
3. **多通道输入**：同时使用密度、引脚、网线三种特征比单一特征效果好
4. **实际应用价值**：在布局阶段就能识别拥塞热点，指导单元移动

## 实际应用

- 布局质量评估：在详细布线前快速评估布局质量
- 迭代优化：将拥塞预测结果反馈给布局引擎进行迭代优化
- 设计空间探索：快速比较不同布局方案的可布线性
