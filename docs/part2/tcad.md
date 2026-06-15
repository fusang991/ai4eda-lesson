# 2.3 TCAD仿真与器件建模

## TCAD仿真和器件建模的背景

### 什么是TCAD？

TCAD（Technology Computer-Aided Design）是利用计算机仿真来模拟半导体器件制造工艺和电学行为的技术。

### TCAD仿真的作用

- **工艺仿真**：模拟离子注入、氧化、刻蚀等工艺步骤
- **器件仿真**：模拟MOSFET等器件的I-V特性、电容特性
- **工艺优化**：在虚拟环境中优化工艺参数
- **缩短研发周期**：减少实际流片次数

### TCAD仿真的挑战

- 仿真速度极慢：一个完整的器件仿真可能需要数小时
- 参数空间大：工艺参数组合数量庞大
- 多物理场耦合：需要同时考虑电学、热学、力学效应

### 深度学习加速TCAD的思路

用神经网络学习TCAD仿真的输入-输出映射关系，替代耗时的物理仿真：

```
传统流程: 工艺参数 -> TCAD仿真(数小时) -> 电学特性
AI加速:   工艺参数 -> 神经网络推理(毫秒)  -> 电学特性
```

## 深度学习的梯度计算和Pytorch自动微分

### 自动微分（Automatic Differentiation）

PyTorch的核心特性之一是自动微分，可以自动计算任意计算图的梯度。

```python
import torch

# 创建需要求导的张量
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 定义计算
y = x[0]**2 + x[1]**3
z = y * 2

# 自动反向传播
z.backward()

# 获取梯度
print(x.grad)  # dz/dx = [4.0, 54.0]
```

### 计算图（Computational Graph）

PyTorch在前向传播时动态构建计算图：

```
x -> [平方] -> x^2
              \
               -> [加法] -> y -> [乘2] -> z
              /
x -> [立方] -> x^3
```

反向传播时沿着计算图反向计算梯度。

### 自动微分在EDA中的应用

在器件建模中，我们不仅需要预测电学特性，还需要计算特性对工艺参数的导数（灵敏度分析）。PyTorch的自动微分可以自动完成这一计算：

```python
# 工艺参数
params = torch.tensor([tox, vth_implant, ...], requires_grad=True)

# 神经网络预测
I_d = model(params)

# 自动计算 dI/dparams（跨导等特性）
I_d.backward()
sensitivity = params.grad
```

## 深度学习用于构造NMOS/PMOS器件模型

### 问题定义

给定工艺参数（如氧化层厚度、注入剂量等），预测NMOS/PMOS器件的电学特性（如I-V曲线）。

### 建模方法

使用前馈神经网络（FNN）建立工艺参数到电学特性的映射：

```
输入特征: [tox, na, nd, vgs, vds, ...]
    |
    v
FNN (多层全连接网络)
    |
    v
输出: [Ids, gm, Cgs, ...]
```

### 模型架构

```python
import torch
import torch.nn as nn

class DeviceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)
```

### 利用跨导概念的建模

跨导 gm = dIds/dVgs 是MOSFET的重要参数。利用PyTorch的自动微分：

```python
Vgs = torch.tensor([vgs_value], requires_grad=True)
other_params = torch.tensor([...])

# 预测 Ids
input_vec = torch.cat([Vgs, other_params])
Ids = model(input_vec)

# 自动计算跨导
Ids.backward()
gm = Vgs.grad  # dIds/dVgs
```

这样，一个模型同时预测I-V特性和跨导，无需额外的有限差分计算。
