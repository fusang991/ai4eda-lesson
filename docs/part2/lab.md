# 2.4 上机实验

## 实验一：PyTorch深度学习框架

### PyTorch安装

```bash
# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CPU版本（无GPU）
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### PyTorch基本功能

```python
import torch

# 创建张量
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.randn(3, 4)        # 随机矩阵
z = torch.zeros(2, 3)         # 零矩阵
w = torch.ones(3, 3)          # 全1矩阵

# GPU操作
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_gpu = x.to(device)

# 基本运算
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a + b)        # 逐元素加法
print(a * b)        # 逐元素乘法
print(torch.dot(a, b))  # 点积
print(a @ b)        # 矩阵乘法（同dot）
```

### PyTorch vs NumPy

```python
import numpy as np
import torch

# NumPy
np_a = np.array([[1, 2], [3, 4]])
np_b = np.array([[5, 6], [7, 8]])
np_c = np_a @ np_b  # 矩阵乘法

# PyTorch
pt_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
pt_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
pt_c = pt_a @ pt_b  # 矩阵乘法

# 互转
np_to_pt = torch.from_numpy(np_a)
pt_to_np = pt_a.numpy()
```

### 使用PyTorch搭建简单神经网络

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型和优化器
model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device).view(-1, 784)
        batch_y = batch_y.to(device)
        
        # 前向传播
        output = model(batch_x)
        loss = criterion(output, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 实验二：ResNet用于手写数字识别

### 数据集准备

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(32),           # MNIST原始28x28，ResNet需要更大输入
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)
```

### 简化版ResNet实现

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        return torch.relu(out)

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)
```

### 训练和评估

```python
model = ResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_x, batch_y in trainloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 测试
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in testloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            pred = output.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
    
    acc = correct / len(testset) * 100
    print(f"Epoch {epoch}: Loss={total_loss/len(trainloader):.4f}, Acc={acc:.2f}%")
```

## 实验三：EDA设计精英挑战赛——TCAD器件建模

### 赛题背景

2022年集成电路EDA设计精英挑战赛的赛题之一：使用深度学习对NMOS/PMOS器件的电学性质进行建模，替代耗时的TCAD仿真。

### 基于FNN的器件建模

```python
class TCADModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# 训练
model = TCADModel(input_dim=5).to(device)  # [tox, Vgs, Vds, Na, Nd]
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(200):
    model.train()
    for batch_x, batch_y in trainloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
```

### 跨导计算（自动微分应用）

```python
# 计算跨导 gm = dIds/dVgs
Vgs = test_data[:, 1].clone().requires_grad_(True)
input_data = test_data.clone()
input_data[:, 1] = Vgs

Ids = model(input_data)
Ids.sum().backward()
gm = Vgs.grad  # 自动计算的跨导
```
