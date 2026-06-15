# 2.2 深度学习发展历程

## 神经网络的发展历程：从全连接到深度学习大模型

### 第一阶段：早期探索（1943-1986）

- **1943年**：McCulloch-Pitts神经元模型
- **1958年**：Rosenblatt感知器
- **1986年**：Rumelhart等人提出反向传播算法

### 第二阶段：寒冬与突破（1990-2011）

- **1990s**：支持向量机（SVM）兴起，神经网络研究遇冷
- **1998年**：LeCun提出LeNet，CNN用于手写数字识别
- **2006年**：Hinton提出深度信念网络（DBN），深度学习复兴

### 第三阶段：深度学习革命（2012-至今）

- **2012年**：AlexNet赢得ImageNet竞赛，深度学习爆发
- **2014年**：GAN（生成对抗网络）提出
- **2015年**：ResNet提出，解决了深层网络训练问题
- **2017年**：Transformer提出（Attention Is All You Need）
- **2018年**：BERT、GPT预训练语言模型
- **2020年**：GPT-3，大语言模型时代
- **2022年**：ChatGPT，AI进入大众视野

## 神经网络的常见分类

### FNN（前馈神经网络，Feedforward Neural Network）

最基础的网络结构，数据单向流动：

```
输入层 -> 隐藏层1 -> 隐藏层2 -> ... -> 输出层
```

- 适用于结构化数据、简单映射关系
- 在EDA中用于器件建模、参数预测

### CNN（卷积神经网络，Convolutional Neural Network）

通过卷积核提取局部特征，擅长处理网格状数据：

```
输入 -> [卷积层 -> 激活 -> 池化层] x N -> 全连接层 -> 输出
```

- 适用于图像处理、2D数据
- 在EDA中用于IR Drop预测、版图分析

### RNN（循环神经网络，Recurrent Neural Network）

具有记忆能力，擅长处理序列数据：

```
x[t] -> 隐藏状态 h[t] -> 输出 y[t]
         |
         +-- h[t-1]（上一时刻的隐藏状态）
```

- 适用于时间序列、自然语言
- 变体：LSTM、GRU解决长期依赖问题

### GNN（图神经网络，Graph Neural Network）

在图结构数据上进行消息传递和特征聚合：

```
对于每个节点 v:
  1. 收集邻居节点特征
  2. 聚合邻居信息
  3. 更新自身特征
```

- 适用于图结构数据
- **在EDA中极为重要**：电路天然具有图结构（网表）

## ResNet残差神经网络

### 残差连接（Skip Connection）

ResNet的核心创新是残差连接，解决了深层网络的退化问题：

```
普通网络: y = F(x)
ResNet:   y = F(x) + x    （跳跃连接）
```

其中 F(x) 是需要学习的残差映射。

### 为什么残差连接有效？

- 如果某层不需要变换，只需学习 F(x) = 0
- 梯度可以通过跳跃连接直接传播，缓解梯度消失
- 使得训练数百层的网络成为可能

### ResNet架构

```
输入 -> [Conv -> BN -> ReLU -> Conv -> BN + 跳跃连接 -> ReLU] x N -> 全局池化 -> FC -> 输出
```

常见变体：ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152

## 常用的深度学习库——Pytorch简介

### PyTorch的核心特点

1. **动态计算图**：运行时构建计算图，调试方便
2. **Pythonic设计**：与NumPy类似的API，学习成本低
3. **强大的GPU加速**：无缝GPU计算
4. **丰富的生态系统**：torchvision, torchaudio, torchtext
5. **工业部署支持**：TorchScript, ONNX

### PyTorch vs TensorFlow

| 特性 | PyTorch | TensorFlow |
|------|---------|------------|
| 计算图 | 动态 | 静态（TF2支持动态） |
| 调试 | 简单（Python原生） | 较复杂 |
| 社区 | 学术界主流 | 工业界主流 |
| 部署 | TorchScript | TF Serving |
| API设计 | Pythonic | 自有风格 |

**在EDA和学术研究中，PyTorch是首选框架。**
