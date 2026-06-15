# 5.6 Transformer模型

## Attention is All You Need——Transformer模型

### 背景

Transformer由Vaswani等人在2017年提出，最初用于机器翻译，后来成为NLP和CV领域的主流架构。

### 核心创新：自注意力机制（Self-Attention）

自注意力机制允许序列中的每个位置关注所有其他位置：

```
输入序列: [x1, x2, ..., xn]

对于每个位置i:
  注意力权重: a_ij = softmax(Q_i * K_j^T / sqrt(d_k))
  输出: y_i = sum(a_ij * V_j)
```

### Q, K, V的计算

```
Q = X * W_Q  (查询矩阵)
K = X * W_K  (键矩阵)
V = X * W_V  (值矩阵)

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

### 多头注意力（Multi-Head Attention）

使用多个注意力头，每个头学习不同的注意力模式：

```
head_i = Attention(Q * W_Qi, K * W_Ki, V * W_Vi)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
```

### Transformer架构

```
输入嵌入 + 位置编码
    |
    v
┌────────────────────────────┐
│ 编码器 (Encoder) x N层:    │
│   多头自注意力              │
│   Add & LayerNorm          │
│   前馈网络 (FFN)           │
│   Add & LayerNorm          │
└────────────────────────────┘
    |
    v
┌────────────────────────────┐
│ 解码器 (Decoder) x N层:    │
│   掩码多头自注意力          │
│   Add & LayerNorm          │
│   交叉注意力 (关注编码器)   │
│   Add & LayerNorm          │
│   前馈网络 (FFN)           │
│   Add & LayerNorm          │
└────────────────────────────┘
    |
    v
输出层
```

### 前馈网络（FFN）

```
FFN(x) = max(0, x * W1 + b1) * W2 + b2

中间维度通常是输入维度的4倍
```

### 位置编码（Positional Encoding）

由于注意力机制没有位置信息，需要显式添加：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Transformer在AI模型中的应用

### 在NLP中的应用

- **BERT**：双向编码器，用于文本理解
- **GPT**：自回归解码器，用于文本生成
- **T5**：编码器-解码器，统一的文本到文本框架

### 在CV中的应用

- **ViT（Vision Transformer）**：将图像分割为patch，用Transformer处理
- **DETR**：Transformer用于目标检测
- **Swin Transformer**：分层的视觉Transformer

### 在EDA中的潜在应用

| 应用 | 说明 |
|------|------|
| 网表编码 | 用Transformer编码电路拓扑 |
| 时序预测 | 序列化的路径分析 |
| 布局优化 | 序列决策问题 |
| 代码生成 | 自动生成RTL代码 |

### PyTorch实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并分头
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=6, d_ff=1024):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

## Transformer vs GNN在EDA中的对比

| 特性 | Transformer | GNN |
|------|-------------|-----|
| 数据结构 | 序列 | 图 |
| 位置编码 | 需要显式编码 | 图结构天然编码 |
| 全局信息 | 自注意力全局建模 | 需要多层消息传递 |
| 计算复杂度 | O(n^2) | O(E) |
| 可扩展性 | 序列长度受限 | 图大小受限 |
| EDA应用 | 序列化电路表示 | 原生电路图表示 |
