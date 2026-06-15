# 4.2 CTS过程详解

## 芯片CTS时钟树综合的过程——从Cluster到Balance

时钟树综合分为两个主要阶段：Cluster（聚类）和Balance（平衡）。

## CTS的目标

### 1. Clock Latency（时钟延迟）

```
目标: 最小化从时钟源到寄存器的最大延迟
方法: 选择合适的Buffer/Inverter级数和尺寸
```

### 2. Clock Skew（时钟偏斜）

```
目标: 最大延迟 - 最小延迟 < 阈值（通常 < 50ps）
方法: DME算法、延迟平衡
```

### 3. DRC（设计规则检查）

```
- 最大转换时间（Max Transition）: 信号上升/下降时间限制
- 最大负载电容（Max Capacitance）: 每个节点的最大电容
- 最大扇出（Max Fanout）: 每个Buffer的最大驱动数
```

### 4. Timing（时序）

```
- 满足建立时间约束
- 满足保持时间约束
- 考虑OCV（On-Chip Variation）的影响
```

## Cluster过程——寄存器聚类

### 寄存器分组

将大量寄存器分成若干组（Cluster），每组内的寄存器在物理位置上相近。

#### K-means聚类

```python
from sklearn.cluster import KMeans
import numpy as np

# 获取所有寄存器的坐标
positions = np.array([[ff.x, ff.y] for ff in flip_flops])

# K-means聚类
n_clusters = len(positions) // 10  # 每组约10个寄存器
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(positions)

# 得到每组的中心
centers = kmeans.cluster_centers_
```

#### 层次聚类（Hierarchical Clustering）

```python
from scipy.cluster.hierarchy import linkage, fcluster

# 计算层次聚类
Z = linkage(positions, method='ward')

# 切割树得到聚类
labels = fcluster(Z, t=n_clusters, criterion='maxclust')
```

### DRC处理

在聚类过程中需要考虑DRC约束：

```
对于每个聚类:
  1. 检查聚类内的最大距离（影响延迟差）
  2. 确保每个Buffer不超过最大扇出
  3. 控制每段连线的负载电容
```

### 时钟树构造

在聚类的基础上构建时钟树：

```
时钟源
  |
  +-- Buffer_1 (驱动Cluster 1)
  |     +-- Buffer_1a (驱动子Cluster 1a)
  |     |     +-- FF1, FF2, FF3
  |     +-- Buffer_1b (驱动子Cluster 1b)
  |           +-- FF4, FF5
  |
  +-- Buffer_2 (驱动Cluster 2)
        +-- ...
```

## Balance过程——时钟树打平

### 时钟树打平的目标

确保从时钟源到所有叶节点（寄存器）的延迟相等。

### 延迟平衡方法

#### 1. 插入Buffer/Inverter

在延迟较小的路径上插入额外的Buffer增加延迟：

```
路径A: 源 -> BUF -> FF1  (延迟 = 20ps)
路径B: 源 -> BUF -> BUF -> FF2  (延迟 = 40ps)

插入Buffer后:
路径A: 源 -> BUF -> BUF -> FF1  (延迟 = 40ps)
路径B: 源 -> BUF -> BUF -> FF2  (延迟 = 40ps)
```

#### 2. 调整Buffer尺寸

使用不同尺寸的Buffer（不同驱动强度）调整延迟：

```
大Buffer: 延迟小，驱动能力强
小Buffer: 延迟大，驱动能力弱
```

#### 3. 调整连线长度

通过增加弯曲（Detour）增加路径延迟。

### 结构优化

平衡后进行结构优化：

```
1. 冗余Buffer移除: 删除不必要的Buffer
2. Buffer尺寸优化: 使用最合适的Buffer尺寸
3. 拓扑优化: 调整树的拓扑结构减少延迟
```

## CTS流程总结

```
输入: 放置后的网表（包含寄存器位置）
    |
    v
1. 寄存器聚类（K-means / 层次聚类）
    |
    v
2. 构建初始时钟树拓扑
    |
    v
3. 插入Buffer/Inverter
    |
    v
4. 延迟平衡（DME算法）
    |
    v
5. DRC检查和修复
    |
    v
6. 时序优化
    |
    v
输出: 时钟树网络（含Buffer/Inverter）
```
