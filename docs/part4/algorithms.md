# 4.3 CTS相关算法

## 寄存器分组和聚类

### K-means聚类

将寄存器按物理位置分为K组：

```
1. 初始化: 随机选择K个中心点
2. 分配: 将每个寄存器分配到最近的中心点
3. 更新: 重新计算每组的中心点
4. 重复2-3直到收敛
```

### 层次聚类

自底向上或自顶向下的聚类方法：

```
自底向上（凝聚）:
  1. 每个寄存器初始为一个单独的组
  2. 合并距离最近的两个组
  3. 重复直到达到目标组数

自顶向下（分裂）:
  1. 所有寄存器初始为一个组
  2. 将最大的组一分为二
  3. 重复直到达到目标组数
```

## 延迟平衡：DME算法

### DME（Deferred-Merge Embedding）算法

DME是CTS中最经典的算法，用于构建零偏斜或有界偏斜的时钟树。

### 核心思想

```
关键观察: 不需要立即确定Buffer的精确位置，
而是延迟到必要时再决定。

DME的两阶段:
  阶段1（Bottom-Up）: 从叶节点向上合并，确定"合并区域"
  阶段2（Top-Down）: 从根节点向下，在合并区域内选择精确位置
```

### 算法流程

```
阶段1 (Bottom-Up):
  对于每对相邻的叶节点 (i, j):
    1. 计算零偏斜合并点 m_ij
       （使得到i和j的延迟相等的点）
    2. 记录合并区域 R_ij = {m_ij}

  对于每对相邻的子树 (T1, T2):
    1. 计算合并点集合（考虑子树的延迟范围）
    2. 合并区域 = 两个子树合并区域的交集

阶段2 (Top-Down):
  从根开始:
    1. 在根的合并区域内选择一个位置
    2. 递归处理子树
    3. 在子树的合并区域内选择最近的位置
```

### Python实现（简化版）

```python
import numpy as np

class DME:
    def __init__(self, sinks):
        """sinks: [(x, y), ...] 叶节点坐标"""
        self.sinks = np.array(sinks)
        self.n = len(sinks)
    
    def zero_skew_merge(self, p1, d1, p2, d2):
        """
        计算零偏斜合并点
        p1, p2: 两个子树的位置
        d1, d2: 两个子树的延迟
        返回: 合并点位置和合并后的延迟
        """
        diff = d1 - d2
        dist = np.linalg.norm(p1 - p2)
        
        # 零偏斜条件: d1 + dist_left = d2 + dist_right
        # dist_left + dist_right = dist
        if d1 >= d2:
            dist_left = (dist - diff) / 2
            dist_right = (dist + diff) / 2
        else:
            dist_left = (dist + diff) / 2
            dist_right = (dist - diff) / 2
        
        # 合并点位置
        t = dist_left / dist
        merge_point = p1 + t * (p2 - p1)
        merge_delay = (d1 + d2 + dist) / 2
        
        return merge_point, merge_delay
    
    def build_tree(self):
        """构建零偏斜时钟树"""
        # 初始化：每个sink是一个子树
        subtrees = [{'pos': self.sinks[i], 'delay': 0, 'children': []} 
                    for i in range(self.n)]
        
        while len(subtrees) > 1:
            # 找最近的两个子树
            min_dist = float('inf')
            best_pair = (0, 1)
            
            for i in range(len(subtrees)):
                for j in range(i + 1, len(subtrees)):
                    d = np.linalg.norm(subtrees[i]['pos'] - subtrees[j]['pos'])
                    if d < min_dist:
                        min_dist = d
                        best_pair = (i, j)
            
            i, j = best_pair
            # 零偏斜合并
            merge_pos, merge_delay = self.zero_skew_merge(
                subtrees[i]['pos'], subtrees[i]['delay'],
                subtrees[j]['pos'], subtrees[j]['delay']
            )
            
            new_subtree = {
                'pos': merge_pos,
                'delay': merge_delay,
                'children': [subtrees[i], subtrees[j]]
            }
            
            # 移除旧子树，添加新子树
            new_subtrees = [s for k, s in enumerate(subtrees) if k != i and k != j]
            new_subtrees.append(new_subtree)
            subtrees = new_subtrees
        
        return subtrees[0]
```

## 时钟布线：Steiner树和FLUTE

### Steiner最小树（SMT）

在欧几里得平面上，连接一组点的最短网络：

```
给定: 一组终端点 {P1, P2, ..., Pn}
求: 一组额外的Steiner点 {S1, S2, ..., Sk}
    使得所有点通过线段连接的总长度最小
```

### FLUTE算法

FLUTE（Fast Lookup Table Based Wirelength Estimation）是一种快速的Steiner树近似算法：

```
1. 使用预计算的查找表
2. 对于给定的终端点，直接查表得到近似Steiner树
3. 时间复杂度: O(n log n)
4. 对于小规模网络（< 9个终端），精确度很高
```

### FLUTE的应用

- CTS中用于估计时钟线长
- 快速评估时钟延迟
- 布局阶段的线长估计

## 零偏斜树、有界偏斜树和有用偏斜

### 零偏斜树（ZST, Zero Skew Tree）

所有叶节点的时钟到达时间完全相同：

```
到达时间(FF1) = 到达时间(FF2) = ... = 到达时间(FFn)
```

- 优点：最严格的时序保证
- 缺点：可能需要更长的线长、更多的Buffer

### 有界偏斜树（BST, Bounded Skew Tree）

允许叶节点到达时间有一定差异：

```
|到达时间(FFi) - 到达时间(FFj)| < SkewBound
```

- 优点：更灵活，可能减少线长和功耗
- 缺点：需要更复杂的优化

### 有用偏斜（Useful Skew）

故意利用正偏斜来改善时序：

```
场景: 发送端FF1比接收端FF2早到达时钟

Setup时间分析:
  有效时钟周期 = T + skew
  给逻辑路径更多的时间
```

有用偏斜可以：
- 放松关键路径的Setup约束
- 提高芯片的最高工作频率
