# 1.4 超图分割算法

## 超图分割在EDA中的应用

超图分割（Hypergraph Partitioning）是EDA中的基础算法，广泛应用于：

- **网表切分（Netlist Partitioning）**：将大规模电路划分为多个子模块
- **布局规划（Floorplanning）**：确定模块的相对位置
- **并行仿真**：将设计分割后进行并行处理
- **时钟域划分**：按功能域划分设计

### 图 vs 超图

- **普通图**：一条边只连接两个节点
- **超图**：一条超边（Hyperedge）可以连接多个节点

在电路网表中，一个信号网络（Net）通常连接多个引脚，因此用超图建模更自然。

## Kernighan-Lin (KL) 算法

### 基本思想

KL算法是一种迭代改进的图分割算法，核心思想是：

1. 将图随机分成两部分（A和B）
2. 交换A和B中的节点对，找到使割边减少最多的交换
3. 重复直到无法改进

### 算法流程

```
输入: 图G=(V,E)，初始分割(A,B)
输出: 改进后的分割(A',B')

1. 计算每个节点的外部代价 D[i] = E[i] - I[i]
   其中 E[i] = 与i相连的在另一部分的边数
        I[i] = 与i相连的在同一部分的边数

2. 对于每一对 (a属于A, b属于B):
   计算交换收益 gain(a,b) = D[a] + D[b] - 2*connect(a,b)

3. 选择收益最大的对进行交换

4. 重复直到所有节点都被交换过

5. 从所有交换步骤中选择累积收益最大的状态
```

### Python实现

```python
def kl_partition(graph, n_nodes):
    """Kernighan-Lin图分割算法"""
    # 初始化随机分割
    half = n_nodes // 2
    A = set(range(half))
    B = set(range(half, n_nodes))
    
    best_cut = count_cut_edges(graph, A, B)
    
    for iteration in range(100):
        # 计算D值
        D = {}
        for node in range(n_nodes):
            if node in A:
                external = sum(1 for n in graph[node] if n in B)
                internal = sum(1 for n in graph[node] if n in A)
            else:
                external = sum(1 for n in graph[node] if n in A)
                internal = sum(1 for n in graph[node] if n in B)
            D[node] = external - internal
        
        # 寻找最佳交换对
        best_gain = 0
        best_pair = None
        for a in A:
            for b in B:
                gain = D[a] + D[b]
                if (a, b) in graph or (b, a) in graph:
                    gain -= 2
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
        
        if best_pair is None or best_gain <= 0:
            break
        
        # 执行交换
        a, b = best_pair
        A.remove(a)
        B.remove(b)
        A.add(b)
        B.add(a)
    
    return A, B
```

## Fiduccia-Mattheyses (FM) 算法

### FM算法的改进

FM算法是KL算法的重要改进：

1. **单节点移动**：每次只移动一个节点（而非交换一对）
2. **桶结构（Bucket Structure）**：用桶数据结构加速增益查找
3. **支持超图**：直接处理超图而非普通图
4. **平衡约束**：可控制分割的平衡性

### 算法流程

```
输入: 超图H=(V,Net)，初始分割(A,B)，平衡因子bal
输出: 改进后的分割(A',B')

1. 初始化:
   - 计算每个节点的初始增益 gain[v]
   - 将节点按增益放入桶结构
   - 标记所有节点为"未锁定"

2. 循环 |V| 次:
   a. 从桶中选择增益最高的可移动节点 v
      （需满足平衡约束）
   b. 移动 v 到另一部分
   c. 锁定 v（本pass不可再移动）
   d. 更新邻居节点的增益
   e. 记录累积增益

3. 选择累积增益最大的前缀状态
   作为本次pass的结果

4. 重复上述pass直到无法改进
```

### FM算法的关键数据结构

```python
class BucketStructure:
    """FM算法的桶结构"""
    def __init__(self, max_gain):
        self.buckets = [[] for _ in range(2 * max_gain + 1)]
        self.offset = max_gain

    def insert(self, node, gain):
        self.buckets[gain + self.offset].append(node)

    def remove(self, node, gain):
        self.buckets[gain + self.offset].remove(node)

    def get_max_gain_node(self):
        for i in range(len(self.buckets) - 1, -1, -1):
            if self.buckets[i]:
                return self.buckets[i][-1], i - self.offset
        return None, None
```

## KL与FM算法对比

| 特性 | KL算法 | FM算法 |
|------|--------|--------|
| 移动方式 | 交换节点对 | 单节点移动 |
| 数据结构 | 简单 | 桶结构 |
| 超图支持 | 不直接支持 | 原生支持 |
| 平衡约束 | 无 | 有 |
| 时间复杂度 | O(n^2 log n) | O(n) per pass |
| 实际效率 | 较低 | 高效 |
