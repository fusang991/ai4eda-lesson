# 3.2 Placement过程详解

## 芯片Placement的过程——从Global Placement到Detailed Placement

布局（Placement）是物理设计中最关键的步骤之一，决定了每个标准单元在芯片上的物理位置。

## 芯片布局的目标

### 1. 线长（Wirelength）

- 目标：最小化总连线长度
- 线长越短，信号传输延迟越小，功耗越低
- 主要评估模型：HPWL和RMST

### 2. 拥塞度（Congestion）

- 目标：避免布线资源不足的区域
- 拥塞度过高会导致布线失败
- 需要在布局阶段就考虑布线的可行性

### 3. PPA（Power, Performance, Area）

- **功耗（Power）**：缩短线长降低开关功耗
- **性能（Performance）**：关键路径延迟优化
- **面积（Area）**：芯片面积利用率

## 线长评估模型

### HPWL（Half-Perimeter Wirelength，半周线长）

最常用的线长评估模型：

```
HPWL(net) = (Xmax - Xmin) + (Ymax - Ymin)

其中：
  Xmax, Xmin = 网络中所有引脚的X坐标最大值和最小值
  Ymax, Ymin = 网络中所有引脚的Y坐标最大值和最小值
```

HPWL是Steiner最小树的一个上界估计，计算简单且效果好。

```python
def compute_hpwl(cells, nets):
    """计算总HPWL"""
    total_hpwl = 0
    for net in nets:
        x_coords = [cells[pin_id]['x'] for pin_id in net]
        y_coords = [cells[pin_id]['y'] for pin_id in net]
        hpwl = (max(x_coords) - min(x_coords)) + \
               (max(y_coords) - min(y_coords))
        total_hpwl += hpwl
    return total_hpwl
```

### RMST（Rectilinear Minimum Spanning Tree，最小生成树）

基于曼哈顿距离的最小生成树：

```
距离(cell_i, cell_j) = |x_i - x_j| + |y_i - y_j|
```

RMST比HPWL更精确，但计算更复杂。

## 时序评估模型

### Elmore延迟模型

RC树结构中，从源到汇的延迟估算：

```
延迟(sink_i) = sum(R_j * C_j)  对于路径上的每一段

其中：
  R_j = 第j段的电阻
  C_j = 第j段的电容（包括扇出负载）
```

### 查找表（LUT, Look-Up Table）

使用预计算的查找表快速评估延迟：

```
输入: 驱动强度、负载电容、线长
输出: 延迟值
```

标准单元库（Library）中包含每个单元在不同输入斜率和输出负载下的延迟数据。

## Global Placement——粗粒度全局布局

### 目标

将所有标准单元放置到芯片上，使得：
- 总线长最小
- 单元分布均匀（密度约束）
- 满足拥塞度要求

### 方法

1. **基于力的方法（Force-Directed）**：
   - 将网线建模为弹簧，单元间的连接产生"吸引力"
   - 添加"斥力"防止单元重叠
   - 求解力的平衡方程

2. **基于解析的方法（Analytical）**：
   - 将布局问题建模为连续优化问题
   - 目标函数：线长 + 密度惩罚
   - 使用梯度下降或共轭梯度求解

### 数学建模

```
minimize:   W(x,y) + lambda * D(x,y)

其中:
  W(x,y) = 线长目标函数（通常是HPWL的平滑近似）
  D(x,y) = 密度惩罚函数（防止单元重叠）
  lambda  = 权重系数
```

## Detailed Placement——细粒度详细布局

### 目标

在Global Placement的结果基础上，进行局部优化：
- 消除单元重叠
- 优化关键路径时序
- 满足所有设计规则

### 方法

1. **交换（Swap）**：交换两个单元的位置
2. **移动（Move）**：将单元移动到相邻空位
3. **链式移动（Chain Move）**：沿链路依次移动多个单元
4. **窗口优化（Window-based）**：在局部窗口内重新布局
