# 3.3 Placement相关算法

## Min-Cut最小割算法

### 基本思想

递归地将芯片划分为更小的区域，同时将单元分配到对应区域：

```
第1次分割: 将芯片分为2个区域
第2次分割: 将每个区域再分为2个 -> 4个区域
第3次分割: 4个 -> 8个区域
...
直到区域大小合适，进行详细布局
```

### 算法流程

```
输入: 网表G, 芯片区域R
输出: 每个单元的位置

1. 如果区域R足够小:
   使用穷举或简单启发式完成布局

2. 将区域R按某个方向（水平或垂直）分成两半

3. 使用FM算法将网表的单元分配到两个半区
   目标: 最小化跨区域的网线数

4. 递归处理两个子区域
```

## 基于模拟退火（SA）的全局布局

### 模拟退火算法

模拟退火是一种通用的随机优化算法，灵感来自金属退火过程：

```
1. 初始化: 随机布局, 温度T = T_max

2. 重复直到温度降至 T_min:
   a. 随机产生一个新的布局（交换两个单元的位置）
   b. 计算代价变化 delta_C = C_new - C_old
   c. 如果 delta_C < 0:
        接受新布局（更好的解）
   d. 否则:
        以概率 exp(-delta_C / T) 接受新布局
   e. 降低温度: T = alpha * T (0 < alpha < 1)

3. 返回找到的最佳布局
```

### 关键参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| T_max | 初始温度 | 使得接受率约80% |
| T_min | 终止温度 | 接近0 |
| alpha | 降温速率 | 0.95 ~ 0.99 |
| 内循环次数 | 每个温度的迭代次数 | 100 * n_cells |

### Python实现

```python
import random
import math

def simulated_annealing_placement(cells, nets, chip_w, chip_h):
    """模拟退火布局"""
    n = len(cells)
    
    # 随机初始布局
    positions = [(random.uniform(0, chip_w), random.uniform(0, chip_h)) 
                 for _ in range(n)]
    
    cost = compute_total_hpwl(positions, cells, nets)
    best_cost = cost
    best_pos = positions[:]
    
    T = 1000.0
    T_min = 0.01
    alpha = 0.995
    
    while T > T_min:
        for _ in range(n * 10):  # 内循环
            # 随机选择一个单元，移动到新位置
            idx = random.randint(0, n - 1)
            old_pos = positions[idx]
            new_pos = (random.uniform(0, chip_w), random.uniform(0, chip_h))
            
            # 计算代价变化
            positions[idx] = new_pos
            new_cost = compute_total_hpwl(positions, cells, nets)
            delta = new_cost - cost
            
            # Metropolis准则
            if delta < 0 or random.random() < math.exp(-delta / T):
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_pos = positions[:]
            else:
                positions[idx] = old_pos  # 恢复
        
        T *= alpha
    
    return best_pos, best_cost
```

## 基于解析法的布局方案

### 仿电场模型（Force-Directed）

将网线建模为弹簧，形成力学系统：

```
对于连接单元i和单元j的网线:
  弹力 F_ij = k * (pos_j - pos_i)
  其中k为弹簧常数（与线的权重相关）
```

全局力平衡方程：
```
C * x = bx
C * y = by

其中C是力常数矩阵（类似图的拉普拉斯矩阵）
```

### 密度约束的处理

纯力驱动的布局会导致单元聚集在芯片中心。需要引入密度惩罚：

```
目标函数 = 线长 + lambda * 密度惩罚

密度惩罚通常使用:
- 二次罚函数
- Bell-shaped函数（钟形分布）
- 非线性惩罚项
```

### 梯度下降求解

将布局问题转化为连续优化问题，使用梯度下降求解：

```
x_new = x_old - learning_rate * dCost/dx

其中:
  dCost/dx = dW/dx + lambda * dD/dx
  dW/dx = 线长的梯度
  dD/dx = 密度的梯度
```

## 求解方法对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| Min-Cut | 快速、层次化 | 分割质量影响最终结果 |
| 模拟退火 | 全局搜索能力强 | 慢，不适合大规模设计 |
| 解析法 | 速度快、可扩展 | 可能陷入局部最优 |
| 强化学习 | 可学习优化策略 | 训练成本高 |
