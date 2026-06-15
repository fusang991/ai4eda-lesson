# 5.2 Routing相关算法

## Maze-Routing Algorithm——迷宫算法

### 基本思想

将布线区域建模为网格图，使用BFS（广度优先搜索）寻找从源到目标的最短路径。

### 算法流程

```
输入: 网格图G, 源点S, 目标点T
输出: 从S到T的最短路径

1. 初始化: 将网格中所有格子标记为"未访问"
2. 将源点S加入队列，标记为"已访问"
3. BFS搜索:
   while 队列不空:
     取出当前格子P
     if P == T:
       回溯路径并返回
     for 每个相邻格子N:
       if N未被访问且N无障碍:
         标记N为已访问
         记录N的前驱为P
         将N加入队列
4. 如果队列为空，搜索失败（无可行路径）
```

### Python实现

```python
from collections import deque

def maze_routing(grid, start, end):
    """
    迷宫布线算法
    grid: 2D网格, 0=可通过, 1=障碍
    start: (row, col) 源点
    end: (row, col) 目标点
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        r, c = queue.popleft()
        
        if (r, c) == end:
            # 回溯路径
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in visited and grid[nr][nc] == 0:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
    
    return None  # 无可行路径

# 示例
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
path = maze_routing(grid, (0, 0), (4, 4))
print("Path:", path)
```

## 基于A*搜索的PathFinder算法

### A*算法

A*是Dijkstra算法的启发式版本，使用估价函数引导搜索：

```
f(n) = g(n) + h(n)

g(n) = 从起点到n的实际代价
h(n) = 从n到终点的启发式估计（如曼哈顿距离）
```

### PathFinder算法

PathFinder是VPR（Versatile Place and Route）中的经典布线算法：

```
核心思想: 允许布线资源被多个网络共享（rip-up and reroute）

迭代过程:
  for 每次迭代:
    for 每个网络:
      1. 使用A*搜索找到最优路径
      2. 如果路径与其他网络冲突:
         允许暂时共享，但增加共享代价
    if 没有冲突:
      布线完成
    else:
      增加共享代价系数，继续下一次迭代
```

### PathFinder的关键公式

```
代价函数: cost(n) = base_cost(n) + history(n) + present(n)

base_cost(n) = 节点n的基本布线代价
history(n) = 历史共享惩罚（累积）
present(n) = 当前迭代的共享惩罚

present(n) = (1 + occupancy(n) / capacity(n)) * sharing_penalty
```

### Python实现

```python
import heapq

def astar_routing(grid, start, end, occupancy):
    """A*布线算法"""
    rows, cols = len(grid), len(grid[0])
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_set = [(0, start)]
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    parent = {start: None}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            neighbor = (current[0]+dr, current[1]+dc)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            
            # 代价 = 基本代价 + 拥塞惩罚
            base_cost = 1
            congestion_penalty = occupancy.get(neighbor, 0) * 0.5
            tentative_g = g_score[current] + base_cost + congestion_penalty
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None
```

## 整数线性规划（ILP）求解Routing问题

### 建模

将布线问题建模为整数线性规划：

```
决策变量: x_e ∈ {0, 1}  (边e是否被使用)

目标函数:
  minimize: sum(c_e * x_e)  总布线代价

约束:
  1. 流量守恒: 对于每个网络，流入每个节点的流量 = 流出的流量
  2. 容量约束: 每条边的使用不超过其容量
  3. 连通性: 源点到汇点必须连通
```

### 优缺点

| 特性 | 说明 |
|------|------|
| 优点 | 可以找到全局最优解 |
| 缺点 | NP-hard问题，大规模设计不可行 |
| 应用 | 小规模、高价值的关键网络 |

## 算法对比

| 算法 | 复杂度 | 最优性 | 适用场景 |
|------|--------|--------|----------|
| Maze (BFS) | O(V+E) | 单网络最优 | 基础布线 |
| A* | O(E log V) | 单网络最优 | 带启发式的布线 |
| PathFinder | 迭代收敛 | 近似最优 | 多网络全局布线 |
| ILP | 指数级 | 全局最优 | 小规模关键网络 |
