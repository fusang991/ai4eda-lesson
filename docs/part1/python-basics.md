# 1.3 Python编程基础

## Python编程语言介绍

Python是AI和科学计算领域的首选编程语言，具有以下特点：

- **简洁易学**：语法清晰，接近自然语言
- **丰富的库生态**：NumPy、PyTorch、SciPy等科学计算库
- **跨平台**：支持Linux、macOS、Windows
- **社区活跃**：大量AI/ML框架和工具

## Python在EDA中的优势

1. **快速原型开发**：快速验证算法想法
2. **与AI框架无缝集成**：PyTorch、TensorFlow等
3. **数据处理能力强**：Pandas、NumPy处理大规模设计数据
4. **可视化便捷**：Matplotlib绘制分析图表

## 常用Python库

### NumPy —— 数学运算库

NumPy是Python科学计算的基础，提供高效的多维数组运算。

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

# 矩阵运算
c = np.dot(a, b)  # 点积: 130

# 创建矩阵
matrix = np.random.rand(10, 10)  # 10x10随机矩阵

# 矩阵转置
matrix_t = matrix.T

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(matrix)
```

**在EDA中的应用**：坐标计算、距离矩阵、HPWL线长计算

### Matplotlib —— 数据可视化

```python
import matplotlib.pyplot as plt

# 绘制布局结果
x_coords = np.random.rand(100) * 1000
y_coords = np.random.rand(100) * 1000

plt.figure(figsize=(10, 10))
plt.scatter(x_coords, y_coords, s=10, c='blue', alpha=0.6)
plt.xlabel('X (um)')
plt.ylabel('Y (um)')
plt.title('Cell Placement Visualization')
plt.grid(True)
plt.savefig('placement.png', dpi=150)
plt.show()
```

### SciPy —— 科学计算库

```python
from scipy import optimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 稀疏矩阵求解（常用于布局的数值求解）
# 构建拉普拉斯矩阵
row = np.array([0, 0, 1, 1, 2])
col = np.array([1, 2, 0, 2, 0])
data = np.array([-1, -1, -1, -1, -1])
L = csr_matrix((data, (row, col)), shape=(3, 3))

# 最优化求解
def objective(x):
    return np.sum(x**2)

result = optimize.minimize(objective, x0=[1.0, 2.0, 3.0])
```

## Python编程核心概念

### 数据类型

```python
# 基本类型
x = 42          # int
y = 3.14        # float
s = "hello"     # str
flag = True     # bool

# 容器类型
lst = [1, 2, 3]           # list
tup = (1, 2, 3)           # tuple
dct = {'a': 1, 'b': 2}   # dict
st = {1, 2, 3}            # set
```

### 函数和类

```python
# 函数
def hpwl(cells, nets):
    """计算半周线长 Half-Perimeter Wirelength"""
    total_hpwl = 0
    for net in nets:
        x_coords = [cells[pin][0] for pin in net]
        y_coords = [cells[pin][1] for pin in net]
        hpwl_val = (max(x_coords) - min(x_coords)) + \
                   (max(y_coords) - min(y_coords))
        total_hpwl += hpwl_val
    return total_hpwl

# 类
class ChipDesign:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self.cells = []
    
    def add_cell(self, cell):
        self.cells.append(cell)
    
    def total_area(self):
        return sum(c['w'] * c['h'] for c in self.cells)
```

### 列表推导式和生成器

```python
# 列表推导式
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# 字典推导式
cell_areas = {cell['name']: cell['w'] * cell['h'] for cell in cells}

# 生成器（处理大数据时节省内存）
def read_netlist(filename):
    with open(filename) as f:
        for line in f:
            yield parse_net(line)
```
