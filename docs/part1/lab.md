# 1.5 上机实验

## 实验一：Linux系统常用命令

### 命令行终端

推荐使用以下终端软件连接服务器：
- **iTerm**（macOS）
- **Xshell**（Windows）
- **Windows Terminal** + SSH

### 常用命令

```bash
# 文件和目录操作
ls          # 列出当前目录文件
ls -la      # 列出详细信息（含隐藏文件）
cd /path    # 切换目录
cd ..       # 返回上级目录
pwd         # 显示当前路径
cp src dst  # 复制文件
mv src dst  # 移动/重命名文件
mkdir dir   # 创建目录
touch file  # 创建空文件

# 查看文件内容
cat file    # 显示文件全部内容
less file   # 分页查看
head -n 20  # 查看前20行
tail -n 20  # 查看后20行
tail -f log # 实时查看日志

# 权限管理
chmod 755 script.sh  # 设置执行权限
```

### Vim文本编辑

```bash
vim file.py    # 打开文件

# 常用操作
i              # 进入插入模式
Esc            # 退出插入模式
:w             # 保存
:q             # 退出
:wq            # 保存并退出
:q!            # 强制退出不保存
/dd            # 搜索
yy             # 复制当前行
p              # 粘贴
dd             # 删除当前行
u              # 撤销
```

## 实验二：Anaconda环境管理

### 创建和管理环境

```bash
# 创建新环境
conda create -n ai4eda python=3.10

# 激活环境
conda activate ai4eda

# 安装常用包
conda install numpy matplotlib scipy pandas jupyter
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 查看已安装包
conda list

# 导出环境
conda env export > environment.yml

# 从文件恢复环境
conda env create -f environment.yml

# 回溯环境历史
conda list --revisions
conda install --revision 5  # 回到第5个版本
```

## 实验三：Python编程实战

### NumPy实战

```python
import numpy as np

# 创建各种数组
zeros = np.zeros((5, 5))
ones = np.ones((3, 3))
identity = np.eye(4)
random_arr = np.random.randn(100)

# 数组运算
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(a + b)       # 矩阵加法
print(a @ b)       # 矩阵乘法
print(np.sum(a))   # 求和
print(np.mean(a))  # 均值
print(np.std(a))   # 标准差

# 广播机制
c = np.array([10, 20])
print(a + c)  # 每行加 [10, 20]
```

### Matplotlib实战

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：散点图
x = np.random.rand(200) * 1000
y = np.random.rand(200) * 1000
axes[0].scatter(x, y, c='steelblue', alpha=0.6, s=20)
axes[0].set_xlabel('X (um)')
axes[0].set_ylabel('Y (um)')
axes[0].set_title('Cell Placement')
axes[0].grid(True, alpha=0.3)

# 右图：直方图
data = np.random.randn(1000)
axes[1].hist(data, bins=50, color='steelblue', alpha=0.7)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution')

plt.tight_layout()
plt.savefig('python_lab.png', dpi=150)
plt.show()
```

### Pandas实战

```python
import pandas as pd

# 创建DataFrame
data = {
    'cell_name': [f'cell_{i}' for i in range(100)],
    'x': np.random.rand(100) * 1000,
    'y': np.random.rand(100) * 1000,
    'width': np.random.uniform(1, 10, 100),
    'height': np.random.uniform(1, 10, 100)
}
df = pd.DataFrame(data)

# 基本操作
print(df.head())
print(df.describe())
print(df[df['width'] > 5])

# 添加计算列
df['area'] = df['width'] * df['height']
df['center_x'] = df['x'] + df['width'] / 2
df['center_y'] = df['y'] + df['height'] / 2

# 分组统计
df['size_category'] = pd.cut(df['area'], bins=3, labels=['small', 'medium', 'large'])
print(df.groupby('size_category')['area'].agg(['count', 'mean', 'sum']))
```

## 实验四：KL/FM算法实现

### 实现Kernighan-Lin算法

```python
import numpy as np
import random

def create_random_graph(n_nodes, n_edges):
    """创建随机图用于测试"""
    graph = {i: set() for i in range(n_nodes)}
    for _ in range(n_edges):
        u, v = random.sample(range(n_nodes), 2)
        graph[u].add(v)
        graph[v].add(u)
    return graph

def count_cut(graph, A, B):
    """计算割边数"""
    cut = 0
    for u in A:
        for v in graph[u]:
            if v in B:
                cut += 1
    return cut

def kl_algorithm(graph, n_nodes, max_iterations=100):
    """KL算法实现"""
    half = n_nodes // 2
    A = set(random.sample(range(n_nodes), half))
    B = set(range(n_nodes)) - A
    
    best_cut = count_cut(graph, A, B)
    print(f"Initial cut: {best_cut}")
    
    for iteration in range(max_iterations):
        # 计算D值
        D = {}
        for node in range(n_nodes):
            own = A if node in A else B
            other = B if node in A else A
            E = sum(1 for n in graph[node] if n in other)
            I = sum(1 for n in graph[node] if n in own)
            D[node] = E - I
        
        # 找最佳交换
        best_gain = 0
        best_pair = None
        for a in A:
            for b in B:
                gain = D[a] + D[b]
                if b in graph[a]:
                    gain -= 2
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
        
        if best_pair is None or best_gain <= 0:
            print(f"Converged at iteration {iteration}")
            break
        
        a, b = best_pair
        A.remove(a); A.add(b)
        B.remove(b); B.add(a)
        
        new_cut = count_cut(graph, A, B)
        print(f"Iteration {iteration}: cut = {new_cut} (gain = {best_gain})")
    
    return A, B

# 测试
graph = create_random_graph(50, 150)
A, B = kl_algorithm(graph, 50)
print(f"Final cut: {count_cut(graph, A, B)}")
```

## 实验五：超算服务器和Slurm

### Slurm作业调度系统

```bash
# 提交交互式作业
srun --partition=gpu --gres=gpu:1 --time=2:00:00 --pty bash

# 提交批处理作业
sbatch train.sh
```

### train.sh 示例

```bash
#!/bin/bash
#SBATCH --job-name=ai4eda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

module load anaconda3
conda activate ai4eda
python train.py --epochs 100 --batch-size 64
```

### 常用Slurm命令

```bash
# 查看作业状态
squeue -u $USER

# 查看集群状态
sinfo

# 取消作业
scancel <job_id>

# 查看作业详情
scontrol show job <job_id>
```
