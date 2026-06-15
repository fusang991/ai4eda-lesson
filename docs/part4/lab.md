# 4.5 上机实验

## 实验一：Python编程实现DME延迟平衡算法

### 实验目标

实现基于DME的时钟树Buffering和Balance过程。

### 完整代码

```python
import numpy as np
import matplotlib.pyplot as plt

class ClockTreeDME:
    def __init__(self, sinks):
        self.sinks = np.array(sinks, dtype=float)
        self.n = len(sinks)
        self.tree_nodes = []  # 所有树节点（包括合并点）
        self.edges = []
    
    def merge(self, node1, node2):
        """零偏斜合并两个节点"""
        p1, d1 = node1['pos'], node1['delay']
        p2, d2 = node2['pos'], node2['delay']
        
        dist = np.linalg.norm(p1 - p2)
        diff = d1 - d2
        
        # 零偏斜合并点
        t = (dist - diff) / (2 * dist) if dist > 0 else 0.5
        t = max(0, min(1, t))
        merge_pos = p1 + t * (p2 - p1)
        merge_delay = (d1 + d2 + dist) / 2
        
        node_id = len(self.tree_nodes)
        new_node = {
            'id': node_id,
            'pos': merge_pos,
            'delay': merge_delay,
            'is_sink': False,
            'children': [node1['id'], node2['id']]
        }
        self.tree_nodes.append(new_node)
        self.edges.append((node_id, node1['id']))
        self.edges.append((node_id, node2['id']))
        
        return new_node
    
    def build_tree(self):
        """构建零偏斜时钟树"""
        # 初始化叶节点
        self.tree_nodes = []
        for i in range(self.n):
            self.tree_nodes.append({
                'id': i,
                'pos': self.sinks[i],
                'delay': 0,
                'is_sink': True,
                'children': []
            })
        
        # 贪心合并
        active = list(range(self.n))
        while len(active) > 1:
            # 找最近的两个节点
            min_dist = float('inf')
            best_i, best_j = 0, 1
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ni = self.tree_nodes[active[i]]
                    nj = self.tree_nodes[active[j]]
                    d = np.linalg.norm(ni['pos'] - nj['pos'])
                    if d < min_dist:
                        min_dist = d
                        best_i, best_j = i, j
            
            # 合并
            ni = self.tree_nodes[active[best_i]]
            nj = self.tree_nodes[active[best_j]]
            new_node = self.merge(ni, nj)
            
            # 更新活跃列表
            active = [a for k, a in enumerate(active) 
                      if k != best_i and k != best_j]
            active.append(new_node['id'])
        
        return self.tree_nodes[active[0]]  # 根节点
    
    def get_skew(self):
        """计算时钟偏斜"""
        sink_delays = [n['delay'] for n in self.tree_nodes if n['is_sink']]
        return max(sink_delays) - min(sink_delays)
    
    def plot(self):
        """可视化时钟树"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 画边
        for parent_id, child_id in self.edges:
            p = self.tree_nodes[parent_id]['pos']
            c = self.tree_nodes[child_id]['pos']
            ax.plot([p[0], c[0]], [p[1], c[1]], 'b-', linewidth=1, alpha=0.7)
        
        # 画节点
        for node in self.tree_nodes:
            if node['is_sink']:
                ax.plot(node['pos'][0], node['pos'][1], 'ro', markersize=8)
            else:
                ax.plot(node['pos'][0], node['pos'][1], 'g^', markersize=6)
        
        # 画根节点
        root = self.tree_nodes[-1]
        ax.plot(root['pos'][0], root['pos'][1], 'bs', markersize=12, label='Root')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Clock Tree (Skew = {self.get_skew():.4f})')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.savefig('clock_tree.png', dpi=150, bbox_inches='tight')
        plt.show()

# 运行示例
np.random.seed(42)
sinks = np.random.rand(20, 2) * 100  # 20个寄存器

dme = ClockTreeDME(sinks)
root = dme.build_tree()
print(f"Clock Skew: {dme.get_skew():.6f}")
print(f"Root Position: ({root['pos'][0]:.2f}, {root['pos'][1]:.2f})")
print(f"Root Delay: {root['delay']:.4f}")
dme.plot()
```

## 实验二：OpenROAD和TritonCTS

### OpenROAD安装

```bash
# 使用Docker
docker pull openroad/centos7-builder-gcc9

# 或从源码编译
git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD.git
cd OpenROAD
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### 在OpenROAD上运行CTS

```tcl
# read_cts_example.tcl

# 读取设计
read_lef tech.lef
read_lef std_cell.lef
read_def placed.def
read_sdc design.sdc

# 配置CTS
configure_cts_characterization -max_cap 1.5 -max_slew 0.2

# 运行CTS
clock_tree_synthesis -root_buf CLKBUF_X3 \
                     -buf_list CLKBUF_X1 CLKBUF_X2 CLKBUF_X3 \
                     -wire_unit 20

# 检查结果
report_cts
report_clock_skew

# 写出结果
write_def cts_result.def
```

### 运行命令

```bash
openroad -no_init -log cts.log read_cts_example.tcl
```

## 实验三：论文复现——基于GNN的时序预测（DAC 2022）

### 数据集准备

```python
import torch
from torch_geometric.data import Data, DataLoader

def create_circuit_graph(netlist, features, timing_labels):
    """将电路网表转换为图数据"""
    edge_index = []
    for net in netlist:
        pins = net['pins']
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                edge_index.append([pins[i], pins[j]])
                edge_index.append([pins[j], pins[i]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(timing_labels, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)
```

### GNN模型

```python
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class TimingPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        return self.head(x).squeeze(-1)
```

### 训练

```python
model = TimingPredictor(in_dim=feature_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 评估
    model.eval()
    with torch.no_grad():
        mae = 0
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            mae += torch.abs(pred - batch.y).mean().item()
    
    print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, "
          f"MAE={mae/len(test_loader):.4f}")
```
