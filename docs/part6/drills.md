# 6.1 DRiLLS: 强化学习用于逻辑综合

## 论文信息

| 项目 | 内容 |
|------|------|
| 标题 | DRiLLS: Deep Reinforcement Learning for Logic Synthesis |
| 作者 | A. Hosny, S. Hashemi, M. Shalan, S. Reda |
| 会议 | ASP-DAC 2020 |
| 机构 | SCALE Lab, Brown University |
| 代码 | https://github.com/scale-lab/DRiLLS |

## 问题背景

逻辑综合需要选择一系列优化操作（rewrite, refactor, resub, balance等）来优化电路。优化序列的选择对最终面积和时序有很大影响，但搜索空间呈指数级增长。

传统方法使用固定的优化脚本（如ABC的resyn2），无法针对不同设计自适应调整。

## 核心方法

### 强化学习建模

| RL要素 | 对应 |
|--------|------|
| 状态 | 电路特征（节点数、边数、逻辑深度、门类型比例等6维） |
| 动作 | 7种优化操作（rewrite, rewrite -z, refactor, refactor -z, resub, resub -z, balance） |
| 奖励 | 面积改善 + 时序改善的组合奖励 |

### A2C (Advantage Actor-Critic) 网络

```
状态(6维) -> 共享层(64) -> Actor头 -> 动作概率
                         -> Critic头 -> 状态价值
```

### 奖励设计

原版使用了精心设计的奖励表：

| 时序满足 | 时序改善 | 面积改善 | 奖励 |
|----------|----------|----------|------|
| 是 | - | 改善 | +3 |
| 是 | - | 不变 | 0 |
| 是 | - | 恶化 | -1 |
| 否 | 改善 | 改善 | +3 |
| 否 | 不变 | 不变 | 0 |
| 否 | 恶化 | 恶化 | -3 |

## 复现实现

### 环境搭建

原版使用TensorFlow 1.x，我们用PyTorch重写。关键改动：
- 使用yosys-abc替代abc（更好的Verilog支持）
- 先用Yosys综合Verilog为BLIF，再用ABC优化
- 用`print_stats`的AND门数作为面积代理

### 工作流程

```
Verilog -> Yosys(synth) -> BLIF -> ABC(strash + 优化序列) -> print_stats
```

### 训练过程

```python
# 核心训练循环
for ep in range(episodes):
    state = env.reset()
    while not done:
        action_probs, value = model(state)
        action = Categorical(action_probs).sample()
        new_state, reward, done = env.step(action)
        # 收集经验...
    # 计算折扣回报并更新网络
    loss = actor_loss + 0.5 * critic_loss
    loss.backward()
    optimizer.step()
```

## 实验结果

```
设计: 8-bit multiplier
初始面积: 506 AND gates
最优面积: 463 AND gates (-8.5%)
最优操作: rewrite -z
训练回合: 20
```

![DRiLLS训练曲线](/part6/drills_training.png)

## 关键发现

1. **A2C能学到有效策略**：Agent逐渐学会偏好`rewrite -z`等面积优化效果好的操作
2. **探索与利用的平衡**：epsilon-贪心策略让Agent在训练早期广泛探索，后期专注利用
3. **奖励设计很重要**：同时考虑面积和时序的奖励表引导Agent找到满足约束的最优解

## 局限性

- 小电路上效果明显，大规模设计需要更多训练
- 每步都需要调用ABC，训练速度受限于工具调用
- 特征维度较低（6维），可能丢失电路结构信息
