# Bridge-DP 训练不收敛分析报告

## 1. 问题描述

**现象**：Bridge-DP 训练 1 个 epoch 后，推理时 `action_head` 输出值极小（±0.025），导致机器人几乎无法运动。而同等训练量的 NavDP（1 epoch 甚至 10 epoch）都能正常输出轨迹并使机器人运动。

**诊断数据**（推理时实际观测值）：

```
decoder output: min=-302, max=427, mean=3.5, std=77.8
layernorm(output): min=-3.84, max=5.41, mean≈0, std=0.99
action_head.weight: shape=[3,384], std=0.026, norm=0.882
action_head.bias: [0.0115, -0.0426, -0.0151]
action_head output (x0_pred): min=-0.0098, max=0.0251
```

**结论**：`action_head` 权重 std=0.026 接近 PyTorch `nn.Linear(384, 3)` 的随机初始化值（Kaiming uniform, std ≈ 1/√384 ≈ 0.051），说明 **action_head 在训练过程中几乎没有学到有效知识**。

---

## 2. 根因分析：NavDP vs Bridge-DP 的训练目标尺度失配

### 2.1 NavDP 训练目标：噪声 ε（自然尺度 ~N(0,1)）

```python
# NavDP 数据集 (navdp_lerobot_dataset.py L794)
pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0  # 差分×4

# NavDP 训练 forward (navdp_policy.py L148-157)
noise = torch.randn(action.shape)                    # 目标: ε ~ N(0,1)
noisy_action = DDPMScheduler.add_noise(action, noise, t)

# NavDP 训练损失 (navdp_trainer.py L138)
loss = (pred_noise - noise).square().mean()           # MSE(pred, ε)
```

**关键**：NavDP 预测的目标是 **噪声 ε ~ N(0,1)**，值域集中在 [-3, 3]。
- `action_head` 的随机初始化 (std≈0.05) 输出的值域约为 [-0.2, 0.2]
- 目标 ε 的值域约为 [-3, 3]
- **初始 MSE loss ≈ 1.0**（可接受的起始值）
- 梯度尺度适中，Adam lr=1e-4 能有效推动收敛

### 2.2 Bridge-DP 训练目标：干净轨迹 x̂₀（绝对坐标，0~10m）

```python
# Bridge-DP 数据集 (bridgedp_lerobot_dataset.py L592)
pred_actions = pred_actions[1:]   # 绝对坐标，不做差分

# Bridge-DP 训练 forward (bridgedp_policy.py L392-394)
x0_ng, ng_time_embed, ng_noisy_embed, _ = self.sample_bridge_noise(
    tensor_label_actions, tensor_point_goal, tensor_theta_g  # 目标: x_0 绝对坐标
)

# Bridge-DP 训练损失 (bridgedp_trainer.py L117)
loss = (x0_pred - x0_target).square().mean()          # MSE(pred, x_0)
```

**关键**：Bridge-DP 预测的目标是 **干净轨迹 x̂₀ 的绝对坐标**。
- `pred_actions` 是从起点到目标的绝对 (x, y, θ) 坐标
- x, y 分量典型范围：0~10 米（目标距离可达 12m）
- θ 分量范围：-π ~ π
- `action_head` 随机初始化输出 ≈ [-0.2, 0.2]
- **初始 MSE loss ≈ 25.0**（目标值 5m 时，(5-0.1)² ≈ 24）

### 2.3 数值对比表

| 指标 | NavDP (predict_noise) | Bridge-DP (predict_x0) |
|------|----------------------|------------------------|
| 预测目标 | 噪声 ε ~ N(0,1) | 绝对坐标 x₀ (0~10m) |
| 目标值域 | [-3, 3] | [0, 10] |
| 初始输出值域 | [-0.2, 0.2] | [-0.2, 0.2] |
| 初始 MSE | ~1.0 | ~25.0 |
| 初始梯度尺度 | ~2.0 | ~10.0 |
| 梯度与 action_head 权重比 | ~40× | ~200× |

---

## 3. 深层原因链分析

### 3.1 问题链路图

```
绝对坐标目标 (0~10m)
    │
    ▼
初始 MSE 巨大 (~25) ──→ 梯度爆炸（初始 grad ~10×正常值）
    │
    ▼
16层 Transformer Decoder 的梯度传播路径长
    │
    ▼
梯度在深层反传时被 LayerNorm 和 GELU 多次截断/缩放
    │
    ▼
action_head 梯度反传至 decoder 深层后接近零
    │
    ▼
decoder 内部层学不到有效表征
    │
    ▼
decoder 输出随机 (std=77.8) ──→ LayerNorm 归一化到 std≈1.0
    │
    ▼
action_head 权重几乎不更新 (std=0.026 ≈ 初始化值)
    │
    ▼
x0_pred ≈ 0 ──→ 机器人无法运动
```

### 3.2 为什么 NavDP 即使 1 epoch 也能动？

1. **目标尺度匹配**：ε ~ N(0,1) 与随机初始化输出 (~0.2) 尺度接近
2. **推理不依赖绝对值**：NavDP 推理时用 `cumsum(pred / 4.0)` 将差分还原为轨迹，即使噪声预测不精确，累积效应也能产生合理轨迹
3. **DDPM scheduler 的稳定性**：DDPM 的 `step()` 方法有内置的方差调度和 clipping，即使噪声预测有偏差也能给出不错的样本

### 3.3 为什么 Bridge-DP 即使多训几个 epoch 也难以收敛？

1. **梯度尺度失配**：action_head 的 MSE 梯度 ∂L/∂w = 2(ŷ-y)·x，当 y≈5m 而 ŷ≈0 时，梯度约为 -10×x（x 是 layernorm 输出 ~1.0），远大于正常范围
2. **LayerNorm 瓶颈**：decoder 输出经过 LayerNorm 后 std 固定为 ~1.0，无论 decoder 内部如何调整，action_head 能看到的输入范围始终被限制在 [-5, 5]。要让 action_head 输出 10m，需要权重达到 ~2.0（当前 0.026），这需要权重增长 80 倍
3. **训练不稳定**：初始大梯度可能导致 Adam 的二阶矩估计 v 在早期就被设定得很大，后续即使梯度合理了，实际步长也被缩小

### 3.4 训练中 forward() 的 Dropout 差异

在训练的 `forward()` 方法中（bridgedp_policy.py L428-457），memory 和 action 嵌入都经过 `self.drop()` (Dropout(0.1))：

```python
ng_cond_embeddings = self.drop(
    torch.cat([ng_time_embed, nogoal_embed, nogoal_embed, nogoal_embed, rgbd_embed, gated_prior], dim=1)
    + cond_pos_embed
)
ng_action_embeddings = self.drop(ng_noisy_embed + out_pos_embed)
```

但在推理的 `predict_x0()` 方法中（bridgedp_policy.py L286-304），**没有 Dropout**：

```python
cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
input_embedding = action_embeds + self.out_pos_embed(action_embeds)
output = self.decoder(tgt=input_embedding, memory=cond_embedding, ...)
```

这个训练/推理不一致虽然是标准做法（eval 模式下 Dropout 自动禁用），但结合 Bridge-DP 的梯度问题，会**加剧训练信号的衰减**——10% 的 token 在每次训练 forward 中被随机丢弃，进一步稀释了本就很弱的有效梯度。

---

## 4. 解决方案

### 方案 A：动作空间归一化（推荐 ⭐⭐⭐⭐⭐）

**核心思路**：在数据集中对绝对坐标做归一化，使训练目标从 [0, 10m] 映射到 [-1, 1]，推理时反归一化。

**修改文件**：`bridgedp_lerobot_dataset.py` 和 `bridgedp_policy.py`

```python
# === 数据集修改 (bridgedp_lerobot_dataset.py) ===

# 在 __init__ 中定义归一化参数
self.action_scale = 5.0   # 经验值：轨迹 x,y 分量最大约 10m，除以 scale 后 [-2, 2]
self.theta_scale = np.pi   # θ 分量范围 [-π, π]

# 在 __getitem__ 中归一化
pred_actions[:, 0:2] = pred_actions[:, 0:2] / self.action_scale
pred_actions[:, 2] = pred_actions[:, 2] / self.theta_scale
augment_actions[:, 0:2] = augment_actions[:, 0:2] / self.action_scale
augment_actions[:, 2] = augment_actions[:, 2] / self.theta_scale

# point_goal 也需要同步归一化
point_goal[0:2] = point_goal[0:2] / self.action_scale
point_goal[2] = point_goal[2] / self.theta_scale

# prior_traj 也需要同步归一化
prior_traj[:, 0:2] = prior_traj[:, 0:2] / self.action_scale
prior_traj[:, 2] = prior_traj[:, 2] / self.theta_scale
```

```python
# === 推理修改 (bridgedp_policy.py predict_x0 / 推理入口) ===

# 推理时 goal 归一化
goal_point_normalized = goal_point.clone()
goal_point_normalized[:, 0:2] /= 5.0
goal_point_normalized[:, 2] /= np.pi

# 推理后反归一化
x0_pred[:, :, 0:2] *= 5.0
x0_pred[:, :, 2] *= np.pi
```

**优势**：
- 训练目标从 [0, 10] 降至 [-2, 2]，与 NavDP 的噪声目标 [-3, 3] 尺度一致
- 初始 MSE 从 ~25 降至 ~1.0
- 不需要修改网络架构
- 不需要修改损失函数
- 完全数学等价，不影响最终精度

**预期效果**：1 epoch 后 action_head 就能输出有意义的值，与 NavDP 收敛速度相当。

---

### 方案 B：action_head 权重专用初始化 + 分层学习率

**核心思路**：加大 action_head 初始化的 std，使其初始输出就能覆盖目标范围；同时给 action_head 更大的学习率。

```python
# === bridgedp_policy.py __init__ ===

# 专用初始化：让 action_head 初始输出能覆盖 [0, 10]
nn.init.xavier_normal_(self.action_head.weight, gain=5.0)
nn.init.zeros_(self.action_head.bias)
```

```python
# === bridgedp_trainer.py create_optimizer ===

# 分层学习率：action_head 用 10× 学习率
action_head_params = list(model_for_optim.action_head.parameters())
other_params = [p for n, p in model_for_optim.named_parameters()
                if 'action_head' not in n and p.requires_grad]

optimizer = torch.optim.Adam([
    {'params': other_params, 'lr': lr},
    {'params': action_head_params, 'lr': lr * 10},
])
```

**优势**：
- 不修改数据流
- action_head 能快速适应目标范围

**劣势**：
- 需要手动调整 gain 和 lr 倍率
- decoder 内部层的梯度问题未完全解决
- 不如方案 A 通用

---

### 方案 C：混合预测策略（predict_v 范式）

**核心思路**：不直接预测 x₀ 或 ε，而是预测 "velocity" v = α·ε - σ·x₀，这是 Stable Diffusion 3 等现代扩散模型常用的技巧。

```python
# 训练目标：v = (1-t)·ε - t·(x₀ - g)
v_target = (1 - t) * noise - t * (x0 - goal)

# 推理时还原：x̂₀ = (1-t)·x_t - σ·v_pred + t·g
# 其中 σ 来自 bridge scheduler
```

**优势**：
- v 的值域在不同 t 下都保持合理范围
- 理论上收敛更稳定

**劣势**：
- 需要较大的代码改动
- 改变了训练目标的数学定义，需要重新推导 bridge scheduler 的 step 公式
- 增加了实现复杂度

---

## 5. 推荐行动计划

1. **优先实施方案 A（动作空间归一化）**
   - 修改量小（数据集 ~20 行，推理 ~10 行）
   - 数学等价，不改变模型含义
   - 与 NavDP 的训练目标尺度对齐
   - 训练后的模型在推理端只需加一步反归一化

2. **如需进一步提升，辅以方案 B 的 action_head 初始化**
   - 可与方案 A 叠加使用
   - Xavier(gain=2.0) + 归一化后，初始输出范围 ~[-2, 2] 与目标 ~[-2, 2] 完美匹配

3. **方案 C 留作后续研究方向**

---

## 6. 验证方法

实施修改后，用以下指标验证训练收敛：

```python
# 在 compute_loss 中加入监控
print(f"action_loss={action_loss.item():.4f}, "
      f"x0_pred range=[{x0_pred_ng.min():.3f}, {x0_pred_ng.max():.3f}], "
      f"x0_target range=[{x0_target_ng.min():.3f}, {x0_target_ng.max():.3f}]")
```

**预期变化**：
- 第 1 步：action_loss 从 ~25 → ~1.0（方案 A 归一化后）
- 100 步后：action_loss < 0.5，x0_pred 范围覆盖 [-2, 2]
- 1 epoch 后：action_loss < 0.1，x0_pred 与 x0_target 有明显相关性
- 推理时：反归一化后 action_head 输出范围 [0, 10m]，机器人正常运动
