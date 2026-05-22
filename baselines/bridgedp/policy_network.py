"""Bridge-DP 策略网络（推理专用）。

独立 nn.Module，完全不依赖 NavDP 或其他 baseline。
与 NavDP_Policy 的核心区别：
1. 噪声调度：DDPMScheduler → BridgeScheduler（布朗桥）
2. 动作空间：增量×4 → 绝对坐标 (x, y, θ)，训练时已归一化
3. 新增模块：PriorEncoder + VisualGate（先验轨迹注入）
4. memory 序列：[time, goal×3, rgbd] → [time, goal×3, rgbd, G·prior×N_p]
5. 预测目标：噪声 ε → 归一化干净轨迹 x̂_0
6. 推理后处理：反归一化 + 三次样条平滑
"""

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

from bridge_scheduler import BridgeScheduler
from policy_backbone import (
    BridgeDP_RGBD_Backbone,
    BridgeDP_ImageGoal_Backbone,
    BridgeDP_PixelGoal_Backbone,
    LearnablePositionalEncoding,
    SinusoidalPosEmb,
)
from prior_encoder import PriorEncoder, VisualGate


class BridgeDP_Policy(nn.Module):
    # ── 动作空间归一化参数（必须与训练数据集 bridgedp_lerobot_dataset.py 保持一致）──
    ACTION_SCALE_XY = 5.0        # x, y 分量的归一化因子
    ACTION_SCALE_THETA = 3.14159  # θ 分量的归一化因子（π）

    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 sigma_base=1.0,
                 sigma_goal=0.1,
                 n_prior_tokens=4,
                 dropout=0.1,
                 channels=3,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.token_dim = token_dim
        self.n_prior_tokens = n_prior_tokens

        # 视觉编码器
        self.rgbd_encoder = BridgeDP_RGBD_Backbone(image_size, token_dim, memory_size, device)
        self.point_encoder = nn.Linear(3, token_dim)
        self.pixel_encoder = BridgeDP_PixelGoal_Backbone(image_size, token_dim, device=device)
        self.image_encoder = BridgeDP_ImageGoal_Backbone(image_size, token_dim, device=device)

        # Transformer Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=token_dim, nhead=heads,
                dim_feedforward=4 * token_dim,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True,
            ),
            num_layers=temporal_depth,
        )
        self.input_embed = nn.Linear(3, token_dim)

        # 位置编码：memory_size*16 + 4(time+goal×3) + n_prior_tokens
        cond_len = memory_size * 16 + 4 + n_prior_tokens
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, cond_len)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)

        # Bridge-DP 专有模块
        self.prior_encoder = PriorEncoder(token_dim, n_prior_tokens, dropout=dropout)
        self.visual_gate = VisualGate(token_dim)
        self.bridge_scheduler = BridgeScheduler(
            num_train_timesteps=100,
            sigma_base=sigma_base,
            sigma_goal=sigma_goal,
        )

        # 因果掩码
        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = (
            self.tgt_mask.float()
            .masked_fill(self.tgt_mask == 0, float('-inf'))
            .masked_fill(self.tgt_mask == 1, float(0.0))
        )

        # Critic 掩码：屏蔽 time + goal×3 + prior tokens
        self.cond_critic_mask = torch.zeros((predict_size, cond_len))
        self.cond_critic_mask[:, 0:4] = float('-inf')
        self.cond_critic_mask[:, 4 + memory_size * 16:] = float('-inf')

    def to(self, device, *args, **kwargs):
        self = super().to(device, *args, **kwargs)
        self.cond_critic_mask = self.cond_critic_mask.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------------
    # 归一化 / 反归一化工具
    # ------------------------------------------------------------------

    def _normalize_action(self, action):
        """将原始绝对坐标归一化到训练空间。

        Args:
            action: (..., 3) 原始坐标 (x, y, θ)。

        Returns:
            归一化后的坐标。
        """
        normed = action.clone()
        normed[..., 0:2] = normed[..., 0:2] / self.ACTION_SCALE_XY
        normed[..., 2] = normed[..., 2] / self.ACTION_SCALE_THETA
        return normed

    def _denormalize_action(self, action):
        """将归一化坐标反归一化到原始物理空间。

        Args:
            action: (..., 3) 归一化坐标。

        Returns:
            原始物理坐标 (x, y, θ)。
        """
        denormed = action.clone()
        denormed[..., 0:2] = denormed[..., 0:2] * self.ACTION_SCALE_XY
        denormed[..., 2] = denormed[..., 2] * self.ACTION_SCALE_THETA
        return denormed

    # ------------------------------------------------------------------
    # 核心网络前向
    # ------------------------------------------------------------------

    def _encode_prior(self, prior_traj):
        """编码先验轨迹并应用视觉门控。"""
        rgbd_embed = self._last_rgbd_embed  # 由调用方设置
        prior_tokens = self.prior_encoder(prior_traj)
        vis_global = rgbd_embed.mean(dim=1)
        gate = self.visual_gate(vis_global)
        return gate * prior_tokens

    def predict_x0(self, noisy_actions, timestep, goal_embed, rgbd_embed, prior_embed):
        """预测归一化干净轨迹 x̂_0。

        注意：输入和输出均在归一化空间中。
        """
        action_embeds = self.input_embed(noisy_actions)
        time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1)

        cond_tokens = torch.cat(
            [time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed, prior_embed],
            dim=1,
        )
        cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
        cond_embedding = cond_embedding.repeat(
            action_embeds.shape[0] // cond_embedding.shape[0], 1, 1
        )

        input_embedding = action_embeds + self.out_pos_embed(action_embeds)

        output = self.decoder(
            tgt=input_embedding, memory=cond_embedding,
            tgt_mask=self.tgt_mask.to(self.device),
        )
        output = self.layernorm(output)
        return self.action_head(output)

    def predict_critic(self, predict_trajectory, rgbd_embed):
        """Critic 评分，不使用先验信息。"""
        repeat_rgbd = rgbd_embed.repeat(predict_trajectory.shape[0], 1, 1)
        nogoal = torch.zeros_like(repeat_rgbd[:, 0:1])
        zero_prior = torch.zeros(
            repeat_rgbd.shape[0], self.n_prior_tokens, self.token_dim,
            device=repeat_rgbd.device,
        )
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_tokens = torch.cat(
            [nogoal, nogoal, nogoal, nogoal, repeat_rgbd, zero_prior], dim=1
        )
        cond_embeddings = cond_tokens + self.cond_pos_embed(cond_tokens)
        critic_output = self.decoder(
            tgt=action_embeddings, memory=cond_embeddings,
            memory_mask=self.cond_critic_mask.to(self.device),
        )
        return self.critic_head(self.layernorm(critic_output).mean(dim=1))[:, 0]

    # ------------------------------------------------------------------
    # 去噪 + 后处理
    # ------------------------------------------------------------------

    def _denoise(self, goal_embed, rgbd_embed, gated_prior,
                 goal_point_normed, theta_g, sample_num):
        """通用去噪流程（在归一化空间中操作）。

        Args:
            goal_embed: 目标嵌入 (B, 1, d)。
            rgbd_embed: RGBD 嵌入 (B, mem*16, d)。
            gated_prior: 门控先验 (B, N_p, d)。
            goal_point_normed: 归一化目标坐标 (B, 3)。
            theta_g: 目标方位角（原始值，不归一化）(B,)。
            sample_num: 候选轨迹数。

        Returns:
            (naction, critic_values): 归一化空间中的轨迹和评分。
        """
        B = goal_point_normed.shape[0]
        # 桥终点 = goal_point（归一化后量级与轨迹末端接近，作为桥的终点锚）
        bridge_endpoint = goal_point_normed  # (B, 3)
        endpoint_for_init = bridge_endpoint.repeat(sample_num, 1)
        naction = self.bridge_scheduler.sample_initial_noise(
            endpoint_for_init, (sample_num * B, self.predict_size, 3), self.device
        )

        self.bridge_scheduler.set_timesteps(self.bridge_scheduler.config.num_train_timesteps)
        endpoint_expanded = bridge_endpoint.unsqueeze(1).expand(
            -1, self.predict_size, -1
        ).repeat(sample_num, 1, 1)
        theta_expanded = theta_g.repeat(sample_num)

        for k in self.bridge_scheduler.timesteps:
            x0_pred = self.predict_x0(
                naction, k.to(self.device).unsqueeze(0),
                goal_embed, rgbd_embed, gated_prior,
            )
            naction = self.bridge_scheduler.step(
                x0_pred, naction, k, endpoint_expanded, theta_expanded,
            )

        critic_values = self.predict_critic(naction, rgbd_embed)
        return naction, critic_values

    def _postprocess(self, naction_normed, critic_values, B, sample_num):
        """反归一化 + 三次样条平滑 + 按 critic 排序。

        Args:
            naction_normed: 归一化空间的轨迹 (B*S, T, 3)。
            critic_values: Critic 评分 (B*S,)。
            B: batch size。
            sample_num: 候选数。

        Returns:
            (all_traj, values, pos, neg): 反归一化后的物理空间轨迹。
        """
        # 反归一化到物理空间
        naction = self._denormalize_action(naction_normed)

        trajectory = _smooth_trajectory_batch(naction)
        trajectory = trajectory.reshape(B, sample_num, self.predict_size, 3)
        critic_values = critic_values.reshape(B, sample_num)

        sorted_pos = (-critic_values).argsort(dim=1)
        sorted_neg = critic_values.argsort(dim=1)
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, 2)
        positive_trajectory = trajectory[batch_idx, sorted_pos[:, 0:2]]
        negative_trajectory = trajectory[batch_idx, sorted_neg[:, 0:2]]

        return (
            trajectory.cpu().numpy(),
            critic_values.cpu().numpy(),
            positive_trajectory.cpu().numpy(),
            negative_trajectory.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # 推理入口
    # ------------------------------------------------------------------

    def predict_pointgoal_action(self, goal_point, input_images, input_depths,
                                  prior_traj=None, theta_g=None, sample_num=16):
        """PointGoal 推理。

        Args:
            goal_point: (B, 3) 原始物理坐标目标。
            input_images: (B, mem, H, W, 3) RGB 图像。
            input_depths: (B, H, W, 1) 深度图。
            prior_traj: (B, T, 3) 原始物理坐标先验轨迹，可选。
            theta_g: (B,) 目标方位角，可选。
            sample_num: 候选轨迹数量。

        Returns:
            (all_trajectory, critic_values, positive_trajectory, negative_trajectory)
        """
        with torch.no_grad():
            tensor_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
            B = tensor_goal.shape[0]

            # 计算 theta_g（在归一化之前，使用原始坐标）
            tensor_theta = (
                torch.as_tensor(theta_g, dtype=torch.float32, device=self.device)
                if theta_g is not None
                else torch.atan2(tensor_goal[:, 1], tensor_goal[:, 0])
            )

            # ── 归一化：将物理坐标映射到训练空间 ──
            tensor_goal_normed = self._normalize_action(tensor_goal)

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            goal_embed = self.point_encoder(tensor_goal_normed).unsqueeze(1)

            # 先验轨迹：归一化后编码
            if prior_traj is not None:
                tensor_prior = torch.as_tensor(prior_traj, dtype=torch.float32, device=self.device)
                tensor_prior = self._normalize_action(tensor_prior)
            else:
                tensor_prior = torch.zeros(B, self.predict_size, 3, device=self.device)

            prior_tokens = self.prior_encoder(tensor_prior)
            gate_value = self.visual_gate(rgbd_embed.mean(dim=1))
            gated_prior = gate_value * prior_tokens

            naction, critic_values = self._denoise(
                goal_embed, rgbd_embed, gated_prior,
                tensor_goal_normed, tensor_theta, sample_num
            )
            return self._postprocess(naction, critic_values, B, sample_num)

    def predict_nogoal_action(self, input_images, input_depths,
                               prior_traj=None, sample_num=16):
        """NoGoal 推理（自由探索）。

        Returns:
            (all_trajectory, critic_values, positive_trajectory, negative_trajectory)
        """
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            B = rgbd_embed.shape[0]
            nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])

            zero_goal = torch.zeros(B, 3, device=self.device)
            zero_theta = torch.zeros(B, device=self.device)

            if prior_traj is not None:
                tensor_prior = torch.as_tensor(prior_traj, dtype=torch.float32, device=self.device)
                tensor_prior = self._normalize_action(tensor_prior)
            else:
                tensor_prior = torch.zeros(B, self.predict_size, 3, device=self.device)

            prior_tokens = self.prior_encoder(tensor_prior)
            gated_prior = self.visual_gate(rgbd_embed.mean(dim=1)) * prior_tokens

            naction, critic_values = self._denoise(
                nogoal_embed, rgbd_embed, gated_prior, zero_goal, zero_theta, sample_num
            )

            # 反归一化
            naction = self._denormalize_action(naction)

            # NoGoal：惩罚过短轨迹
            trajectory = _smooth_trajectory_batch(naction)
            trajectory = trajectory.reshape(B, sample_num, self.predict_size, 3)
            critic_values = critic_values.reshape(B, sample_num)
            traj_length = trajectory[:, :, -1, 0:2].norm(dim=-1)
            critic_values[traj_length < 1.0] -= 10.0

            sorted_pos = (-critic_values).argsort(dim=1)
            sorted_neg = critic_values.argsort(dim=1)
            batch_idx = torch.arange(B).unsqueeze(1).expand(-1, 2)
            positive_trajectory = trajectory[batch_idx, sorted_pos[:, 0:2]]
            negative_trajectory = trajectory[batch_idx, sorted_neg[:, 0:2]]

            return (
                trajectory.cpu().numpy(),
                critic_values.cpu().numpy(),
                positive_trajectory.cpu().numpy(),
                negative_trajectory.cpu().numpy(),
            )

    def predict_imagegoal_action(self, goal_image, input_images, input_depths,
                                  prior_traj=None, sample_num=16):
        """ImageGoal 推理。

        Returns:
            (all_trajectory, critic_values, positive_trajectory, negative_trajectory)
        """
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            B = rgbd_embed.shape[0]
            imagegoal_embed = self.image_encoder(
                np.concatenate((goal_image, input_images[:, -1]), axis=-1)
            ).unsqueeze(1)

            zero_goal = torch.zeros(B, 3, device=self.device)
            zero_theta = torch.zeros(B, device=self.device)

            if prior_traj is not None:
                tensor_prior = torch.as_tensor(prior_traj, dtype=torch.float32, device=self.device)
                tensor_prior = self._normalize_action(tensor_prior)
            else:
                tensor_prior = torch.zeros(B, self.predict_size, 3, device=self.device)

            prior_tokens = self.prior_encoder(tensor_prior)
            gated_prior = self.visual_gate(rgbd_embed.mean(dim=1)) * prior_tokens

            naction, critic_values = self._denoise(
                imagegoal_embed, rgbd_embed, gated_prior, zero_goal, zero_theta, sample_num
            )
            return self._postprocess(naction, critic_values, B, sample_num)

    def predict_pixelgoal_action(self, goal_image, input_images, input_depths,
                                  prior_traj=None, sample_num=16):
        """PixelGoal 推理。

        Returns:
            (all_trajectory, critic_values, positive_trajectory, negative_trajectory)
        """
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            B = rgbd_embed.shape[0]
            pixelgoal_embed = self.pixel_encoder(
                np.concatenate((goal_image[:, :, :, None], input_images[:, -1]), axis=-1)
            ).unsqueeze(1)

            zero_goal = torch.zeros(B, 3, device=self.device)
            zero_theta = torch.zeros(B, device=self.device)

            if prior_traj is not None:
                tensor_prior = torch.as_tensor(prior_traj, dtype=torch.float32, device=self.device)
                tensor_prior = self._normalize_action(tensor_prior)
            else:
                tensor_prior = torch.zeros(B, self.predict_size, 3, device=self.device)

            prior_tokens = self.prior_encoder(tensor_prior)
            gated_prior = self.visual_gate(rgbd_embed.mean(dim=1)) * prior_tokens

            naction, critic_values = self._denoise(
                pixelgoal_embed, rgbd_embed, gated_prior, zero_goal, zero_theta, sample_num
            )
            return self._postprocess(naction, critic_values, B, sample_num)


def _smooth_trajectory_batch(trajectories: torch.Tensor) -> torch.Tensor:
    """对 batch 轨迹做三次样条平滑（替代 NavDP 的 cumsum/4）。

    Bridge-DP 输出绝对坐标，需要样条平滑保证 C2 连续性。

    Args:
        trajectories: (B, T, 3) 绝对坐标轨迹。

    Returns:
        smoothed: (B, T, 3) 平滑后轨迹。
    """
    device = trajectories.device
    B, T, D = trajectories.shape
    result = torch.zeros_like(trajectories)
    t_in = np.linspace(0, 1, T)

    for b in range(B):
        traj_np = trajectories[b].cpu().numpy()
        for d in range(D):
            cs = CubicSpline(t_in, traj_np[:, d])
            result[b, :, d] = torch.from_numpy(cs(t_in).astype(np.float32)).to(device)

    return result
