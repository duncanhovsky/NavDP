"""Standalone Bridge-DP inference policy for the NavDP evaluation server.

The model outputs 24 absolute trajectory control points.  They are raw curve
samples, not fixed-speed executable waypoints.  The agent wrapper converts them
to execution waypoints after candidate scoring.
"""

import numpy as np
import torch
import torch.nn as nn

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
    ACTION_SCALE_XY = 5.0
    ACTION_SCALE_THETA = 3.14159

    def __init__(
        self,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        sigma_base=1.0,
        sigma_goal=0.1,
        sigma_floor=None,
        nogoal_front_distance=0.8,
        nogoal_sigma_start=0.03,
        nogoal_sigma_x_end=0.35,
        nogoal_sigma_y_end=0.80,
        nogoal_sigma_theta_end=0.60,
        nogoal_sigma_power=2.0,
        bridge_scale_invariant_sigma=False,
        bridge_anisotropic_xy=True,
        bridge_normal_sigma_ratio=0.25,
        bridge_tangent_sigma_ratio=0.03,
        bridge_theta_sigma_ratio=0.05,
        n_prior_tokens=4,
        dropout=0.1,
        channels=3,
        enable_trajectory_normalization=False,
        trajectory_norm_target_distance=2.0,
        trajectory_norm_min_distance_m=0.10,
        trajectory_norm_eps=1e-6,
        enable_scale_condition_token=False,
        scale_condition_clamp_min_m=0.10,
        scale_condition_clamp_max_m=20.0,
        enable_scale_rgbd_film=False,
        scale_rgbd_film_alpha=1.0,
        scale_rgbd_film_zero_init=True,
        scale_rgbd_film_use_layernorm=True,
        enable_goal_consistency_score=False,
        goal_consistency_terminal_weight=1.0,
        goal_consistency_path_weight=0.2,
        num_train_timesteps=100,
        num_inference_timesteps=100,
        use_prior_traj=False,
        device='cuda:0',
    ):
        super().__init__()
        _ = channels
        self.device = device
        self._device = torch.device(device)
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.token_dim = token_dim
        self.n_prior_tokens = n_prior_tokens
        self.dropout = dropout

        self.action_scale_xy = self.ACTION_SCALE_XY
        self.action_scale_theta = self.ACTION_SCALE_THETA
        self.nogoal_front_distance = nogoal_front_distance
        self.enable_trajectory_normalization = enable_trajectory_normalization
        self.trajectory_norm_target_distance = float(trajectory_norm_target_distance)
        self.trajectory_norm_min_distance_m = float(trajectory_norm_min_distance_m)
        self.trajectory_norm_eps = float(trajectory_norm_eps)
        self.enable_scale_condition_token = enable_scale_condition_token
        self.scale_condition_clamp_min_m = float(scale_condition_clamp_min_m)
        self.scale_condition_clamp_max_m = float(scale_condition_clamp_max_m)
        self.n_scale_tokens = 1 if enable_scale_condition_token else 0
        self.enable_scale_rgbd_film = enable_scale_rgbd_film
        self.scale_rgbd_film_alpha = float(scale_rgbd_film_alpha)
        self.scale_rgbd_film_zero_init = scale_rgbd_film_zero_init
        self.scale_rgbd_film_use_layernorm = scale_rgbd_film_use_layernorm
        self.enable_goal_consistency_score = enable_goal_consistency_score
        self.goal_consistency_terminal_weight = float(goal_consistency_terminal_weight)
        self.goal_consistency_path_weight = float(goal_consistency_path_weight)
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.use_prior_traj = use_prior_traj

        self.rgbd_encoder = BridgeDP_RGBD_Backbone(image_size, token_dim, memory_size, device)
        self.point_encoder = nn.Linear(3, token_dim)
        self.pixel_encoder = BridgeDP_PixelGoal_Backbone(image_size, token_dim, device=device)
        self.image_encoder = BridgeDP_ImageGoal_Backbone(image_size, token_dim, device=device)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=temporal_depth)
        self.input_embed = nn.Linear(3, token_dim)

        cond_len = memory_size * 16 + 4 + self.n_scale_tokens + n_prior_tokens
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, cond_len)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.drop = nn.Dropout(dropout)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        self.action_head = nn.Linear(token_dim, 3)
        self.delta_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        self.pixel_aux_head = nn.Linear(token_dim, 3)
        self.image_aux_head = nn.Linear(token_dim, 3)

        if self.enable_scale_condition_token:
            self.scale_encoder = nn.Sequential(
                nn.Linear(4, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
        if self.enable_scale_rgbd_film:
            self.scale_rgbd_film = nn.Sequential(
                nn.Linear(4, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, 2 * token_dim),
            )
            if self.scale_rgbd_film_use_layernorm:
                self.scale_rgbd_film_norm = nn.LayerNorm(token_dim)
            if self.scale_rgbd_film_zero_init:
                nn.init.zeros_(self.scale_rgbd_film[-1].weight)
                nn.init.zeros_(self.scale_rgbd_film[-1].bias)

        self.prior_encoder = PriorEncoder(
            embed_dim=token_dim,
            n_prior_tokens=n_prior_tokens,
            dropout=dropout,
        )
        self.visual_gate = VisualGate(gate_dim=token_dim)
        self.bridge_scheduler = BridgeScheduler(
            num_train_timesteps=self.num_train_timesteps,
            sigma_base=sigma_base,
            sigma_goal=sigma_goal,
            sigma_floor=sigma_floor,
            nogoal_front_distance=nogoal_front_distance,
            nogoal_sigma_start=nogoal_sigma_start,
            nogoal_sigma_x_end=nogoal_sigma_x_end,
            nogoal_sigma_y_end=nogoal_sigma_y_end,
            nogoal_sigma_theta_end=nogoal_sigma_theta_end,
            nogoal_sigma_power=nogoal_sigma_power,
            bridge_scale_invariant_sigma=bridge_scale_invariant_sigma,
            bridge_anisotropic_xy=bridge_anisotropic_xy,
            bridge_normal_sigma_ratio=bridge_normal_sigma_ratio,
            bridge_tangent_sigma_ratio=bridge_tangent_sigma_ratio,
            bridge_theta_sigma_ratio=bridge_theta_sigma_ratio,
        )

        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = (
            self.tgt_mask.float()
            .masked_fill(self.tgt_mask == 0, float('-inf'))
            .masked_fill(self.tgt_mask == 1, float(0.0))
        )

        self.cond_critic_mask = torch.zeros((predict_size, cond_len))
        rgbd_start = 4 + self.n_scale_tokens
        self.cond_critic_mask[:, 0:rgbd_start] = float('-inf')
        self.cond_critic_mask[:, rgbd_start + memory_size * 16:] = float('-inf')

    def to(self, device, *args, **kwargs):
        self = super().to(device, *args, **kwargs)
        self.device = device
        self._device = torch.device(device)
        self.cond_critic_mask = self.cond_critic_mask.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        for module in (self.rgbd_encoder, self.pixel_encoder, self.image_encoder):
            if hasattr(module, 'device'):
                module.device = device
        return self

    def _normalize_action(self, action):
        normed = action.clone()
        normed[..., 0:2] = normed[..., 0:2] / self.action_scale_xy
        normed[..., 2] = normed[..., 2] / self.action_scale_theta
        return normed

    def _denormalize_action(self, action):
        denormed = action.clone()
        denormed[..., 0:2] = denormed[..., 0:2] * self.action_scale_xy
        denormed[..., 2] = denormed[..., 2] * self.action_scale_theta
        return denormed

    def _normalize_trajectory_action(self, action, traj_distance_m):
        normed = action.clone()
        distances = traj_distance_m.to(device=normed.device, dtype=normed.dtype).view(-1)
        scale = self.trajectory_norm_target_distance / distances.clamp(min=self.trajectory_norm_eps)
        if normed.dim() == 3:
            scale = scale.view(-1, 1, 1)
        elif normed.dim() == 2:
            scale = scale.view(-1, 1)
        else:
            while scale.dim() < normed[..., 0:2].dim():
                scale = scale.unsqueeze(-1)
        normed[..., 0:2] = normed[..., 0:2] * scale
        if normed.shape[-1] >= 3:
            normed[..., 2] = normed[..., 2] / self.action_scale_theta
        return normed

    def _denormalize_trajectory_action(self, action, traj_distance_m):
        denormed = action.clone()
        distances = traj_distance_m.to(device=denormed.device, dtype=denormed.dtype).view(-1)
        scale = distances / max(self.trajectory_norm_target_distance, self.trajectory_norm_eps)
        if denormed.dim() == 3:
            scale = scale.view(-1, 1, 1)
        elif denormed.dim() == 2:
            scale = scale.view(-1, 1)
        else:
            while scale.dim() < denormed[..., 0:2].dim():
                scale = scale.unsqueeze(-1)
        denormed[..., 0:2] = denormed[..., 0:2] * scale
        if denormed.shape[-1] >= 3:
            denormed[..., 2] = denormed[..., 2] * self.action_scale_theta
        return denormed

    def _default_nogoal_distance(self, batch_size, device, dtype=torch.float32):
        distance = self.nogoal_front_distance * self.action_scale_xy
        return torch.full((batch_size,), distance, device=device, dtype=dtype)

    def _build_scale_features(self, traj_distance_m):
        d = traj_distance_m.to(device=self._device, dtype=torch.float32).view(-1)
        d = d.clamp(
            min=max(self.scale_condition_clamp_min_m, self.trajectory_norm_eps),
            max=self.scale_condition_clamp_max_m,
        )
        target = max(self.trajectory_norm_target_distance, self.trajectory_norm_eps)
        return torch.stack([d, torch.log(d), d.new_full(d.shape, target) / d, d / target], dim=-1)

    def _build_scale_token(self, traj_distance_m, like_token=None):
        if not self.enable_scale_condition_token:
            if like_token is not None:
                return like_token.new_zeros((like_token.shape[0], 0, like_token.shape[-1]))
            return torch.zeros((traj_distance_m.shape[0], 0, self.token_dim), device=self._device)
        token = self.scale_encoder(self._build_scale_features(traj_distance_m)).unsqueeze(1)
        if like_token is not None:
            token = token.to(device=like_token.device, dtype=like_token.dtype)
        return token

    def _apply_scale_rgbd_film(self, rgbd_embed, traj_distance_m):
        if not self.enable_scale_rgbd_film:
            return rgbd_embed
        feat = self._build_scale_features(traj_distance_m)
        batch_size = rgbd_embed.shape[0]
        if feat.shape[0] == 1 and batch_size > 1:
            feat = feat.expand(batch_size, -1)
        elif feat.shape[0] != batch_size:
            raise ValueError(
                f"scale batch size {feat.shape[0]} does not match rgbd batch size {batch_size}"
            )
        gamma_beta = self.scale_rgbd_film(feat).to(device=rgbd_embed.device, dtype=rgbd_embed.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        base = self.scale_rgbd_film_norm(rgbd_embed) if self.scale_rgbd_film_use_layernorm else rgbd_embed
        return rgbd_embed + self.scale_rgbd_film_alpha * (base * gamma.unsqueeze(1) + beta.unsqueeze(1))

    def _make_prior_tokens(self, tensor_prior, rgbd_embed):
        if self.use_prior_traj:
            prior_tokens = self.prior_encoder(tensor_prior)
            gate = self.visual_gate(rgbd_embed.mean(dim=1))
            return gate * prior_tokens
        return torch.zeros(
            tensor_prior.shape[0],
            self.n_prior_tokens,
            self.token_dim,
            device=rgbd_embed.device,
            dtype=rgbd_embed.dtype,
        )

    def predict_x0(self, noisy_actions, timestep, goal_embed, rgbd_embed, prior_embed, scale_embed=None):
        action_embeds = self.input_embed(noisy_actions)
        time_embeds = self.time_emb(timestep.to(self._device).view(-1)).unsqueeze(1)
        cond_batch = goal_embed.shape[0]
        if time_embeds.shape[0] == 1 and cond_batch > 1:
            time_embeds = time_embeds.expand(cond_batch, -1, -1)

        if scale_embed is None:
            scale_embed = self._build_scale_token(
                self._default_nogoal_distance(cond_batch, self._device),
                like_token=goal_embed,
            )

        cond_tokens = torch.cat(
            [time_embeds, scale_embed, goal_embed, goal_embed, goal_embed, rgbd_embed, prior_embed],
            dim=1,
        )
        cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
        cond_embedding = cond_embedding.repeat(action_embeds.shape[0] // cond_embedding.shape[0], 1, 1)

        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(
            tgt=input_embedding,
            memory=cond_embedding,
            tgt_mask=self.tgt_mask.to(self._device),
        )
        return self.action_head(self.layernorm(output))

    def predict_critic(self, predict_trajectory, rgbd_embed, scale_embed=None):
        repeat_factor = max(1, predict_trajectory.shape[0] // rgbd_embed.shape[0])
        repeat_rgbd_embed = rgbd_embed.repeat(repeat_factor, 1, 1)
        nogoal_embed = torch.zeros_like(repeat_rgbd_embed[:, 0:1])

        if scale_embed is None:
            scale_embed = self._build_scale_token(
                self._default_nogoal_distance(rgbd_embed.shape[0], rgbd_embed.device, rgbd_embed.dtype),
                like_token=rgbd_embed[:, 0:1],
            )
        repeat_scale_embed = scale_embed.repeat(repeat_factor, 1, 1)

        zero_prior = torch.zeros(
            repeat_rgbd_embed.shape[0],
            self.n_prior_tokens,
            self.token_dim,
            device=repeat_rgbd_embed.device,
            dtype=repeat_rgbd_embed.dtype,
        )
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_tokens = torch.cat(
            [
                nogoal_embed,
                repeat_scale_embed,
                nogoal_embed,
                nogoal_embed,
                nogoal_embed,
                repeat_rgbd_embed,
                zero_prior,
            ],
            dim=1,
        )
        cond_embeddings = cond_tokens + self.cond_pos_embed(cond_tokens)
        critic_output = self.decoder(
            tgt=action_embeddings,
            memory=cond_embeddings,
            memory_mask=self.cond_critic_mask.to(self._device),
        )
        return self.critic_head(self.layernorm(critic_output).mean(dim=1))[:, 0]

    def _apply_goal_consistency_score(self, critic_values, trajectories, goals, origins=None):
        if not self.enable_goal_consistency_score:
            return critic_values
        if goals.dim() == 3:
            goals = goals[:, -1, :]
        if origins is None:
            origins = torch.zeros_like(goals)
        elif origins.dim() == 3:
            origins = origins[:, -1, :]

        terminal_err = torch.norm(trajectories[:, -1, :2] - goals[:, :2], dim=-1)
        path_len = torch.norm(trajectories[:, 1:, :2] - trajectories[:, :-1, :2], dim=-1).sum(dim=-1)
        goal_dist = torch.norm(goals[:, :2] - origins[:, :2], dim=-1)
        penalty = (
            self.goal_consistency_terminal_weight * terminal_err
            + self.goal_consistency_path_weight * torch.relu(path_len - goal_dist)
        )
        return critic_values - penalty

    def _reshape_candidates(self, trajectory_flat, values_flat, batch_size, sample_num):
        trajectory = trajectory_flat.reshape(sample_num, batch_size, self.predict_size, 3).permute(1, 0, 2, 3)
        values = values_flat.reshape(sample_num, batch_size).permute(1, 0)
        return trajectory, values

    def _select_topk(self, trajectory, values, topk=2):
        topk = min(topk, trajectory.shape[1])
        sorted_pos = (-values).argsort(dim=1)
        sorted_neg = values.argsort(dim=1)
        batch_idx = torch.arange(trajectory.shape[0], device=trajectory.device).unsqueeze(1).expand(-1, topk)
        positive = trajectory[batch_idx, sorted_pos[:, :topk]]
        negative = trajectory[batch_idx, sorted_neg[:, :topk]]
        return positive, negative

    def _to_numpy_response(self, trajectory, values, positive, negative):
        return (
            trajectory.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
            positive.detach().cpu().numpy(),
            negative.detach().cpu().numpy(),
        )

    def _prepare_prior(self, prior_traj, batch_size, distance_for_norm=None):
        if prior_traj is None:
            return torch.zeros(batch_size, self.predict_size, 3, device=self._device)
        tensor_prior = torch.as_tensor(prior_traj, dtype=torch.float32, device=self._device)
        if self.enable_trajectory_normalization and distance_for_norm is not None:
            return self._normalize_trajectory_action(tensor_prior, distance_for_norm)
        return self._normalize_action(tensor_prior)

    def predict_pointgoal_action(self, goal_point, input_images, input_depths, prior_traj=None, theta_g=None, sample_num=16):
        with torch.no_grad():
            tensor_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self._device)
            B = tensor_goal.shape[0]
            goal_distance_m = torch.norm(tensor_goal[:, :2], dim=-1)
            distance_for_norm = goal_distance_m.clamp(min=self.trajectory_norm_min_distance_m)
            arrived_mask = goal_distance_m < self.trajectory_norm_min_distance_m

            if theta_g is not None:
                tensor_theta = torch.as_tensor(theta_g, dtype=torch.float32, device=self._device)
            else:
                tensor_theta = torch.atan2(tensor_goal[:, 1], tensor_goal[:, 0])

            if self.enable_trajectory_normalization:
                tensor_goal_n = self._normalize_trajectory_action(tensor_goal, distance_for_norm)
            else:
                tensor_goal_n = self._normalize_action(tensor_goal)

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            goal_embed = self.point_encoder(tensor_goal_n).unsqueeze(1)
            scale_embed = self._build_scale_token(distance_for_norm, like_token=goal_embed)
            rgbd_embed = self._apply_scale_rgbd_film(rgbd_embed, distance_for_norm)

            tensor_prior = self._prepare_prior(prior_traj, B, distance_for_norm)
            gated_prior = self._make_prior_tokens(tensor_prior, rgbd_embed)

            origin = torch.zeros_like(tensor_goal_n)
            naction = self.bridge_scheduler.sample_initial_noise_ordered(
                goal=tensor_goal_n.repeat(sample_num, 1),
                origin=origin.repeat(sample_num, 1),
                shape=(sample_num * B, self.predict_size, 3),
                device=self._device,
            )

            self.bridge_scheduler.set_timesteps(self.num_inference_timesteps)
            goal_repeated = tensor_goal_n.repeat(sample_num, 1)
            origin_repeated = origin.repeat(sample_num, 1)
            theta_repeated = tensor_theta.repeat(sample_num)
            if self.enable_trajectory_normalization:
                distance_repeated = distance_for_norm.repeat(sample_num)

            for k in self.bridge_scheduler.timesteps:
                x0_pred = self.predict_x0(
                    naction,
                    k.to(self._device).unsqueeze(0),
                    goal_embed,
                    rgbd_embed,
                    gated_prior,
                    scale_embed,
                )
                naction = self.bridge_scheduler.step_trajectory(
                    x0_pred,
                    naction,
                    k,
                    goal=goal_repeated,
                    theta_g=theta_repeated,
                    origin=origin_repeated,
                    mode="pointgoal",
                )

            critic_values = self.predict_critic(naction, rgbd_embed, scale_embed)
            score_values = self._apply_goal_consistency_score(
                critic_values, naction, goal_repeated, origin_repeated
            )

            if self.enable_trajectory_normalization:
                trajectory_flat = self._denormalize_trajectory_action(naction, distance_repeated)
            else:
                trajectory_flat = self._denormalize_action(naction)

            trajectory, values = self._reshape_candidates(trajectory_flat, score_values, B, sample_num)
            if arrived_mask.any():
                trajectory[arrived_mask] = 0.0
            positive, negative = self._select_topk(trajectory, values)
            return self._to_numpy_response(trajectory, values, positive, negative)

    def predict_nogoal_action(self, input_images, input_depths, prior_traj=None, sample_num=16):
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            B = rgbd_embed.shape[0]
            nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
            nogoal_distance_m = self._default_nogoal_distance(B, self._device, dtype=rgbd_embed.dtype)
            scale_embed = self._build_scale_token(nogoal_distance_m, like_token=nogoal_embed)
            rgbd_embed = self._apply_scale_rgbd_film(rgbd_embed, nogoal_distance_m)

            tensor_prior = self._prepare_prior(prior_traj, B)
            gated_prior = self._make_prior_tokens(tensor_prior, rgbd_embed)

            naction = self.bridge_scheduler.sample_initial_noise_nogoal(
                (sample_num * B, self.predict_size, 3),
                self._device,
                dtype=rgbd_embed.dtype,
            )
            self.bridge_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.bridge_scheduler.timesteps:
                x0_pred = self.predict_x0(
                    naction,
                    k.to(self._device).unsqueeze(0),
                    nogoal_embed,
                    rgbd_embed,
                    gated_prior,
                    scale_embed,
                )
                naction = self.bridge_scheduler.step_trajectory(x0_pred, naction, k, mode="nogoal")

            critic_values = self.predict_critic(naction, rgbd_embed, scale_embed)
            trajectory_flat = self._denormalize_action(naction)
            trajectory, values = self._reshape_candidates(trajectory_flat, critic_values, B, sample_num)
            traj_length = torch.norm(trajectory[:, :, -1, 0:2], dim=-1)
            values = values.clone()
            values[traj_length < 1.0] -= 10.0
            positive, negative = self._select_topk(trajectory, values)
            return self._to_numpy_response(trajectory, values, positive, negative)

    def _predict_with_goal_embed(self, goal_embed, rgbd_embed, prior_traj=None, sample_num=16):
        B = rgbd_embed.shape[0]
        nogoal_distance_m = self._default_nogoal_distance(B, self._device, dtype=rgbd_embed.dtype)
        scale_embed = self._build_scale_token(nogoal_distance_m, like_token=goal_embed)
        rgbd_embed = self._apply_scale_rgbd_film(rgbd_embed, nogoal_distance_m)
        tensor_prior = self._prepare_prior(prior_traj, B)
        gated_prior = self._make_prior_tokens(tensor_prior, rgbd_embed)

        naction = self.bridge_scheduler.sample_initial_noise_nogoal(
            (sample_num * B, self.predict_size, 3),
            self._device,
            dtype=rgbd_embed.dtype,
        )
        self.bridge_scheduler.set_timesteps(self.num_inference_timesteps)
        for k in self.bridge_scheduler.timesteps:
            x0_pred = self.predict_x0(
                naction,
                k.to(self._device).unsqueeze(0),
                goal_embed,
                rgbd_embed,
                gated_prior,
                scale_embed,
            )
            naction = self.bridge_scheduler.step_trajectory(x0_pred, naction, k, mode="nogoal")

        critic_values = self.predict_critic(naction, rgbd_embed, scale_embed)
        trajectory_flat = self._denormalize_action(naction)
        trajectory, values = self._reshape_candidates(trajectory_flat, critic_values, B, sample_num)
        positive, negative = self._select_topk(trajectory, values)
        return self._to_numpy_response(trajectory, values, positive, negative)

    def predict_imagegoal_action(self, goal_image, input_images, input_depths, prior_traj=None, sample_num=16):
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            imagegoal_embed = self.image_encoder(
                np.concatenate((goal_image, input_images[:, -1]), axis=-1)
            ).unsqueeze(1)
            return self._predict_with_goal_embed(imagegoal_embed, rgbd_embed, prior_traj, sample_num)

    def predict_pixelgoal_action(self, goal_image, input_images, input_depths, prior_traj=None, sample_num=16):
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            pixelgoal_embed = self.pixel_encoder(
                np.concatenate((goal_image[:, :, :, None], input_images[:, -1]), axis=-1)
            ).unsqueeze(1)
            return self._predict_with_goal_embed(pixelgoal_embed, rgbd_embed, prior_traj, sample_num)
