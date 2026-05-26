"""Bridge-DP trajectory-time bridge scheduler.

This standalone scheduler mirrors the current InternNav Bridge-DP semantics
without depending on the InternNav package.  Unlike the old endpoint-only
bridge, each predicted control point has an ordered trajectory-time position:
waypoint i represents tau=(i+1)/T along the generated curve.
"""

from typing import Optional

import torch


class BridgeScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 100,
        sigma_base: float = 1.0,
        sigma_goal: float = 0.5,
        sigma_floor: Optional[float] = None,
        nogoal_front_distance: float = 0.8,
        nogoal_sigma_start: float = 0.03,
        nogoal_sigma_x_end: float = 0.35,
        nogoal_sigma_y_end: float = 0.80,
        nogoal_sigma_theta_end: float = 0.60,
        nogoal_sigma_power: float = 2.0,
        bridge_scale_invariant_sigma: bool = False,
        bridge_anisotropic_xy: bool = True,
        bridge_normal_sigma_ratio: float = 0.25,
        bridge_tangent_sigma_ratio: float = 0.03,
        bridge_theta_sigma_ratio: float = 0.05,
        bridge_envelope_frontload: float = 0.0,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.sigma_base = sigma_base
        self.sigma_goal = sigma_goal
        self.sigma_floor = sigma_goal if sigma_floor is None else sigma_floor
        self.nogoal_front_distance = nogoal_front_distance
        self.nogoal_sigma_start = nogoal_sigma_start
        self.nogoal_sigma_x_end = nogoal_sigma_x_end
        self.nogoal_sigma_y_end = nogoal_sigma_y_end
        self.nogoal_sigma_theta_end = nogoal_sigma_theta_end
        self.nogoal_sigma_power = nogoal_sigma_power
        self.bridge_scale_invariant_sigma = bridge_scale_invariant_sigma
        self.bridge_anisotropic_xy = bridge_anisotropic_xy
        self.bridge_normal_sigma_ratio = bridge_normal_sigma_ratio
        self.bridge_tangent_sigma_ratio = bridge_tangent_sigma_ratio
        self.bridge_theta_sigma_ratio = bridge_theta_sigma_ratio
        self.bridge_envelope_frontload = float(bridge_envelope_frontload)
        if self.bridge_envelope_frontload < 0.0:
            raise ValueError("bridge_envelope_frontload must be non-negative")
        self._timesteps: Optional[torch.Tensor] = None

    @property
    def timesteps(self) -> torch.Tensor:
        if self._timesteps is None:
            self.set_timesteps(self.num_train_timesteps)
        return self._timesteps

    @property
    def config(self):
        return _SchedulerConfig(self.num_train_timesteps)

    def set_timesteps(self, num_inference_steps: int) -> None:
        self._timesteps = torch.arange(num_inference_steps - 1, -1, -1).long()

    def _normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (timesteps.float() + 1.0) / self.num_train_timesteps

    def direction_adaptive_exponent(self, theta_g: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.3 * torch.cos(theta_g)

    def variance(self, t_norm: torch.Tensor, theta_g: torch.Tensor) -> torch.Tensor:
        p = self.direction_adaptive_exponent(theta_g)
        t_prod = (t_norm * (1.0 - t_norm)).clamp(min=1e-8)
        bridge_term = (self.sigma_base ** 2) * (t_prod ** p)
        elastic_term = (t_norm ** 2) * (self.sigma_goal ** 2)
        return bridge_term + elastic_term

    def std(self, t_norm: torch.Tensor, theta_g: torch.Tensor) -> torch.Tensor:
        return self.variance(t_norm, theta_g).sqrt()

    def trajectory_time(
        self,
        predict_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.linspace(
            1.0 / float(predict_size),
            1.0,
            predict_size,
            device=device,
            dtype=dtype,
        ).view(1, predict_size, 1)

    def trajectory_time_envelopes(
        self,
        tau: torch.Tensor,
        p: torch.Tensor,
    ):
        t_prod = (tau * (1.0 - tau)).clamp(min=0.0)
        symmetric = (t_prod / 0.25).clamp(min=0.0).pow(p)
        if self.bridge_envelope_frontload == 0.0:
            return symmetric, symmetric

        q = 1.0 + self.bridge_envelope_frontload
        turn_peak = (q**q) / ((q + 1.0) ** (q + 1.0))
        turn_prod = (tau * (1.0 - tau).pow(q)).clamp(min=0.0)
        turn = (turn_prod / turn_peak).clamp(min=0.0).pow(p)
        return symmetric, turn

    def _batch_endpoint(
        self,
        value: Optional[torch.Tensor],
        batch_size: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dim, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if value.dim() == 3:
            value = value[:, -1, :]
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        return value

    def bridge_mean_ordered(
        self,
        goal: torch.Tensor,
        origin: Optional[torch.Tensor],
        shape: tuple,
    ) -> torch.Tensor:
        B, T_pred, dim = shape
        device = goal.device
        dtype = goal.dtype
        goal = self._batch_endpoint(goal, B, dim, device, dtype)
        origin = self._batch_endpoint(origin, B, dim, device, dtype)
        tau = self.trajectory_time(T_pred, device, dtype)
        return origin.unsqueeze(1) + tau * (goal.unsqueeze(1) - origin.unsqueeze(1))

    def bridge_mean_nogoal(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        B, T_pred, dim = shape
        tau = self.trajectory_time(T_pred, device, dtype)
        front = torch.zeros(B, dim, device=device, dtype=dtype)
        front[:, 0] = self.nogoal_front_distance
        return tau * front.unsqueeze(1)

    def pointgoal_noise_params(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
    ):
        B, _, dim = shape
        goal_b = self._batch_endpoint(goal, B, dim, device, dtype)
        origin_b = self._batch_endpoint(origin, B, dim, device, dtype)
        mu = self.bridge_mean_ordered(goal_b, origin_b, shape)

        if dim >= 2:
            vec_xy = goal_b[:, :2] - origin_b[:, :2]
        else:
            vec_xy = torch.zeros(B, 2, device=device, dtype=dtype)
        dist = torch.norm(vec_xy, dim=-1)

        if theta_g is None:
            theta_g = torch.atan2(vec_xy[:, 1], vec_xy[:, 0])
        else:
            theta_g = theta_g.to(device=device, dtype=dtype).view(-1)
            if theta_g.shape[0] == 1 and B > 1:
                theta_g = theta_g.expand(B)

        dist_safe = dist.clamp(min=1e-6)
        tangent = vec_xy / dist_safe.view(B, 1)
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
        zero_dist = dist <= 1e-6
        if zero_dist.any():
            default_tangent = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
            default_normal = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
            tangent = torch.where(zero_dist.view(B, 1), default_tangent.view(1, 2), tangent)
            normal = torch.where(zero_dist.view(B, 1), default_normal.view(1, 2), normal)

        tau = self.trajectory_time(shape[1], device, dtype)
        p = self.direction_adaptive_exponent(theta_g).view(B, 1, 1)
        shape_tau, turn_shape_tau = self.trajectory_time_envelopes(tau, p)
        dist_scale = dist.view(B, 1, 1)

        sigma_normal = dist_scale * self.bridge_normal_sigma_ratio * turn_shape_tau
        sigma_tangent = dist_scale * self.bridge_tangent_sigma_ratio * shape_tau
        sigma_theta = dist_scale * self.bridge_theta_sigma_ratio * turn_shape_tau
        if not self.bridge_anisotropic_xy:
            sigma_tangent = sigma_normal

        sigma_diag = torch.zeros(shape, device=device, dtype=dtype)
        if dim >= 1:
            sigma_diag[..., 0:1] = sigma_tangent
        if dim >= 2:
            sigma_diag[..., 1:2] = sigma_normal
        if dim >= 3:
            sigma_diag[..., 2:3] = sigma_theta

        return mu, tangent, normal, sigma_tangent, sigma_normal, sigma_theta, sigma_diag

    def sample_pointgoal_bridge_noise(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ):
        B, _, dim = shape
        mu, tangent, normal, sigma_tangent, sigma_normal, sigma_theta, sigma_diag = (
            self.pointgoal_noise_params(shape, device, dtype, goal, theta_g, origin)
        )
        eps = torch.randn(shape, device=device, dtype=dtype) if noise is None else noise.to(device=device, dtype=dtype)

        bridge_noise = torch.zeros(shape, device=device, dtype=dtype)
        if dim >= 2:
            xy_noise = (
                tangent.view(B, 1, 2) * sigma_tangent * eps[..., 0:1]
                + normal.view(B, 1, 2) * sigma_normal * eps[..., 1:2]
            )
            bridge_noise[..., :2] = xy_noise
        elif dim == 1:
            bridge_noise[..., 0:1] = sigma_tangent * eps[..., 0:1]

        if dim >= 3:
            bridge_noise[..., 2:3] = sigma_theta * eps[..., 2:3]

        return bridge_noise, mu, sigma_diag

    def trajectory_std(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        mode: str = "pointgoal",
    ) -> torch.Tensor:
        B, T_pred, dim = shape
        tau = self.trajectory_time(T_pred, device, dtype)

        if mode == "nogoal":
            grow = tau.pow(self.nogoal_sigma_power)
            end = torch.tensor(
                [self.nogoal_sigma_x_end, self.nogoal_sigma_y_end, self.nogoal_sigma_theta_end],
                device=device,
                dtype=dtype,
            ).view(1, 1, 3)
            if dim != 3:
                end = end[..., :dim]
            start = torch.full_like(end, self.nogoal_sigma_start)
            sigma = start + (end - start) * grow
            return sigma.expand(B, T_pred, dim).clamp(min=self.sigma_floor)

        if self.bridge_scale_invariant_sigma:
            _, _, _, _, _, _, sigma_diag = self.pointgoal_noise_params(
                shape, device, dtype, goal=goal, theta_g=theta_g, origin=origin
            )
            return sigma_diag

        if theta_g is None:
            goal_b = self._batch_endpoint(goal, B, dim, device, dtype)
            origin_b = self._batch_endpoint(origin, B, dim, device, dtype)
            vec = goal_b - origin_b
            theta_g = torch.atan2(vec[:, 1], vec[:, 0])
        else:
            theta_g = theta_g.to(device=device, dtype=dtype).view(-1)
            if theta_g.shape[0] == 1 and B > 1:
                theta_g = theta_g.expand(B)

        p = self.direction_adaptive_exponent(theta_g).view(B, 1, 1)
        t_prod = (tau * (1.0 - tau)).clamp(min=0.0)
        var = (self.sigma_base ** 2) * (t_prod ** p) + (self.sigma_floor ** 2)
        return var.sqrt().expand(B, T_pred, dim)

    def add_noise_trajectory(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        mode: str = "pointgoal",
        noise: Optional[torch.Tensor] = None,
    ):
        B, _, dim = x0.shape
        device = x0.device
        dtype = x0.dtype
        s_norm = self._normalized_time(timesteps.to(device)).to(dtype=dtype).view(-1, 1, 1)
        if s_norm.shape[0] == 1 and B > 1:
            s_norm = s_norm.expand(B, 1, 1)

        if mode == "nogoal":
            noise = torch.randn_like(x0) if noise is None else noise
            mu = self.bridge_mean_nogoal(x0.shape, device, dtype)
            sigma = self.trajectory_std(x0.shape, device, dtype, mode="nogoal")
            noisy = (1.0 - s_norm) * x0 + s_norm * mu + s_norm.clamp(min=1e-6) * sigma * noise
            return noisy, noise, mu, sigma, s_norm

        if origin is None:
            origin = torch.zeros(B, dim, device=device, dtype=dtype)

        if self.bridge_scale_invariant_sigma:
            bridge_noise, mu, sigma = self.sample_pointgoal_bridge_noise(
                x0.shape, device, dtype, goal=goal, theta_g=theta_g, origin=origin, noise=noise
            )
            noisy = (1.0 - s_norm) * x0 + s_norm * mu + s_norm.clamp(min=1e-6) * bridge_noise
            return noisy, bridge_noise, mu, sigma, s_norm

        noise = torch.randn_like(x0) if noise is None else noise
        mu = self.bridge_mean_ordered(goal, origin, x0.shape)
        sigma = self.trajectory_std(
            x0.shape, device, dtype, goal=goal, theta_g=theta_g, origin=origin, mode="pointgoal"
        )
        noisy = (1.0 - s_norm) * x0 + s_norm * mu + s_norm.clamp(min=1e-6) * sigma * noise
        return noisy, noise, mu, sigma, s_norm

    def add_noise(
        self,
        x0: torch.Tensor,
        goal: torch.Tensor,
        theta_g: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noisy, _, _, _, _ = self.add_noise_trajectory(
            x0, timesteps, goal=goal, theta_g=theta_g, mode="pointgoal", noise=noise
        )
        return noisy

    def step_trajectory(
        self,
        x0_pred: torch.Tensor,
        x_s: torch.Tensor,
        timestep: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        theta_g: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        mode: str = "pointgoal",
        eta: float = 0.0,
    ) -> torch.Tensor:
        device = x_s.device
        dtype = x_s.dtype
        timestep = timestep.to(device)
        s_norm = self._normalized_time(timestep).to(dtype=dtype)
        dt = 1.0 / self.num_train_timesteps

        if s_norm.numel() == 1 and s_norm.item() <= dt + 1e-6:
            return x0_pred

        B, _, dim = x_s.shape
        s = s_norm.view(-1, 1, 1)
        if s.shape[0] == 1 and B > 1:
            s = s.expand(B, 1, 1)
        s_prev = (s - dt).clamp(min=0.0)

        if mode == "nogoal":
            mu = self.bridge_mean_nogoal(x_s.shape, device, dtype)
            sigma = self.trajectory_std(x_s.shape, device, dtype, mode="nogoal")
        else:
            if origin is None:
                origin = torch.zeros(B, dim, device=device, dtype=dtype)
            mu = self.bridge_mean_ordered(goal, origin, x_s.shape)
            sigma = None
            if not self.bridge_scale_invariant_sigma:
                sigma = self.trajectory_std(
                    x_s.shape, device, dtype, goal=goal, theta_g=theta_g, origin=origin, mode="pointgoal"
                )

        mean_s = (1.0 - s) * x0_pred + s * mu
        residual = x_s - mean_s
        x_prev = (1.0 - s_prev) * x0_pred + s_prev * mu + (s_prev / s.clamp(min=1e-6)) * residual

        if eta > 0.0:
            if mode == "pointgoal" and self.bridge_scale_invariant_sigma:
                extra_noise, _, _ = self.sample_pointgoal_bridge_noise(
                    x_s.shape, device, dtype, goal=goal, theta_g=theta_g, origin=origin
                )
                x_prev = x_prev + eta * s_prev * extra_noise
            else:
                if sigma is None:
                    sigma = self.trajectory_std(
                        x_s.shape, device, dtype, goal=goal, theta_g=theta_g, origin=origin, mode=mode
                    )
                x_prev = x_prev + eta * s_prev * sigma * torch.randn_like(x_prev)

        return x_prev

    def step(
        self,
        x0_pred: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        goal: torch.Tensor,
        theta_g: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        return self.step_trajectory(
            x0_pred, x_t, timestep, goal=goal, theta_g=theta_g, mode="pointgoal", eta=eta
        )

    def sample_initial_noise(
        self,
        goal: torch.Tensor,
        shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        noise = torch.randn(shape, device=device)
        if goal.dim() == 2:
            goal = goal.unsqueeze(1)
        goal = goal.expand(shape)
        return goal + self.sigma_goal * noise

    def sample_initial_noise_ordered(
        self,
        goal: torch.Tensor,
        origin: torch.Tensor,
        shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        B, _, _ = shape
        if origin.dim() == 1:
            origin = origin.unsqueeze(0).expand(B, -1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0).expand(B, -1)

        if self.bridge_scale_invariant_sigma:
            bridge_noise, mu, _ = self.sample_pointgoal_bridge_noise(
                shape, device, goal.dtype, goal=goal, origin=origin
            )
            return mu + bridge_noise

        mu = self.bridge_mean_ordered(goal, origin, shape)
        goal_vec = goal - origin
        theta_g = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])
        sigma_per_point = self.trajectory_std(
            shape, device, goal.dtype, goal=goal, theta_g=theta_g, origin=origin, mode="pointgoal"
        )
        noise = torch.randn(shape, device=device, dtype=goal.dtype)
        return mu + sigma_per_point * noise

    def sample_initial_noise_nogoal(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        mu = self.bridge_mean_nogoal(shape, device, dtype)
        sigma = self.trajectory_std(shape, device, dtype, mode="nogoal")
        return mu + sigma * torch.randn(shape, device=device, dtype=dtype)


class _SchedulerConfig:
    def __init__(self, num_train_timesteps: int) -> None:
        self.num_train_timesteps = num_train_timesteps
