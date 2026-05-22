"""方向自适应弹性布朗桥调度器。

本模块实现 Bridge-DP 的核心噪声调度逻辑，替代 NavDP 使用的
``diffusers.DDPMScheduler``。

前向加噪（训练）::

    x_t = (1-t) * x_0 + t * g + σ(t; θ_g) * ε

方差调度::

    σ²(t; θ_g) = σ²_base · [t(1-t)]^{p(θ_g)} + t² · σ²_goal
    p(θ_g) = 0.5 + 0.3 · cos(θ_g)

反向去噪（推理，DDIM 确定性采样）::

    x_{t-Δt} = (t-Δt)/t · x_t + Δt/t · x̂_0
"""

from typing import Optional

import torch


class BridgeScheduler:
    """方向自适应弹性布朗桥噪声调度器。

    与 ``diffusers.DDPMScheduler`` 接口对齐，但内部实现为布朗桥 SDE。

    Attributes:
        num_train_timesteps: 训练扩散总步数（离散化步数），默认 10。
        sigma_base: 数据驱动的基础方差常数。
        sigma_goal: 弹性尾端松弛方差。
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        sigma_base: float = 1.0,
        sigma_goal: float = 0.1,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.sigma_base = sigma_base
        self.sigma_goal = sigma_goal
        self._timesteps: Optional[torch.Tensor] = None

    @property
    def timesteps(self) -> torch.Tensor:
        """返回推理时间步序列（从 T-1 到 0）。"""
        if self._timesteps is None:
            self.set_timesteps(self.num_train_timesteps)
        return self._timesteps

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时的时间步序列。

        Args:
            num_inference_steps: 推理步数。
        """
        self._timesteps = torch.arange(num_inference_steps - 1, -1, -1).long()

    def _normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        """将离散时间步转换为归一化连续时间 t ∈ (0, 1]。

        离散步 k ∈ {0, ..., T-1} 映射为 t = (k+1) / T。
        """
        return (timesteps.float() + 1.0) / self.num_train_timesteps

    def direction_adaptive_exponent(self, theta_g: torch.Tensor) -> torch.Tensor:
        """计算方向自适应指数 p(θ_g) = 0.5 + 0.3 · cos(θ_g)。"""
        return 0.5 + 0.3 * torch.cos(theta_g)

    def variance(
        self,
        t_norm: torch.Tensor,
        theta_g: torch.Tensor,
    ) -> torch.Tensor:
        """计算方向自适应桥方差 σ²(t; θ_g)。

        σ²(t; θ_g) = σ²_base · [t(1-t)]^{p(θ_g)} + t² · σ²_goal
        """
        p = self.direction_adaptive_exponent(theta_g)
        t_prod = (t_norm * (1.0 - t_norm)).clamp(min=1e-8)
        bridge_term = (self.sigma_base ** 2) * (t_prod ** p)
        elastic_term = (t_norm ** 2) * (self.sigma_goal ** 2)
        return bridge_term + elastic_term

    def std(
        self,
        t_norm: torch.Tensor,
        theta_g: torch.Tensor,
    ) -> torch.Tensor:
        """计算标准差 σ(t; θ_g) = √(σ²(t; θ_g))。"""
        return self.variance(t_norm, theta_g).sqrt()

    def add_noise(
        self,
        x0: torch.Tensor,
        goal: torch.Tensor,
        theta_g: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """布朗桥前向加噪：x_t = (1-t)·x_0 + t·g + σ(t; θ_g)·ε。

        Args:
            x0: 干净轨迹 (B, T_pred, 3)。
            goal: 目标位置 (B, 3) 或 (B, 1, 3)。
            theta_g: 目标方位角 (B,)。
            timesteps: 离散时间步 (B,)。
            noise: 可选的预生成高斯噪声。

        Returns:
            含噪轨迹 x_t (B, T_pred, 3)。
        """
        if noise is None:
            noise = torch.randn_like(x0)

        t_norm = self._normalized_time(timesteps).view(-1, 1, 1).to(x0.device)
        theta_g_expanded = theta_g.view(-1, 1, 1)

        if goal.dim() == 2:
            goal = goal.unsqueeze(1)
        goal = goal.expand_as(x0)

        bridge_mean = (1.0 - t_norm) * x0 + t_norm * goal
        sigma = self.std(t_norm, theta_g_expanded)

        return bridge_mean + sigma * noise

    def step(
        self,
        x0_pred: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        goal: torch.Tensor,
        theta_g: torch.Tensor,
    ) -> torch.Tensor:
        """布朗桥 DDIM 确定性反向去噪一步。

        x_{t-Δt} = (t-Δt)/t · x_t + Δt/t · x̂_0

        保留 x_t，轨迹形状灵活（可绕障），起终点约束由训练时的桥端点隐式保证。
        """
        t_norm = self._normalized_time(timestep).float().to(x_t.device)
        dt = 1.0 / self.num_train_timesteps

        if t_norm.item() <= dt + 1e-6:
            return x0_pred

        t_prev = t_norm - dt
        coeff_xt = t_prev / t_norm
        coeff_x0 = dt / t_norm

        while coeff_xt.dim() < x_t.dim():
            coeff_xt = coeff_xt.unsqueeze(-1)
            coeff_x0 = coeff_x0.unsqueeze(-1)

        return coeff_xt * x_t + coeff_x0 * x0_pred

    def sample_initial_noise(
        self,
        goal: torch.Tensor,
        shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """从目标附近采样推理初始噪声 x_T ~ N(g, σ²_goal · I)。

        Args:
            goal: 目标位置 (B, 3) 或 (B, 1, 3)。
            shape: 输出形状 (B, T_pred, 3)。
            device: 计算设备。

        Returns:
            初始含噪轨迹 x_T (B, T_pred, 3)。
        """
        noise = torch.randn(shape, device=device)

        if goal.dim() == 2:
            goal = goal.unsqueeze(1)
        goal = goal.expand(shape)

        return goal + self.sigma_goal * noise

    @property
    def config(self):
        """兼容 DDPMScheduler 的 config 访问模式。"""
        return _SchedulerConfig(self.num_train_timesteps)


class _SchedulerConfig:
    """轻量 config 容器，模仿 diffusers SchedulerConfig 的属性访问。"""

    def __init__(self, num_train_timesteps: int) -> None:
        self.num_train_timesteps = num_train_timesteps
