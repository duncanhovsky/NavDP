"""先验轨迹编码器与视觉门控模块。

Bridge-DP 独有的两个组件：

1. PriorEncoder: 将上一帧预测的先验轨迹编码为 N_p 个 token。
2. VisualGate: 基于当前视觉观测生成门控值 G ∈ (0, 1)，控制先验注入强度。
"""

import torch
import torch.nn as nn


class PriorEncoder(nn.Module):
    """轻量 Transformer 编码器，将先验轨迹编码为固定数量的 token。

    架构：线性投影 → 可学习位置编码 → 2 层 Transformer Encoder → 可学习查询压缩

    Attributes:
        n_prior_tokens: 输出 token 数量 N_p。
        embed_dim: token 嵌入维度 d。
    """

    def __init__(
        self,
        embed_dim: int = 384,
        n_prior_tokens: int = 4,
        action_dim: int = 3,
        num_layers: int = 2,
        nhead: int = 4,
        max_traj_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_prior_tokens = n_prior_tokens

        # 轨迹点线性投影：(x, y, θ) → d 维
        self.input_proj = nn.Linear(action_dim, embed_dim)

        # 可学习位置编码
        self.pos_embed = nn.Embedding(max_traj_len, embed_dim)

        # 2 层 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 可学习查询 token，用于将 T 个输入压缩为 N_p 个输出
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_prior_tokens, embed_dim) * 0.02
        )

        # 交叉注意力：query_tokens attend to encoded trajectory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, prior_traj: torch.Tensor) -> torch.Tensor:
        """编码先验轨迹为 N_p 个 token。

        Args:
            prior_traj: 先验轨迹 (B, T, 3)。

        Returns:
            先验 token (B, N_p, embed_dim)。
        """
        B, T, _ = prior_traj.shape

        # 线性投影 + 位置编码
        positions = torch.arange(T, device=prior_traj.device)
        h = self.input_proj(prior_traj) + self.pos_embed(positions).unsqueeze(0)

        # Transformer 自注意力编码
        h = self.transformer(h)  # (B, T, d)

        # 可学习查询压缩：T → N_p
        queries = self.query_tokens.expand(B, -1, -1)  # (B, N_p, d)

        # 交叉注意力
        attn_out, _ = self.cross_attn(
            query=queries,
            key=h,
            value=h,
        )
        queries = self.cross_norm(queries + attn_out)

        # FFN
        ffn_out = self.cross_ffn(queries)
        output = self.ffn_norm(queries + ffn_out)

        return output  # (B, N_p, d)


class VisualGate(nn.Module):
    """视觉门控模块，基于当前视觉特征生成先验可信度门控值。

    G = σ(MLP(h_vis)) ∈ (0, 1)

    场景变化大时 G→0（忽略过时先验），场景稳定时 G→1。

    Attributes:
        gate_dim: 输入视觉特征维度。
    """

    def __init__(
        self,
        gate_dim: int = 384,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 初始化偏置为正值，使训练初期 G ≈ σ(1.0) ≈ 0.73
        nn.init.constant_(self.mlp[-1].bias, 1.0)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """计算门控值。

        Args:
            visual_features: 全局视觉特征 (B, gate_dim)。

        Returns:
            门控值 G (B, 1, 1)。
        """
        gate = torch.sigmoid(self.mlp(visual_features))  # (B, 1)
        return gate.unsqueeze(-1)  # (B, 1, 1)
