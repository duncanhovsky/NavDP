"""Bridge-DP 视觉编码器。

独立实现，与 NavDP baseline 完全解耦。
复用预训练的 DepthAnything V2 编码器（共享的预训练权重）。

架构与 NavDP 视觉编码器一致：
- RGBD 双流编码器：DINOv2 (RGB) + DepthAnything (Depth) → Transformer 压缩
- ImageGoal 编码器：6 通道输入（goal+current）→ DINOv2
- PixelGoal 编码器：4 通道输入（mask+current）→ DINOv2
- 辅助模块：SinusoidalPosEmb, LearnablePositionalEncoding
"""

import math

import torch
import torch.nn as nn

from depth_anything.depth_anything_v2.dpt import DepthAnythingV2


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码，用于扩散时间步嵌入。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码。"""

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_encoding = self.position_embedding(position_ids)
        return position_encoding


class BridgeDP_RGBD_Backbone(nn.Module):
    """RGBD 双流视觉编码器。

    使用预训练的 DINOv2 (冻结) 编码 RGB 图像，
    使用可训练的 DepthAnything 编码深度图像，
    通过 Transformer Decoder 将 (memory_size+1)*256 个 token
    压缩为 memory_size*16 个 token。
    """

    def __init__(self,
                 image_size=224,
                 embed_size=384,
                 memory_size=8,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size

        model_configs = {
            'vits': {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384],
            }
        }

        # RGB 流：冻结的 DINOv2 预训练模型
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.rgb_model.eval()
        self.preprocess_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        )
        self.preprocess_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        )

        # Depth 流：可训练的 DepthAnything
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()

        # Transformer Decoder 压缩
        self.former_query = LearnablePositionalEncoding(384, self.memory_size * 16)
        self.former_pe = LearnablePositionalEncoding(384, (self.memory_size + 1) * 256)
        self.former_net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(384, 8, batch_first=True), 2
        )
        self.project_layer = nn.Linear(384, embed_size)

    def forward(self, images, depths):
        with torch.no_grad():
            if len(images.shape) == 4:
                tensor_images = torch.as_tensor(
                    images, dtype=torch.float32, device=self.device
                ).permute(0, 3, 1, 2)
                tensor_images = tensor_images.reshape(
                    -1, 3, self.image_size, self.image_size
                )
                tensor_norm_images = (
                    tensor_images
                    - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)
                ) / self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(
                    tensor_norm_images
                )[0]
            elif len(images.shape) == 5:
                tensor_images = torch.as_tensor(
                    images, dtype=torch.float32, device=self.device
                ).permute(0, 1, 4, 2, 3)
                B, T, C, H, W = tensor_images.shape
                tensor_images = tensor_images.reshape(
                    -1, 3, self.image_size, self.image_size
                )
                tensor_norm_images = (
                    tensor_images
                    - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)
                ) / self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(
                    tensor_norm_images
                )[0].reshape(B, T * 256, -1)

        if len(depths.shape) == 4:
            tensor_depths = torch.as_tensor(
                depths, dtype=torch.float32, device=self.device
            ).permute(0, 3, 1, 2)
            tensor_depths = tensor_depths.reshape(
                -1, 1, self.image_size, self.image_size
            )
            tensor_depths = torch.concat(
                [tensor_depths, tensor_depths, tensor_depths], dim=1
            )
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
        elif len(depths.shape) == 5:
            tensor_depths = torch.as_tensor(
                depths, dtype=torch.float32, device=self.device
            ).permute(0, 1, 4, 2, 3)
            B, T, C, H, W = tensor_depths.shape
            tensor_depths = tensor_depths.reshape(
                -1, 1, self.image_size, self.image_size
            )
            tensor_depths = torch.concat(
                [tensor_depths, tensor_depths, tensor_depths], dim=1
            )
            depth_token = self.depth_model.get_intermediate_layers(
                tensor_depths
            )[0].reshape(B, T * 256, -1)

        former_token = (
            torch.concat((image_token, depth_token), dim=1)
            + self.former_pe(torch.concat((image_token, depth_token), dim=1))
        )
        former_query = self.former_query(
            torch.zeros(
                (image_token.shape[0], self.memory_size * 16, 384),
                device=self.device,
            )
        )
        memory_token = self.former_net(former_query, former_token)
        memory_token = self.project_layer(memory_token)
        return memory_token


class BridgeDP_ImageGoal_Backbone(nn.Module):
    """ImageGoal 编码器：6 通道（goal image + current image）→ DINOv2。"""

    def __init__(self,
                 image_size=224,
                 embed_size=384,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size

        model_configs = {
            'vits': {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384],
            }
        }
        self.imagegoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.imagegoal_encoder = self.imagegoal_encoder.pretrained.float()
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=6,
            out_channels=self.imagegoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.imagegoal_encoder.patch_embed.proj.kernel_size,
            stride=self.imagegoal_encoder.patch_embed.proj.stride,
            padding=self.imagegoal_encoder.patch_embed.proj.padding,
        )
        self.imagegoal_encoder.eval()
        self.project_layer = nn.Linear(384, embed_size)

    def forward(self, images):
        with torch.no_grad():
            assert len(images.shape) == 4  # B, H, W, C
            tensor_images = torch.as_tensor(
                images, dtype=torch.float32, device=self.device
            ).permute(0, 3, 1, 2)
            image_token = self.imagegoal_encoder.get_intermediate_layers(
                tensor_images
            )[0].mean(dim=1)
            image_token = self.project_layer(image_token)
            return image_token


class BridgeDP_PixelGoal_Backbone(nn.Module):
    """PixelGoal 编码器：4 通道（pixel mask + current image）→ DINOv2。"""

    def __init__(self,
                 image_size=224,
                 embed_size=384,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size

        model_configs = {
            'vits': {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384],
            }
        }
        self.pixelgoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.pixelgoal_encoder = self.pixelgoal_encoder.pretrained.float()
        self.pixelgoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=4,
            out_channels=self.pixelgoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.pixelgoal_encoder.patch_embed.proj.kernel_size,
            stride=self.pixelgoal_encoder.patch_embed.proj.stride,
            padding=self.pixelgoal_encoder.patch_embed.proj.padding,
        )
        self.pixelgoal_encoder.eval()
        self.project_layer = nn.Linear(384, embed_size)

    def forward(self, images):
        with torch.no_grad():
            assert len(images.shape) == 4  # B, H, W, C
            tensor_images = torch.as_tensor(
                images, dtype=torch.float32, device=self.device
            ).permute(0, 3, 1, 2)
            image_token = self.pixelgoal_encoder.get_intermediate_layers(
                tensor_images
            )[0].mean(dim=1)
            image_token = self.project_layer(image_token)
            return image_token
