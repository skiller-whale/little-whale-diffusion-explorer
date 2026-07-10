"""Small static-shape diffusion U-Net designed for ONNX Runtime Web."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 5


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time(F.silu(time))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class Attention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        hidden = self.norm(x).reshape(batch, channels, height * width)
        query, key, value = self.qkv(hidden).chunk(3, dim=1)
        scale = channels ** -0.5
        weights = torch.softmax(torch.bmm(query.transpose(1, 2), key) * scale, dim=-1)
        attended = torch.bmm(value, weights.transpose(1, 2))
        return x + self.output(attended).reshape(batch, channels, height, width)


class WhaleUNet(nn.Module):
    """Approximately 1.7M parameters / 6.5 MB in float32."""
    def __init__(self, base: int = 24, time_dim: int = 160):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(3, base, 3, padding=1)
        self.d1a, self.d1b = ResBlock(base, base, time_dim), ResBlock(base, base, time_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.d2a, self.d2b = ResBlock(base * 2, base * 2, time_dim), ResBlock(base * 2, base * 2, time_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.d3a, self.d3b = ResBlock(base * 4, base * 4, time_dim), ResBlock(base * 4, base * 4, time_dim)
        self.mid1 = ResBlock(base * 4, base * 4, time_dim)
        self.attn = Attention(base * 4)
        self.mid2 = ResBlock(base * 4, base * 4, time_dim)
        self.u3a, self.u3b = ResBlock(base * 8, base * 4, time_dim), ResBlock(base * 4, base * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.u2a, self.u2b = ResBlock(base * 4, base * 2, time_dim), ResBlock(base * 2, base * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.u1a, self.u1b = ResBlock(base * 2, base, time_dim), ResBlock(base, base, time_dim)
        self.output_norm = nn.GroupNorm(groups(base), base)
        self.output = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, sample: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        time = self.time_mlp(timestep_embedding)
        x1 = self.d1b(self.d1a(self.input(sample), time), time)
        x2 = self.d2b(self.d2a(self.down1(x1), time), time)
        x3 = self.d3b(self.d3a(self.down2(x2), time), time)
        hidden = self.mid2(self.attn(self.mid1(x3, time)), time)
        hidden = self.u3b(self.u3a(torch.cat((hidden, x3), dim=1), time), time)
        hidden = self.u2b(self.u2a(torch.cat((self.up2(hidden), x2), dim=1), time), time)
        hidden = self.u1b(self.u1a(torch.cat((self.up1(hidden), x1), dim=1), time), time)
        return self.output(F.silu(self.output_norm(hidden)))
