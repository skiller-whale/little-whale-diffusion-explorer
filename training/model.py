"""A compact four-level U-Net for 64 px browser diffusion."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def groups(channels: int) -> int:
    for value in (8, 6, 4, 3, 2):
        if channels % value == 0: return value
    return 1


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
        return self.conv2(F.silu(self.norm2(hidden))) + self.skip(x)


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
        weights = torch.softmax(torch.bmm(query.transpose(1, 2), key) * channels ** -0.5, dim=-1)
        return x + self.output(torch.bmm(value, weights.transpose(1, 2))).reshape(batch, channels, height, width)


class WhaleUNet(nn.Module):
    """Wider at 64 px, capped at 128 channels to remain practical in WASM."""
    def __init__(self, time_dim: int = 192, condition_dim: int = 16):
        super().__init__()
        c1, c2, c3, c4 = 32, 64, 96, 128
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.condition_mlp = nn.Sequential(nn.Linear(condition_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(3, c1, 3, padding=1)
        self.d1a, self.d1b = ResBlock(c1, c1, time_dim), ResBlock(c1, c1, time_dim)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.d2a, self.d2b = ResBlock(c2, c2, time_dim), ResBlock(c2, c2, time_dim)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.d3a, self.d3b = ResBlock(c3, c3, time_dim), ResBlock(c3, c3, time_dim)
        self.down3 = nn.Conv2d(c3, c4, 4, stride=2, padding=1)
        self.d4a, self.d4b = ResBlock(c4, c4, time_dim), ResBlock(c4, c4, time_dim)
        self.mid1, self.attn, self.mid2 = ResBlock(c4, c4, time_dim), Attention(c4), ResBlock(c4, c4, time_dim)
        self.u4a, self.u4b = ResBlock(c4 * 2, c4, time_dim), ResBlock(c4, c4, time_dim)
        self.up3 = nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1)
        self.u3a, self.u3b = ResBlock(c3 * 2, c3, time_dim), ResBlock(c3, c3, time_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.u2a, self.u2b = ResBlock(c2 * 2, c2, time_dim), ResBlock(c2, c2, time_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.u1a, self.u1b = ResBlock(c1 * 2, c1, time_dim), ResBlock(c1, c1, time_dim)
        self.output_norm = nn.GroupNorm(groups(c1), c1)
        self.output = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, sample: torch.Tensor, timestep_embedding: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        time = self.time_mlp(timestep_embedding) + self.condition_mlp(conditioning)
        x1 = self.d1b(self.d1a(self.input(sample), time), time)
        x2 = self.d2b(self.d2a(self.down1(x1), time), time)
        x3 = self.d3b(self.d3a(self.down2(x2), time), time)
        x4 = self.d4b(self.d4a(self.down3(x3), time), time)
        hidden = self.mid2(self.attn(self.mid1(x4, time)), time)
        hidden = self.u4b(self.u4a(torch.cat((hidden, x4), dim=1), time), time)
        hidden = self.u3b(self.u3a(torch.cat((self.up3(hidden), x3), dim=1), time), time)
        hidden = self.u2b(self.u2a(torch.cat((self.up2(hidden), x2), dim=1), time), time)
        hidden = self.u1b(self.u1a(torch.cat((self.up1(hidden), x1), dim=1), time), time)
        return self.output(F.silu(self.output_norm(hidden)))
