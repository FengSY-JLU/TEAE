from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.epsilon)
        return (
            normalized * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depthwise_expansion: int = 2,
        ffn_expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        depthwise_channels = channels * depthwise_expansion
        ffn_channels = channels * ffn_expansion

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(
            channels, depthwise_channels, kernel_size=1, bias=True
        )
        self.depthwise = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            kernel_size=3,
            padding=1,
            groups=depthwise_channels,
            bias=True,
        )
        self.gate1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                depthwise_channels // 2,
                depthwise_channels // 2,
                kernel_size=1,
                bias=True,
            ),
        )
        self.conv2 = nn.Conv2d(
            depthwise_channels // 2, channels, kernel_size=1, bias=True
        )
        self.dropout1 = (
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        )

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(
            channels, ffn_channels, kernel_size=1, bias=True
        )
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(
            ffn_channels // 2, channels, kernel_size=1, bias=True
        )
        self.dropout2 = (
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value = self.norm1(x)
        value = self.conv1(value)
        value = self.depthwise(value)
        value = self.gate1(value)
        value = value * self.sca(value)
        value = self.conv2(value)
        value = self.dropout1(value)
        x = residual + value * self.beta

        residual = x
        value = self.norm2(x)
        value = self.conv3(value)
        value = self.gate2(value)
        value = self.conv4(value)
        value = self.dropout2(value)
        return residual + value * self.gamma


class RawAnchoredNAFNet(nn.Module):
    def __init__(
        self,
        *,
        width: int = 32,
        encoder_blocks: Sequence[int] = (2, 2, 4, 8),
        middle_blocks: int = 12,
        decoder_blocks: Sequence[int] = (2, 2, 2, 2),
        gate_floor: float = 0.05,
        gate_ceiling: float = 1.0,
    ) -> None:
        super().__init__()
        if len(encoder_blocks) != len(decoder_blocks):
            raise ValueError("Encoder/decoder stage counts must match.")
        if not 0.0 <= gate_floor <= gate_ceiling <= 1.0:
            raise ValueError("Require 0 <= gate_floor <= gate_ceiling <= 1.")

        self.stage_count = len(encoder_blocks)
        self.padder_size = 2 ** self.stage_count
        self.gate_floor = float(gate_floor)
        self.gate_ceiling = float(gate_ceiling)

        self.intro = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width
        for block_count in encoder_blocks:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(channels) for _ in range(int(block_count))]
                )
            )
            self.downs.append(
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=2,
                    stride=2,
                )
            )
            channels *= 2

        self.middle = nn.Sequential(
            *[NAFBlock(channels) for _ in range(int(middle_blocks))]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for block_count in decoder_blocks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(channels) for _ in range(int(block_count))]
                )
            )

        self.residual_head = nn.Conv2d(
            width, 3, kernel_size=3, padding=1
        )
        self.gate_head = nn.Conv2d(
            width, 1, kernel_size=3, padding=1
        )

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h)), height, width

    def forward(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError("raw must be [B,3,H,W].")
        padded, height, width = self._pad(raw)

        value = self.intro(padded)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):

            value = checkpoint(
                encoder,
                value,
                use_reentrant=False,
            )

            skips.append(value)

            value = down(value)

        value = checkpoint(
            self.middle,
            value,
            use_reentrant=False,
        )

        for up, decoder, skip in zip(
            self.ups, self.decoders, reversed(skips)
        ):
            value = up(value)
            if value.shape[-2:] != skip.shape[-2:]:
                value = F.interpolate(
                    value,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            value = checkpoint(
                decoder,
                value + skip,
                use_reentrant=False,
            )

        residual = torch.tanh(self.residual_head(value))
        gate_unit = torch.sigmoid(self.gate_head(value))
        gate = (
            self.gate_floor
            + (self.gate_ceiling - self.gate_floor) * gate_unit
        )
        enhanced = torch.clamp(padded + gate * residual, 0.0, 1.0)

        return {
            "enhanced": enhanced[..., :height, :width],
            "residual": residual[..., :height, :width],
            "gate": gate[..., :height, :width],
        }
