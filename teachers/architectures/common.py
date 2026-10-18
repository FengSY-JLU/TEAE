from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageNetNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (
            image.float() - self.mean
        ) / self.std


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )


def output_dict(
    logits: torch.Tensor,
    feature_map: torch.Tensor,
    *,
    feature_pyramid: tuple[torch.Tensor, ...] | list[torch.Tensor] = (),
) -> Dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    feature = F.adaptive_avg_pool2d(
        feature_map.float(),
        output_size=1,
    ).flatten(1)

    pyramid = tuple(feature_pyramid) or (feature_map,)
    return {
        "logits": logits,
        "feature_map": feature_map,
        "feature": feature,
        "feature_pyramid": pyramid,
    }
