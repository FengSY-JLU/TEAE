from __future__ import annotations

from pathlib import Path
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from .common import (
    ConvNormAct,
    ImageNetNormalize,
    output_dict,
)


def load_state(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


class ASPPPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        # A global 1x1 pooled tensor is unsafe with BatchNorm when the
        # final training batch has size one. Use a bias convolution and ReLU.
        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.adaptive_avg_pool2d(x, 1)
        value = self.project(value)
        return F.interpolate(
            value,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: tuple[int, int, int] = (6, 12, 18),
    ) -> None:
        super().__init__()

        branches: List[nn.Module] = [
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            )
        ]

        for rate in rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        branches.append(
            ASPPPooling(in_channels, out_channels)
        )

        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            ConvNormAct(
                out_channels * len(branches),
                out_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(
            torch.cat(
                [branch(x) for branch in self.branches],
                dim=1,
            )
        )


class DeepLabV3PlusResNet50(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__()

        backbone = resnet50(
            weights=None,
            replace_stride_with_dilation=[
                False,
                False,
                True,
            ],
        )
        if checkpoint_path is not None:
            incompatible = backbone.load_state_dict(
                load_state(Path(checkpoint_path)),
                strict=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "ResNet50 checkpoint failed strict loading"
                )

        self.normalize = ImageNetNormalize()
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.aspp = ASPP(2048, 256)
        self.low_projection = ConvNormAct(
            256,
            48,
            kernel_size=1,
            padding=0,
        )
        self.decoder = nn.Sequential(
            ConvNormAct(256 + 48, 256),
            ConvNormAct(256, 256),
        )
        self.classifier = nn.Conv2d(
            256,
            int(num_classes),
            kernel_size=1,
        )

    def forward(self, image: torch.Tensor):
        input_size = image.shape[-2:]
        x = self.normalize(image)

        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)

        high = self.aspp(high)
        high = F.interpolate(
            high,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low = self.low_projection(low)

        feature_map = self.decoder(
            torch.cat([high, low], dim=1)
        )
        logits = self.classifier(feature_map)
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return output_dict(
            logits,
            feature_map,
            feature_pyramid=(low, high, feature_map),
        )
