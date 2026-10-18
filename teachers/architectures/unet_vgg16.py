from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from .common import (
    DoubleConv,
    ImageNetNormalize,
    output_dict,
)


def load_torch_state(path: Path) -> Any:
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


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.reduce(x)
        return self.conv(
            torch.cat([x, skip], dim=1)
        )


class UNetVGG16(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__()

        backbone = vgg16(weights=None)
        if checkpoint_path is not None:
            incompatible = backbone.load_state_dict(
                load_torch_state(Path(checkpoint_path)),
                strict=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "VGG16 checkpoint failed strict loading"
                )

        features = backbone.features

        self.enc1 = nn.Sequential(*features[0:4])
        self.pool1 = features[4]
        self.enc2 = nn.Sequential(*features[5:9])
        self.pool2 = features[9]
        self.enc3 = nn.Sequential(*features[10:16])
        self.pool3 = features[16]
        self.enc4 = nn.Sequential(*features[17:23])
        self.pool4 = features[23]
        self.enc5 = nn.Sequential(*features[24:30])

        del backbone

        self.normalize = ImageNetNormalize()
        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 64)

        self.feature_head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(
            256,
            int(num_classes),
            kernel_size=1,
        )

    def forward(self, image: torch.Tensor):
        x = self.normalize(image)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        feature_map = self.feature_head(d1)
        logits = self.classifier(feature_map)

        return output_dict(
            logits,
            feature_map,
            feature_pyramid=(e2, e3, e4, feature_map),
        )
