from __future__ import annotations

from pathlib import Path
from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    ConvNormAct,
    ImageNetNormalize,
    output_dict,
)


def load_state(path: Path):
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


class HRNetW18Segmentation(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        checkpoint_path: Path | None = None,
        model_name: str = "hrnet_w18.ms_aug_in1k",
        decoder_dim: int = 128,
    ) -> None:
        super().__init__()

        self.normalize = ImageNetNormalize()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
        )
        if checkpoint_path is not None:
            incompatible = self.backbone.load_state_dict(
                load_state(Path(checkpoint_path)),
                strict=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "HRNet features-only checkpoint failed strict loading"
                )

        channels = list(
            self.backbone.feature_info.channels()
        )

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    int(channel),
                    int(decoder_dim),
                    kernel_size=1,
                )
                for channel in channels
            ]
        )
        self.fusion = nn.Sequential(
            ConvNormAct(
                int(decoder_dim) * len(channels),
                256,
            ),
            ConvNormAct(256, 256),
        )
        self.classifier = nn.Conv2d(
            256,
            int(num_classes),
            kernel_size=1,
        )

    def forward(self, image: torch.Tensor):
        input_size = image.shape[-2:]
        features = self.backbone(
            self.normalize(image)
        )

        if len(features) != len(self.projections):
            raise RuntimeError(
                "Unexpected HRNet feature count"
            )

        target_size = features[1].shape[-2:]
        projected: List[torch.Tensor] = []

        for projection, feature in zip(
            self.projections,
            features,
        ):
            value = projection(feature)
            value = F.interpolate(
                value,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            projected.append(value)

        feature_map = self.fusion(
            torch.cat(projected, dim=1)
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
            feature_pyramid=tuple(features) + (feature_map,),
        )
