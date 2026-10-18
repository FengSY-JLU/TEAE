from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel

from .common import (
    ConvNormAct,
    ImageNetNormalize,
    output_dict,
)


class SegFormerB2(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        pretrained_dir: Path,
        decoder_dim: int = 256,
    ) -> None:
        super().__init__()

        self.normalize = ImageNetNormalize()
        config = SegformerConfig.from_pretrained(
            str(Path(pretrained_dir).expanduser().resolve()),
            local_files_only=True,
        )
        self.backbone = SegformerModel(config)

        hidden_sizes = list(
            self.backbone.config.hidden_sizes
        )

        if len(hidden_sizes) != 4:
            raise ValueError(
                f"Expected four SegFormer stages, got {hidden_sizes}"
            )

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    int(decoder_dim),
                    kernel_size=1,
                )
                for channels in hidden_sizes
            ]
        )
        self.fusion = nn.Sequential(
            ConvNormAct(
                int(decoder_dim) * 4,
                int(decoder_dim),
            ),
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Conv2d(
            int(decoder_dim),
            int(num_classes),
            kernel_size=1,
        )

    def forward(self, image: torch.Tensor):
        input_size = image.shape[-2:]
        x = self.normalize(image)

        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = list(outputs.hidden_states or ())
        four_dimensional = [
            tensor
            for tensor in hidden_states
            if torch.is_tensor(tensor) and tensor.dim() == 4
        ]

        # Match stages by their configured channel counts. This remains
        # compatible if a transformers release exposes additional states.
        feature_maps = []
        search_start = 0
        for expected_channels in self.backbone.config.hidden_sizes:
            matched = None
            for position in range(
                search_start,
                len(four_dimensional),
            ):
                candidate = four_dimensional[position]
                if candidate.shape[1] == expected_channels:
                    matched = candidate
                    search_start = position + 1
                    break
            if matched is None:
                raise RuntimeError(
                    "Could not match SegFormer stage channels "
                    f"{list(self.backbone.config.hidden_sizes)} from "
                    f"{[tuple(x.shape) for x in four_dimensional]}"
                )
            feature_maps.append(matched)

        target_size = feature_maps[0].shape[-2:]
        projected: List[torch.Tensor] = []

        for projection, feature in zip(
            self.projections,
            feature_maps,
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
            feature_pyramid=tuple(feature_maps) + (feature_map,),
        )
