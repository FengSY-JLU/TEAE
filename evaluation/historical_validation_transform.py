from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import ColorJitter


def _resize_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]

    mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=size,
        mode="nearest",
    )[0, 0].long()

    return image, mask


def _pad_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    minimum_height: int,
    minimum_width: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    pad_height = max(0, minimum_height - height)
    pad_width = max(0, minimum_width - width)

    if pad_height == 0 and pad_width == 0:
        return image, mask

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    image = F.pad(
        image,
        (left, right, top, bottom),
        mode="replicate",
    )
    mask = F.pad(
        mask,
        (left, right, top, bottom),
        mode="constant",
        value=int(ignore_index),
    )

    return image, mask


def _resize_long_side(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    long_side: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    scale = float(long_side) / float(max(height, width))

    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))

    return _resize_pair(
        image,
        mask,
        (target_height, target_width),
    )


class TrainSegmentationTransform:
    def __init__(
        self,
        *,
        crop_size: int = 384,
        scale_min: float = 0.75,
        scale_max: float = 1.50,
        horizontal_flip_probability: float = 0.5,
        color_jitter_probability: float = 0.8,
        ignore_index: int = 255,
    ) -> None:
        self.crop_size = int(crop_size)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.horizontal_flip_probability = float(
            horizontal_flip_probability
        )
        self.color_jitter_probability = float(
            color_jitter_probability
        )
        self.ignore_index = int(ignore_index)

        self.color_jitter = ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03,
        )

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = random.uniform(
            self.scale_min,
            self.scale_max,
        )
        long_side = max(
            32,
            int(round(self.crop_size * scale)),
        )

        image, mask = _resize_long_side(
            image,
            mask,
            long_side=long_side,
        )

        if random.random() < self.horizontal_flip_probability:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        if random.random() < self.color_jitter_probability:
            image = self.color_jitter(image).clamp(0.0, 1.0)

        image, mask = _pad_pair(
            image,
            mask,
            minimum_height=self.crop_size,
            minimum_width=self.crop_size,
            ignore_index=self.ignore_index,
        )

        height, width = image.shape[-2:]
        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)

        image = image[
            :,
            top : top + self.crop_size,
            left : left + self.crop_size,
        ]
        mask = mask[
            top : top + self.crop_size,
            left : left + self.crop_size,
        ]

        return image.contiguous(), mask.contiguous()


class ValidationSegmentationTransform:
    def __init__(
        self,
        *,
        size: int = 384,
        ignore_index: int = 255,
    ) -> None:
        self.size = int(size)
        self.ignore_index = int(ignore_index)

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = _resize_long_side(
            image,
            mask,
            long_side=self.size,
        )
        image, mask = _pad_pair(
            image,
            mask,
            minimum_height=self.size,
            minimum_width=self.size,
            ignore_index=self.ignore_index,
        )

        height, width = image.shape[-2:]
        top = max(0, (height - self.size) // 2)
        left = max(0, (width - self.size) // 2)

        image = image[
            :,
            top : top + self.size,
            left : left + self.size,
        ]
        mask = mask[
            top : top + self.size,
            left : left + self.size,
        ]

        return image.contiguous(), mask.contiguous()
