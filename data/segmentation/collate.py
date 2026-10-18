from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from .suim_mask_codec import SUIM_IGNORE_INDEX


def _validate_sample(sample: Dict[str, Any]) -> None:
    image = sample["image"]
    mask = sample["mask"]

    if not torch.is_tensor(image) or image.dim() != 3:
        raise ValueError("sample['image'] must be Tensor [3,H,W]")

    if image.shape[0] != 3:
        raise ValueError("sample['image'] must have 3 channels")

    if not torch.is_tensor(mask) or mask.dim() != 2:
        raise ValueError("sample['mask'] must be Tensor [H,W]")

    if tuple(image.shape[-2:]) != tuple(mask.shape):
        raise ValueError(
            "Image/mask shape mismatch inside collate: "
            f"{tuple(image.shape[-2:])} vs {tuple(mask.shape)}"
        )


def seg_bank_collate_fn(
    batch: Sequence[Dict[str, Any]],
    ignore_index: int = SUIM_IGNORE_INDEX,
) -> Dict[str, Any]:
    """
    Collate for proxy-bank generation.

    It intentionally requires B=1 and performs no resize or padding, so each
    bank sample retains its original spatial resolution.
    """
    if len(batch) != 1:
        raise ValueError(
            "Seg proxy-bank generation requires batch_size=1, "
            f"got {len(batch)}"
        )

    sample = batch[0]
    _validate_sample(sample)

    image = sample["image"].unsqueeze(0).contiguous()
    mask = sample["mask"].unsqueeze(0).long().contiguous()

    return {
        "image": image,
        "mask": mask,
        "valid_pixel_mask": (mask != int(ignore_index)),
        "class_present": sample["class_present"].unsqueeze(0),
        "image_path": sample["image_path"],
        "mask_path": sample["mask_path"],
        "source_id": sample["source_id"],
        "dataset": sample["dataset"],
        "split": sample["split"],
        "original_size": [sample["original_size"]],
        "index": torch.tensor([sample["index"]], dtype=torch.long),
        "decode_meta": [sample["decode_meta"]],
    }


def seg_pad_collate_fn(
    batch: Sequence[Dict[str, Any]],
    ignore_index: int = SUIM_IGNORE_INDEX,
) -> Dict[str, Any]:
    """
    Pad a variable-size segmentation batch to the maximum height/width.

    Images are padded with zero, masks with ``ignore_index``. The returned
    ``valid_pixel_mask`` excludes both padding and ignored source pixels.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    for sample in batch:
        _validate_sample(sample)

    max_height = max(int(sample["image"].shape[-2]) for sample in batch)
    max_width = max(int(sample["image"].shape[-1]) for sample in batch)

    batch_size = len(batch)
    image_dtype = batch[0]["image"].dtype
    image_device = batch[0]["image"].device

    images = torch.zeros(
        batch_size,
        3,
        max_height,
        max_width,
        dtype=image_dtype,
        device=image_device,
    )
    masks = torch.full(
        (batch_size, max_height, max_width),
        fill_value=int(ignore_index),
        dtype=torch.long,
        device=image_device,
    )
    valid_pixel_mask = torch.zeros(
        batch_size,
        max_height,
        max_width,
        dtype=torch.bool,
        device=image_device,
    )

    for batch_index, sample in enumerate(batch):
        image = sample["image"]
        mask = sample["mask"].long()
        height, width = mask.shape

        images[batch_index, :, :height, :width] = image
        masks[batch_index, :height, :width] = mask
        valid_pixel_mask[batch_index, :height, :width] = (
            mask != int(ignore_index)
        )

    return {
        "image": images.contiguous(),
        "mask": masks.contiguous(),
        "valid_pixel_mask": valid_pixel_mask.contiguous(),
        "class_present": torch.stack(
            [sample["class_present"] for sample in batch],
            dim=0,
        ),
        "image_path": [sample["image_path"] for sample in batch],
        "mask_path": [sample["mask_path"] for sample in batch],
        "source_id": [sample["source_id"] for sample in batch],
        "dataset": [sample["dataset"] for sample in batch],
        "split": [sample["split"] for sample in batch],
        "original_size": [sample["original_size"] for sample in batch],
        "index": torch.tensor(
            [sample["index"] for sample in batch],
            dtype=torch.long,
        ),
        "decode_meta": [sample["decode_meta"] for sample in batch],
        "padded_size": (max_height, max_width),
    }
