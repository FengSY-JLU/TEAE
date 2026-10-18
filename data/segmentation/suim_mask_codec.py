from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


PathLike = Union[str, Path]

SUIM_CLASS_NAMES: Tuple[str, ...] = (
    "BW",  # 0: Background waterbody
    "HD",  # 1: Human divers
    "PF",  # 2: Plants / sea-grass
    "WR",  # 3: Wrecks / ruins
    "RO",  # 4: Robots / instruments
    "RI",  # 5: Reefs and invertebrates
    "FV",  # 6: Fish and vertebrates
    "SR",  # 7: Sand / sea-floor / rocks
)

SUIM_CLASS_DESCRIPTIONS: Tuple[str, ...] = (
    "Background waterbody",
    "Human divers",
    "Plants / sea-grass",
    "Wrecks / ruins",
    "Robots / instruments",
    "Reefs and invertebrates",
    "Fish and vertebrates",
    "Sand / sea-floor / rocks",
)

SUIM_FOREGROUND_CLASS_IDS: Dict[str, int] = {
    "HD": 1,
    "PF": 2,
    "WR": 3,
    "RO": 4,
    "RI": 5,
    "FV": 6,
    "SR": 7,
}

SUIM_NUM_CLASSES = len(SUIM_CLASS_NAMES)
SUIM_IGNORE_INDEX = 255

SUIM_CANONICAL_RGB = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [0, 255, 255],
        [255, 0, 0],
        [255, 0, 255],
        [255, 255, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class TrainMaskDecodeResult:
    mask: np.ndarray
    ambiguous_mask: np.ndarray
    exact_palette_ratio: float
    non_exact_palette_ratio: float
    ambiguous_pixel_count: int
    ambiguous_pixel_ratio: float


@dataclass(frozen=True)
class TestMaskDecodeResult:
    mask: np.ndarray
    overlap_mask: np.ndarray
    overlap_pixel_count: int
    overlap_pixel_ratio: float
    non_binary_pixel_count: int


def _require_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "SUIM train mask must be RGB [H,W,3], "
            f"got shape={tuple(rgb.shape)}"
        )

    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            raise TypeError(
                f"SUIM RGB mask must be uint8/integer, got {rgb.dtype}"
            )

    return np.ascontiguousarray(rgb)


def decode_train_rgb_mask(
    rgb: np.ndarray,
    ambiguity_margin: float = 8.0,
    ignore_index: int = SUIM_IGNORE_INDEX,
) -> TrainMaskDecodeResult:
    """
    Decode a SUIM RGB mask into class ids 0..7.

    Each RGB channel is thresholded at 128. This is equivalent to nearest
    neighbour assignment to the eight vertices of the binary RGB cube.
    Pixels close to the 127.5 decision boundary can optionally be marked
    with ``ignore_index``.
    """
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be non-negative")

    if not 0 <= int(ignore_index) <= 255:
        raise ValueError("ignore_index must fit uint8")

    rgb = _require_rgb(rgb)
    rgb_float = rgb.astype(np.float32)

    bits = rgb >= 128
    class_mask = (
        bits[..., 0].astype(np.uint8) * 4
        + bits[..., 1].astype(np.uint8) * 2
        + bits[..., 2].astype(np.uint8)
    )

    quantized_rgb = bits.astype(np.uint8) * 255
    exact_palette_pixels = np.all(rgb == quantized_rgb, axis=-1)

    if ambiguity_margin > 0:
        boundary_margin = np.min(
            np.abs(rgb_float - 127.5),
            axis=-1,
        )
        ambiguous_mask = boundary_margin < float(ambiguity_margin)
    else:
        ambiguous_mask = np.zeros(class_mask.shape, dtype=bool)

    output_mask = class_mask.copy()
    output_mask[ambiguous_mask] = np.uint8(ignore_index)

    pixel_count = int(output_mask.size)
    exact_ratio = float(exact_palette_pixels.mean())
    ambiguous_count = int(ambiguous_mask.sum())

    return TrainMaskDecodeResult(
        mask=np.ascontiguousarray(output_mask),
        ambiguous_mask=np.ascontiguousarray(ambiguous_mask),
        exact_palette_ratio=exact_ratio,
        non_exact_palette_ratio=float(1.0 - exact_ratio),
        ambiguous_pixel_count=ambiguous_count,
        ambiguous_pixel_ratio=float(
            ambiguous_count / max(pixel_count, 1)
        ),
    )


def load_train_semantic_mask(
    path: PathLike,
    ambiguity_margin: float = 8.0,
    ignore_index: int = SUIM_IGNORE_INDEX,
) -> TrainMaskDecodeResult:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)

    return decode_train_rgb_mask(
        rgb=rgb,
        ambiguity_margin=ambiguity_margin,
        ignore_index=ignore_index,
    )


def _load_binary_mask(
    path: PathLike,
    threshold: int = 128,
) -> Tuple[np.ndarray, int]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.uint8, copy=True)

    unique_values = np.unique(array)
    canonical = np.isin(unique_values, np.asarray([0, 255], dtype=np.uint8))
    non_binary_pixel_count = 0

    if not bool(canonical.all()):
        valid_pixels = np.logical_or(array == 0, array == 255)
        non_binary_pixel_count = int((~valid_pixels).sum())

    foreground = array >= int(threshold)

    return np.ascontiguousarray(foreground), non_binary_pixel_count


def load_test_semantic_mask(
    mask_paths: Mapping[str, PathLike],
    expected_size: Optional[Tuple[int, int]] = None,
    binary_threshold: int = 128,
    ignore_index: int = SUIM_IGNORE_INDEX,
    strict_overlap: bool = True,
    strict_binary: bool = True,
) -> TestMaskDecodeResult:
    """
    Merge the seven SUIM TEST foreground masks into one 8-class mask.

    Missing category files are treated as data errors. If overlap checking is
    relaxed, overlapping pixels are assigned ``ignore_index`` rather than
    being silently overwritten.
    """
    missing = set(SUIM_FOREGROUND_CLASS_IDS) - set(mask_paths)
    if missing:
        raise KeyError(
            f"Missing SUIM TEST mask paths for classes: {sorted(missing)}"
        )

    binary_masks: Dict[str, np.ndarray] = {}
    non_binary_pixel_count = 0
    resolved_shape: Optional[Tuple[int, int]] = expected_size

    for class_name in SUIM_FOREGROUND_CLASS_IDS:
        binary_mask, non_binary_count = _load_binary_mask(
            mask_paths[class_name],
            threshold=binary_threshold,
        )
        non_binary_pixel_count += int(non_binary_count)

        if resolved_shape is None:
            resolved_shape = tuple(binary_mask.shape)

        if tuple(binary_mask.shape) != tuple(resolved_shape):
            raise ValueError(
                f"TEST mask shape mismatch for {class_name}: "
                f"{tuple(binary_mask.shape)} vs {tuple(resolved_shape)}"
            )

        binary_masks[class_name] = binary_mask

    if resolved_shape is None:
        raise RuntimeError("No SUIM TEST masks were loaded")

    if strict_binary and non_binary_pixel_count > 0:
        raise ValueError(
            "SUIM TEST masks contain values other than 0/255: "
            f"{non_binary_pixel_count} pixels"
        )

    occupancy = np.zeros(resolved_shape, dtype=np.uint8)
    for binary_mask in binary_masks.values():
        occupancy += binary_mask.astype(np.uint8)

    overlap_mask = occupancy > 1
    overlap_pixel_count = int(overlap_mask.sum())

    if strict_overlap and overlap_pixel_count > 0:
        raise ValueError(
            "Overlapping SUIM TEST semantic masks detected: "
            f"{overlap_pixel_count} pixels"
        )

    semantic_mask = np.zeros(resolved_shape, dtype=np.uint8)

    for class_name, class_id in SUIM_FOREGROUND_CLASS_IDS.items():
        semantic_mask[binary_masks[class_name]] = np.uint8(class_id)

    if overlap_pixel_count > 0:
        semantic_mask[overlap_mask] = np.uint8(ignore_index)

    pixel_count = int(semantic_mask.size)

    return TestMaskDecodeResult(
        mask=np.ascontiguousarray(semantic_mask),
        overlap_mask=np.ascontiguousarray(overlap_mask),
        overlap_pixel_count=overlap_pixel_count,
        overlap_pixel_ratio=float(
            overlap_pixel_count / max(pixel_count, 1)
        ),
        non_binary_pixel_count=int(non_binary_pixel_count),
    )


def validate_semantic_mask(
    mask: Union[np.ndarray, torch.Tensor],
    num_classes: int = SUIM_NUM_CLASSES,
    ignore_index: int = SUIM_IGNORE_INDEX,
) -> None:
    tensor = torch.as_tensor(mask)

    if tensor.dim() != 2:
        raise ValueError(
            f"Semantic mask must be [H,W], got {tuple(tensor.shape)}"
        )

    if tensor.numel() == 0:
        raise ValueError("Semantic mask is empty")

    allowed = torch.logical_or(
        torch.logical_and(tensor >= 0, tensor < int(num_classes)),
        tensor == int(ignore_index),
    )

    if not bool(allowed.all().item()):
        invalid_values = torch.unique(tensor[~allowed]).cpu().tolist()
        raise ValueError(
            "Semantic mask contains invalid values: "
            f"{invalid_values[:20]}"
        )


def compute_class_present(
    mask: Union[np.ndarray, torch.Tensor],
    num_classes: int = SUIM_NUM_CLASSES,
    ignore_index: int = SUIM_IGNORE_INDEX,
) -> torch.Tensor:
    tensor = torch.as_tensor(mask).long()
    validate_semantic_mask(
        tensor,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    present = [
        bool((tensor == class_id).any().item())
        for class_id in range(int(num_classes))
    ]

    return torch.tensor(present, dtype=torch.bool)
