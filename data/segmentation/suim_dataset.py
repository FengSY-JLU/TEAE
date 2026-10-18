from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .suim_mask_codec import (
    SUIM_CLASS_NAMES,
    SUIM_FOREGROUND_CLASS_IDS,
    SUIM_IGNORE_INDEX,
    compute_class_present,
    load_test_semantic_mask,
    load_train_semantic_mask,
    validate_semantic_mask,
)


PathLike = Union[str, Path]
JointTransform = Callable[
    [torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


def _collect_files_by_stem(
    folder: Path,
    allowed_suffixes: Sequence[str],
) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    allowed = {suffix.lower() for suffix in allowed_suffixes}
    result: Dict[str, Path] = {}

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue

        if path.stem in result:
            raise RuntimeError(
                f"Duplicate file stem {path.stem!r} in {folder}: "
                f"{result[path.stem].name!r}, {path.name!r}"
            )

        result[path.stem] = path.resolve()

    if not result:
        raise RuntimeError(
            f"No files with suffixes {sorted(allowed)} found in {folder}"
        )

    return result


def _validate_exact_stem_match(
    reference: Mapping[str, Path],
    candidate: Mapping[str, Path],
    reference_name: str,
    candidate_name: str,
) -> None:
    reference_stems = set(reference)
    candidate_stems = set(candidate)

    missing = sorted(reference_stems - candidate_stems)
    orphan = sorted(candidate_stems - reference_stems)

    if missing or orphan:
        raise RuntimeError(
            f"SUIM file pairing failed between {reference_name} and "
            f"{candidate_name}: missing={missing[:20]}, orphan={orphan[:20]}"
        )


def _read_rgb_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    return tensor.contiguous()


TRAIN_SIZE_POLICIES = (
    "strict",
    "suim_crop_mask_bottom_55",
)


def _align_train_mask_to_image(
    mask: torch.Tensor,
    image_size: Tuple[int, int],
    source_id: str,
    policy: str,
    expected_extra_height: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Align known SUIM train/validation mask-size anomalies.

    The audited anomaly has identical image/mask widths while the mask is
    exactly 55 pixels taller. Visual inspection shows that the correct repair
    is to keep the top image-height rows and remove 55 rows from the mask
    bottom. No resizing is performed.
    """
    if mask.dim() != 2:
        raise ValueError(
            f"Train mask must be [H,W], got {tuple(mask.shape)}"
        )

    image_height, image_width = map(int, image_size)
    mask_height, mask_width = map(int, mask.shape)

    meta: Dict[str, Any] = {
        "alignment_applied": False,
        "alignment_operation": "none",
        "alignment_policy": policy,
        "original_mask_size": [mask_height, mask_width],
        "aligned_mask_size": [mask_height, mask_width],
        "crop_top": 0,
        "crop_bottom": 0,
        "crop_left": 0,
        "crop_right": 0,
    }

    if (mask_height, mask_width) == (image_height, image_width):
        return mask.contiguous(), meta

    if policy == "strict":
        raise ValueError(
            f"Image/mask size mismatch for {source_id}: "
            f"image={(image_height, image_width)}, "
            f"mask={(mask_height, mask_width)}"
        )

    if policy != "suim_crop_mask_bottom_55":
        raise ValueError(
            f"Unknown train size-mismatch policy: {policy!r}. "
            f"Expected one of {TRAIN_SIZE_POLICIES}."
        )

    valid_known_anomaly = (
        mask_width == image_width
        and mask_height == image_height + int(expected_extra_height)
        and int(expected_extra_height) == 55
    )

    if not valid_known_anomaly:
        raise ValueError(
            f"Unrecognized SUIM size mismatch for {source_id}: "
            f"image={(image_height, image_width)}, "
            f"mask={(mask_height, mask_width)}. "
            "Automatic repair is allowed only when widths match and the "
            f"mask is exactly {expected_extra_height} pixels taller."
        )

    aligned = mask[:image_height, :].contiguous()

    if tuple(aligned.shape) != (image_height, image_width):
        raise RuntimeError(
            f"Alignment failed for {source_id}: got {tuple(aligned.shape)}"
        )

    meta.update(
        {
            "alignment_applied": True,
            "alignment_operation": "crop_mask_bottom",
            "aligned_mask_size": [image_height, image_width],
            "crop_bottom": int(expected_extra_height),
        }
    )

    return aligned, meta


class SUIMDataset(Dataset):
    """
    SUIM semantic-segmentation dataset.

    ``split`` supports:
        - ``train_val``: RGB multiclass BMP masks.
        - ``test``: seven per-class binary BMP masks.

    Returned sample:
        image:          float32 [3,H,W] in [0,1]
        mask:           int64 [H,W], values 0..7 or ignore_index
        class_present:  bool [8]
        image_path:     str
        mask_path:      str (train_val) or dict[str, str] (test)
        source_id:      str
        dataset:        "SUIM"
        split:          "train_val" or "test"
        original_size:  (H,W)
        index:          int
        decode_meta:    dict
    """

    def __init__(
        self,
        root: PathLike,
        split: str = "train_val",
        transform: Optional[JointTransform] = None,
        ambiguity_margin: float = 8.0,
        ignore_index: int = SUIM_IGNORE_INDEX,
        binary_threshold: int = 128,
        strict_test_overlap: bool = True,
        strict_test_binary: bool = True,
        train_size_mismatch_policy: str = "suim_crop_mask_bottom_55",
        train_mask_extra_height: int = 55,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()

        split = split.lower()
        if split == "test":
            canonical_split = "test"
        elif split in {"train_val", "trainval", "train-val"}:
            canonical_split = "train_val"
        else:
            raise ValueError(
                "split must be 'train_val' or 'test', "
                f"got {split!r}"
            )

        if max_samples is not None and int(max_samples) <= 0:
            raise ValueError("max_samples must be positive when provided")

        self.root = Path(root).expanduser().resolve()
        self.split = canonical_split
        self.transform = transform
        self.ambiguity_margin = float(ambiguity_margin)
        self.ignore_index = int(ignore_index)
        self.binary_threshold = int(binary_threshold)
        self.strict_test_overlap = bool(strict_test_overlap)
        self.strict_test_binary = bool(strict_test_binary)
        self.train_size_mismatch_policy = str(
            train_size_mismatch_policy
        ).lower()
        self.train_mask_extra_height = int(train_mask_extra_height)

        if self.train_size_mismatch_policy not in TRAIN_SIZE_POLICIES:
            raise ValueError(
                "train_size_mismatch_policy must be one of "
                f"{TRAIN_SIZE_POLICIES}, got "
                f"{self.train_size_mismatch_policy!r}"
            )

        if self.train_mask_extra_height <= 0:
            raise ValueError("train_mask_extra_height must be positive")

        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

        if self.split == "train_val":
            samples = self._index_train_val()
        else:
            samples = self._index_test()

        if max_samples is not None:
            samples = samples[: int(max_samples)]

        if not samples:
            raise RuntimeError("SUIM dataset contains no indexed samples")

        self.samples = samples

    @property
    def class_names(self) -> Tuple[str, ...]:
        return SUIM_CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return len(SUIM_CLASS_NAMES)

    def _index_train_val(self) -> List[Dict[str, Any]]:
        image_root = self.root / "train_val" / "images"
        mask_root = self.root / "train_val" / "masks"

        image_map = _collect_files_by_stem(
            image_root,
            allowed_suffixes=(".jpg", ".jpeg", ".png"),
        )
        mask_map = _collect_files_by_stem(
            mask_root,
            allowed_suffixes=(".bmp",),
        )

        _validate_exact_stem_match(
            image_map,
            mask_map,
            "train_val/images",
            "train_val/masks",
        )

        return [
            {
                "source_id": stem,
                "image_path": image_map[stem],
                "mask_path": mask_map[stem],
            }
            for stem in sorted(image_map)
        ]

    def _index_test(self) -> List[Dict[str, Any]]:
        image_root = self.root / "TEST" / "images"
        mask_root = self.root / "TEST" / "masks"

        image_map = _collect_files_by_stem(
            image_root,
            allowed_suffixes=(".jpg", ".jpeg", ".png"),
        )

        class_maps: Dict[str, Dict[str, Path]] = {}

        for class_name in SUIM_FOREGROUND_CLASS_IDS:
            class_map = _collect_files_by_stem(
                mask_root / class_name,
                allowed_suffixes=(".bmp",),
            )

            _validate_exact_stem_match(
                image_map,
                class_map,
                "TEST/images",
                f"TEST/masks/{class_name}",
            )

            class_maps[class_name] = class_map

        samples: List[Dict[str, Any]] = []

        for stem in sorted(image_map):
            samples.append(
                {
                    "source_id": stem,
                    "image_path": image_map[stem],
                    "mask_paths": {
                        class_name: class_maps[class_name][stem]
                        for class_name in SUIM_FOREGROUND_CLASS_IDS
                    },
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.samples[index]
        image_path: Path = record["image_path"]
        image = _read_rgb_image(image_path)
        _, height, width = image.shape

        if self.split == "train_val":
            mask_path: Path = record["mask_path"]
            decoded = load_train_semantic_mask(
                mask_path,
                ambiguity_margin=self.ambiguity_margin,
                ignore_index=self.ignore_index,
            )
            mask = torch.from_numpy(decoded.mask.copy()).long()
            mask, alignment_meta = _align_train_mask_to_image(
                mask=mask,
                image_size=(height, width),
                source_id=str(record["source_id"]),
                policy=self.train_size_mismatch_policy,
                expected_extra_height=self.train_mask_extra_height,
            )

            decode_meta = {
                "exact_palette_ratio": decoded.exact_palette_ratio,
                "non_exact_palette_ratio": decoded.non_exact_palette_ratio,
                "ambiguous_pixel_count": decoded.ambiguous_pixel_count,
                "ambiguous_pixel_ratio": decoded.ambiguous_pixel_ratio,
                "alignment": alignment_meta,
            }
            mask_path_value: Union[str, Dict[str, str]] = str(mask_path)

        else:
            mask_paths: Dict[str, Path] = record["mask_paths"]
            decoded = load_test_semantic_mask(
                mask_paths=mask_paths,
                expected_size=(height, width),
                binary_threshold=self.binary_threshold,
                ignore_index=self.ignore_index,
                strict_overlap=self.strict_test_overlap,
                strict_binary=self.strict_test_binary,
            )
            mask = torch.from_numpy(decoded.mask.copy()).long()

            decode_meta = {
                "overlap_pixel_count": decoded.overlap_pixel_count,
                "overlap_pixel_ratio": decoded.overlap_pixel_ratio,
                "non_binary_pixel_count": decoded.non_binary_pixel_count,
                "alignment": {
                    "alignment_applied": False,
                    "alignment_operation": "none",
                    "alignment_policy": "strict",
                    "original_mask_size": [height, width],
                    "aligned_mask_size": [height, width],
                    "crop_top": 0,
                    "crop_bottom": 0,
                    "crop_left": 0,
                    "crop_right": 0,
                },
            }
            mask_path_value = {
                class_name: str(path)
                for class_name, path in mask_paths.items()
            }

        if tuple(mask.shape) != (height, width):
            raise ValueError(
                f"Image/mask size mismatch for {record['source_id']}: "
                f"image={(height, width)}, mask={tuple(mask.shape)}"
            )

        validate_semantic_mask(
            mask,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        original_size = (height, width)

        if self.transform is not None:
            transformed = self.transform(image, mask)

            if (
                not isinstance(transformed, (tuple, list))
                or len(transformed) != 2
            ):
                raise TypeError(
                    "Segmentation transform must return (image, mask)"
                )

            image, mask = transformed

            if not torch.is_tensor(image) or not torch.is_tensor(mask):
                raise TypeError(
                    "Segmentation transform must return tensors"
                )

            if image.dim() != 3 or image.shape[0] != 3:
                raise ValueError(
                    "Transformed image must be [3,H,W], "
                    f"got {tuple(image.shape)}"
                )

            if mask.dim() != 2:
                raise ValueError(
                    "Transformed mask must be [H,W], "
                    f"got {tuple(mask.shape)}"
                )

            if tuple(image.shape[-2:]) != tuple(mask.shape):
                raise ValueError(
                    "Transformed image/mask size mismatch: "
                    f"{tuple(image.shape[-2:])} vs {tuple(mask.shape)}"
                )

            image = image.float().contiguous()
            mask = mask.long().contiguous()

            validate_semantic_mask(
                mask,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )

        class_present = compute_class_present(
            mask,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        return {
            "image": image,
            "mask": mask,
            "class_present": class_present,
            "image_path": str(image_path),
            "mask_path": mask_path_value,
            "source_id": str(record["source_id"]),
            "dataset": "SUIM",
            "split": self.split,
            "original_size": original_size,
            "index": int(index),
            "decode_meta": decode_meta,
        }
