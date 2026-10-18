from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .collate import seg_bank_collate_fn, seg_pad_collate_fn
from .suim_dataset import JointTransform, SUIMDataset
from .suim_mask_codec import SUIM_IGNORE_INDEX


PathLike = Union[str, Path]


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_suim_loader(
    root: PathLike,
    split: str = "train_val",
    batch_size: int = 1,
    workers: int = 4,
    shuffle: Optional[bool] = None,
    transform: Optional[JointTransform] = None,
    collate_mode: str = "bank",
    ambiguity_margin: float = 8.0,
    ignore_index: int = SUIM_IGNORE_INDEX,
    binary_threshold: int = 128,
    strict_test_overlap: bool = True,
    strict_test_binary: bool = True,
    train_size_mismatch_policy: str = "suim_crop_mask_bottom_55",
    train_mask_extra_height: int = 55,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    seed: int = 20260724,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if workers < 0:
        raise ValueError("workers must be non-negative")

    collate_mode = collate_mode.lower()

    if collate_mode == "bank":
        if batch_size != 1:
            raise ValueError(
                "collate_mode='bank' requires batch_size=1"
            )
        collate_fn = partial(
            seg_bank_collate_fn,
            ignore_index=ignore_index,
        )
        default_shuffle = False

    elif collate_mode in {"pad", "train"}:
        collate_fn = partial(
            seg_pad_collate_fn,
            ignore_index=ignore_index,
        )
        default_shuffle = split.lower() != "test"

    else:
        raise ValueError(
            "collate_mode must be 'bank' or 'pad', "
            f"got {collate_mode!r}"
        )

    if shuffle is None:
        shuffle = default_shuffle

    dataset = SUIMDataset(
        root=root,
        split=split,
        transform=transform,
        ambiguity_margin=ambiguity_margin,
        ignore_index=ignore_index,
        binary_threshold=binary_threshold,
        strict_test_overlap=strict_test_overlap,
        strict_test_binary=strict_test_binary,
        train_size_mismatch_policy=train_size_mismatch_policy,
        train_mask_extra_height=train_mask_extra_height,
        max_samples=max_samples,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "collate_fn": collate_fn,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }

    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**loader_kwargs)
