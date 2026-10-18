from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_TEACHERS = (
    "unet_vgg16",
    "segformer_b2",
    "hrnet_w18",
)
HELDOUT_LEARNER = "deeplabv3plus_resnet50"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping checkpoint: {path}")
    return value


def build_architecture(
    name: str,
    *,
    num_classes: int,
    backbone_asset_root: Path,
) -> nn.Module:
    if name == "unet_vgg16":
        from .architectures.unet_vgg16 import UNetVGG16
        return UNetVGG16(
            num_classes=num_classes,
            checkpoint_path=None,
        )
    if name == "segformer_b2":
        from .architectures.segformer_b2 import SegFormerB2
        return SegFormerB2(
            num_classes=num_classes,
            pretrained_dir=(
                backbone_asset_root / "segformer_b2_mit_b2"
            ),
        )
    if name == "hrnet_w18":
        from .architectures.hrnet_w18 import HRNetW18Segmentation
        return HRNetW18Segmentation(
            num_classes=num_classes,
            checkpoint_path=None,
        )
    if name == HELDOUT_LEARNER:
        from .architectures.deeplabv3plus import DeepLabV3PlusResNet50
        return DeepLabV3PlusResNet50(
            num_classes=num_classes,
            checkpoint_path=None,
        )
    raise KeyError(name)


class FrozenTeacherEnsemble(nn.Module):
    def __init__(
        self,
        *,
        teachers: Mapping[str, nn.Module],
        metadata: Mapping[str, Mapping[str, Any]],
    ) -> None:
        super().__init__()
        self.teachers = nn.ModuleDict(OrderedDict(teachers))
        self.metadata = {
            name: dict(value)
            for name, value in metadata.items()
        }
        for teacher in self.teachers.values():
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)

    @property
    def teacher_names(self) -> tuple[str, ...]:
        return tuple(self.teachers.keys())

    def train(self, mode: bool = True) -> "FrozenTeacherEnsemble":
        super().train(False)
        for teacher in self.teachers.values():
            teacher.eval()
        return self

    def forward_teacher(
        self,
        name: str,
        image: torch.Tensor,
    ) -> Mapping[str, Any]:
        if name not in self.teachers:
            raise KeyError(name)
        h, w = image.shape[-2:]

        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32

        if (new_h, new_w) != (h, w):
            image = F.interpolate(
                image,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )

        output = self.teachers[name](image)
        if not isinstance(output, Mapping):
            raise TypeError(
                f"{name}: expected mapping output, got {type(output)}"
            )
        if "logits" not in output or "feature_pyramid" not in output:
            raise RuntimeError(
                f"{name}: missing logits/feature_pyramid output"
            )
        return output


def load_frozen_teacher_ensemble(
    *,
    project_root: Path,
    device: torch.device,
    teacher_names: Sequence[str] = TRAIN_TEACHERS,
    num_classes: int = 8,
    allow_heldout: bool = False,
) -> FrozenTeacherEnsemble:
    root = Path(project_root).expanduser().resolve()
    manifest_path = root / "assets" / "teachers" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {
        str(entry["name"]): dict(entry)
        for entry in manifest.get("models", [])
    }

    names = tuple(str(name) for name in teacher_names)
    if HELDOUT_LEARNER in names and not allow_heldout:
        raise RuntimeError(
            "Held-out DeepLab cannot be loaded into development training."
        )
    if not allow_heldout and any(
        entries[name].get("heldout", False) for name in names
    ):
        raise RuntimeError("Held-out learner leaked into development ensemble.")

    backbone_root = (
        root / "assets" / "backbones" / "seg_teachers"
    ).resolve()
    if not backbone_root.is_dir():
        raise FileNotFoundError(backbone_root)

    teachers: OrderedDict[str, nn.Module] = OrderedDict()
    metadata: dict[str, dict[str, Any]] = {}

    for name in names:
        if name not in entries:
            raise KeyError(f"Teacher absent from manifest: {name}")
        entry = entries[name]
        checkpoint_path = (
            root / "assets" / "teachers" / name / "final.pth"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_sha = file_sha256(checkpoint_path)
        expected_sha = str(entry["sha256"])
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{name}: checkpoint SHA mismatch "
                f"{actual_sha} != {expected_sha}"
            )

        checkpoint = safe_torch_load(checkpoint_path)
        if checkpoint.get("model_name") != name:
            raise RuntimeError(f"{name}: model_name mismatch")
        if checkpoint.get("training_scope") != "split":
            raise RuntimeError(f"{name}: expected split-scope checkpoint")
        if int(checkpoint.get("epoch", -1)) != 80:
            raise RuntimeError(f"{name}: expected epoch 80 final checkpoint")

        model = build_architecture(
            name,
            num_classes=num_classes,
            backbone_asset_root=backbone_root,
        )
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"{name}: strict state loading failed")
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError(f"{name}: teacher is not fully frozen")

        teachers[name] = model
        metadata[name] = {
            "name": name,
            "checkpoint": str(checkpoint_path),
            "sha256": actual_sha,
            "epoch": 80,
            "training_scope": "split",
            "heldout": bool(entry.get("heldout", False)),
        }

    ensemble = FrozenTeacherEnsemble(
        teachers=teachers,
        metadata=metadata,
    ).to(device)
    ensemble.eval()
    return ensemble
