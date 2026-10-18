"""Frozen SUIM Protocol-A class-present per-image metric.
Extracted verbatim in executable logic from source with verified SHA256.
Class 0 contributes when present in GT; label 255 is ignored.
Not a global confusion-matrix metric.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

def require(x, msg):
    if not x:
        raise RuntimeError(msg)


def resize_logits(logits, mask):

    if logits.shape[-2:] != mask.shape[-2:]:

        logits = F.interpolate(
            logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return logits


def gt_present_metrics(
    logits,
    mask,
):

    logits = resize_logits(
        logits,
        mask,
    )

    pred = logits.argmax(dim=1)

    target = mask.long()

    valid = (
        (target != 255)
        & (target >= 0)
        & (target < 8)
    )

    ious = []
    dices = []

    for cls in range(8):

        gt = (
            (target == cls)
            & valid
        )

        # Preserve the original evaluation protocol's
        # GT-present semantics.
        if not bool(gt.any().item()):
            continue

        pd = (
            (pred == cls)
            & valid
        )

        inter = (
            gt & pd
        ).sum().double()

        union = (
            gt | pd
        ).sum().double()

        gt_n = gt.sum().double()
        pd_n = pd.sum().double()

        require(
            float(union.item()) > 0.0,
            "Invalid GT-present union.",
        )

        iou = inter / union

        dice_den = gt_n + pd_n

        require(
            float(dice_den.item()) > 0.0,
            "Invalid Dice denominator.",
        )

        dice = (
            2.0 * inter
            / dice_den
        )

        ious.append(
            float(iou.item())
        )

        dices.append(
            float(dice.item())
        )

    require(
        len(ious) > 0,
        "No GT-present class.",
    )

    require(
        len(ious) == len(dices),
        "Metric length drift.",
    )

    return (
        float(np.mean(ious)),
        float(np.mean(dices)),
    )


def mean_teacher_metric(
    scores,
    key,
):

    return float(
        np.mean([
            float(scores[name][key])
            for name in TEACHERS_3T
        ])
    )
