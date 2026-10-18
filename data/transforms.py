from __future__ import annotations

import random
import torch
import torch.nn.functional as F


def _reflect_pad_in_chunks(
    image,
    pad_right,
    pad_bottom,
):
    remaining_right = int(pad_right)
    remaining_bottom = int(pad_bottom)
    result = image

    while remaining_right > 0 or remaining_bottom > 0:
        height, width = result.shape[-2:]

        if (
            (remaining_right > 0 and width <= 1)
            or
            (remaining_bottom > 0 and height <= 1)
        ):
            return F.pad(
                result,
                (0, remaining_right, 0, remaining_bottom),
                mode="replicate",
            )

        step_right = min(
            remaining_right,
            max(0, width-1)
        )
        step_bottom = min(
            remaining_bottom,
            max(0, height-1)
        )

        result = F.pad(
            result,
            (0, step_right, 0, step_bottom),
            mode="reflect",
        )

        remaining_right -= step_right
        remaining_bottom -= step_bottom

    return result



def paired_random_crop_flip(
    input_image,
    target_image,
    *,
    crop_size,
    horizontal_flip_probability=0.5,
):

    if input_image.shape != target_image.shape:
        raise ValueError(
            "paired shape mismatch"
        )

    h,w=input_image.shape[-2:]

    input_image=_reflect_pad_in_chunks(
        input_image,
        max(0,crop_size-w),
        max(0,crop_size-h),
    )

    target_image=_reflect_pad_in_chunks(
        target_image,
        max(0,crop_size-w),
        max(0,crop_size-h),
    )

    h,w=input_image.shape[-2:]

    top=random.randint(
        0,
        h-crop_size
    )

    left=random.randint(
        0,
        w-crop_size
    )

    input_image=input_image[
        :,
        top:top+crop_size,
        left:left+crop_size
    ]

    target_image=target_image[
        :,
        top:top+crop_size,
        left:left+crop_size
    ]


    if random.random() < horizontal_flip_probability:
        input_image=torch.flip(
            input_image,
            dims=(-1,)
        )
        target_image=torch.flip(
            target_image,
            dims=(-1,)
        )


    return (
        input_image.contiguous(),
        target_image.contiguous(),
    )

