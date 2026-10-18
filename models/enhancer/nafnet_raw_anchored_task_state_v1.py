from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.enhancer.nafnet_raw_anchored import NAFBlock


class TaskStateSpatialFiLMV1(nn.Module):
    """
    Content-only spatial task-state modulation.

    Critical control semantics:
      1. to_affine has NO bias.
      2. to_affine weight is exactly zero initialized.
      3. task_state=None is exact identity.
      4. task_state=zeros is exact identity for ALL parameter values.

    Therefore ZeroState cannot become an unconditional
    learnable extra-capacity path during training.
    """

    def __init__(
        self,
        *,
        state_channels: int,
        feature_channels: int,
        state_norm_groups: int = 32,
    ) -> None:
        super().__init__()

        if state_channels <= 0:
            raise ValueError(
                "state_channels must be positive."
            )

        if feature_channels <= 0:
            raise ValueError(
                "feature_channels must be positive."
            )

        if (
            state_channels
            % state_norm_groups
            != 0
        ):
            raise ValueError(
                "state_channels must be divisible "
                "by state_norm_groups."
            )

        self.state_channels = int(
            state_channels
        )

        self.feature_channels = int(
            feature_channels
        )

        self.state_norm = nn.GroupNorm(
            num_groups=int(
                state_norm_groups
            ),
            num_channels=int(
                state_channels
            ),
            affine=False,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # bias=False makes ZeroState a permanent exact identity
        # control, not merely an initialization identity.
        # ----------------------------------------------------
        self.to_affine = nn.Conv2d(
            int(state_channels),
            int(feature_channels) * 2,
            kernel_size=1,
            bias=False,
        )

        nn.init.zeros_(
            self.to_affine.weight
        )

    def forward(
        self,
        feature: torch.Tensor,
        task_state: torch.Tensor | None,
    ) -> torch.Tensor:

        if task_state is None:
            return feature

        if task_state.ndim != 4:
            raise ValueError(
                "task_state must be [B,C,H,W]."
            )

        if (
            task_state.shape[0]
            != feature.shape[0]
        ):
            raise RuntimeError(
                "task_state/feature batch mismatch."
            )

        if (
            task_state.shape[1]
            != self.state_channels
        ):
            raise RuntimeError(
                "task_state channel mismatch: "
                f"expected {self.state_channels}, "
                f"got {task_state.shape[1]}."
            )

        if (
            feature.shape[1]
            != self.feature_channels
        ):
            raise RuntimeError(
                "feature channel mismatch: "
                f"expected {self.feature_channels}, "
                f"got {feature.shape[1]}."
            )

        state = task_state

        if (
            state.shape[-2:]
            != feature.shape[-2:]
        ):
            state = F.interpolate(
                state,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        state = self.state_norm(
            state.float()
        ).to(
            dtype=feature.dtype
        )

        affine = self.to_affine(
            state
        )

        gamma, beta = torch.chunk(
            affine,
            chunks=2,
            dim=1,
        )

        return (
            feature
            * (
                1.0
                + gamma
            )
            + beta
        )


class RawAnchoredNAFNetTaskStateV1(nn.Module):
    """
    RawAnchoredNAFNet with HRNet multi-scale current-task-state
    conditioning.

    Default HRNet mapping:

      HRNet L0 (64)   -> NAF encoder stage 0
      HRNet L1 (128)  -> NAF encoder stage 1
      HRNet L2 (256)  -> NAF encoder stage 2
      HRNet L3 (512)  -> NAF encoder stage 3
      HRNet L4 (1024) -> NAF middle

    HRNet L5 is intentionally NOT used here.
    L5 remains the desired-transition OTD feature.

    All task-state modules are constructed AFTER the original
    NAFNet core so core initialization/loading remains matched.
    """

    def __init__(
        self,
        *,
        width: int = 32,
        encoder_blocks: Sequence[int] = (
            2, 2, 4, 8
        ),
        middle_blocks: int = 12,
        decoder_blocks: Sequence[int] = (
            2, 2, 2, 2
        ),
        gate_floor: float = 0.05,
        gate_ceiling: float = 1.0,
        task_state_channels: Sequence[int] = (
            64,
            128,
            256,
            512,
            1024,
        ),
        state_norm_groups: int = 32,
    ) -> None:
        super().__init__()

        if (
            len(encoder_blocks)
            != len(decoder_blocks)
        ):
            raise ValueError(
                "Encoder/decoder stage counts must match."
            )

        if not (
            0.0
            <= gate_floor
            <= gate_ceiling
            <= 1.0
        ):
            raise ValueError(
                "Require "
                "0 <= gate_floor <= gate_ceiling <= 1."
            )

        self.stage_count = len(
            encoder_blocks
        )

        if (
            len(task_state_channels)
            != self.stage_count + 1
        ):
            raise ValueError(
                "task_state_channels must contain "
                "one entry per encoder stage plus middle."
            )

        self.padder_size = (
            2 ** self.stage_count
        )

        self.gate_floor = float(
            gate_floor
        )

        self.gate_ceiling = float(
            gate_ceiling
        )

        self.task_state_channels = tuple(
            int(x)
            for x in task_state_channels
        )

        # ====================================================
        # Original RawAnchoredNAFNet core.
        # Construction order intentionally preserved.
        # ====================================================

        self.intro = nn.Conv2d(
            3,
            width,
            kernel_size=3,
            padding=1,
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width

        for block_count in encoder_blocks:

            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            channels
                        )
                        for _ in range(
                            int(
                                block_count
                            )
                        )
                    ]
                )
            )

            self.downs.append(
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=2,
                    stride=2,
                )
            )

            channels *= 2

        self.middle = nn.Sequential(
            *[
                NAFBlock(
                    channels
                )
                for _ in range(
                    int(
                        middle_blocks
                    )
                )
            ]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for block_count in decoder_blocks:

            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.PixelShuffle(
                        2
                    ),
                )
            )

            channels //= 2

            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            channels
                        )
                        for _ in range(
                            int(
                                block_count
                            )
                        )
                    ]
                )
            )

        self.residual_head = nn.Conv2d(
            width,
            3,
            kernel_size=3,
            padding=1,
        )

        self.gate_head = nn.Conv2d(
            width,
            1,
            kernel_size=3,
            padding=1,
        )

        nn.init.zeros_(
            self.residual_head.weight
        )

        nn.init.zeros_(
            self.residual_head.bias
        )

        nn.init.zeros_(
            self.gate_head.weight
        )

        nn.init.zeros_(
            self.gate_head.bias
        )

        # ====================================================
        # Task-state modules constructed LAST.
        # ====================================================

        encoder_feature_channels = [
            int(width)
            * (
                2 ** idx
            )
            for idx in range(
                self.stage_count
            )
        ]

        middle_feature_channels = (
            int(width)
            * (
                2 ** self.stage_count
            )
        )

        self.task_state_films = (
            nn.ModuleList(
                [
                    TaskStateSpatialFiLMV1(
                        state_channels=
                            self.task_state_channels[
                                idx
                            ],
                        feature_channels=
                            encoder_feature_channels[
                                idx
                            ],
                        state_norm_groups=
                            int(
                                state_norm_groups
                            ),
                    )
                    for idx in range(
                        self.stage_count
                    )
                ]
            )
        )

        self.task_state_middle_film = (
            TaskStateSpatialFiLMV1(
                state_channels=
                    self.task_state_channels[
                        -1
                    ],
                feature_channels=
                    middle_feature_channels,
                state_norm_groups=
                    int(
                        state_norm_groups
                    ),
            )
        )

    def _pad(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        int,
        int,
    ]:

        height, width = x.shape[
            -2:
        ]

        pad_h = (
            self.padder_size
            - height
            % self.padder_size
        ) % self.padder_size

        pad_w = (
            self.padder_size
            - width
            % self.padder_size
        ) % self.padder_size

        return (
            F.pad(
                x,
                (
                    0,
                    pad_w,
                    0,
                    pad_h,
                ),
            ),
            height,
            width,
        )

    def _validate_task_state(
        self,
        task_state_pyramid:
            Sequence[torch.Tensor]
            | None,
        batch_size: int,
    ) -> None:

        if task_state_pyramid is None:
            return

        if (
            len(task_state_pyramid)
            != self.stage_count + 1
        ):
            raise ValueError(
                "Expected "
                f"{self.stage_count + 1} "
                "task-state tensors."
            )

        for level, (
            state,
            expected_channels,
        ) in enumerate(
            zip(
                task_state_pyramid,
                self.task_state_channels,
            )
        ):

            if state.ndim != 4:
                raise ValueError(
                    f"task state L{level} "
                    "must be [B,C,H,W]."
                )

            if (
                state.shape[0]
                != batch_size
            ):
                raise ValueError(
                    f"task state L{level} "
                    "batch mismatch."
                )

            if (
                state.shape[1]
                != expected_channels
            ):
                raise ValueError(
                    f"task state L{level} "
                    "channel mismatch: "
                    f"expected "
                    f"{expected_channels}, "
                    f"got "
                    f"{state.shape[1]}."
                )

    def forward(
        self,
        raw: torch.Tensor,
        *,
        task_state_pyramid:
            Sequence[torch.Tensor]
            | None = None,
    ) -> dict[
        str,
        torch.Tensor,
    ]:

        if (
            raw.ndim != 4
            or raw.shape[1] != 3
        ):
            raise ValueError(
                "raw must be [B,3,H,W]."
            )

        self._validate_task_state(
            task_state_pyramid,
            int(
                raw.shape[0]
            ),
        )

        padded, height, width = (
            self._pad(
                raw
            )
        )

        value = self.intro(
            padded
        )

        skips = []

        for stage_index, (
            encoder,
            down,
        ) in enumerate(
            zip(
                self.encoders,
                self.downs,
            )
        ):

            value = checkpoint(
                encoder,
                value,
                use_reentrant=False,
            )

            state = (
                None
                if task_state_pyramid
                is None
                else task_state_pyramid[
                    stage_index
                ]
            )

            value = (
                self.task_state_films[
                    stage_index
                ](
                    value,
                    state,
                )
            )

            skips.append(
                value
            )

            value = down(
                value
            )

        value = checkpoint(
            self.middle,
            value,
            use_reentrant=False,
        )

        middle_state = (
            None
            if task_state_pyramid
            is None
            else task_state_pyramid[
                -1
            ]
        )

        value = (
            self.task_state_middle_film(
                value,
                middle_state,
            )
        )

        for up, decoder, skip in zip(
            self.ups,
            self.decoders,
            reversed(
                skips
            ),
        ):

            value = up(
                value
            )

            if (
                value.shape[-2:]
                != skip.shape[-2:]
            ):
                value = F.interpolate(
                    value,
                    size=
                        skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            value = checkpoint(
                decoder,
                value + skip,
                use_reentrant=False,
            )

        residual = torch.tanh(
            self.residual_head(
                value
            )
        )

        gate_unit = torch.sigmoid(
            self.gate_head(
                value
            )
        )

        gate = (
            self.gate_floor
            +
            (
                self.gate_ceiling
                - self.gate_floor
            )
            * gate_unit
        )

        enhanced = torch.clamp(
            padded
            + gate
            * residual,
            0.0,
            1.0,
        )

        return {
            "enhanced":
                enhanced[
                    ...,
                    :height,
                    :width,
                ],

            "residual":
                residual[
                    ...,
                    :height,
                    :width,
                ],

            "gate":
                gate[
                    ...,
                    :height,
                    :width,
                ],
        }


class TEAETaskStateStudentEnhancerV1(
    nn.Module
):
    def __init__(
        self,
        backbone:
            RawAnchoredNAFNetTaskStateV1,
    ) -> None:
        super().__init__()

        self.backbone = backbone

    def forward(
        self,
        raw: torch.Tensor,
        *,
        task_state_pyramid:
            Sequence[torch.Tensor]
            | None = None,
    ) -> dict[
        str,
        torch.Tensor,
    ]:

        from models.teae.inference_output import resolve_enhanced_image

        raw = resolve_enhanced_image(
            raw
        )

        return self.backbone(
            raw,
            task_state_pyramid=
                task_state_pyramid,
        )
