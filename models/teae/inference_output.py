"""Only the original output resolution helper, isolated for deployment.
No private training or research code is included.
"""
from __future__ import annotations
from typing import Mapping
import torch

def resolve_enhanced_image(
    output: torch.Tensor | Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Normalize enhancer outputs to a ``[B,3,H,W]`` image tensor."""

    if isinstance(output, Mapping):
        if "enhanced" not in output:
            raise KeyError("Enhancer output must contain 'enhanced'.")
        output = output["enhanced"]
    if not torch.is_tensor(output):
        raise TypeError("Enhancer output must be a tensor or mapping.")
    if output.ndim != 4 or output.shape[1] != 3:
        raise ValueError("Enhanced image must have shape [B,3,H,W].")
    return output
