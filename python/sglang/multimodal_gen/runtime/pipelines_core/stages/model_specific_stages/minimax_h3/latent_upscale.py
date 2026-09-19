# SPDX-License-Identifier: Apache-2.0
"""Shared config for the learned latent upscale between denoise and decode.

Two places need to agree on the post-upscale geometry: the decoding stage, which
performs the upscale, and validate_final_outputs_sync, which probes the finished
MP4 against the resolved request. They run in different processes, so a value
stashed in batch.extra by one is not visible to the other -- both must derive the
geometry from the same pure function of the config instead.

The scale is read per request from a file when SGLANG_H3_LATENT_UPSCALE_FILE
points at one, falling back to SGLANG_H3_LATENT_UPSCALE. Per-request is safe here
in a way it is not for the resolution ladder: this runs after denoise, so the 8
ranks never build different packed sequences from it.
"""
from __future__ import annotations

import os

LATENT_CELL_PX = 16


def latent_upscale_scale() -> float:
    """Spatial scale factor; <= 1.0 means the upscale is off."""
    scale = float(os.environ.get("SGLANG_H3_LATENT_UPSCALE", "0") or 0)
    path = os.environ.get("SGLANG_H3_LATENT_UPSCALE_FILE")
    if path and os.path.exists(path):
        try:
            scale = float(open(path).read().strip() or 0)
        except (OSError, ValueError):
            return scale
    return scale


def upscaled_latent_hw(h: int, w: int, scale: float) -> tuple[int, int]:
    """Target latent cells for a spatial upscale.

    Rounded down to even counts: one latent cell is 16 px, so an even count keeps
    the decoded canvas on the 32 px alignment the upscaler's author recommends.
    """
    return (int(round(h * scale)) & ~1, int(round(w * scale)) & ~1)


def upscaled_pixel_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Decoded canvas in pixels after the upscale, from the pre-upscale canvas.

    Takes and returns pixels because that is what the resolved plan and ffprobe
    both speak, but converts through latent cells so the even-count rounding
    above is applied exactly once.
    """
    h_out, w_out = upscaled_latent_hw(
        height // LATENT_CELL_PX, width // LATENT_CELL_PX, scale
    )
    return (w_out * LATENT_CELL_PX, h_out * LATENT_CELL_PX)


__all__ = [
    "LATENT_CELL_PX",
    "latent_upscale_scale",
    "upscaled_latent_hw",
    "upscaled_pixel_size",
]
