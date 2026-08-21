# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math


def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def _minimax_h3_unshift_sigma(sigma: float, shift_scale: float) -> float:
    return sigma / (shift_scale - (shift_scale - 1.0) * sigma)


def _minimax_h3_shift_sigma(sigma: float, shift_scale: float) -> float:
    return shift_scale * sigma / (1.0 + (shift_scale - 1.0) * sigma)


def minimax_h3_pinned_sigmas(
    *,
    video_timesteps: list[float],
    video_shift_scale: float,
    audio_shift_scale: float,
) -> dict[str, list[float]]:
    """Build both sigma schedules from a pinned video timestep grid.

    Published grids are video timesteps on the 0-1000 scale, so 1000 * sigma.
    Audio stays derived rather than pinned: the sampler over-steps audio
    whenever the two grids disagree. A terminal 0 is appended when omitted.
    """
    if not video_timesteps:
        raise ValueError("MiniMax H3 pinned timesteps must not be empty")
    sigmas = [float(value) / 1000.0 for value in video_timesteps]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in sigmas):
        raise ValueError("MiniMax H3 pinned timesteps must fall within [0, 1000]")
    if sigmas[0] != 1.0:
        raise ValueError("MiniMax H3 pinned timesteps must start at 1000")
    if any(later >= earlier for earlier, later in zip(sigmas, sigmas[1:])):
        raise ValueError("MiniMax H3 pinned timesteps must be strictly descending")
    if sigmas[-1] > 0.0:
        sigmas.append(0.0)

    unshifted = [
        _minimax_h3_unshift_sigma(sigma, video_shift_scale) for sigma in sigmas
    ]
    return {
        "video": sigmas,
        "audio": [
            _minimax_h3_shift_sigma(sigma, audio_shift_scale) for sigma in unshifted
        ],
    }


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
) -> list[float]:
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    import torch

    # The rectified-flow sigma range is fixed at [1.0, 0.0].
    base = torch.linspace(
        1.0,
        0.0,
        int(num_steps),
        device="cpu",
        dtype=torch.float32,
    )
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    # A one-point request is still exactly one point.  Normal serving uses
    # multiple points, but preserving the requested cardinality keeps
    # ``num_inference_steps`` the sole schedule-size control.
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]
