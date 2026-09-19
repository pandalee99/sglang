# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENT: run the early denoise steps on a coarser latent grid.

Motivation. Generating natively at 2K measures 9.80x the per-step cost of 768p
(3915.4ms vs 399.7ms at 5s, 8 forwards, cache off) while carrying real extra
detail -- 6.02x/73.94x the energy above the lanczos ceiling f=0.25, confirmed as
structure rather than grain because the high band's adjacent-frame coherence
(0.8382) sits at 0.973 of its own mid band (0.8613). Paying that on every step is
what puts 2K out of the real-time budget, so the question is whether the detail
can be bought on only the last few steps.

This is also the only way to obtain PAIRED (768p, 2K) training data. Generating
the two resolutions independently does NOT produce pairs: measured best aligned
SSIM is 0.8120 against a 0.9934 same-clip neighbour reference, with per-probe
temporal offsets disagreeing (-5/-3/+15), so the trajectories genuinely diverge
from the first step. Sharing the early trajectory and branching late constructs
the correspondence instead of hoping for it.

Mechanism: run steps [0, k) on a coarse grid, resize the latent, run [k, n) on
the requested grid. The sigma schedule is split at k, not resampled, so both
segments see exactly the timesteps the schedule prescribes.

Gated by SGLANG_H3_LADDER_STEPS (0/unset = shipped behaviour, untouched).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
)

MINIMAX_H3_LADDER_PATCH = (1, 2, 2)
MINIMAX_H3_LADDER_LATENT_CHANNELS = 24

# Cached across requests: loading the 345M-param resizer costs more than running it.
_LEARNED_UPSCALER = None


def _env_or_file(env_key: str, file_key: str, default: str) -> str:
    """Read a knob from $env_key, overridden by the contents of $file_key's path.

    The file form exists so a sweep does not cost a 98 s cold start per arm. It is
    safe for the two renoise knobs specifically: every rank reads the same file and
    the sweep only writes it while the server is idle between requests, so the 8
    ranks never disagree within one request. Do NOT extend this to a knob that
    changes the packed layout mid-request -- that is why the shipped comment calls
    these launch-time.
    """
    value = os.environ.get(env_key, default)
    path = os.environ.get(file_key)
    if path and os.path.exists(path):
        try:
            text = open(path).read().strip()
        except OSError:
            return value
        if text:
            return text
    return value


def minimax_h3_ladder_steps() -> int:
    """Number of trailing steps to run at the requested resolution.

    0 (the default) disables the ladder entirely: no extra branch is built and
    the denoise loop runs exactly as shipped.
    """
    return int(float(_env_or_file(
        "SGLANG_H3_LADDER_STEPS", "SGLANG_H3_LADDER_STEPS_FILE", "0")))


def minimax_h3_ladder_coarse_grid(latent_h: int, latent_w: int) -> tuple[int, int]:
    """Halve a latent grid, keeping both axes even.

    Patchify requires even latent dims, and rounding down to even lands exactly
    on the shipped geometry for the case that matters: the 2K grid 96x170 gives
    48x84, which is 768x1344 -- the only resolution this checkpoint is tuned at.
    """
    coarse_h = max(2, (int(latent_h) // 2) // 2 * 2)
    coarse_w = max(2, (int(latent_w) // 2) // 2 * 2)
    return coarse_h, coarse_w


def minimax_h3_resize_video_rows(
    rows: torch.Tensor,
    *,
    latent_t: int,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    """Move packed video rows between latent grids, preserving frame count.

    Rows are patchified latents, so they cannot be interpolated directly -- each
    row interleaves a 2x2 spatial patch across its 96 values. Unpatchify to
    [B,C,T,H,W] first, resize the spatial axes only, then re-patchify.

    Trilinear with the temporal size pinned rather than bilinear per frame: it is
    one kernel over the whole clip and, with size[0] == T, is exactly separable
    in time, so no frame is blended into its neighbours.
    """
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    channels = MINIMAX_H3_LADDER_LATENT_CHANNELS
    latent = minimax_h3_unpatchify_video_tokens(
        rows,
        latent_shape=[int(latent_t), src_h // 2, src_w // 2, channels],
        patch_size=list(MINIMAX_H3_LADDER_PATCH),
    )
    if (src_h, src_w) != (dst_h, dst_w):
        latent = F.interpolate(
            latent.to(torch.float32),
            size=(int(latent_t), dst_h, dst_w),
            mode="trilinear",
            align_corners=False,
        )
    return minimax_h3_patchify_video_latent(
        latent, patch_size=list(MINIMAX_H3_LADDER_PATCH)
    )


def minimax_h3_ladder_upsample_mode() -> str:
    """Which upsampler carries the handoff latent to the fine grid.

    "trilinear" (default) is the shipped local average. "learned" runs the
    community LatentResizer3D instead.

    Why this knob exists: the learned upsampler was shown to reproduce native 2K's
    high-frequency content on TEXTURE (hi1/hi2 6.13x/73.93x against a size-matched
    lanczos, matching native 2K's own 6.02x/73.94x) with zero refinement steps --
    but NOT on burned-in text. Measured on four animated arms with Chinese
    subtitles: it narrows glyph edges only ~20% and makes strokes 1-2 px THICKER,
    while native 2K draws them at 5-6 px. It is a conv net over latents with no
    notion of a character, so it can sharpen a blob it was handed and cannot
    recover a stroke the 768p grid never resolved. The DiT can. Feeding the DiT a
    better-than-trilinear starting point is what makes spending fine steps on it
    worth the cost.

    "surgery" is "learned" with its low band replaced by trilinear's, which is
    the only one of the three measured to beat both baselines against a paired
    reference (see `minimax_h3_ladder_band_surgery`).
    """
    return _env_or_file(
        "SGLANG_H3_LADDER_UPSAMPLE", "SGLANG_H3_LADDER_UPSAMPLE_FILE", "trilinear"
    ).strip().lower()


def minimax_h3_learned_resize_video_rows(
    rows: torch.Tensor,
    *,
    latent_t: int,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    """Move packed video rows between latent grids via the learned upscaler.

    Same contract as `minimax_h3_resize_video_rows` so the two are interchangeable
    at the handoff: unpatchify, resize spatially, re-patchify, frame count intact.

    The (x-mean)/std wrapper is mandatory and non-obvious. These rows hold
    NORMALIZED latents (decode's `_reverse_normalize_latents` is what makes them
    raw), and the upscaler's hardcoded LATENTS_MEAN/STD equal our video_vae
    config's to max|diff| = 0.000e+00, so the wrapper looks like a second
    normalization. It is not: the pipeline's normalized latents measure per-channel
    std 0.7957 rather than 1.0, because LATENTS_STD are corpus-wide statistics and
    a single clip is less dispersed than the corpus. Skipping the wrapper drives
    the output's per-channel statistics 6.80 away from the input's against 0.32
    through it, and decodes to mean luma 193.7 vs 98.6 on a FIXED seed -- brightness
    moving on a fixed seed is off-distribution, not sharper.

    Runs on every rank rather than on rank 0 plus a broadcast: the rows are
    replicated here, the net is deterministic, and all ranks load the same weights,
    so each arrives at the same tensor without a collective.
    """
    import sys

    if "/results" not in sys.path:
        sys.path.insert(0, "/results")
    from h3_upscaler import latent_norm_stats, load_upscaler

    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    latent = minimax_h3_unpatchify_video_tokens(
        rows,
        latent_shape=[int(latent_t), src_h // 2, src_w // 2,
                      MINIMAX_H3_LADDER_LATENT_CHANNELS],
        patch_size=list(MINIMAX_H3_LADDER_PATCH),
    )
    if (src_h, src_w) != (dst_h, dst_w):
        global _LEARNED_UPSCALER
        if _LEARNED_UPSCALER is None:
            ckpt = os.environ.get(
                "SGLANG_H3_LATENT_UPSCALE_CKPT",
                "/models/h3_latent_upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors",
            )
            _LEARNED_UPSCALER = load_upscaler(
                ckpt, device=latent.device, dtype=latent.dtype
            )
        mean, std = latent_norm_stats(latent.device, latent.dtype)
        # The net conditions on `scale - 1`, so it needs the scale it is actually
        # being asked for; dst/src on the h axis is that number by construction.
        scale = dst_h / src_h
        with torch.inference_mode():
            out = _LEARNED_UPSCALER(
                (latent - mean) / std,
                scale=scale,
                target_size=(int(latent_t), dst_h, dst_w),
            )
            latent = (out * std + mean).to(rows.dtype)
    return minimax_h3_patchify_video_latent(
        latent, patch_size=list(MINIMAX_H3_LADDER_PATCH)
    )


def minimax_h3_resize_keyframe_rows(
    rows: torch.Tensor,
    *,
    n_keyframes: int,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    """Resize keyframe condition rows to a different latent grid.

    Condition rows are stacked per keyframe, each a single-frame latent, so the
    same row transform applies with latent_t=1 per keyframe. Splitting instead of
    treating the stack as one clip keeps the resize from mixing separate
    keyframes together along the temporal axis.
    """
    if n_keyframes <= 0:
        raise ValueError(f"n_keyframes must be positive, got {n_keyframes}")
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    rows_per_keyframe = (src_h // 2) * (src_w // 2)
    if int(rows.shape[0]) != rows_per_keyframe * n_keyframes:
        raise ValueError(
            f"keyframe rows {int(rows.shape[0])} != {n_keyframes} x "
            f"{rows_per_keyframe} for src grid {src_h}x{src_w}"
        )
    return torch.cat(
        [
            minimax_h3_resize_video_rows(
                rows[i * rows_per_keyframe : (i + 1) * rows_per_keyframe],
                latent_t=1,
                src_hw=(src_h, src_w),
                dst_hw=dst_hw,
            )
            for i in range(int(n_keyframes))
        ]
    )


def minimax_h3_ladder_noise(
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    seed: int,
    row_width: int,
) -> torch.Tensor:
    """Fresh unit-variance noise for the coarse grid.

    Downsampling the request's fine-grid noise would be wrong: trilinear
    averaging cuts the variance about 4x, and the flow-matching schedule assumes
    a unit-variance start, so the first sigma would no longer correspond to the
    noise actually present. Drawing fresh noise at the coarse shape is what a
    native run at that resolution does.

    Drawn the way latent_preparation.py draws it -- randn on the RAW latent
    tensor [1, 24, T, H, W] and then patchified -- not directly on the packed row
    shape. Both give unit variance, but randn fills in tensor order, so the two
    orders scatter the same numbers to different rows and the coarse segment then
    follows a DIFFERENT trajectory from a native run at the same seed. Measured
    cost of getting this wrong: the coarse product matched an independent 768p run
    at only 0.73 SSIM, worse than full 2K's own 0.85 against it -- i.e. no pair at
    all, which is the one thing this ladder exists to construct.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latent = torch.randn(
        1,
        MINIMAX_H3_LADDER_LATENT_CHANNELS,
        int(latent_t),
        int(latent_h),
        int(latent_w),
        generator=generator,
        dtype=torch.float32,
    )
    rows = minimax_h3_patchify_video_latent(
        latent, patch_size=list(MINIMAX_H3_LADDER_PATCH)
    ).to(torch.float32)
    if int(rows.shape[1]) != int(row_width):
        raise ValueError(
            f"coarse noise row width {int(rows.shape[1])} != {int(row_width)}"
        )
    return rows


def minimax_h3_ladder_renoise_sigma() -> float:
    """Sigma to re-noise to for the self-bootstrap mode. 0/unset = off.

    Splitting the schedule by STEP INDEX does not work on this model, measured:
    flow_shift 12 back-loads the transport so hard that step 6 of 8 still sits at
    sigma 0.8000 (the last single step carries 63.2% of the total). Every
    step-index split therefore hands the fine segment a noise-dominated state,
    and upsampling makes that noise smooth rather than white -- off-distribution
    whatever its variance. Measured cost: adjacent-frame mean|delta| 8.1 (6+2) and
    10.7 (4+4) grey levels against 0.14 for a native run.

    Re-noising instead reaches a LOW sigma by construction: run the whole
    schedule at the coarse grid to a finished latent, upsample that, then add
    fresh white noise to a chosen small sigma and denoise a few fine steps. The
    state handed over is then mostly clean signal plus genuinely white noise,
    which is what the model was trained on.
    """
    return float(_env_or_file(
        "SGLANG_H3_LADDER_RENOISE", "SGLANG_H3_LADDER_RENOISE_FILE", "0"))


def minimax_h3_ladder_renoise(
    rows: torch.Tensor, *, sigma: float, seed: int
) -> torch.Tensor:
    """Re-noise a finished latent to `sigma` in the schedule's own convention.

    This path's states satisfy x = (1-sigma)*x0 + sigma*eps (video_sigma_t is
    1-sigma in the denoise loop), so reconstructing that mixture -- rather than
    the x0 + sigma*eps convention other schedulers use -- is what makes the fine
    segment's first timestep mean what the model thinks it means.
    """
    sigma = float(sigma)
    if not 0.0 < sigma < 1.0:
        raise ValueError(f"renoise sigma must be in (0, 1), got {sigma}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    eps = torch.randn(
        tuple(rows.shape), generator=generator, dtype=torch.float32
    ).to(device=rows.device, dtype=rows.dtype)
    return (1.0 - sigma) * rows + sigma * eps


def minimax_h3_ladder_renoise_schedule(sigma: float, steps: int) -> list[float]:
    """Uniform-in-sigma descent from `sigma` to 0 over `steps` model calls.

    Uniform rather than shift-warped: the shift exists to concentrate effort near
    sigma 1 where the trajectory is still being chosen, and this segment starts
    well below that -- it is refining an already-decided image, so every step here
    should carry the same amount of transport.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"renoise schedule needs at least 1 step, got {steps}")
    return [float(sigma) * (1.0 - i / steps) for i in range(steps)] + [0.0]


def minimax_h3_ladder_progressive() -> bool:
    """Hand off a CLEAN x0 prediction mid-schedule instead of a noisy state.

    Third ladder mode, and the only one that spends exactly the schedule's own
    step count. The step-index split fails because it upsamples the state
    x = (1-sigma)*x0 + sigma*eps while sigma is still ~0.8, and a local average
    turns that eps smooth rather than white. The renoise ladder fixes the
    whiteness but pays the WHOLE schedule at the coarse grid and then extra fine
    steps on top, so it can only ever refine a decision already made at 768p.

    This mode splits the difference: run the head at the coarse grid but let its
    last step land on the x0 prediction rather than on x_k, upsample that clean
    latent, then add fresh white noise back to the schedule's own sigma_k. The
    upsampler sees no noise at all (its measured in-distribution case) and the
    fine segment sees clean signal plus genuinely white noise -- while the
    schedule's remaining transport, which flow_shift 12 concentrates almost
    entirely in the tail (step 8 of 8 alone carries 63.2%), happens at 2K.
    """
    return _env_or_file(
        "SGLANG_H3_LADDER_PROGRESSIVE", "SGLANG_H3_LADDER_PROGRESSIVE_FILE", "0"
    ).strip().lower() not in ("", "0", "false", "no", "off")


def minimax_h3_ladder_x0_schedule(
    sigmas: list[float], fine_steps: int
) -> list[float]:
    """Coarse video schedule whose final step lands on the x0 PREDICTION.

    Same split point as `minimax_h3_ladder_split_sigmas`, but the coarse tail
    sigma is replaced by 0.0. That costs nothing and adds no forward: the loop
    builds `sigma_ratio = sigmas[step+1]/sigmas[step]`, so a trailing zero makes
    the last step's ratio 0 and leaves `state = x + sigma*velocity`, which in this
    schedule's convention x = (1-sigma)*x0 + sigma*eps is exactly x0. The
    timesteps the model is shown come from `sigmas[:-1]` and are untouched, so
    every coarse step still runs at the sigma the scheduler chose.

    The one step that changes meaning is the last coarse one: instead of Euler's
    deterministic hop to x_k it jumps to x0 and the caller noises back to sigma_k
    with a fresh draw. That is a stochastic step, not a shortcut -- the total
    forward count over both segments is still the requested n.
    """
    n_steps = len(sigmas) - 1
    if not 0 < int(fine_steps) < n_steps:
        raise ValueError(
            f"fine_steps must be in (0, {n_steps}), got {int(fine_steps)}"
        )
    split = n_steps - int(fine_steps)
    return [float(v) for v in sigmas[:split]] + [0.0]


def minimax_h3_ladder_surgery_gain() -> float:
    """High-band gain for upsample mode "surgery". Default is the measured b*.

    b* = <ho,hg>/|ho|^2 is the fidelity-optimal scalar, and it landed at 0.687 and
    0.666 on two clips of different content and seed -- so 0.687 is a calibrated
    default rather than a guess. Shares the env keys with decode's post-hoc
    surgery so an A/B sweeps both paths from one file.
    """
    return float(_env_or_file(
        "SGLANG_H3_UPSCALE_HI_GAIN", "SGLANG_H3_UPSCALE_HI_GAIN_FILE", "0.687"))


def minimax_h3_ladder_band_surgery(
    learned: torch.Tensor,
    trilinear: torch.Tensor,
    *,
    latent_t: int,
    dst_hw: tuple[int, int],
    gain: float,
) -> torch.Tensor:
    """Trilinear's low band plus `gain` x the learned upsampler's high band.

    Measured against a true paired reference (e9_band_surgery.py, two clips): the
    learned upsampler scores 3.96 dB BELOW plain trilinear even though its high
    band correlates BETTER with the truth (0.539 vs 0.380), because it rewrites
    the whole latent and damages the LOW band the coarse input already carried
    exactly (low-band corr 0.99970 -> 0.95438). Splicing the other way beats both
    baselines: 31.52 dB at gain 0.687 against trilinear's 30.78.

    The band edge is a downsample-and-back round trip rather than an FFT cutoff,
    so it coincides exactly with "what the coarse grid could already represent" --
    which is what makes one scalar gain transfer across clips.
    """
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    channels = MINIMAX_H3_LADDER_LATENT_CHANNELS
    shape = [int(latent_t), dst_h // 2, dst_w // 2, channels]
    patch = list(MINIMAX_H3_LADDER_PATCH)
    lat_learned = minimax_h3_unpatchify_video_tokens(
        learned, latent_shape=shape, patch_size=patch
    ).to(torch.float32)
    lat_tri = minimax_h3_unpatchify_video_tokens(
        trilinear, latent_shape=shape, patch_size=patch
    ).to(torch.float32)

    def low_band(x: torch.Tensor) -> torch.Tensor:
        half = F.interpolate(
            x,
            size=(x.shape[2], max(1, x.shape[3] // 2), max(1, x.shape[4] // 2)),
            mode="area",
        )
        return F.interpolate(
            half, size=x.shape[2:], mode="trilinear", align_corners=False
        )

    spliced = low_band(lat_tri) + float(gain) * (lat_learned - low_band(lat_learned))
    return minimax_h3_patchify_video_latent(
        spliced.to(learned.dtype), patch_size=patch
    )


def minimax_h3_ladder_noise_attenuation(
    *,
    latent_t: int,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    row_width: int,
    seed: int = 0,
) -> float:
    """Std ratio of white noise pushed through the handoff resize.

    Measured rather than assumed: the exact factor depends on the grid pair and on
    trilinear's kernel, and the whole correction below is only as good as this
    number. Calibrating on white noise is the right probe because the state's
    noise component IS white in latent space -- it was drawn with randn.
    """
    n_rows = int(latent_t) * (int(src_hw[0]) // 2) * (int(src_hw[1]) // 2)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    probe = torch.randn(
        (n_rows, int(row_width)), generator=generator, dtype=torch.float32
    )
    out = minimax_h3_resize_video_rows(
        probe, latent_t=int(latent_t), src_hw=src_hw, dst_hw=dst_hw
    )
    src_std = float(probe.std())
    return float(out.std()) / src_std if src_std > 0 else 1.0


def minimax_h3_ladder_restore_noise(
    rows: torch.Tensor,
    *,
    sigma: float,
    attenuation: float,
    seed: int,
) -> torch.Tensor:
    """Top the handoff state's noise component back up to its sigma.

    Upsampling is a local average, so it passes the smooth partly-denoised
    component through (measured 0.995 on a smooth field) while attenuating the
    high-frequency noise component (measured 0.652 on white noise, a 57.5%
    variance deficit). The fine segment is then handed a state quieter than the
    sigma the scheduler tells it it is at, so it removes more than is there.
    Measured cost of leaving this uncorrected: adjacent-frame mean|delta| of 8-11
    grey levels on a static head-and-shoulders shot, against 0.14 for a native
    run -- whole-frame per-frame jitter, and NOT fixed by spending more fine
    steps (4 fine steps scored 0.8249 mid-band coherence vs 2 steps' 0.8222).

    In this schedule's convention x = (1-sigma)*x0 + sigma*eps, so adding fresh
    noise of std sigma*sqrt(1-attenuation^2) makes the total noise variance
    sigma^2*(a^2 + 1 - a^2) = sigma^2 again. Fresh noise rather than a rescale of
    what survived: rescaling would amplify the surviving low-frequency noise by
    1/a and leave the missing high frequencies missing.
    """
    deficit = 1.0 - float(attenuation) ** 2
    if deficit <= 0.0:
        return rows
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    extra = torch.randn(
        tuple(rows.shape), generator=generator, dtype=torch.float32
    ).to(device=rows.device, dtype=rows.dtype)
    return rows + float(sigma) * (deficit**0.5) * extra


def minimax_h3_ladder_split_sigmas(
    sigmas: list[float], fine_steps: int
) -> tuple[list[float], list[float]]:
    """Split one n-step schedule into a coarse prefix and a fine suffix.

    A schedule of n steps has n+1 sigmas. Splitting at k=n-fine_steps shares the
    boundary sigma between the two segments, so the coarse segment ends exactly
    where the fine one begins and the union runs the same n timesteps the
    scheduler prescribed. Resampling two shorter schedules instead would change
    every timestep, making the ladder's output incomparable to a native run.
    """
    n_steps = len(sigmas) - 1
    if not 0 < int(fine_steps) < n_steps:
        raise ValueError(
            f"fine_steps must be in (0, {n_steps}), got {int(fine_steps)}"
        )
    split = n_steps - int(fine_steps)
    return [float(v) for v in sigmas[: split + 1]], [
        float(v) for v in sigmas[split:]
    ]


def minimax_h3_ladder_coarse_keyframe(
    keyframe: Mapping[str, Any] | None,
    *,
    coarse_hw: tuple[int, int],
) -> dict[str, Any] | None:
    """Restate a keyframe payload on the coarse grid.

    The payload is not just rows: condition noise augmentation draws its noise at
    each condition's own ``latent_h``/``latent_w`` (see
    ``minimax_h3_imgvid_cond_noise_aug_rows``), and ``_imgvid_condition_shapes``
    reads the same fields off the per-keyframe entries. Resizing rows while
    leaving those dims at the fine values would let the aug step build noise of
    the wrong shape, so every copy of the grid has to move together.

    Semantic metadata (anchor indices, frame_count) describes time, not space,
    and is carried through untouched.
    """
    if keyframe is None:
        return None
    coarse_h, coarse_w = int(coarse_hw[0]), int(coarse_hw[1])
    src_h, src_w = int(keyframe["latent_h"]), int(keyframe["latent_w"])
    src_hw = (src_h, src_w)
    entries = keyframe.get("keyframes")
    if isinstance(entries, list) and entries:
        n_keyframes = len(entries)
    else:
        # Legacy payloads carry no per-keyframe entries, so the anchor count comes
        # from the row count -- the same derivation _imgvid_condition_shapes uses.
        # Assuming one anchor here would silently resize only the first frame.
        rows_per_keyframe = (src_h // 2) * (src_w // 2)
        n_rows = int(keyframe["rows"].shape[0])
        if rows_per_keyframe <= 0 or n_rows % rows_per_keyframe:
            raise ValueError(
                f"legacy keyframe rows {n_rows} do not split into frames of "
                f"{rows_per_keyframe} for grid {src_h}x{src_w}"
            )
        n_keyframes = n_rows // rows_per_keyframe
    coarse = dict(keyframe)
    coarse["rows"] = minimax_h3_resize_keyframe_rows(
        keyframe["rows"],
        n_keyframes=n_keyframes,
        src_hw=src_hw,
        dst_hw=(coarse_h, coarse_w),
    )
    coarse["latent_h"] = coarse_h
    coarse["latent_w"] = coarse_w
    if isinstance(entries, list) and entries:
        coarse["keyframes"] = [
            {**entry, "latent_h": coarse_h, "latent_w": coarse_w} for entry in entries
        ]
    return coarse


__all__ = [
    "MINIMAX_H3_LADDER_LATENT_CHANNELS",
    "MINIMAX_H3_LADDER_PATCH",
    "minimax_h3_ladder_band_surgery",
    "minimax_h3_ladder_coarse_grid",
    "minimax_h3_ladder_coarse_keyframe",
    "minimax_h3_ladder_noise",
    "minimax_h3_ladder_noise_attenuation",
    "minimax_h3_ladder_progressive",
    "minimax_h3_ladder_renoise",
    "minimax_h3_ladder_renoise_schedule",
    "minimax_h3_ladder_renoise_sigma",
    "minimax_h3_ladder_restore_noise",
    "minimax_h3_ladder_split_sigmas",
    "minimax_h3_ladder_steps",
    "minimax_h3_ladder_surgery_gain",
    "minimax_h3_ladder_upsample_mode",
    "minimax_h3_ladder_x0_schedule",
    "minimax_h3_learned_resize_video_rows",
    "minimax_h3_resize_keyframe_rows",
    "minimax_h3_resize_video_rows",
]
