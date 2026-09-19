# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import (
    get_replica_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context,
    autocast_enabled_for_device,
    resolve_decode_precision,
    resolve_precision,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.latent_upscale import (
    latent_upscale_scale,
    upscaled_latent_hw,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    ActiveTargetCompiledCallable,
)

logger = init_logger(__name__)


def _required_tensor(value, path: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} must be a torch.Tensor")
    return value


@contextmanager
def _deterministic_audio_decode_context():
    """Deterministic-algorithm scope for the fp32 audio-VAE decode.

    Without it, cuDNN picks conv algorithms from free-workspace state, so the
    same audio latent decodes to different bytes on a server process's first
    request than on every later one. Deterministic algorithms with TF32 off
    keep cuDNN speed (unlike the encode-side context, which disables cuDNN);
    if first-request divergence ever reappears, escalate to
    reference_encoding._AudioVAEDeterminismContext.
    """
    b = torch.backends
    saved = (
        b.cudnn.allow_tf32,
        b.cuda.matmul.allow_tf32,
        b.cudnn.deterministic,
        b.cudnn.benchmark,
    )
    b.cudnn.allow_tf32 = False
    b.cuda.matmul.allow_tf32 = False
    b.cudnn.deterministic = True
    b.cudnn.benchmark = False
    try:
        yield
    finally:
        (
            b.cudnn.allow_tf32,
            b.cuda.matmul.allow_tf32,
            b.cudnn.deterministic,
            b.cudnn.benchmark,
        ) = saved


@functools.lru_cache(maxsize=None)
def _cached_decode_mean_std(
    mean_values: tuple[float, ...],
    std_values: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device/dtype-keyed mean/std tensors, built once per distinct combination.

    mean_values/std_values come from the loaded arch_config and are fixed
    for the process lifetime, so the same (values, device, dtype) always
    reconstructs an identical tensor; cache it instead of rebuilding it on
    every decode call.
    """
    mean = torch.as_tensor(mean_values, device=device, dtype=dtype)
    std = torch.as_tensor(std_values, device=device, dtype=dtype)
    return mean, std


def _reverse_normalize_latents(
    latents: torch.Tensor,
    *,
    mean_values,
    std_values,
    name: str,
) -> torch.Tensor:
    mean, std = _cached_decode_mean_std(
        tuple(mean_values), tuple(std_values), latents.device, latents.dtype
    )
    if mean.ndim != 1:
        raise ValueError(f"{name}.latents_mean must be 1-D, got {tuple(mean.shape)}")
    if std.ndim != 1:
        raise ValueError(f"{name}.latents_std must be 1-D, got {tuple(std.shape)}")
    if mean.shape != std.shape:
        raise ValueError(
            f"{name} latent normalization shape mismatch: "
            f"mean={tuple(mean.shape)} std={tuple(std.shape)}"
        )
    if latents.ndim < 2:
        raise ValueError(f"{name} latents must have a channel dimension")
    if int(latents.shape[1]) != int(mean.shape[0]):
        raise ValueError(
            f"{name} latent normalization channel mismatch: "
            f"latents.shape[1]={int(latents.shape[1])} mean_len={int(mean.shape[0])}"
        )
    view_shape = [1] * latents.ndim
    view_shape[1] = int(mean.shape[0])
    # Out of place on purpose. batch.latents / batch.audio_latents are
    # inference tensors allocated inside the denoising stage's InferenceMode,
    # while --vae-cpu-offload runs this stage under torch.inference_mode(False)
    # (PipelineExecutor._stage_needs_version_counters), where writing to one in
    # place raises. Keep the mul-then-add rounding order rather than addcmul so
    # the result does not shift with FMA contraction.
    return latents * std.view(*view_shape) + mean.view(*view_shape)


# Keyed by checkpoint path, not a single slot: the checkpoint is selectable per
# request (see _upscale_ckpt_path) and reloading 345M parameters on every swap
# would dominate an A/B whose arms differ by a few seconds.
_LATENT_UPSCALERS: dict[str, tuple] = {}
_SHIPPED_UPSCALER_CKPT = (
    "/models/h3_latent_upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors"
)


def _is_dump_owner() -> bool:
    """True on exactly one rank. Mirrors the audio-decode owner test below."""
    replica_group = get_replica_group() if model_parallel_is_initialized() else None
    return replica_group is None or replica_group.rank_in_group == 0


def _latent_dump_path(dump: str, batch: Req, latents: torch.Tensor) -> str:
    """Where one request's latent dump goes.

    A plain suffix was fine while the dump served single-arm offline probes, but
    collecting a training corpus issues many requests against one server and every
    one of them would land on the same path. When `dump` names a DIRECTORY, write
    one file per request inside it keyed by seed and latent shape, which is enough
    to identify a pair member and cheap to enumerate later. Otherwise keep the old
    single-file behaviour so existing probes are unaffected.

    Takes the tensor being dumped rather than reading `batch.latents`: this stage is
    handed the visual latent as an argument and the two are not guaranteed to be the
    same object, so keying the filename off the batch could name a file after a
    shape it does not contain.
    """
    if os.path.isdir(dump):
        seed = getattr(batch, "seed", None)
        h, w = int(latents.shape[-2]), int(latents.shape[-1])
        stem = f"seed{seed}_{h}x{w}"
        path = os.path.join(dump, f"{stem}.pt")
        # Same seed at the same grid twice means a repeat, and silently
        # overwriting would make a corpus smaller than its manifest claims.
        n = 1
        while os.path.exists(path):
            path = os.path.join(dump, f"{stem}_{n}.pt")
            n += 1
        return path
    return f"{dump}.pre_upscale.pt"


def _upscale_ckpt_path() -> str:
    """Which upscaler checkpoint this request uses.

    Read from SGLANG_H3_LATENT_UPSCALE_CKPT_FILE per request when that file exists,
    falling back to the env var and then to the shipped checkpoint. Same rationale
    as the scale knob -- this runs after denoise, so the 8 ranks cannot disagree
    about a packed sequence because of it.

    Measured reason a per-request swap is worth the code: content is reproducible
    only WITHIN a server instance. Two identical requests (same seed, same config,
    no upscaler) score ssim 0.9161 / gradient agreement 0.7848 against each other
    across a restart, while back-to-back inside one instance are bit-identical. So
    an A/B whose arms sit in different instances cannot resolve anything smaller
    than that, and comparing two checkpoints was exactly that A/B.
    """
    path = os.environ.get("SGLANG_H3_LATENT_UPSCALE_CKPT_FILE")
    if path and os.path.exists(path):
        try:
            picked = open(path).read().strip()
        except OSError:
            picked = ""
        if picked:
            return picked
    return os.environ.get("SGLANG_H3_LATENT_UPSCALE_CKPT", _SHIPPED_UPSCALER_CKPT)


def _latent_upscaler(device: torch.device, dtype: torch.dtype):
    """Load a LatentResizer3D, cached per checkpoint path.

    Returns (model, composition) where composition holds how this checkpoint was
    TRAINED, taken from the checkpoint itself rather than from the environment.
    Getting that pairing wrong does not fail loudly -- a residual-trained network
    run in direct mode emits a correction with no baseline under it, and a
    direct-trained one run in residual mode doubles the picture -- and a per-request
    checkpoint swap makes an env var the wrong place to hold it, since the two would
    desync on exactly the arm being measured. train_upscaler.py records its flags in
    the .pt, so the checkpoint is self-describing; the env vars remain the fallback
    for the shipped .safetensors, which carries no such record.
    """
    ckpt = _upscale_ckpt_path()
    if ckpt not in _LATENT_UPSCALERS:
        import sys

        if "/results" not in sys.path:
            sys.path.insert(0, "/results")
        from h3_upscaler import load_upscaler

        comp = {"residual": _upscale_residual(),
                "project_low": _upscale_project_low()}
        if ckpt.endswith(".pt"):
            try:
                trained_args = torch.load(ckpt, map_location="cpu").get("args") or {}
            except (OSError, RuntimeError):
                trained_args = {}
            if trained_args:
                comp = {"residual": bool(trained_args.get("residual", 0)),
                        "project_low": bool(trained_args.get("project_low", 0))}
        _LATENT_UPSCALERS[ckpt] = (
            load_upscaler(ckpt, device=device, dtype=dtype), comp,
        )
        logger.info("[MiniMaxH3LatentUpscale] loaded %s (residual=%s project_low=%s)",
                    ckpt, comp["residual"], comp["project_low"])
    return _LATENT_UPSCALERS[ckpt]


def _latent_norm_stats(device: torch.device, dtype: torch.dtype):
    """Per-channel (mean, std) the upscaler was trained against, as 5D views."""
    import sys

    if "/results" not in sys.path:
        sys.path.insert(0, "/results")
    from h3_upscaler import latent_norm_stats

    return latent_norm_stats(device, dtype)


def _upscale_residual() -> bool:
    """Compose the upscaler's output as trilinear + f(x) rather than as f(x).

    Must match how the loaded checkpoint was TRAINED, which is why this is a
    separate knob from the checkpoint path: running a residual-trained network in
    direct mode emits a correction with no baseline under it, and running a
    direct-trained one in residual mode doubles the picture. Neither fails loudly.

    Measured reason the training uses residual form at all: on held-out crops
    against a native-2K truth, plain trilinear scores L1 0.218 while the shipped
    checkpoint scores 0.286 and its downsample-consistency is twice as bad (0.223
    vs 0.108). Direct prediction therefore starts below a parameterless baseline,
    for the reason _band_surgery_gain() documents -- the network rewrites the low
    band interpolation already carries exactly. With conv_out zero-initialized the
    residual form starts exactly AT trilinear, so training can only improve on it
    and what it learns is confined to the high frequency that is genuinely missing.

    Off by default: the shipped checkpoint is a direct predictor.
    """
    return os.environ.get("SGLANG_H3_UPSCALE_RESIDUAL", "0").strip().lower() not in (
        "", "0", "false", "no", "off",
    )


def _upscale_project_low() -> bool:
    """Force the upscale's LOW band to come from interpolation, not the network.

    Must match training, for the same reason _upscale_residual() must. Measured
    reason it exists: the first fine-tuned checkpoint beat every baseline on the
    synthetic pair (latent PSNR 37.98 vs trilinear 35.21 vs shipped 31.33,
    high-band correlation with the truth 0.751 vs 0.474) and then FAILED the
    real-video content gate, scoring 0.766 ssim / 0.218 gradient agreement against
    its own 768p source where a lanczos control on identical geometry scores
    0.994 / 0.989. The probe that localized it: about half the learned correction's
    energy sits in the LOW band, and the low band is what determines where content
    is. So the network was displacing structure while improving every number that
    does not measure position.

    Projecting the low band onto the interpolated baseline makes that structurally
    impossible instead of merely penalized -- the hard form of the soft band surgery
    _band_surgery_gain() documents, and DDNM's range-null decomposition.

    Off by default: the shipped checkpoint predicts the whole latent.
    """
    return os.environ.get("SGLANG_H3_UPSCALE_PROJECT_LOW", "0").strip().lower() not in (
        "", "0", "false", "no", "off",
    )


def _band_surgery_gain() -> float:
    """Gain applied to the learned upscaler's HIGH band; 0 disables the surgery.

    Measured reason this exists (e9_band_surgery.py, two clips). Against a true
    paired reference the learned upscale scores 3.96 dB BELOW plain trilinear
    while its high band correlates BETTER with the truth (0.539 vs 0.380). Both
    hold at once because the network rewrites the whole latent and damages the LOW
    band, which the coarse input already carried exactly: low-band correlation
    falls 0.99970 -> 0.95438, and since the low band holds nearly all the energy
    that alone costs the 4 dB. The control that pins the cause is the opposite
    splice -- learned's low band with trilinear's high band still scores 26.86,
    so the high frequency was never the problem.

    So: keep trilinear's low band, take only the learned high band, scaled. The
    closed-form fidelity optimum is 0.687 on one clip and 0.666 on another, and
    the resulting arm beats BOTH baselines (31.52 vs trilinear 30.78 vs learned
    26.82; +0.74 dB over trilinear on both clips).

    Only valid at scale <= 2. At 2.5x and 3x the learned high band's error exceeds
    1.0 -- worse than emitting nothing in that band -- and the optimum collapses to
    ~0.22, so the caller gates on scale rather than trusting this value blindly.

    Read from a file when SGLANG_H3_UPSCALE_HI_GAIN_FILE points at one, so an A/B
    does not cost a cold start per arm. Safe per-request for the same reason the
    upscale scale is: this runs after denoise, so the 8 ranks never build different
    packed sequences from it.
    """
    gain = float(os.environ.get("SGLANG_H3_UPSCALE_HI_GAIN", "0") or 0)
    path = os.environ.get("SGLANG_H3_UPSCALE_HI_GAIN_FILE")
    if path and os.path.exists(path):
        try:
            return float(open(path).read().strip() or 0)
        except (OSError, ValueError):
            return gain
    return gain


def _split_bands(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(low, high) split at half the current grid, matching e9_band_surgery.py.

    The band edge is defined by a downsample-and-back round trip rather than an
    FFT cutoff so it coincides exactly with "what the coarse input could already
    represent" -- the same definition the offline measurement used, which is what
    makes the 0.687 gain transferable to this code path.
    """
    t, h, w = int(x.shape[2]), int(x.shape[-2]), int(x.shape[-1])
    flat = x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], h, w)
    lo = torch.nn.functional.interpolate(
        flat, size=(max(h // 2, 1), max(w // 2, 1)), mode="area"
    )
    lo = torch.nn.functional.interpolate(
        lo, size=(h, w), mode="bilinear", align_corners=False
    )
    lo = lo.reshape(x.shape[0], t, x.shape[1], h, w).permute(0, 2, 1, 3, 4)
    return lo.contiguous(), x - lo


def _maybe_upscale_latents(batch: Req, latents: torch.Tensor) -> torch.Tensor:
    """Learned latent upscale between denoise and decode. Off unless requested.

    Goes through the author's own (x-mean)/std wrapper rather than feeding these
    latents to the network directly. That is not obvious and was measured, because
    the tempting argument is wrong: this stage holds latents in NORMALIZED units
    (_reverse_normalize_latents() below is what makes them raw), the upscaler's
    hardcoded LATENTS_MEAN/STD equal our video_vae config's to max|diff| =
    0.000e+00, so the wrapper looks like a second normalization.

    It is not. The pipeline's normalized latents measure per-channel std 0.7957,
    not 1.0 -- LATENTS_STD are dataset-wide statistics and one clip is less
    dispersed than the corpus -- so they are NOT in the trained space. Feeding
    them directly drives the output's per-channel statistics 6.80 away from the
    input's, against 0.32 through the wrapper (e3_noise_blindness.py). The direct
    form decodes to mean luma 193.7 vs 98.6 on a fixed seed and laplacian
    variance 2703 vs a lanczos control's 5.4: off-distribution, not sharper.

    The scale is the spatial factor (2.0 => 768p latents decode at 2K). The
    temporal axis is deliberately untouched; the network preserves it and the
    audio latents are a separate 32-channel tensor that never sees this.

    Read from SGLANG_H3_LATENT_UPSCALE_FILE per request when that file exists,
    falling back to SGLANG_H3_LATENT_UPSCALE. A launch-time-only knob would cost
    a 98 s cold start per arm, and the scale ladder needs several arms; unlike the
    resolution ladder this is safe to vary per request because it runs after
    denoise, so the 8 ranks never build different packed sequences from it.
    """
    scale = latent_upscale_scale()

    dump = os.environ.get("SGLANG_H3_LATENT_DUMP")
    if dump and _is_dump_owner():
        # Dumped BEFORE the scale gate, so `scale <= 1.0` still yields a latent.
        # That combination is the one a training corpus needs: a clean natively
        # generated latent with no upscale applied, which is the ground-truth half
        # of a (coarse, fine) pair. Gating this behind the upscale -- as it was --
        # meant the only obtainable dumps had already been through the very
        # operator under test.
        #
        # Rank-guarded: this stage is REPLICATED, so all 8 ranks hold the same
        # latents and would race on one path.
        torch.save({"latents": latents.detach().cpu(),
                    "latent_h": int(latents.shape[-2]),
                    "latent_w": int(latents.shape[-1]),
                    "seed": getattr(batch, "seed", None)},
                   _latent_dump_path(dump, batch, latents))

    if scale <= 1.0:
        return latents

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    )

    h_in, w_in = int(latents.shape[-2]), int(latents.shape[-1])
    # Rounding lives in latent_upscale.py because video_adapter's output check
    # has to predict this exact canvas from the pre-upscale pixel size.
    h_out, w_out = upscaled_latent_hw(h_in, w_in, scale)

    model, comp = _latent_upscaler(latents.device, latents.dtype)
    norm_mean, norm_std = _latent_norm_stats(latents.device, latents.dtype)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        x_norm = (latents - norm_mean) / norm_std
        out = model(x_norm, scale=scale,
                    target_size=(int(latents.shape[2]), h_out, w_out))
        if comp["residual"]:
            # Bilinear per frame, matching the training baseline exactly. A 3D
            # trilinear here would blend neighbouring frames into the baseline and
            # the network's learned correction would then be sitting on a different
            # foundation than the one it was fitted against.
            b_, c_, t_, h_, w_ = x_norm.shape
            flat = x_norm.permute(0, 2, 1, 3, 4).reshape(b_ * t_, c_, h_, w_)
            base = torch.nn.functional.interpolate(
                flat.float(), size=(h_out, w_out), mode="bilinear",
                align_corners=False,
            ).to(out.dtype)
            base = base.reshape(b_, t_, c_, h_out, w_out).permute(0, 2, 1, 3, 4)
            out = base + out
            logger.info("[MiniMaxH3LatentUpscale] residual composition: "
                        "bilinear baseline + learned correction")
            if comp["project_low"]:
                # _split_bands' edge is a downsample-and-back round trip, the same
                # definition the training projection uses, so the split transfers.
                base_lo, _ = _split_bands(base)
                _, out_hi = _split_bands(out)
                out = base_lo + out_hi
                logger.info("[MiniMaxH3LatentUpscale] low band projected onto the "
                            "bilinear baseline; only the high band is learned")
        out = out * norm_std + norm_mean

        hi_gain = _band_surgery_gain()
        # Gated on scale, not just on the gain: past 2x the learned high band's
        # error exceeds 1.0 (worse than leaving that band empty) and the fidelity
        # optimum falls to ~0.22, so a gain tuned at 2x would be actively harmful
        # if it silently carried over to a 3x request.
        if hi_gain > 0 and scale <= 2.0:
            tri = torch.nn.functional.interpolate(
                latents.float(),
                size=(int(latents.shape[2]), h_out, w_out),
                mode="trilinear", align_corners=False,
            ).to(out.dtype)
            tri_lo, _ = _split_bands(tri)
            _, learned_hi = _split_bands(out)
            out = tri_lo + hi_gain * learned_hi
            logger.info(
                "[MiniMaxH3LatentUpscale] band surgery: trilinear low band + "
                "%.3f x learned high band", hi_gain,
            )
    end.record()
    torch.cuda.synchronize()
    logger.info(
        "[MiniMaxH3LatentUpscale] %dx%d -> %dx%d latent (%dx%d -> %dx%d px) "
        "scale=%.3f in %.3f s",
        h_in, w_in, h_out, w_out, h_in * 16, w_in * 16, h_out * 16, w_out * 16,
        scale, start.elapsed_time(end) / 1000.0,
    )

    # The crop below trusts the denoise state for the target canvas. Leaving it
    # at the pre-upscale grid would crop a 2K decode back to the 768p canvas,
    # i.e. keep only the top-left quarter of the frame.
    state = batch.extra.get(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY)
    if state is not None:
        state["latent_h"] = h_out
        state["latent_w"] = w_out
    return out.to(latents.dtype)


def _crop_to_target_canvas(batch: Req, frames: torch.Tensor) -> torch.Tensor:
    """Crop decoded frames [B,C,T,H,W] back to the target canvas.

    The visual VAE pads the latent grid to its tile multiples (padding lands
    at the bottom/right), so a non-tile-aligned geometry decodes larger than the
    requested canvas (e.g. 1344x768 for a 1280x704 target). Target dims come
    from the direct-mode denoise state (latent_h/w * 16); requests without
    that state keep the raw decode.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    )

    state = batch.extra.get(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY)
    if state is None:
        return frames
    target_h = int(state["latent_h"]) * 16
    target_w = int(state["latent_w"]) * 16
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h < target_h or w < target_w:
        raise ValueError(
            f"decoded frames {h}x{w} smaller than target canvas {target_h}x{target_w}"
        )
    if h == target_h and w == target_w:
        return frames
    return frames[..., :target_h, :target_w]


def _canonical_visual_video_frames(
    frames: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    if frames.ndim == 4:
        if int(frames.shape[0]) % batch_size != 0:
            raise ValueError(
                f"Decoded visual video shape {tuple(frames.shape)} is incompatible "
                f"with batch_size={batch_size}"
            )
        frames = frames.reshape(
            batch_size, int(frames.shape[0]) // batch_size, *frames.shape[1:]
        )
        frames = frames.transpose(1, 2)
    elif frames.ndim == 5:
        if int(frames.shape[0]) != batch_size:
            raise ValueError(
                f"Decoded visual video batch mismatch: frames.shape[0]={int(frames.shape[0])} "
                f"batch_size={batch_size}"
            )
    else:
        raise ValueError(
            f"Decoded visual video shape {tuple(frames.shape)} is not supported"
        )
    return frames


def _canonical_output_audio_waveform(
    audio_waveform: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    """Project audio-VAE-native ``[C, 1, L]`` audio to output ``[1, C, L]``.

    The audio VAE treats stereo channels as its decoder batch and returns
    ``[2, 1, samples]`` for MiniMax H3's one generated sample.  The generic output
    path instead selects generated samples along dimension zero.  Keep the audio VAE
    tensor unchanged for decoder artifacts, then make the singleton generated-
    sample dimension explicit only at the ``OutputBatch`` boundary.
    """
    if audio_waveform.ndim != 3:
        raise ValueError(
            "Decoded audio VAE waveform must be [C, 1, L], got "
            f"{tuple(audio_waveform.shape)}"
        )
    if batch_size != 1:
        raise ValueError(
            "MiniMax H3 audio VAE output only supports one generated sample, "
            f"got visual batch_size={batch_size}"
        )
    if int(audio_waveform.shape[1]) != 1:
        raise ValueError(
            "Decoded audio VAE waveform must have shape [C, 1, L], got "
            f"{tuple(audio_waveform.shape)}"
        )
    return audio_waveform.permute(1, 0, 2).contiguous()


_MINIMAX_H3_DECODER_TASKS = frozenset({"t2va", "fl2va", "ref2va"})
_MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY = "minimax_h3_canonical_request"
_MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY = "minimax_h3_resolved_plan"


def _minimax_h3_decoder_task(batch: Req) -> str | None:
    """Return the validated request task used for output-decoder routing.

    Debug requests have no canonical task and retain the
    generic decoder.
    """

    extra = getattr(batch, "extra", None)
    if not isinstance(extra, Mapping):
        return None
    canonical = extra.get(_MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY)
    if canonical is not None and not isinstance(canonical, Mapping):
        raise ValueError("minimax_h3_canonical_request must be a mapping")
    canonical_task = canonical.get("task") if isinstance(canonical, Mapping) else None
    resolved = extra.get(_MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY)
    resolved_task = getattr(resolved, "task", None) if resolved is not None else None
    if canonical_task is not None and resolved_task is not None:
        if str(canonical_task) != str(resolved_task):
            raise ValueError(
                "MiniMax H3 decoder task mismatch between canonical request and "
                "resolved plan"
            )
    task_value = resolved_task if resolved_task is not None else canonical_task
    if task_value is None:
        return None
    if not isinstance(task_value, str) or task_value not in _MINIMAX_H3_DECODER_TASKS:
        raise ValueError(f"unsupported MiniMax H3 decoder task {task_value!r}")
    return task_value


class MiniMaxH3DecodingStage(DecodingStage):
    def __init__(self, video_vae, audio_vae) -> None:
        super().__init__(vae=video_vae, component_name="video_vae")
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self._compiled_audio_vae_decode = ActiveTargetCompiledCallable()

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    @property
    def parallelism_type(self) -> StageParallelismType:
        # Every decode-group rank owns a subset of visual VAE tiles. The GPU
        # worker only materializes/saves the final OutputBatch on world rank 0.
        return StageParallelismType.REPLICATED

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        video_vae_dtype = resolve_precision(
            server_args, "video_vae", precision_attr="vae_precision"
        )
        audio_vae_dtype = resolve_precision(
            server_args, "audio_vae", precision_attr="audio_vae_precision"
        )
        uses = [
            ComponentUse(stage_name, "video_vae", target_dtype=video_vae_dtype),
        ]
        uses.append(ComponentUse(stage_name, "audio_vae", target_dtype=audio_vae_dtype))
        return uses

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "audio_latents",
            batch.audio_latents,
            [V.is_tensor, V.with_dims(3)],
        )
        return result

    def verify_output(
        self, batch: OutputBatch, server_args: ServerArgs
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, [V.is_tensor, V.with_dims(5)])
        result.add_check("audio", batch.audio, [V.is_tensor, V.with_dims(3)])
        result.add_check("audio_sample_rate", batch.audio_sample_rate, V.positive_int)
        return result

    def _decode_audio(
        self,
        audio_latent: torch.Tensor,
        server_args: ServerArgs,
    ) -> dict:
        with self.use_declared_component(
            component_name="audio_vae",
            module=self.audio_vae,
        ) as audio_vae:
            assert audio_vae is not None
            self.audio_vae = audio_vae
            if audio_vae.training:
                audio_vae.eval()
            audio_arch_config = server_args.pipeline_config.audio_vae_config.arch_config
            audio_decode_latent = _reverse_normalize_latents(
                audio_latent,
                mean_values=audio_arch_config.latents_mean,
                std_values=audio_arch_config.latents_std,
                name="audio_vae",
            )
            audio_vae_dtype = resolve_precision(
                server_args, "audio_vae", precision_attr="audio_vae_precision"
            )
            audio_autocast_enabled = autocast_enabled_for_device(
                audio_latent, audio_vae_dtype, server_args.disable_autocast
            )
            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=audio_vae_dtype,
                    enabled=audio_autocast_enabled,
                )
                if audio_latent.is_cuda
                else nullcontext()
            )
            with _deterministic_audio_decode_context(), autocast_context:
                audio_decode = self._get_vae_decode_fn(
                    audio_vae,
                    server_args,
                    decode_fn=audio_vae.decode,
                    compiled_callable=self._compiled_audio_vae_decode,
                )
                waveform = _required_tensor(
                    audio_decode(audio_decode_latent), "audio_vae.decode"
                )
            return {
                "waveform": waveform,
                "sample_rate": int(audio_vae.sample_rate),
            }

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        _minimax_h3_decoder_task(batch)
        visual_latent = _required_tensor(batch.latents, "batch.latents")
        audio_latent = _required_tensor(batch.audio_latents, "batch.audio_latents")
        if visual_latent.ndim != 5:
            raise ValueError("batch.latents must be [B, C, T, H, W]")
        if audio_latent.ndim != 3:
            raise ValueError(
                "batch.audio_latents must be [audio_channel, latent_dim, T]"
            )

        # Before _reverse_normalize_latents, because the upscaler works in the
        # same normalized space these latents are still in.
        visual_latent = _maybe_upscale_latents(batch, visual_latent)

        if self.video_vae is None:
            raise RuntimeError("MiniMax H3 tasks require the video_vae output decoder")
        with self.use_declared_component(
            component_name="video_vae",
            module=self.video_vae,
        ) as selected_video_vae:
            if selected_video_vae is None:
                raise RuntimeError("video_vae became unavailable during decode")
            self.video_vae = selected_video_vae
            if selected_video_vae.training:
                selected_video_vae.eval()
            visual_arch_config = server_args.pipeline_config.vae_config.arch_config
            visual_decode_latent = _reverse_normalize_latents(
                visual_latent,
                mean_values=visual_arch_config.latents_mean,
                std_values=visual_arch_config.latents_std,
                name="video_vae",
            )
            video_vae_dtype = resolve_decode_precision(server_args, "video_vae")
            visual_autocast_enabled = autocast_enabled_for_device(
                visual_latent, video_vae_dtype, server_args.disable_autocast
            )
            if visual_autocast_enabled:
                selected_video_vae.prepare_decoder_autocast_weights(video_vae_dtype)
            with autocast_context(
                video_vae_dtype,
                server_args.disable_autocast,
                enabled=visual_autocast_enabled,
            ):
                video_decode = self._get_vae_decode_fn(
                    selected_video_vae,
                    server_args,
                    decode_fn=selected_video_vae.decode_base,
                )
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    visual_frames = video_decode(visual_decode_latent)
                visual_frames = selected_video_vae.processor.revert_tensor(
                    visual_frames
                )
                visual_frames = _required_tensor(
                    visual_frames,
                    "video_vae.processor.revert_tensor",
                )
                visual_frames = _canonical_visual_video_frames(
                    visual_frames, batch_size=int(visual_latent.shape[0])
                )
                visual_frames = _crop_to_target_canvas(batch, visual_frames)
                if (
                    visual_frames.dtype != torch.float32
                    or not visual_frames.is_contiguous()
                ):
                    canonical_frames = torch.empty_like(
                        visual_frames,
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    )
                    canonical_frames.copy_(visual_frames)
                    visual_frames = canonical_frames

        # Audio VAE weights are replicated. Decode on replica rank 0 and broadcast
        # only within the request's replica, excluding independent DP replicas.
        replica_group = get_replica_group() if model_parallel_is_initialized() else None
        is_audio_owner = replica_group is None or replica_group.rank_in_group == 0
        owner_exception = None
        owner_error = None
        audio_payload = None
        if is_audio_owner:
            try:
                audio_payload = self._decode_audio(audio_latent, server_args)
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        if replica_group is not None:
            owner_error = replica_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if replica_group is not None:
            audio_payload = replica_group.broadcast_tensor_dict(audio_payload, src=0)
        if not isinstance(audio_payload, dict):
            raise RuntimeError("MiniMax H3 audio decode produced no output payload")
        audio_waveform = _required_tensor(
            audio_payload.get("waveform"), "audio_vae.decode"
        )
        audio_sample_rate = int(audio_payload["sample_rate"])

        visual_frames = server_args.pipeline_config.post_decoding(
            visual_frames, server_args
        )
        output_audio_waveform = _canonical_output_audio_waveform(
            audio_waveform, batch_size=int(visual_frames.shape[0])
        )
        return OutputBatch(
            output=visual_frames,
            audio=output_audio_waveform,
            audio_sample_rate=audio_sample_rate,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=None,
            metrics=batch.metrics,
            noise_pred=None,
        )


__all__ = [
    "MiniMaxH3DecodingStage",
]
