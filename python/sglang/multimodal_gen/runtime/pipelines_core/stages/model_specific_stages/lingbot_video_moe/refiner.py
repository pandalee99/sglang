# SPDX-License-Identifier: Apache-2.0
"""LingBot-Video MoE refiner: an in-process super-resolution second pass.

Ports the reference runner's refiner flow (``lingbot_video/runner.py``):
decode the base pass, bicubic-upscale the pixels to the refiner resolution,
VAE re-encode, flow-noise the latent to ``t_thresh``, then run a short
truncated sigma schedule with the ``refiner/`` DiT. TI2V requests keep the
clean first frame pinned at the refiner resolution, and the refiner
conditions on text only (zero negative embeddings, like the reference
``null_cond_clone_zero`` default).

In-process differences from the reference (which round-trips through a saved
mp4 between two processes): no codec re-compression, no fps resampling (the
latent frame count is preserved), and the deterministic VAE posterior mean is
used instead of sampling it.
"""

from __future__ import annotations

import copy
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.ti2v import (
    should_apply_lingbot_ti2v,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

logger = init_logger(__name__)


def _truthy_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def lingbot_refiner_skipped(batch: Req | None = None) -> bool:
    """Per-request refiner opt-out (mirrors the SANA-WM skip convention)."""
    if os.getenv("SGLANG_LINGBOT_VIDEO_SKIP_REFINER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    if batch is None:
        return False
    extra = getattr(batch, "extra", None) or {}
    diffusers_kwargs = extra.get("diffusers_kwargs", {})
    if not isinstance(diffusers_kwargs, dict):
        diffusers_kwargs = {}
    return any(
        _truthy_flag(value)
        for value in (
            extra.get("skip_refiner"),
            diffusers_kwargs.get("skip_refiner"),
        )
    )


def validate_refiner_sigmas(
    sigmas: np.ndarray, t_thresh: float | None = None
) -> np.ndarray:
    arr = np.asarray(list(sigmas), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("refiner sigma schedule must be a non-empty 1D list")
    if not np.all(np.isfinite(arr)):
        raise ValueError("refiner sigma schedule contains non-finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(
            f"refiner sigma schedule values must be in [0, 1], got {arr.tolist()}"
        )
    if arr.size > 1 and not np.all(np.diff(arr) < 0.0):
        raise ValueError(
            f"refiner sigma schedule must be strictly descending, got {arr.tolist()}"
        )
    if t_thresh is not None and abs(float(arr[0]) - float(t_thresh)) > 1e-6:
        raise ValueError(
            f"refiner sigma schedule must start at t_thresh={float(t_thresh)}, "
            f"got {float(arr[0])}"
        )
    return arr


def compute_refiner_sigmas(
    *,
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    t_thresh: float,
    tail_steps: int = 0,
) -> np.ndarray:
    """Truncated shifted sigma schedule starting at ``t_thresh`` (reference port)."""
    t_value = float(t_thresh)
    if not (0.0 < t_value <= 1.0):
        raise ValueError(f"refiner t_thresh must lie in (0, 1], got {t_value}")
    steps = int(num_inference_steps)
    if steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
    tail = int(tail_steps or 0)
    if tail < 0:
        raise ValueError(f"refiner_sigma_tail_steps must be >= 0, got {tail}")

    base = np.linspace(float(sigma_max), float(sigma_min), steps + 1).copy()[:-1]
    shift_value = float(shift)
    shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
    eps = 1e-6
    sigmas = shifted[shifted <= t_value + eps]
    if sigmas.size == 0 or abs(float(sigmas[0]) - t_value) > eps:
        sigmas = np.concatenate([[t_value], sigmas])
    if tail > 0:
        start = float(sigmas[-1])
        stop = min(float(sigma_min), start)
        extra = np.linspace(start, stop, tail + 2, dtype=np.float64)[1:-1]
        sigmas = np.concatenate([sigmas, extra])
    return validate_refiner_sigmas(sigmas, t_value).astype(np.float32)


def prepare_refiner_latent(
    x_up: torch.Tensor,
    noise: torch.Tensor,
    t_thresh: float | torch.Tensor,
) -> torch.Tensor:
    """Flow-noise the encoded upscaled video to the refiner start level."""
    if not torch.is_tensor(t_thresh):
        t_thresh = torch.tensor(float(t_thresh), device=x_up.device, dtype=x_up.dtype)
    while t_thresh.ndim < x_up.ndim:
        t_thresh = t_thresh.view(*t_thresh.shape, *([1] * (x_up.ndim - t_thresh.ndim)))
    return (1.0 - t_thresh) * x_up + t_thresh * noise


def resize_video_pixels(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bicubic per-frame resize of a ``[B, C, T, H, W]`` video in ``[0, 1]``."""
    if video.ndim != 5:
        raise ValueError(
            f"video tensor must have shape [B,C,T,H,W], got {tuple(video.shape)}"
        )
    bsz, channels, frames, _height, _width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, _height, _width)
    resized = F.interpolate(
        flat, size=(height, width), mode="bicubic", align_corners=False
    )
    resized = resized.clamp(0.0, 1.0)
    return (
        resized.reshape(bsz, frames, channels, height, width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def crop_condition_image_to_geometry(
    image: Image.Image | list[Image.Image],
    target_height: int,
    target_width: int,
    geometry_height: int,
    geometry_width: int,
) -> torch.Tensor:
    """Clean first frame center-cropped to the base video's aspect ratio.

    The refiner latent grid is encoded from the upscaled base video, so the
    injected frame-0 condition must share that video's geometry, not the raw
    image's (reference ``load_first_frame_condition_tensor``). Returns a
    ``(1, C, 1, H, W)`` float tensor in ``[0, 1]``.
    """
    if isinstance(image, list):
        image = image[0]
    image = image.convert("RGB")
    image_width, image_height = image.size
    geometry_aspect = float(geometry_width) / float(geometry_height)
    image_aspect = float(image_width) / float(image_height)
    if image_aspect > geometry_aspect:
        crop_height = image_height
        crop_width = max(1, int(round(crop_height * geometry_aspect)))
        left = int(round((image_width - crop_width) / 2.0))
        top = 0
    else:
        crop_width = image_width
        crop_height = max(1, int(round(crop_width / geometry_aspect)))
        left = 0
        top = int(round((image_height - crop_height) / 2.0))
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    crop = crop.resize((target_width, target_height), resample=Image.BICUBIC)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    frame = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return frame.permute(1, 0, 2, 3).unsqueeze(0)


def _transformer_timestep(
    timestep: torch.Tensor, transformer_dtype: torch.dtype
) -> torch.Tensor:
    sigma = timestep.float() / 1000.0
    if transformer_dtype in {torch.bfloat16, torch.float16}:
        sigma = sigma.to(transformer_dtype)
    return (sigma * 1000.0).float()


def _transformer_autocast(device: torch.device, transformer_dtype: torch.dtype):
    if device.type != "cuda" or transformer_dtype not in {
        torch.bfloat16,
        torch.float16,
    }:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=transformer_dtype)


class LingBotVideoRefinerStage(PipelineStage):
    """Run the LingBot-Video ``refiner/`` DiT between denoise and decode.

    Skipped when the refiner is not loaded or the request opts out; the
    decode stage then sees the untouched base latents.
    """

    def __init__(self, transformer_2, scheduler, vae, text_encoding_stage) -> None:
        super().__init__()
        self.transformer_2 = transformer_2
        self.scheduler = scheduler
        self.vae = vae
        # Composed for TI2V: the refiner conditions on text only, so it
        # re-encodes the prompt without the image through the same stage.
        self.text_encoding_stage = text_encoding_stage

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        vae_dtype = resolve_precision(server_args, "vae", precision_attr="vae_precision")
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="vae",
                target_dtype=vae_dtype,
            ),
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer_2",
                phase="transformer_2",
                memory_intensive=True,
            ),
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if self.transformer_2 is None or lingbot_refiner_skipped(batch):
            return batch
        if get_sp_world_size() > 1:
            raise NotImplementedError(
                "The LingBot-Video refiner does not support sequence "
                "parallelism yet; run it with sp_world_size == 1."
            )

        config = server_args.pipeline_config
        device = get_local_torch_device()
        target_dtype = resolve_precision(server_args, "dit", precision_attr="dit_precision")

        refiner_height = int(config.refiner_height)
        refiner_width = int(config.refiner_width)
        if refiner_height % 16 or refiner_width % 16:
            raise ValueError(
                "refiner_height and refiner_width must be multiples of 16, got "
                f"{refiner_height}x{refiner_width}."
            )

        x_up = self._upscale_and_reencode(batch, server_args, refiner_height, refiner_width)

        cond_latent = None
        if should_apply_lingbot_ti2v(batch, server_args):
            cond_latent = self._encode_refiner_condition_latent(
                batch, server_args, refiner_height, refiner_width
            )
            x_up[:, :, : cond_latent.shape[2]] = cond_latent

        prompt_embeds, prompt_mask = self._refiner_prompt_embeds(
            batch, device, target_dtype
        )

        latents = self._refine(
            batch=batch,
            server_args=server_args,
            x_up=x_up,
            cond_latent=cond_latent,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            device=device,
            target_dtype=target_dtype,
        )

        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.height = refiner_height
        batch.width = refiner_width
        return batch

    def _decode_scale_and_shift(
        self, server_args: ServerArgs, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arch = server_args.pipeline_config.vae_config.arch_config
        mean = torch.tensor(arch.latents_mean, device=device, dtype=torch.float32)
        std = torch.tensor(arch.latents_std, device=device, dtype=torch.float32)
        return mean.view(1, -1, 1, 1, 1), std.view(1, -1, 1, 1, 1)

    def _upscale_and_reencode(
        self,
        batch: Req,
        server_args: ServerArgs,
        refiner_height: int,
        refiner_width: int,
    ) -> torch.Tensor:
        """Base latents -> pixels -> bicubic upscale -> VAE re-encode."""
        device = get_local_torch_device()
        vae_dtype = resolve_precision(server_args, "vae", precision_attr="vae_precision")
        mean, std = self._decode_scale_and_shift(server_args, device)
        with self.use_declared_component(
            component_name="vae",
            module=self.vae,
            target_dtype=vae_dtype,
        ) as vae:
            assert vae is not None
            self.vae = vae
            vae_latents = (
                batch.latents.to(device=device, dtype=torch.float32) * std + mean
            )
            decoded = vae.decode(vae_latents.to(vae_dtype))
            if isinstance(decoded, torch.Tensor):
                frames = decoded
            elif isinstance(decoded, tuple):
                frames = decoded[0]
            else:
                frames = decoded.sample
            frames = frames.float().clamp_(-1.0, 1.0)
            frames = (frames + 1.0) / 2.0
            frames = resize_video_pixels(frames, refiner_height, refiner_width)
            encoded = vae.encode((frames * 2.0 - 1.0).to(vae_dtype))
        z = encoded.mean.float()
        return (z - mean) / std

    def _encode_refiner_condition_latent(
        self,
        batch: Req,
        server_args: ServerArgs,
        refiner_height: int,
        refiner_width: int,
    ) -> torch.Tensor:
        """Clean first-frame latent at refiner resolution (base geometry crop)."""
        device = get_local_torch_device()
        vae_dtype = resolve_precision(server_args, "vae", precision_attr="vae_precision")
        pixel = crop_condition_image_to_geometry(
            batch.condition_image,
            target_height=refiner_height,
            target_width=refiner_width,
            geometry_height=int(batch.height),
            geometry_width=int(batch.width),
        )
        mean, std = self._decode_scale_and_shift(server_args, device)
        with self.use_declared_component(
            component_name="vae",
            module=self.vae,
            target_dtype=vae_dtype,
        ) as vae:
            assert vae is not None
            self.vae = vae
            norm_pixel = pixel.to(device=device, dtype=torch.float32) * 2.0 - 1.0
            z = vae.encode(norm_pixel.to(vae_dtype)).mean.float()
        cond = (z - mean) / std
        return cond[:, :, 0:1].contiguous()

    def _refiner_prompt_embeds(
        self, batch: Req, device: torch.device, target_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positive embeddings for the refiner pass (text only).

        T2V reuses the base pass embeddings. TI2V re-encodes the prompt
        WITHOUT the image (reference: the ti2v condition cache carries image
        tokens; the refiner conditions on text only).
        """
        if batch.condition_image is None:
            return batch.prompt_embeds[0], batch.prompt_attention_mask
        return self.text_encoding_stage._encode_prompt(
            batch.prompt, device, target_dtype, images=None
        )

    def _refine(
        self,
        *,
        batch: Req,
        server_args: ServerArgs,
        x_up: torch.Tensor,
        cond_latent: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        config = server_args.pipeline_config
        scheduler = copy.deepcopy(self.scheduler)
        sigmas = compute_refiner_sigmas(
            sigma_max=float(scheduler.sigma_max),
            sigma_min=float(scheduler.sigma_min),
            num_inference_steps=int(config.refiner_num_inference_steps),
            shift=float(config.refiner_flow_shift),
            t_thresh=float(config.refiner_t_thresh),
            tail_steps=int(config.refiner_sigma_tail_steps),
        )
        scheduler.set_timesteps(
            int(sigmas.shape[0]), device=device, sigmas=sigmas, shift=1.0
        )

        noise = torch.randn(
            x_up.shape, device=device, dtype=x_up.dtype, generator=batch.generator
        )
        latents = prepare_refiner_latent(x_up, noise, float(config.refiner_t_thresh))
        if cond_latent is not None:
            latents[:, :, : cond_latent.shape[2]] = cond_latent

        guidance_scale = float(config.refiner_guidance_scale)
        do_cfg = guidance_scale > 1.0
        prompt_embeds = prompt_embeds.to(device=device, dtype=target_dtype)
        prompt_mask = prompt_mask.to(device=device)
        if do_cfg:
            # Reference refiner default (null_cond_clone_zero): the negative
            # branch uses zero embeddings with the positive mask.
            negative_embeds = torch.zeros_like(prompt_embeds)
            negative_mask = prompt_mask.clone()

        with self.use_declared_component(
            component_name="transformer_2",
            module=self.transformer_2,
        ) as transformer:
            assert transformer is not None
            self.transformer_2 = transformer
            for timestep in scheduler.timesteps:
                timestep_batch = (
                    _transformer_timestep(timestep, target_dtype).expand(1).to(device)
                )
                latent_model_input = latents.to(target_dtype)
                with _transformer_autocast(device, target_dtype):
                    noise_pred = transformer(
                        latent_model_input,
                        timestep_batch,
                        prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                    ).float()
                if do_cfg:
                    with _transformer_autocast(device, target_dtype):
                        noise_pred_uncond = transformer(
                            latent_model_input,
                            timestep_batch,
                            negative_embeds,
                            encoder_attention_mask=negative_mask,
                        ).float()
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )
                latents = scheduler.step(
                    noise_pred,
                    timestep,
                    latents,
                    return_dict=False,
                    generator=batch.generator,
                )[0]
                if cond_latent is not None:
                    latents[:, :, : cond_latent.shape[2]] = cond_latent
        return latents
