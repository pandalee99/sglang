# SPDX-License-Identifier: Apache-2.0
"""LingBot-Video TI2V (first-frame conditioning) helpers.

The reference pipeline (``lingbot_video/pipeline_lingbot_video_i2v.py``) uses
the condition frame twice: as a visual input to the Qwen3-VL prompt encoder,
and as a clean latent written into the start of the diffusion latent before
sampling and after every scheduler step, so the fixed frame stays clean while
the rest denoises against it through attention.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_parallel_rank,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

# Qwen3-VL vision tokenization bounds for the condition frame.
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> tuple[int, int]:
    """Qwen-VL smart resize: factor-aligned dims within the token budget."""
    max_pixels = (
        max_pixels if max_pixels is not None else IMAGE_MAX_TOKEN_NUM * factor**2
    )
    min_pixels = (
        min_pixels if min_pixels is not None else IMAGE_MIN_TOKEN_NUM * factor**2
    )
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels.")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}.")

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def preprocess_condition_pixel(
    image: Image.Image | list[Image.Image], height: int, width: int
) -> torch.Tensor:
    """Scale-to-cover + center-crop the condition frame.

    Returns a ``(1, C, 1, H, W)`` float tensor in ``[0, 1]`` (the reference
    ``preprocess_image``).
    """
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError(
                f"LingBot-Video TI2V takes exactly one condition image, "
                f"got {len(image)}."
            )
        image = image[0]
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"LingBot-Video TI2V expects a PIL condition image, "
            f"got {type(image).__name__}."
        )
    raw = (
        torch.from_numpy(np.array(image.convert("RGB")))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    old_h, old_w = raw.shape[-2:]
    scale = max(height / old_h, width / old_w)
    new_h = max(math.ceil(old_h * scale), height)
    new_w = max(math.ceil(old_w * scale), width)
    resized = F.interpolate(
        raw, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    top = int(round((new_h - height) / 2.0))
    left = int(round((new_w - width) / 2.0))
    cropped = resized[:, :, top : top + height, left : left + width].float() / 255.0
    return cropped.unsqueeze(2)


def condition_pixel_to_vlm_image(
    pixel: torch.Tensor, patch_factor: int
) -> Image.Image:
    """First frame of the pixel tensor as a smart-resized PIL for Qwen3-VL."""
    frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    image = Image.fromarray(array, mode="RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(height, width, factor=patch_factor)
    return image.resize((resized_width, resized_height))


def vision_patch_factor(text_encoder: Any, processor: Any) -> int:
    """Resolve the VLM image alignment factor (vision patch * spatial merge)."""
    for obj in (
        getattr(getattr(text_encoder, "config", None), "vision_config", None),
        getattr(processor, "image_processor", None),
    ):
        patch = getattr(obj, "patch_size", None)
        if patch is not None:
            return int(patch) * SPATIAL_MERGE_SIZE
    return 16 * SPATIAL_MERGE_SIZE


def should_apply_lingbot_ti2v(batch: Req, server_args: ServerArgs) -> bool:
    """Whether the request should use the LingBot-Video first-frame path."""
    return bool(
        server_args.pipeline_config.task_type == ModelTaskType.TI2V
        and batch.condition_image is not None
        and isinstance(server_args.pipeline_config, LingBotVideoMoEPipelineConfig)
    )


class LingBotVideoDenoisingStage(DenoisingStage):
    """DenoisingStage with LingBot-Video first-frame latent conditioning.

    Text-only requests are untouched (both hooks gate on the condition
    image), so the same stage serves T2V, T2I, and TI2V.
    """

    def _prepare_denoising_loop(
        self, batch: Req, server_args: ServerArgs
    ) -> DenoisingContext:
        ctx = super()._prepare_denoising_loop(batch, server_args)
        if not should_apply_lingbot_ti2v(batch, server_args):
            return ctx
        cond_latent = self._encode_condition_latent(batch, server_args)
        # After temporal SP sharding only rank 0 holds latent frame 0; the
        # other ranks keep z=None so the per-step re-splice is a no-op there.
        if getattr(batch, "did_sp_shard_latents", False) and get_sp_parallel_rank():
            return ctx
        cond_latent = cond_latent.to(
            device=ctx.latents.device, dtype=ctx.latents.dtype
        )
        ctx.latents[:, :, : cond_latent.shape[2]] = cond_latent
        ctx.z = cond_latent
        return ctx

    def _encode_condition_latent(
        self, batch: Req, server_args: ServerArgs
    ) -> torch.Tensor:
        """VAE-encode the preprocessed condition frame into DiT latent space.

        Uses the deterministic posterior mean (like the Wan TI2V path; the
        reference samples the posterior, which only differs by the tiny
        posterior noise) and the same latents_mean/latents_std normalization
        the decode side inverts.
        """
        assert batch.image_latent is None, "TI2V task should not have image latents"
        assert self.vae is not None, "VAE is not provided for TI2V task"
        pixel = batch.preprocessed_image
        if pixel is None:
            raise ValueError(
                "LingBot-Video TI2V requires the text-encoding stage to set "
                "batch.preprocessed_image from the condition image."
            )
        vae_dtype = resolve_precision(server_args, "vae", precision_attr="vae_precision")
        with self.use_declared_component(
            component_name="vae",
            module=self.vae,
            target_dtype=vae_dtype,
        ) as vae:
            assert vae is not None
            self.vae = vae
            norm_pixel = (
                pixel.to(device=get_local_torch_device(), dtype=torch.float32) * 2.0
                - 1.0
            )
            z = vae.encode(norm_pixel.to(vae_dtype)).mean.float()

        arch = server_args.pipeline_config.vae_config.arch_config
        mean = torch.tensor(arch.latents_mean, device=z.device, dtype=torch.float32)
        std = torch.tensor(arch.latents_std, device=z.device, dtype=torch.float32)
        mean = mean.view(1, -1, 1, 1, 1)
        std = std.view(1, -1, 1, 1, 1)
        return (z - mean) / std

    def post_forward_for_ti2v_task(
        self, batch: Req, server_args: ServerArgs, reserved_frames_mask, latents, z
    ):
        latents = super().post_forward_for_ti2v_task(
            batch, server_args, reserved_frames_mask, latents, z
        )
        if z is not None and should_apply_lingbot_ti2v(batch, server_args):
            latents[:, :, : z.shape[2]] = z.to(latents.dtype)
        return latents
