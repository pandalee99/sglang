# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from ..constants import MINIMAX_H3_SIGMAS_EXTRA_KEY

logger = init_logger(__name__)


class MiniMaxH3TimestepPreparationStage(PipelineStage):
    deduplicated_tensor_tree_output_fields = ("timesteps", "sigmas")
    deduplicated_extra_tensor_tree_output_keys = (MINIMAX_H3_SIGMAS_EXTRA_KEY,)

    def __init__(self, sigma_shift_scales=None) -> None:
        super().__init__()
        # Per-model sigma shift override (model_index.json "_minimax_h3" release
        # block, sigma_shift_scales): the schedule constants are a MODEL
        # serving contract — fl2va and ref2va use video 12 / audio 3 by default.
        self.sigma_shift_scales = sigma_shift_scales

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} is a MiniMax H3 contract stage "
                "and has no implementation yet."
            )
        self._generate_sigmas_from_plan(batch, plan, server_args)
        self._publish_native_timestep_state(batch)
        return batch

    def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} is a MiniMax H3 contract stage "
                "and has no implementation yet."
            )
        return (
            tuple(server_args.minimax_h3_pinned_timesteps or ()),
            batch.num_inference_steps,
            plan.flow_shift,
            plan.audio_flow_shift,
            plan.default_flow_shift,
            plan.default_audio_flow_shift,
            self.freeze_for_dedup(self.sigma_shift_scales),
        )

    @staticmethod
    def _publish_native_timestep_state(batch: Req) -> None:
        sigmas = batch.extra.get(MINIMAX_H3_SIGMAS_EXTRA_KEY)
        if not isinstance(sigmas, dict):
            raise ValueError("MiniMax H3 sigma schedules must be a mapping")
        video_sigmas = sigmas.get("video")
        audio_sigmas = sigmas.get("audio")
        if (
            not isinstance(video_sigmas, list)
            or not isinstance(audio_sigmas, list)
            or len(video_sigmas) != len(audio_sigmas)
            or len(video_sigmas) < 2
        ):
            raise ValueError(
                "MiniMax H3 video/audio sigma schedules must be equal-length lists"
            )
        batch.sigmas = list(video_sigmas)
        batch.timesteps = torch.tensor(
            [1.0 - float(sigma) for sigma in video_sigmas[:-1]],
            dtype=torch.float32,
        )

    def _generate_sigmas_from_plan(
        self, batch: Req, plan, server_args: ServerArgs
    ) -> None:
        """Generate the fixed per-modality float32 time-shift schedules."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_pinned_sigmas,
            minimax_h3_time_shift_sigmas,
        )

        if MINIMAX_H3_SIGMAS_EXTRA_KEY in batch.extra:
            return
        scales = self._resolved_shift_scales(plan)
        pinned_timesteps = server_args.minimax_h3_pinned_timesteps
        if pinned_timesteps:
            sigmas = minimax_h3_pinned_sigmas(
                video_timesteps=list(pinned_timesteps),
                video_shift_scale=scales["video"],
                audio_shift_scale=scales["audio"],
            )
            self._adopt_pinned_step_count(batch, num_steps=len(sigmas["video"]) - 1)
            batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = sigmas
            return
        requested_num_steps = self._resolved_num_steps(batch)
        batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = {
            modality: minimax_h3_time_shift_sigmas(
                num_steps=requested_num_steps,
                shift_scale=scales[modality],
            )
            for modality in ("video", "audio")
        }

    @staticmethod
    def _adopt_pinned_step_count(batch: Req, *, num_steps: int) -> None:
        requested_num_steps = batch.num_inference_steps
        if (
            isinstance(requested_num_steps, int)
            and not isinstance(requested_num_steps, bool)
            and requested_num_steps != num_steps
        ):
            logger.warning(
                "Ignoring num_inference_steps=%d: --minimax-h3-pinned-timesteps "
                "fixes the schedule at %d model calls.",
                requested_num_steps,
                num_steps,
            )
        batch.num_inference_steps = num_steps

    @staticmethod
    def _resolved_num_steps(batch: Req) -> int:
        requested_num_steps = getattr(batch, "num_inference_steps", None)
        if requested_num_steps is None:
            sampling = getattr(batch, "sampling_params", None)
            requested_num_steps = getattr(sampling, "num_inference_steps", None)
        if requested_num_steps is None:
            requested_num_steps = 50
        if (
            isinstance(requested_num_steps, bool)
            or not isinstance(requested_num_steps, int)
            or requested_num_steps <= 0
        ):
            raise ValueError(
                "num_inference_steps must be a positive integer, got "
                f"{requested_num_steps!r}"
            )
        return requested_num_steps

    def _resolved_shift_scales(self, plan) -> dict[str, float]:
        model_scales = self.sigma_shift_scales
        if model_scales is not None and not isinstance(model_scales, Mapping):
            raise ValueError("model sigma_shift_scales must be an object")

        def resolved_scale(
            *, modality: str, request_value, task_default: float
        ) -> float:
            value = request_value
            source = (
                "request "
                f"{'flow_shift' if modality == 'video' else 'audio_flow_shift'}"
            )
            if value is None and model_scales is not None:
                value = model_scales.get(modality)
                source = f"model sigma_shift_scales.{modality}"
            if value is None:
                value = task_default
                source = (
                    "task default "
                    f"{'flow_shift' if modality == 'video' else 'audio_flow_shift'}"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{source} must be a positive finite number")
            scale = float(value)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{source} must be a positive finite number")
            return scale

        return {
            "video": resolved_scale(
                modality="video",
                request_value=plan.flow_shift,
                task_default=plan.default_flow_shift,
            ),
            "audio": resolved_scale(
                modality="audio",
                request_value=plan.audio_flow_shift,
                task_default=plan.default_audio_flow_shift,
            ),
        }

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("timesteps", batch.timesteps, V.none_or_tensor)
        result.add_check("sigmas", batch.sigmas, V.none_or_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        result.add_check("sigmas", batch.sigmas, V.list_not_empty)
        result.add_check(
            MINIMAX_H3_SIGMAS_EXTRA_KEY,
            batch.extra.get(MINIMAX_H3_SIGMAS_EXTRA_KEY),
            lambda value: isinstance(value, Mapping),
        )
        return result


__all__ = ["MiniMaxH3TimestepPreparationStage"]
