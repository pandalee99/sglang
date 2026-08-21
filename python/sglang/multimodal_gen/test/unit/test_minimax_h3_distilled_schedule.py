# SPDX-License-Identifier: Apache-2.0
"""Pinned distilled schedules and per-step LoRA strength for MiniMax H3."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_pinned_sigmas,
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# The published 4-step turbo grid, as MiniMax states it on the 0-1000 scale.
OFFICIAL_4_STEP_TIMESTEPS = [1000.0, 972.973, 923.077, 800.0]
MINIMAX_H3_VIDEO_SHIFT = 12.0
MINIMAX_H3_AUDIO_SHIFT = 3.0


def _pinned(timesteps):
    return minimax_h3_pinned_sigmas(
        video_timesteps=timesteps,
        video_shift_scale=MINIMAX_H3_VIDEO_SHIFT,
        audio_shift_scale=MINIMAX_H3_AUDIO_SHIFT,
    )


def _validate_pinned(timesteps):
    ServerArgs._validate_minimax_h3_pinned_timesteps(
        SimpleNamespace(minimax_h3_pinned_timesteps=timesteps)
    )


def _validate_strength(**overrides):
    args = SimpleNamespace(
        minimax_h3_lora_strength_schedule=None,
        lora_path=None,
        lora_merge_mode="dynamic",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    ServerArgs._validate_minimax_h3_lora_strength_schedule(args)


@pytest.mark.parametrize(
    ("modality", "shift_scale"),
    (("video", MINIMAX_H3_VIDEO_SHIFT), ("audio", MINIMAX_H3_AUDIO_SHIFT)),
)
def test_published_grid_matches_the_shift_derived_schedule(modality, shift_scale):
    """The published turbo grid is the shift formula at NFE=4, not a new curve.

    MiniMax documents the two grids as timestep lists, which reads as data the
    schedule helper cannot produce. Both are the uniform grid q_i = (N - i) / N
    under each modality's own shift, so a rewrite that drifts from the helper
    also drifts from the checkpoint the adapter was distilled against.
    """
    pinned = _pinned(OFFICIAL_4_STEP_TIMESTEPS)[modality]
    derived = minimax_h3_time_shift_sigmas(num_steps=5, shift_scale=shift_scale)
    assert pinned == pytest.approx(derived, abs=1e-6)


def test_audio_grid_is_derived_from_the_video_grid():
    """Audio rides its own clock; pinning it separately desynchronizes it.

    MiniMax's sampler over-steps the audio stream when the grids disagree, so
    the audio schedule stays a function of the video grid and the two shifts.
    """
    pinned = _pinned([1000.0, 640.0, 320.0])
    assert pinned["video"] == pytest.approx([1.0, 0.64, 0.32, 0.0])
    assert pinned["audio"] == pytest.approx([1.0, 0.307692, 0.105263, 0.0], abs=1e-6)


def test_terminal_zero_is_appended_so_entry_count_is_the_call_count():
    pinned = _pinned(OFFICIAL_4_STEP_TIMESTEPS)
    assert len(pinned["video"]) == len(OFFICIAL_4_STEP_TIMESTEPS) + 1
    assert pinned["video"][-1] == 0.0 and pinned["audio"][-1] == 0.0
    # An already-terminated grid is not terminated twice.
    assert _pinned([1000.0, 500.0, 0.0])["video"] == pytest.approx([1.0, 0.5, 0.0])


def _prep_stage(sigma_shift_scales=None):
    # Skip PipelineStage.__init__, which reaches for the global ServerArgs.
    stage = object.__new__(MiniMaxH3TimestepPreparationStage)
    stage.sigma_shift_scales = sigma_shift_scales
    return stage


_SHIFT_PLAN = SimpleNamespace(
    flow_shift=None,
    audio_flow_shift=None,
    default_flow_shift=MINIMAX_H3_VIDEO_SHIFT,
    default_audio_flow_shift=MINIMAX_H3_AUDIO_SHIFT,
)


def test_pinned_timesteps_replace_the_derived_schedule():
    """The flag has to reach the schedule, and fix the model-call count with it.

    Guards the wiring, not the math: a pinned grid that never reaches
    batch.extra leaves the derived schedule in place, which still generates and
    still sounds plausible while running the adapter off its distilled points.
    """
    batch = SimpleNamespace(extra={}, num_inference_steps=50)
    stage = _prep_stage()
    stage._generate_sigmas_from_plan(
        batch,
        _SHIFT_PLAN,
        SimpleNamespace(minimax_h3_pinned_timesteps=OFFICIAL_4_STEP_TIMESTEPS),
    )
    sigmas = batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
    expected = _pinned(OFFICIAL_4_STEP_TIMESTEPS)
    assert sigmas["video"] == pytest.approx(expected["video"])
    assert sigmas["audio"] == pytest.approx(expected["audio"])
    assert batch.num_inference_steps == len(OFFICIAL_4_STEP_TIMESTEPS)


def test_absent_pinned_timesteps_keep_the_step_count_in_charge():
    batch = SimpleNamespace(extra={}, num_inference_steps=7)
    stage = _prep_stage()
    stage._generate_sigmas_from_plan(
        batch,
        _SHIFT_PLAN,
        SimpleNamespace(minimax_h3_pinned_timesteps=None),
    )
    sigmas = batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
    assert sigmas["video"] == pytest.approx(
        minimax_h3_time_shift_sigmas(num_steps=7, shift_scale=MINIMAX_H3_VIDEO_SHIFT)
    )
    assert batch.num_inference_steps == 7


@pytest.mark.parametrize(
    "timesteps",
    (
        [800.0, 640.0],  # must open at pure noise
        [1000.0, 640.0, 640.0],  # must strictly descend
        [1000.0, 320.0, 640.0],
        [1000.0, float("nan")],
        [1000.0, 1200.0],  # outside the training scale
        [],
    ),
)
def test_startup_rejects_an_unusable_pinned_grid(timesteps):
    with pytest.raises(ValueError):
        _validate_pinned(timesteps)


def test_absent_pinned_grid_leaves_the_derived_schedule_alone():
    _validate_pinned(None)


def test_lora_strength_schedule_requires_the_dynamic_merge_path():
    """Merging would restore and re-add DiT weights at every step boundary."""
    with pytest.raises(ValueError, match="dynamic"):
        _validate_strength(
            minimax_h3_lora_strength_schedule=[1.0, 0.0],
            lora_path="turbo.safetensors",
            lora_merge_mode="merge",
        )


def test_lora_strength_schedule_requires_an_adapter():
    with pytest.raises(ValueError, match="--lora-path"):
        _validate_strength(minimax_h3_lora_strength_schedule=[1.0, 0.0])


def test_lora_strength_schedule_rejects_non_finite_entries():
    with pytest.raises(ValueError, match="finite"):
        _validate_strength(
            minimax_h3_lora_strength_schedule=[1.0, float("nan")],
            lora_path="turbo.safetensors",
        )


def test_absent_lora_strength_schedule_is_not_validated():
    _validate_strength(lora_merge_mode="merge")


def _stage(layers):
    return SimpleNamespace(
        pipeline=lambda: SimpleNamespace(lora_layers=layers),
        server_args=SimpleNamespace(minimax_h3_lora_strength_schedule=None),
    )


def test_lora_strength_schedule_must_cover_every_model_call():
    stage = _stage({"blocks.0.attn.qkv_proj": SimpleNamespace(lora_A=object())})
    stage.server_args.minimax_h3_lora_strength_schedule = [1.0] * 4
    with pytest.raises(ValueError, match="8 model calls"):
        MiniMaxH3DenoisingStage._resolve_lora_strength_schedule(stage, 8)


def test_lora_strength_schedule_requires_wrapped_layers():
    stage = _stage({})
    stage.server_args.minimax_h3_lora_strength_schedule = [1.0, 0.0]
    with pytest.raises(ValueError, match="LoRA-wrapped"):
        MiniMaxH3DenoisingStage._resolve_lora_strength_schedule(stage, 2)


def _dynamic_lora_layer():
    torch.manual_seed(0)
    layer = LinearWithLoRA(nn.Linear(4, 4, bias=False), lora_rank=2, lora_alpha=2)
    layer.set_lora_weights(
        torch.randn(2, 4),
        torch.randn(4, 2),
        strength=1.0,
        clear_existing=True,
        merge_weights=False,
    )
    return layer


def _pipeline_with(layer):
    return SimpleNamespace(
        _get_target_lora_layers=lambda target: ([("transformer", {"l": layer})], None),
        is_lora_merged={},
        cur_adapter_strength={},
    )


def test_zero_strength_yields_the_base_output_and_is_reversible():
    """A zero-strength step costs nothing, and a later step is not stuck off."""
    layer = _dynamic_lora_layer()
    pipeline = _pipeline_with(layer)
    x = torch.randn(3, 4)
    base_out = layer.base_layer(x)

    LoRAPipeline.set_lora_strength(pipeline, strength=0.0, target="transformer")
    assert layer.disable_lora is True
    assert torch.allclose(layer(x), base_out, atol=1e-6)

    LoRAPipeline.set_lora_strength(pipeline, strength=1.0, target="transformer")
    assert layer.disable_lora is False
    assert not torch.allclose(layer(x), base_out, atol=1e-4)


def test_set_lora_strength_rejects_a_merged_target():
    layer = _dynamic_lora_layer()
    pipeline = _pipeline_with(layer)
    pipeline.is_lora_merged["transformer"] = True
    with pytest.raises(ValueError, match="dynamic"):
        LoRAPipeline.set_lora_strength(pipeline, strength=0.0, target="transformer")
