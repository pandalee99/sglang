# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LingBot-Video TI2V first-frame conditioning.

Covers the smart-resize / cover-crop preprocessing math, the request gate
(only a TI2V-typed LingBot config with a condition image engages the path),
and the Qwen3-VL prompt template contract for image-conditioned requests.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_dense import (
    LingBotVideoDensePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    IMG_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE,
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.ti2v import (
    condition_pixel_to_vlm_image,
    preprocess_condition_pixel,
    should_apply_lingbot_ti2v,
    smart_resize,
)


def _pil(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(
        rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8), mode="RGB"
    )


class TestSmartResize(unittest.TestCase):
    """Derived-property tests for the Qwen-VL token-budget resize math."""

    def test_dims_are_factor_aligned_and_at_least_factor(self):
        h, w = smart_resize(50, 1000, factor=32)
        self.assertEqual(h % 32, 0)
        self.assertEqual(w % 32, 0)
        self.assertGreaterEqual(h, 32)
        self.assertGreaterEqual(w, 32)

    def test_oversized_image_shrinks_into_max_pixels(self):
        max_pixels = 64 * 32**2
        h, w = smart_resize(4096, 4096, factor=32, max_pixels=max_pixels)
        self.assertLessEqual(h * w, max_pixels)
        self.assertEqual(h % 32, 0)
        self.assertEqual(w % 32, 0)

    def test_tiny_image_grows_to_min_pixels(self):
        min_pixels = 16 * 32**2
        h, w = smart_resize(33, 33, factor=32, min_pixels=min_pixels)
        self.assertGreaterEqual(h * w, min_pixels)

    def test_extreme_aspect_ratio_rejected(self):
        with self.assertRaises(ValueError):
            smart_resize(10, 10 * 201, factor=32)


class TestPreprocessConditionPixel(unittest.TestCase):
    def test_cover_crop_shape_and_range(self):
        pixel = preprocess_condition_pixel(_pil(640, 360), height=480, width=832)
        self.assertEqual(tuple(pixel.shape), (1, 3, 1, 480, 832))
        self.assertEqual(pixel.dtype, torch.float32)
        self.assertGreaterEqual(float(pixel.min()), 0.0)
        self.assertLessEqual(float(pixel.max()), 1.0)

    def test_single_element_list_unwrapped(self):
        pixel = preprocess_condition_pixel([_pil(100, 100)], height=64, width=64)
        self.assertEqual(tuple(pixel.shape), (1, 3, 1, 64, 64))

    def test_multi_image_list_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_condition_pixel(
                [_pil(64, 64), _pil(64, 64)], height=64, width=64
            )

    def test_vlm_image_is_patch_factor_aligned(self):
        pixel = preprocess_condition_pixel(_pil(640, 360), height=480, width=832)
        vlm_image = condition_pixel_to_vlm_image(pixel, patch_factor=32)
        self.assertEqual(vlm_image.width % 32, 0)
        self.assertEqual(vlm_image.height % 32, 0)


class TestShouldApplyLingBotTI2V(unittest.TestCase):
    """The gate must require: LingBot config, TI2V task, and an image."""

    def _server_args(self, config):
        return SimpleNamespace(pipeline_config=config)

    def test_dense_config_with_image_applies(self):
        batch = SimpleNamespace(condition_image=_pil(64, 64))
        self.assertTrue(
            should_apply_lingbot_ti2v(
                batch, self._server_args(LingBotVideoDensePipelineConfig())
            )
        )

    def test_text_only_request_does_not_apply(self):
        batch = SimpleNamespace(condition_image=None)
        self.assertFalse(
            should_apply_lingbot_ti2v(
                batch, self._server_args(LingBotVideoDensePipelineConfig())
            )
        )

    def test_moe_t2v_config_does_not_apply(self):
        # The MoE config stays T2V until its PR opts in; an image on a T2V
        # server must not silently engage the conditioning path.
        batch = SimpleNamespace(condition_image=_pil(64, 64))
        self.assertFalse(
            should_apply_lingbot_ti2v(
                batch, self._server_args(LingBotVideoMoEPipelineConfig())
            )
        )


class _RecordingProcessor:
    """Captures the processor call so the template contract can be asserted."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}


class TestPromptTemplateWithImage(unittest.TestCase):
    def _stage(self, processor):
        stage = object.__new__(LingBotVideoTextEncodingStage)
        stage.tokenizers = [processor]
        stage.token_length = 128
        stage.prompt_template = PROMPT_TEMPLATE
        return stage

    def test_image_request_prefixes_vision_tokens(self):
        processor = _RecordingProcessor()
        stage = self._stage(processor)
        image = _pil(64, 64)

        stage._build_prompt_inputs("a cat", images=[image])

        call = processor.calls[-1]
        self.assertEqual(
            call["text"], [PROMPT_TEMPLATE.format(IMG_PROMPT_TEMPLATE + "a cat")]
        )
        self.assertEqual(call["images"], [image])

    def test_text_only_request_has_no_vision_tokens(self):
        processor = _RecordingProcessor()
        stage = self._stage(processor)

        stage._build_prompt_inputs("a cat")

        call = processor.calls[-1]
        self.assertEqual(call["text"], [PROMPT_TEMPLATE.format("a cat")])
        self.assertIsNone(call["images"])


if __name__ == "__main__":
    unittest.main()
