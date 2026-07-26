# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LingBot-Video MoE refiner stage helpers.

Pins the reference-ported math (truncated sigma schedule, flow noising,
pixel upscale, geometry-consistent condition crop) and the opt-in/opt-out
bookkeeping. All external-literal defaults come from the reference runner
CLI (t_thresh 0.85, 8 steps, 2 tail steps, 1088x1920, guidance 3.0,
shift 3.0).
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.refiner import (
    compute_refiner_sigmas,
    crop_condition_image_to_geometry,
    lingbot_refiner_skipped,
    prepare_refiner_latent,
    resize_video_pixels,
)


class TestComputeRefinerSigmas(unittest.TestCase):
    """Derived properties of the truncated shifted schedule (reference port)."""

    def _sigmas(self, **overrides):
        kwargs = dict(
            sigma_max=1.0,
            sigma_min=0.001,
            num_inference_steps=8,
            shift=3.0,
            t_thresh=0.85,
            tail_steps=2,
        )
        kwargs.update(overrides)
        return compute_refiner_sigmas(**kwargs)

    def test_schedule_starts_at_t_thresh_and_descends(self):
        sigmas = self._sigmas()
        self.assertAlmostEqual(float(sigmas[0]), 0.85, places=5)
        self.assertTrue(np.all(np.diff(sigmas) < 0.0))
        self.assertTrue(np.all(sigmas >= 0.0))
        self.assertTrue(np.all(sigmas <= 1.0))

    def test_truncation_drops_sigmas_above_t_thresh(self):
        sigmas = self._sigmas(tail_steps=0)
        self.assertTrue(np.all(sigmas <= 0.85 + 1e-6))

    def test_tail_appends_extra_low_noise_steps(self):
        base = self._sigmas(tail_steps=0)
        tailed = self._sigmas(tail_steps=2)
        self.assertEqual(tailed.size, base.size + 2)
        np.testing.assert_allclose(tailed[: base.size], base, rtol=1e-6)
        self.assertTrue(np.all(tailed[base.size :] < base[-1]))

    def test_invalid_t_thresh_rejected(self):
        with self.assertRaises(ValueError):
            self._sigmas(t_thresh=0.0)
        with self.assertRaises(ValueError):
            self._sigmas(t_thresh=1.5)

    def test_invalid_steps_rejected(self):
        with self.assertRaises(ValueError):
            self._sigmas(num_inference_steps=0)
        with self.assertRaises(ValueError):
            self._sigmas(tail_steps=-1)


class TestPrepareRefinerLatent(unittest.TestCase):
    def test_flow_interpolation_formula(self):
        x = torch.zeros(1, 4, 3, 8, 8)
        noise = torch.ones_like(x)
        latent = prepare_refiner_latent(x, noise, 0.85)
        torch.testing.assert_close(latent, torch.full_like(x, 0.85))

    def test_signal_preserved_at_zero_threshold_limit(self):
        x = torch.randn(1, 4, 3, 8, 8)
        noise = torch.randn_like(x)
        latent = prepare_refiner_latent(x, noise, 1e-6)
        torch.testing.assert_close(latent, x, atol=1e-4, rtol=1e-4)


class TestResizeVideoPixels(unittest.TestCase):
    def test_bicubic_upscale_shape_and_clamp(self):
        video = torch.rand(1, 3, 5, 32, 48)
        resized = resize_video_pixels(video, height=64, width=96)
        self.assertEqual(tuple(resized.shape), (1, 3, 5, 64, 96))
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_non_video_rank_rejected(self):
        with self.assertRaises(ValueError):
            resize_video_pixels(torch.rand(3, 32, 48), height=64, width=96)


class TestCropConditionImageToGeometry(unittest.TestCase):
    def _image(self, width, height):
        rng = np.random.default_rng(0)
        return Image.fromarray(
            rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8), mode="RGB"
        )

    def test_wide_image_cropped_to_base_geometry(self):
        # Image wider than the base 832x480 geometry: height kept, width cropped.
        pixel = crop_condition_image_to_geometry(
            self._image(2000, 500),
            target_height=1088,
            target_width=1920,
            geometry_height=480,
            geometry_width=832,
        )
        self.assertEqual(tuple(pixel.shape), (1, 3, 1, 1088, 1920))
        self.assertGreaterEqual(float(pixel.min()), 0.0)
        self.assertLessEqual(float(pixel.max()), 1.0)

    def test_tall_image_cropped_to_base_geometry(self):
        pixel = crop_condition_image_to_geometry(
            self._image(500, 2000),
            target_height=1088,
            target_width=1920,
            geometry_height=480,
            geometry_width=832,
        )
        self.assertEqual(tuple(pixel.shape), (1, 3, 1, 1088, 1920))


class TestRefinerBookkeeping(unittest.TestCase):
    def test_refiner_defaults_mirror_reference_cli(self):
        config = LingBotVideoMoEPipelineConfig()
        self.assertFalse(config.load_refiner)
        self.assertEqual(config.refiner_height, 1088)
        self.assertEqual(config.refiner_width, 1920)
        self.assertEqual(config.refiner_num_inference_steps, 8)
        self.assertEqual(config.refiner_guidance_scale, 3.0)
        self.assertEqual(config.refiner_flow_shift, 3.0)
        self.assertEqual(config.refiner_t_thresh, 0.85)
        self.assertEqual(config.refiner_sigma_tail_steps, 2)

    def test_dense_config_inherits_refiner_off_by_default(self):
        self.assertFalse(LingBotVideoDensePipelineConfig().load_refiner)

    def test_per_request_skip_flag(self):
        self.assertFalse(lingbot_refiner_skipped(SimpleNamespace(extra=None)))
        self.assertTrue(
            lingbot_refiner_skipped(SimpleNamespace(extra={"skip_refiner": True}))
        )
        self.assertTrue(
            lingbot_refiner_skipped(
                SimpleNamespace(extra={"diffusers_kwargs": {"skip_refiner": "true"}})
            )
        )
        self.assertFalse(
            lingbot_refiner_skipped(SimpleNamespace(extra={"skip_refiner": False}))
        )


if __name__ == "__main__":
    unittest.main()
