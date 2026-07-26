# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LingBot-Video Dense variant.

Registry resolution (dense vs MoE detector), dense arch defaults, and the
request-driven T2I mode (num_frames == 1) of the dense sampling params.
"""

import json
import os
import tempfile
import unittest

from sglang.multimodal_gen.configs.models.dits.lingbot_video_dense import (
    LingBotVideoDenseArchConfig,
    LingBotVideoDenseConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_dense import (
    LingBotVideoDensePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_dense import (
    DEFAULT_NEGATIVE_PROMPT_IMAGE,
    LingBotVideoDenseSamplingParams,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    DEFAULT_NEGATIVE_PROMPT,
    LingBotVideoMoESamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.registry import _get_config_info, get_model_info

_LINGBOT_MODULE_SUBDIRS = (
    "scheduler",
    "text_encoder",
    "processor",
    "transformer",
    "vae",
)


def _make_model_dir(root: str, repo_name: str) -> str:
    model_dir = os.path.join(root, repo_name)
    os.makedirs(model_dir)
    with open(
        os.path.join(model_dir, "model_index.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"_class_name": "LingBotVideoPipeline", "_diffusers_version": "0.37.1"},
            f,
        )
    for subdir in _LINGBOT_MODULE_SUBDIRS:
        os.mkdir(os.path.join(model_dir, subdir))
    return model_dir


class TestLingBotVideoDenseRegistry(unittest.TestCase):
    """The dense and MoE detectors must resolve to their own config classes."""

    def _resolve(self, repo_name: str):
        get_model_info.cache_clear()
        _get_config_info.cache_clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _make_model_dir(tmpdir, repo_name)
            return get_model_info(model_dir, backend="sglang")

    def test_dense_path_resolves_dense_configs(self):
        info = self._resolve("lingbot-video-dense-1.3b")
        self.assertEqual(info.pipeline_cls.__name__, "LingBotVideoPipeline")
        self.assertIs(info.pipeline_config_cls, LingBotVideoDensePipelineConfig)
        self.assertIs(info.sampling_param_cls, LingBotVideoDenseSamplingParams)

    def test_moe_path_still_resolves_moe_configs(self):
        info = self._resolve("lingbot-video-moe-30b-a3b")
        self.assertEqual(info.pipeline_cls.__name__, "LingBotVideoPipeline")
        self.assertIs(info.pipeline_config_cls, LingBotVideoMoEPipelineConfig)
        self.assertIs(info.sampling_param_cls, LingBotVideoMoESamplingParams)


class TestLingBotVideoDenseConfigs(unittest.TestCase):
    def test_dense_arch_defaults_disable_moe(self):
        arch = LingBotVideoDenseArchConfig()
        self.assertEqual(arch.num_experts, 0)
        self.assertEqual(arch.depth, 24)
        self.assertEqual(arch.axes_lens, (8192, 1024, 1024))
        self.assertIsNone(arch.n_shared_experts)
        self.assertIsNone(arch.n_group)
        self.assertIsNone(arch.topk_group)
        self.assertEqual(arch.routed_scaling_factor, 1.0)
        self.assertEqual(arch.mlp_only_layers, ())

    def test_pipeline_config_uses_dense_dit_and_wan_vae(self):
        config = LingBotVideoDensePipelineConfig()
        self.assertIsInstance(config.dit_config, LingBotVideoDenseConfig)
        # One dense checkpoint serves T2V/TI2V/T2I request-driven; TI2V
        # accepts-but-does-not-require the condition image.
        self.assertEqual(config.task_type, ModelTaskType.TI2V)
        # TI2V VAE-encodes the condition frame, so the dense config must
        # re-enable the encoder the MoE (T2V) __post_init__ turns off.
        self.assertTrue(config.vae_config.load_encoder)
        self.assertTrue(config.vae_config.load_decoder)


class TestLingBotVideoDenseSamplingParamsDataType(unittest.TestCase):
    """num_frames == 1 is a T2I request: IMAGE data_type + still-image negative."""

    def test_single_frame_sets_image_data_type_and_negative(self):
        params = LingBotVideoDenseSamplingParams(prompt="test", num_frames=1)
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.IMAGE)
        self.assertTrue(
            params.output_file_name.endswith((".png", ".jpg", ".jpeg", ".webp")),
            f"Expected image extension, got: {params.output_file_name}",
        )
        self.assertEqual(params.negative_prompt, DEFAULT_NEGATIVE_PROMPT_IMAGE)

    def test_single_frame_keeps_custom_negative(self):
        params = LingBotVideoDenseSamplingParams(
            prompt="test", num_frames=1, negative_prompt="my negative"
        )
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.IMAGE)
        self.assertEqual(params.negative_prompt, "my negative")

    def test_multi_frame_keeps_video_data_type_and_negative(self):
        params = LingBotVideoDenseSamplingParams(prompt="test", num_frames=81)
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.VIDEO)
        self.assertEqual(params.negative_prompt, DEFAULT_NEGATIVE_PROMPT)

    def test_default_num_frames_is_video(self):
        params = LingBotVideoDenseSamplingParams(prompt="test")
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.VIDEO)


if __name__ == "__main__":
    unittest.main()
