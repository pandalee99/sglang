# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LingBot-Video MoE support.

Pins the review-sensitive pieces that fail only at runtime otherwise:
registry resolution, the router's biased-selection / unbiased-gating
asymmetry, the fused-w13 cache invalidation, and the request-driven T2I
mode of the sampling params.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.lingbot_video_moe import (
    LingBotVideoMoEArchConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NEGATIVE_PROMPT_IMAGE,
    LingBotVideoMoESamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.registry import _get_config_info, get_model_info
from sglang.multimodal_gen.runtime.layers.moe import (
    LingBotVideoRouter,
    LingBotVideoSparseMoeBlock,
)
from sglang.multimodal_gen.runtime.models.dits import (
    lingbot_video_moe as dits_lingbot_video_moe,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_video_moe import (
    LingBotVideoAttention,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    PROMPT_TEMPLATE,
    LingBotVideoTextEncodingStage,
)

_LINGBOT_MODULE_SUBDIRS = (
    "scheduler",
    "text_encoder",
    "processor",
    "transformer",
    "vae",
)


class TestLingBotVideoMoERegistry(unittest.TestCase):
    def test_moe_path_resolves_moe_configs(self):
        get_model_info.cache_clear()
        _get_config_info.cache_clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "lingbot-video-moe-30b-a3b")
            os.makedirs(model_dir)
            with open(
                os.path.join(model_dir, "model_index.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "_class_name": "LingBotVideoPipeline",
                        "_diffusers_version": "0.37.1",
                    },
                    f,
                )
            for subdir in _LINGBOT_MODULE_SUBDIRS:
                os.mkdir(os.path.join(model_dir, subdir))
            info = get_model_info(model_dir, backend="sglang")
        self.assertEqual(info.pipeline_cls.__name__, "LingBotVideoPipeline")
        self.assertIs(info.pipeline_config_cls, LingBotVideoMoEPipelineConfig)
        self.assertIs(info.sampling_param_cls, LingBotVideoMoESamplingParams)

    def test_arch_config_defaults(self):
        arch = LingBotVideoMoEArchConfig()
        self.assertEqual(arch.num_experts, 128)
        # Declared fallback: a checkpoint json without the key must not
        # AttributeError at model init.
        self.assertEqual(arch.mlp_only_layers, ())


class TestLingBotVideoRouterAsymmetry(unittest.TestCase):
    """Selection uses bias-added scores; gating gathers the bias-free scores.

    This is the reference TokenChoiceTopKRouter's load-balancing asymmetry
    and the easiest property to silently break when porting: collapsing the
    two score tensors into one changes outputs without erroring.
    """

    def test_bias_changes_selection_but_not_gate_weights(self):
        hidden, num_experts = 4, 4
        router = LingBotVideoRouter(
            hidden_size=hidden,
            num_experts=num_experts,
            top_k=2,
            score_func="sigmoid",
            norm_topk_prob=False,
            n_group=None,
            topk_group=None,
            route_scale=1.0,
        )
        with torch.no_grad():
            # Expert e scores the one-hot token e-th coordinate: logits equal
            # weight rows; craft descending raw scores for a fixed token.
            router.weight.copy_(
                torch.tensor(
                    [
                        [4.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                        [-2.0, 0.0, 0.0, 0.0],
                        [-4.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
            # Bias promotes expert 3 into the top-2 despite its low raw score.
            router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))

        tokens = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        top_indices, top_scores = router(tokens)

        self.assertEqual(set(top_indices[0].tolist()), {0, 3})
        raw_scores = torch.sigmoid(torch.tensor([4.0, -4.0]))
        picked = {
            int(idx): float(score)
            for idx, score in zip(top_indices[0], top_scores[0])
        }
        # Gate weights are the UNBIASED sigmoid scores, not the biased ones.
        self.assertAlmostEqual(picked[0], float(raw_scores[0]), places=5)
        self.assertAlmostEqual(picked[3], float(raw_scores[1]), places=5)


class TestFusedW13Cache(unittest.TestCase):
    def _block(self):
        return LingBotVideoSparseMoeBlock(
            hidden_size=8,
            intermediate_size=4,
            num_experts=2,
            top_k=1,
            score_func="sigmoid",
            norm_topk_prob=False,
            n_group=None,
            topk_group=None,
            routed_scaling_factor=1.0,
            n_shared_experts=None,
        )

    def test_cache_hit_reuses_tensor(self):
        block = self._block()
        first = block._get_fused_w13()
        self.assertIs(block._get_fused_w13(), first)
        self.assertEqual(first.shape, (2, 8, 8))  # (E, 2*I, H)

    def test_cache_invalidates_when_weights_move(self):
        block = self._block()
        first = block._get_fused_w13()
        with torch.no_grad():
            # Simulate a reload/offload re-upload: fresh storage, new data_ptr.
            block.experts.w1 = torch.nn.Parameter(block.experts.w1.detach().clone())
        second = block._get_fused_w13()
        self.assertIsNot(second, first)


class _PassThroughLinear:
    """Stands in for the parallel Linear layers: returns (x, None)."""

    def __call__(self, x):
        return x, None


class _RecordingAttention:
    """Identity attention that records every (shape, mask) it is given."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, attn_mask=None):
        self.calls.append(
            (tuple(q.shape), None if attn_mask is None else attn_mask.clone())
        )
        return v


class TestAttentionBatchIsolation(unittest.TestCase):
    """Regression for the B>1 cross-sample attention leak.

    The original port flattened a batch into one (1, B*S) sequence with only
    a key-padding mask, so samples attended across batch boundaries (the
    reference packed path isolates samples via varlen cu_seqlens). The fix
    keeps RoPE on the flattened per-token layout but must run attention per
    sample with that sample's own mask.
    """

    def _attention(self, num_heads: int, head_dim: int) -> LingBotVideoAttention:
        attn = object.__new__(LingBotVideoAttention)
        attn.local_num_heads = num_heads
        attn.head_dim = head_dim
        attn.to_q = _PassThroughLinear()
        attn.to_k = _PassThroughLinear()
        attn.to_v = _PassThroughLinear()
        attn.norm_q = lambda t: t
        attn.norm_k = lambda t: t
        attn.to_out = _PassThroughLinear()
        attn.attn = _RecordingAttention()
        return attn

    def _forward(self, attn, x, mask):
        rope_shapes = []

        def identity_rope(t, cos, sin, is_neox_style):
            rope_shapes.append(tuple(t.shape))
            return t

        batch, seq_len, hidden = x.shape
        freqs = torch.zeros(batch * seq_len, attn.head_dim // 2)
        with patch.object(
            dits_lingbot_video_moe, "_apply_rotary_emb", side_effect=identity_rope
        ):
            out = attn.forward(x, (freqs, freqs), attention_mask=mask)
        return out, rope_shapes

    def test_batched_samples_attend_per_sample(self):
        num_heads, head_dim, batch, seq_len = 2, 4, 2, 6
        attn = self._attention(num_heads, head_dim)
        x = torch.randn(batch, seq_len, num_heads * head_dim)
        mask = torch.zeros(batch, 1, 1, seq_len, dtype=torch.bool)
        mask[0, ..., :4] = True
        mask[1, ..., :6] = True

        out, rope_shapes = self._forward(attn, x, mask)

        # RoPE stays on the flattened per-token layout (matches cos/sin).
        self.assertEqual(
            rope_shapes,
            [(1, batch * seq_len, num_heads, head_dim)] * 2,
        )
        # Attention never sees the flattened sequence: one call per sample,
        # each with that sample's own mask.
        self.assertEqual(len(attn.attn.calls), batch)
        for i, (shape, sample_mask) in enumerate(attn.attn.calls):
            self.assertEqual(shape, (1, seq_len, num_heads, head_dim))
            torch.testing.assert_close(sample_mask, mask[i : i + 1])
        # Identity attention round-trips the input in sample order.
        self.assertEqual(tuple(out.shape), (batch, seq_len, num_heads * head_dim))
        torch.testing.assert_close(out, x)

    def test_single_sample_path_unchanged(self):
        num_heads, head_dim, seq_len = 2, 4, 5
        attn = self._attention(num_heads, head_dim)
        x = torch.randn(1, seq_len, num_heads * head_dim)

        out, rope_shapes = self._forward(attn, x, mask=None)

        self.assertEqual(rope_shapes, [(1, seq_len, num_heads, head_dim)] * 2)
        self.assertEqual(len(attn.attn.calls), 1)
        self.assertIsNone(attn.attn.calls[0][1])
        torch.testing.assert_close(out, x)


class _FakeBatchEncoding(dict):
    def to(self, _device):
        return self


class _FakeQwenProcessor:
    """Tokenizes to a fixed width per call type (full prompt vs crop prefix)."""

    def __init__(self, prompt_width: int, prefix_width: int, true_len: int):
        self.prompt_width = prompt_width
        self.prefix_width = prefix_width
        self.true_len = true_len

    def __call__(self, **kwargs):
        if "max_length" in kwargs:  # _build_prompt_inputs (full prompt, padded)
            width = self.prompt_width
            mask = torch.zeros(1, width, dtype=torch.long)
            mask[0, : self.true_len] = 1
        else:  # _compute_crop_start (template prefix, no padding)
            width = self.prefix_width
            mask = torch.ones(1, width, dtype=torch.long)
        return _FakeBatchEncoding(
            input_ids=torch.zeros(1, width, dtype=torch.long),
            attention_mask=mask,
        )


class TestTextEncodingCropAndTrim(unittest.TestCase):
    """Pins the template-crop + right-padding-trim arithmetic.

    An off-by-one here silently feeds template tokens (or drops caption
    tokens) into the DiT — no error, just degraded conditioning.
    """

    def _stage(self, processor, encoder) -> LingBotVideoTextEncodingStage:
        stage = object.__new__(LingBotVideoTextEncodingStage)
        stage.text_encoders = [encoder]
        stage.tokenizers = [processor]
        stage.token_length = 128
        stage.hidden_state_skip_layer = 0
        stage.prompt_template = PROMPT_TEMPLATE
        stage._crop_start = None
        return stage

    def test_crop_start_then_pad_trim(self):
        prompt_width, prefix_width, true_len, channels = 10, 3, 8, 4
        hidden = torch.arange(prompt_width, dtype=torch.float32)
        hidden = hidden.view(1, prompt_width, 1).expand(1, prompt_width, channels)

        def encoder(**kwargs):
            return SimpleNamespace(hidden_states=[hidden])

        stage = self._stage(
            _FakeQwenProcessor(prompt_width, prefix_width, true_len), encoder
        )
        embeds, mask = stage._encode_prompt(
            "a structured caption", torch.device("cpu"), torch.float32
        )

        # Template prefix cropped (3), then right padding trimmed against the
        # cropped mask (8 true - 3 cropped = 5 tokens survive): tokens 3..7.
        self.assertEqual(tuple(embeds.shape), (1, true_len - prefix_width, channels))
        torch.testing.assert_close(embeds, hidden[:, prefix_width:true_len])
        self.assertEqual(int(mask.sum()), true_len - prefix_width)
        # crop_start is cached after the first computation.
        self.assertEqual(stage._compute_crop_start(), prefix_width)


class TestTextEncodingCheckInputs(unittest.TestCase):
    def test_frame_count_contract(self):
        check = LingBotVideoTextEncodingStage.check_inputs
        check(480, 832, 1)  # T2I
        check(480, 832, 81)  # 4n+1
        with self.assertRaises(ValueError):
            check(480, 832, 82)
        with self.assertRaises(ValueError):
            check(480, 830, 81)  # width not %16


class TestPipelineConfigDecodeMath(unittest.TestCase):
    def test_decode_scale_and_shift_invert_vae_normalization(self):
        config = LingBotVideoMoEPipelineConfig()
        scale, shift = config.get_decode_scale_and_shift(
            torch.device("cpu"), torch.float32, vae=None
        )
        arch = config.vae_config.arch_config
        std = torch.tensor(arch.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
        mean = torch.tensor(arch.latents_mean, dtype=torch.float32).view(
            1, -1, 1, 1, 1
        )
        torch.testing.assert_close(scale, 1.0 / std)
        torch.testing.assert_close(shift, mean)

    def test_latents_stay_fp32(self):
        config = LingBotVideoMoEPipelineConfig()
        self.assertEqual(config.get_latent_dtype(torch.bfloat16), torch.float32)


class TestLingBotVideoMoESamplingParamsDataType(unittest.TestCase):
    """num_frames == 1 is a T2I request: IMAGE data_type + still-image negative."""

    def test_single_frame_sets_image_data_type_and_negative(self):
        params = LingBotVideoMoESamplingParams(prompt="test", num_frames=1)
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.IMAGE)
        self.assertTrue(
            params.output_file_name.endswith((".png", ".jpg", ".jpeg", ".webp")),
            f"Expected image extension, got: {params.output_file_name}",
        )
        self.assertEqual(params.negative_prompt, DEFAULT_NEGATIVE_PROMPT_IMAGE)

    def test_single_frame_keeps_custom_negative(self):
        params = LingBotVideoMoESamplingParams(
            prompt="test", num_frames=1, negative_prompt="my negative"
        )
        params._set_output_file_name()
        self.assertEqual(params.negative_prompt, "my negative")

    def test_multi_frame_keeps_video_data_type_and_negative(self):
        params = LingBotVideoMoESamplingParams(prompt="test", num_frames=81)
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.VIDEO)
        self.assertEqual(params.negative_prompt, DEFAULT_NEGATIVE_PROMPT)


if __name__ == "__main__":
    unittest.main()
