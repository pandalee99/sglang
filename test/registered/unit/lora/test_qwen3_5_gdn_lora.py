"""
Unit tests for GDN (GatedDeltaNet) LoRA support in the Qwen3.5/3.6 family.

HF-format adapters (e.g. the LingBot-Video rewriter LoRA for Qwen3.6-27B)
attach LoRA to the HF checkpoint-layout GDN projections (in_proj_qkv +
in_proj_z, in_proj_b + in_proj_a), while SGLang serves them as the fused
in_proj_qkvz / in_proj_ba Linears. These tests pin the cross-file
bookkeeping and the weight-stacking transforms that bridge the two layouts.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.utils import (
    _KNOWN_LORA_TARGET_MODULES,
    get_normalized_target_modules,
    get_stacked_multiply,
)
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.test.test_utils import CustomTestCase

HIDDEN = 32
RANK = 4
NUM_V_HEADS = 6
KEY_DIM = 16
VALUE_DIM = 48

_LAYER_PREFIX = "base_model.model.model.language_model.layers.0.linear_attn."


def _adapter() -> LoRAAdapter:
    # The normalization helpers only touch the weights dict, so a bare
    # instance is enough — no config or backend needed.
    return object.__new__(LoRAAdapter)


class TestGdnLoraBookkeeping(CustomTestCase):
    """Guards "someone extended X without updating Y" across the LoRA stack.

    in_proj_ba support spans four files (model supported_lora_modules and
    get_hidden_dim, _KNOWN_LORA_TARGET_MODULES, stacked-multiply table,
    adapter weight normalization); dropping any one of them fails at server
    startup only, which no other CPU test covers.
    """

    def test_in_proj_ba_is_known_target_module(self):
        self.assertIn("in_proj_ba", _KNOWN_LORA_TARGET_MODULES)

    def test_in_proj_ba_stacked_multiply(self):
        self.assertEqual(get_stacked_multiply("in_proj_ba"), 2)

    def test_in_proj_ba_supported_by_qwen3_5(self):
        self.assertIn("in_proj_ba", Qwen3_5ForCausalLM.supported_lora_modules)

    def test_hf_gdn_leaf_names_normalize_to_fused_modules(self):
        # Leaf names as they appear in HF-format adapter configs.
        normalized = get_normalized_target_modules(
            ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
        )
        self.assertEqual(normalized, {"in_proj_qkvz", "in_proj_ba", "out_proj"})

    def test_get_hidden_dim_in_proj_ba(self):
        model = object.__new__(Qwen3_5ForCausalLM)
        # Qwen3.6-27B text-config values.
        model.config = SimpleNamespace(
            hidden_size=5120,
            head_dim=256,
            num_attention_heads=24,
            num_key_value_heads=4,
            linear_key_head_dim=128,
            linear_num_key_heads=16,
            linear_value_head_dim=128,
            linear_num_value_heads=48,
        )
        # One scalar b and one scalar a per value head: 2 * 48.
        self.assertEqual(model.get_hidden_dim("in_proj_ba", 0), (5120, 96))


class TestGdnLoraWeightNormalization(CustomTestCase):
    """Derived-property tests for the HF-split → fused stacking layouts.

    The stacked A buffer pairs one rank-block per fused slice (q/k/v/z or
    b/a) while B is the plain output-dim concatenation; getting either the
    block order or the qkv A-sharing wrong produces silently wrong LoRA
    output, not an error.
    """

    def test_hf_split_qkv_z_stacks_into_qkvz(self):
        a_qkv = torch.randn(RANK, HIDDEN)
        a_z = torch.randn(RANK, HIDDEN)
        b_qkv = torch.randn(2 * KEY_DIM + VALUE_DIM, RANK)
        b_z = torch.randn(VALUE_DIM, RANK)
        weights = {
            _LAYER_PREFIX + "in_proj_qkv.lora_A.weight": a_qkv.clone(),
            _LAYER_PREFIX + "in_proj_z.lora_A.weight": a_z.clone(),
            _LAYER_PREFIX + "in_proj_qkv.lora_B.weight": b_qkv.clone(),
            _LAYER_PREFIX + "in_proj_z.lora_B.weight": b_z.clone(),
        }

        _adapter()._normalize_in_proj_qkvz(weights)

        a_name = _LAYER_PREFIX + "in_proj_qkvz.lora_A.weight"
        b_name = _LAYER_PREFIX + "in_proj_qkvz.lora_B.weight"
        self.assertEqual(set(weights), {a_name, b_name})
        # q/k/v slices share the in_proj_qkv A block; z keeps its own.
        stacked_a = weights[a_name]
        self.assertEqual(stacked_a.shape, (4 * RANK, HIDDEN))
        for slice_idx in range(3):
            torch.testing.assert_close(
                stacked_a[slice_idx * RANK : (slice_idx + 1) * RANK], a_qkv
            )
        torch.testing.assert_close(stacked_a[3 * RANK :], a_z)
        # B is the plain output concatenation [B_qkv; B_z].
        stacked_b = weights[b_name]
        self.assertEqual(stacked_b.shape, (2 * KEY_DIM + 2 * VALUE_DIM, RANK))
        torch.testing.assert_close(stacked_b[: 2 * KEY_DIM + VALUE_DIM], b_qkv)
        torch.testing.assert_close(stacked_b[2 * KEY_DIM + VALUE_DIM :], b_z)

    def test_hf_split_b_a_stacks_into_ba(self):
        a_b = torch.randn(RANK, HIDDEN)
        a_a = torch.randn(RANK, HIDDEN)
        b_b = torch.randn(NUM_V_HEADS, RANK)
        b_a = torch.randn(NUM_V_HEADS, RANK)
        weights = {
            _LAYER_PREFIX + "in_proj_b.lora_A.weight": a_b.clone(),
            _LAYER_PREFIX + "in_proj_a.lora_A.weight": a_a.clone(),
            _LAYER_PREFIX + "in_proj_b.lora_B.weight": b_b.clone(),
            _LAYER_PREFIX + "in_proj_a.lora_B.weight": b_a.clone(),
        }

        _adapter()._normalize_in_proj_ba(weights)

        a_name = _LAYER_PREFIX + "in_proj_ba.lora_A.weight"
        b_name = _LAYER_PREFIX + "in_proj_ba.lora_B.weight"
        self.assertEqual(set(weights), {a_name, b_name})
        # Slice order must match the fused Linear: b first, then a.
        stacked_a = weights[a_name]
        self.assertEqual(stacked_a.shape, (2 * RANK, HIDDEN))
        torch.testing.assert_close(stacked_a[:RANK], a_b)
        torch.testing.assert_close(stacked_a[RANK:], a_a)
        stacked_b = weights[b_name]
        self.assertEqual(stacked_b.shape, (2 * NUM_V_HEADS, RANK))
        torch.testing.assert_close(stacked_b[:NUM_V_HEADS], b_b)
        torch.testing.assert_close(stacked_b[NUM_V_HEADS:], b_a)

    def test_merged_in_proj_ba_repeats_lora_a(self):
        a = torch.randn(RANK, HIDDEN)
        b = torch.randn(2 * NUM_V_HEADS, RANK)
        weights = {
            _LAYER_PREFIX + "in_proj_ba.lora_A.weight": a.clone(),
            _LAYER_PREFIX + "in_proj_ba.lora_B.weight": b.clone(),
        }

        _adapter()._normalize_in_proj_ba(weights)

        stacked_a = weights[_LAYER_PREFIX + "in_proj_ba.lora_A.weight"]
        self.assertEqual(stacked_a.shape, (2 * RANK, HIDDEN))
        torch.testing.assert_close(stacked_a[:RANK], a)
        torch.testing.assert_close(stacked_a[RANK:], a)
        # lora_B is already full-output-dim: untouched.
        torch.testing.assert_close(
            weights[_LAYER_PREFIX + "in_proj_ba.lora_B.weight"], b
        )

    def test_lone_split_half_is_left_untouched(self):
        # A b-half without its a-half (or qkv without z) must not be
        # half-stacked into a fused weight with a garbage slice.
        a_b = torch.randn(RANK, HIDDEN)
        weights = {_LAYER_PREFIX + "in_proj_b.lora_A.weight": a_b.clone()}

        _adapter()._normalize_in_proj_ba(weights)

        self.assertEqual(
            set(weights), {_LAYER_PREFIX + "in_proj_b.lora_A.weight"}
        )
        torch.testing.assert_close(
            weights[_LAYER_PREFIX + "in_proj_b.lora_A.weight"], a_b
        )


if __name__ == "__main__":
    unittest.main()
