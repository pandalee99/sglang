"""
Unit tests for PEFT regex-string ``target_modules`` support in LoRAConfig.

peft >= 0.17 can save ``target_modules`` as a fullmatch regex over module
paths instead of a suffix list (e.g. the LingBot-Video rewriter LoRA for
Qwen3.6-27B ships one). SGLang previously hard-failed on any string other
than "all"/"all-linear"; LoRAConfig now expands a regex against the
adapter's own safetensors weight names into the leaf-module list the rest
of the LoRA stack consumes.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from sglang.srt.lora.lora_config import LoRAConfig, expand_regex_target_modules
from sglang.test.test_utils import CustomTestCase

# The target_modules regex shipped by the LingBot-Video rewriter adapter
# (robbyant/lingbot-video-rewriter-lora): language-model-only, attention +
# MLP + the HF-layout GDN projections.
REWRITER_REGEX = (
    r"^(model\.language_model(?=\.).*\.(in_proj_a|up_proj|v_proj|k_proj|o_proj"
    r"|gate_proj|out_proj|in_proj_qkv|in_proj_b|q_proj|down_proj|in_proj_z))$"
)

_LM_PREFIX = "base_model.model.model.language_model.layers.0."


def _weight_names():
    return [
        _LM_PREFIX + "self_attn.q_proj.lora_A.weight",
        _LM_PREFIX + "self_attn.q_proj.lora_B.weight",
        _LM_PREFIX + "linear_attn.in_proj_qkv.lora_A.weight",
        _LM_PREFIX + "linear_attn.in_proj_b.lora_B.weight",
        _LM_PREFIX + "mlp.down_proj.lora_A.weight",
        # Vision tower: matches the leaf-name alternation but not the
        # language-model prefix — the regex must exclude it.
        "base_model.model.model.visual.blocks.0.mlp.down_proj.lora_A.weight",
        # Not a LoRA weight (modules_to_save style): skipped entirely.
        _LM_PREFIX + "self_attn.q_proj.weight",
    ]


class TestExpandRegexTargetModules(CustomTestCase):
    def test_rewriter_regex_expands_to_language_model_leaves(self):
        resolved = expand_regex_target_modules(REWRITER_REGEX, _weight_names())
        self.assertEqual(
            resolved, ["down_proj", "in_proj_b", "in_proj_qkv", "q_proj"]
        )

    def test_no_match_raises(self):
        with self.assertRaises(ValueError):
            expand_regex_target_modules(
                r"^model\.nonexistent\..*$", _weight_names()
            )


class TestLoRAConfigTargetModulesResolution(CustomTestCase):
    def _write_adapter(self, tmpdir: str, target_modules) -> str:
        with open(
            os.path.join(tmpdir, "adapter_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "target_modules": target_modules,
                    "r": 32,
                    "lora_alpha": 64,
                    "peft_type": "LORA",
                },
                f,
            )
        tensors = {
            _LM_PREFIX + "self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
            _LM_PREFIX + "self_attn.q_proj.lora_B.weight": torch.zeros(4, 2),
            _LM_PREFIX + "linear_attn.in_proj_qkv.lora_A.weight": torch.zeros(2, 4),
            _LM_PREFIX + "linear_attn.in_proj_z.lora_A.weight": torch.zeros(2, 4),
        }
        save_file(tensors, os.path.join(tmpdir, "adapter_model.safetensors"))
        return tmpdir

    def test_regex_target_modules_resolved_from_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(self._write_adapter(tmpdir, REWRITER_REGEX))
        self.assertEqual(
            config.target_modules, ["in_proj_qkv", "in_proj_z", "q_proj"]
        )

    def test_list_target_modules_pass_through(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(
                self._write_adapter(tmpdir, ["q_proj", "v_proj"])
            )
        self.assertEqual(config.target_modules, ["q_proj", "v_proj"])

    def test_all_linear_shorthand_passes_through(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(self._write_adapter(tmpdir, "all-linear"))
        self.assertEqual(config.target_modules, "all-linear")

    def test_regex_without_adapter_path_raises(self):
        with self.assertRaises(ValueError):
            LoRAConfig.from_dict(
                {
                    "target_modules": REWRITER_REGEX,
                    "r": 32,
                    "lora_alpha": 64,
                }
            )


if __name__ == "__main__":
    unittest.main()
