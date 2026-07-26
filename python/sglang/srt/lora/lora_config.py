# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import glob
import json
import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Union

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# PEFT shorthand strings resolved downstream against the loaded base model.
PEFT_TARGET_MODULES_SHORTHANDS = ("all", "all-linear")


def _lora_module_path(weight_name: str) -> Optional[str]:
    """Module path of a LoRA weight key, PEFT wrapper prefix stripped.

    ``base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight``
    → ``model.language_model.layers.0.self_attn.q_proj``. Returns None for
    non-LoRA keys (e.g. modules_to_save or embedding weights).
    """
    if ".lora_" not in weight_name:
        return None
    module_path = weight_name.rsplit(".lora_", 1)[0]
    peft_prefix = "base_model.model."
    if module_path.startswith(peft_prefix):
        module_path = module_path[len(peft_prefix) :]
    return module_path


def expand_regex_target_modules(
    pattern: str, weight_names: Iterable[str]
) -> List[str]:
    """Expand a PEFT regex-string ``target_modules`` into leaf module names.

    peft >= 0.17 may save ``target_modules`` as a fullmatch regex over module
    paths of the wrapped model instead of a suffix list. The adapter's own
    weight keys are the ground truth for which modules carry LoRA weights, so
    match the regex against them and reduce to leaf names, which is the form
    the rest of the LoRA stack consumes.
    """
    compiled = re.compile(pattern)
    leaves = set()
    for name in weight_names:
        module_path = _lora_module_path(name)
        if module_path is None:
            continue
        if compiled.fullmatch(module_path):
            leaves.add(module_path.split(".")[-1])
    if not leaves:
        raise ValueError(
            f"target_modules regex {pattern!r} matched none of the adapter's "
            "LoRA weight names; the adapter config and weights disagree."
        )
    return sorted(leaves)


class LoRAConfig:
    def __init__(
        self,
        path: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        added_tokens_config: Optional[Dict] = None,
        base_vocab_size: Optional[int] = None,
    ) -> None:
        self.path = path

        if config_dict is not None:
            self.hf_config = config_dict
            self.added_tokens_config = added_tokens_config
        else:
            self.hf_config = self.get_lora_config()
            self.added_tokens_config = self.get_added_tokens_config()

        self.target_modules = self._resolve_target_modules(
            self.hf_config["target_modules"]
        )
        self.r = self.hf_config["r"]
        self.lora_alpha = self.hf_config["lora_alpha"]
        self.use_dora = self.hf_config.get("use_dora", False)

        # Filter fake added tokens: tokens with ID < base_vocab_size are already
        # part of the base vocabulary and should not be treated as added tokens.
        # This commonly happens when added_tokens.json is copied from the base
        # model's tokenizer.
        if self.added_tokens_config and base_vocab_size is not None:
            self.added_tokens_config = {
                token: token_id
                for token, token_id in self.added_tokens_config.items()
                if token_id >= base_vocab_size
            }

        self.lora_added_tokens_size = (
            len(self.added_tokens_config) if self.added_tokens_config is not None else 0
        )

    def _resolve_target_modules(
        self, target_modules: Union[str, List[str], None]
    ) -> Union[str, List[str], None]:
        """Resolve a PEFT regex-string ``target_modules`` into a module list.

        Lists and the "all"/"all-linear" shorthands pass through unchanged;
        any other string is a peft >= 0.17 regex and is expanded against the
        adapter's own LoRA weight names.
        """
        if not isinstance(target_modules, str):
            return target_modules
        if target_modules in PEFT_TARGET_MODULES_SHORTHANDS:
            return target_modules
        if self.path is None:
            raise ValueError(
                f"target_modules is a regex ({target_modules!r}) but the "
                "LoRAConfig was built from an in-memory dict with no adapter "
                "path to expand it against; pass target_modules as a list "
                "of module names instead."
            )
        weight_names = self._read_adapter_weight_names()
        resolved = expand_regex_target_modules(target_modules, weight_names)
        logger.info(
            "Expanded regex target_modules of LoRA adapter %s to %s",
            self.path,
            resolved,
        )
        return resolved

    def _read_adapter_weight_names(self) -> List[str]:
        """Read the adapter's weight names from its safetensors headers."""
        if not os.path.isdir(self.path):
            weights_dir = snapshot_download(
                self.path, allow_patterns=["*.safetensors"]
            )
        else:
            weights_dir = self.path
        safetensors_files = sorted(
            glob.glob(os.path.join(weights_dir, "*.safetensors"))
        )
        if not safetensors_files:
            raise ValueError(
                f"LoRA adapter {self.path} declares target_modules as a regex, "
                "which SGLang expands against the adapter's safetensors weight "
                "names, but no *.safetensors file was found. Convert the "
                "adapter to safetensors or rewrite target_modules as a list."
            )
        from safetensors import safe_open

        weight_names: List[str] = []
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt") as f:
                weight_names.extend(f.keys())
        return weight_names

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict,
        added_tokens_config: Optional[Dict] = None,
        base_vocab_size: Optional[int] = None,
    ) -> "LoRAConfig":
        return cls(
            config_dict=config_dict,
            added_tokens_config=added_tokens_config,
            base_vocab_size=base_vocab_size,
        )

    def get_lora_config(self, dummy=False):
        if dummy:
            raise NotImplementedError()
        else:
            if not os.path.isdir(self.path):
                weights_dir = snapshot_download(self.path, allow_patterns=["*.json"])
            else:
                weights_dir = self.path
            config_name = "adapter_config.json"
            with open(os.path.join(weights_dir, config_name), "r") as f:
                return json.load(f)

    def get_added_tokens_config(self):
        """Load added tokens from the LoRA adapter if the file exists."""
        # Determine the weights directory
        if not os.path.isdir(self.path):
            weights_dir = snapshot_download(self.path, allow_patterns=["*.json"])
        else:
            weights_dir = self.path

        # Construct the path to added_tokens.json
        added_tokens_path = os.path.join(weights_dir, "added_tokens.json")

        # Return None if the file doesn't exist (optional for standard LoRA adapters)
        if not os.path.exists(added_tokens_path):
            return None

        # Load and return the added tokens
        try:
            with open(added_tokens_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse added_tokens.json: {e}")
            return None
