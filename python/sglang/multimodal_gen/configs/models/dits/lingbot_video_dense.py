# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.dits.lingbot_video_moe import (
    LingBotVideoMoEArchConfig,
)


@dataclass
class LingBotVideoDenseArchConfig(LingBotVideoMoEArchConfig):
    """LingBot-Video Dense 1.3B: every block uses the dense MLP.

    Same unified LingBotVideo DiT architecture as the MoE variant;
    ``num_experts == 0`` routes every block through ``LingBotVideoMLP``.
    Defaults mirror the released ``lingbot-video-dense-1.3b`` transformer
    config; the checkpoint's ``config.json`` supersedes them at load time.
    """

    depth: int = 24
    axes_lens: tuple[int, ...] = (8192, 1024, 1024)

    num_experts: int = 0
    moe_intermediate_size: int = 512
    n_shared_experts: int | None = None
    n_group: int | None = None
    topk_group: int | None = None
    routed_scaling_factor: float = 1.0
    # Declared here (the MoE config relies on the checkpoint json providing it)
    # so a config.json without the key still loads with no MLP-only overrides.
    mlp_only_layers: tuple[int, ...] = ()


@dataclass
class LingBotVideoDenseConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LingBotVideoDenseArchConfig)
    prefix: str = "LingBotVideo"
