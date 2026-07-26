# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits import LingBotVideoDenseConfig
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)


@dataclass
class LingBotVideoDensePipelineConfig(LingBotVideoMoEPipelineConfig):
    """LingBot-Video Dense 1.3B (T2V, plus request-driven T2I via num_frames == 1).

    Same pipeline topology as the MoE variant (Qwen3-VL text encoder, Wan VAE
    decoder, FlowUniPC scheduler, unified LingBotVideo DiT); only the DiT arch
    defaults and the guidance default differ.
    """

    dit_config: DiTConfig = field(default_factory=LingBotVideoDenseConfig)
    embedded_cfg_scale: float = 3.0
