# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits import LingBotVideoDenseConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)


@dataclass
class LingBotVideoDensePipelineConfig(LingBotVideoMoEPipelineConfig):
    """LingBot-Video Dense 1.3B (T2V / TI2V, plus request-driven T2I).

    Same pipeline topology as the MoE variant (Qwen3-VL text encoder, Wan VAE,
    FlowUniPC scheduler, unified LingBotVideo DiT); only the DiT arch defaults
    and the guidance default differ. task_type is TI2V because one dense
    checkpoint serves all modes request-driven: a condition image makes the
    request TI2V, no image is plain T2V, and num_frames == 1 is T2I.
    """

    task_type: ModelTaskType = ModelTaskType.TI2V
    dit_config: DiTConfig = field(default_factory=LingBotVideoDenseConfig)
    embedded_cfg_scale: float = 3.0
    # LingBot does its own scale-to-cover + center-crop preprocessing in the
    # text-encoding stage; the Wan-specific TI2V branch must not run.
    skip_input_image_preprocess: bool = True

    def __post_init__(self):
        super().__post_init__()
        # TI2V VAE-encodes the condition frame, so the decoder-only default
        # from the MoE (T2V) config is not enough.
        self.vae_config.load_encoder = True
