from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.adapter.base import (
    AdapterArchConfig,
    AdapterConfig,
)


@dataclass
class LTX2ConnectorArchConfig(AdapterArchConfig):
    # V1 (LTX-2) transformer-based connector settings
    audio_connector_attention_head_dim: int = 128
    audio_connector_num_attention_heads: int = 30
    audio_connector_num_layers: int = 2
    audio_connector_num_learnable_registers: int = 128
    caption_channels: int = 3840
    causal_temporal_positioning: bool = False
    connector_rope_base_seq_len: int = 4096
    rope_double_precision: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "split"
    text_proj_in_factor: int = 49
    video_connector_attention_head_dim: int = 128
    video_connector_num_attention_heads: int = 30
    video_connector_num_layers: int = 2
    video_connector_num_learnable_registers: int = 128

    # V2 (LTX-2.3) linear projection settings
    # V2 uses simple linear projections instead of transformer blocks
    use_v2_connectors: bool = False  # Auto-detected from checkpoint
    video_hidden_size: int = 4096  # Output dim for video_aggregate_embed
    audio_hidden_size: int = 2048  # Output dim for audio_aggregate_embed
    # Input dim is caption_channels * text_proj_in_factor (3840 * 49 = 188160)


@dataclass
class LTX2ConnectorConfig(AdapterConfig):

    arch_config: AdapterArchConfig = field(default_factory=LTX2ConnectorArchConfig)

    prefix: str = "LTX2"
