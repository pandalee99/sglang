# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    DEFAULT_NEGATIVE_PROMPT,
)
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

# Still-image default: drops the temporal/motion block and the video-only
# terms of DEFAULT_NEGATIVE_PROMPT that cannot apply to a single frame.
DEFAULT_NEGATIVE_PROMPT_IMAGE = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "pillarboxed", "side bars", "portrait image in landscape frame"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "distorted reflections"]}}'


@dataclass
class LingBotVideoDenseSamplingParams(SamplingParams):
    # prompt must be a structured-JSON caption; raw free-text is out-of-distribution.
    num_frames: int = 81
    height: int = 480
    width: int = 832
    fps: int = 24
    num_inference_steps: int = 40
    guidance_scale: float = 3.0
    flow_shift: float = 3.0
    negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT
    seed: int = 42

    def _set_output_file_name(self) -> None:
        # One dense checkpoint serves T2V and T2I; mode is request-driven:
        # num_frames == 1 means T2I (the reference runner forces num_frames=1
        # for --mode t2i). Flip data_type before the base derives the file
        # name so the extension and save path agree, and swap the video
        # default negative prompt for the still-image default.
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
            if self.negative_prompt == DEFAULT_NEGATIVE_PROMPT:
                self.negative_prompt = DEFAULT_NEGATIVE_PROMPT_IMAGE
        super()._set_output_file_name()
