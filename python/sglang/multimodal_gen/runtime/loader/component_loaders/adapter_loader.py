from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _detect_v2_connectors(safetensors_path: str) -> bool:
    """Detect V2 (LTX-2.3) connectors by checking for video_aggregate_embed key.

    V1 (LTX-2) has: text_proj_in, video_connector.*, audio_connector.*
    V2 (LTX-2.3) has: video_aggregate_embed, audio_aggregate_embed
    """
    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
    return any("video_aggregate_embed" in k for k in keys)


class AdapterLoader(ComponentLoader):
    """Loader for small adapter-style modules (e.g., LTX-2 connectors).

    This loader intentionally avoids FSDP sharding and just:
    1) Instantiates the module from `config.json`.
    2) Loads a single safetensors state_dict.
    """

    component_names = ["connectors"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        config = get_diffusers_component_config(component_path=component_model_path)

        cls_name = config.pop("_class_name", None)
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        config.pop("_diffusers_version", None)
        config.pop("_name_or_path", None)

        server_args.model_paths["connectors"] = component_model_path

        # Get safetensors file path first to detect version
        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")
        if len(safetensors_list) != 1:
            raise ValueError(
                f"Found {len(safetensors_list)} safetensors files in {component_model_path}, expected 1"
            )

        # Detect V2 connectors (LTX-2.3) by checking checkpoint keys
        is_v2 = _detect_v2_connectors(safetensors_list[0])
        if is_v2:
            logger.info("Detected V2 (LTX-2.3) connectors, using LTX2TextConnectorsV2")
            cls_name = "LTX2TextConnectorsV2"
            # Add V2 flag to config
            config["use_v2_connectors"] = True

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        target_device = get_local_torch_device()
        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        from types import SimpleNamespace

        with set_default_torch_dtype(default_dtype), skip_init_modules():
            connector_cfg = SimpleNamespace(**config)
            model = model_cls(connector_cfg).to(
                device=target_device, dtype=default_dtype
            )

        loaded = safetensors_load_file(safetensors_list[0])
        missing_keys, unexpected_keys = model.load_state_dict(loaded, strict=False)

        if missing_keys:
            logger.warning(f"Connector missing keys: {missing_keys}")
        if unexpected_keys:
            logger.debug(f"Connector unexpected keys: {unexpected_keys}")

        return model
