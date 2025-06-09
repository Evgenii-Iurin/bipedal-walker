from pathlib import Path
from typing import Union, Dict, Any

import yaml
from pydantic import BaseModel
from rl_trainer.base.types import ConfigOptions, ConfigOptionT, ConfigSourceT
from rl_trainer.base.registry import get_config_model, get_config_adapter
from rl_trainer.algorithms.vanilla_ppo.config import VanillaPPOConfig  # noqa: F401
from rl_trainer.algorithms.custom_ppo.config import CustomPPOConfig  # noqa: F401
from rl_trainer.adapters import StableBaselinesAdapterConfig  # noqa: F401


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_config(src: ConfigSourceT, option: ConfigOptionT) -> BaseModel:
    """
    *src* may be:
        - str / Path: treated as YAML file path
        - dict : already-loaded YAML
    Returns the instantiated ModelConfig subclass.
    """
    data: Dict[str, Any] = load_config(src) if not isinstance(src, dict) else src

    name_field = "$name" if "$name" in data else "name"
    if name_field not in data:
        raise ValueError("YAML must contain either 'name' or '$name' field")

    if "kwargs" not in data:
        data["kwargs"] = {}

    if option == ConfigOptions.MODEL_CONFIG:
        cfg_cls = get_config_model(data[name_field])
    elif option == ConfigOptions.ADAPTER:
        cfg_cls = get_config_adapter(data[name_field])
    else:
        raise ValueError(f"Unknown config option: {option}. Expected one of {list(ConfigOptions)}")
    return cfg_cls(**data)
