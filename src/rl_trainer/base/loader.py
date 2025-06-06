from pathlib import Path
from typing import Union, Dict, Any

import yaml
from pydantic import BaseModel

from rl_trainer.base.registry import get_config_class


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_model_config(src: Union[str, Path, Dict[str, Any]]) -> BaseModel:
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

    cfg_cls = get_config_class(data[name_field])
    return cfg_cls(**data)
