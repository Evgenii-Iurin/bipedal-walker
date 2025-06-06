from typing import Dict, Type
from pydantic import BaseModel

_CONFIGS: Dict[str, Type[BaseModel]] = {}


def register_config(cls: Type[BaseModel]) -> Type[BaseModel]:
    """
    Decorator: @register_config
    Registers `cls` under the `name` attribute or field default.
    """
    cfg_name = getattr(cls, "name", None)

    if not cfg_name and hasattr(cls, "model_fields") and "name" in cls.model_fields:
        field_info = cls.model_fields["name"]
        cfg_name = field_info.default

    if not cfg_name:
        raise AttributeError(f"{cls.__name__} is missing a 'name' attribute or field")

    if cfg_name in _CONFIGS:
        raise RuntimeError(f"Duplicate config name '{cfg_name}'")

    _CONFIGS[cfg_name] = cls
    return cls


def get_config_class(name: str) -> Type[BaseModel]:
    try:
        return _CONFIGS[name]
    except KeyError:
        raise KeyError(f"Unknown config name '{name}'. " f"Available: {list(_CONFIGS)}") from None
