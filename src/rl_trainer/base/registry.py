from typing import Dict, Type
from rl_trainer.base.types import ConfigOptions, ConfigOptionT
from pydantic import BaseModel

_MODEL_CONFIGS: Dict[str, Type[BaseModel]] = {}

_ADAPTER_CONFIGS: Dict[str, Type[BaseModel]] = {}


def register_config(mode: ConfigOptionT):
    """
    Decorator: @register_config
    Registers `cls` under the `name` attribute or field default.
    """

    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:

        cfg_name = getattr(cls, "name", None)

        if not cfg_name and hasattr(cls, "model_fields") and "name" in cls.model_fields:
            field_info = cls.model_fields["name"]
            cfg_name = field_info.default

        if not cfg_name:
            raise AttributeError(f"{cls.__name__} is missing a 'name' attribute or field")

        if mode == ConfigOptions.MODEL_CONFIG:
            if cfg_name in _MODEL_CONFIGS:
                raise RuntimeError(f"Duplicate config name '{cfg_name}'")
            _MODEL_CONFIGS[cfg_name] = cls

        elif mode == ConfigOptions.ADAPTER:
            if cfg_name in _ADAPTER_CONFIGS:
                raise RuntimeError(f"Duplicate adapter config name '{cfg_name}'")
            _ADAPTER_CONFIGS[cfg_name] = cls

        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'model' or 'adapter'.")

        return cls

    return decorator


def get_config_model(name: str) -> Type[BaseModel]:
    try:
        return _MODEL_CONFIGS[name]
    except KeyError:
        raise KeyError(f"Unknown config name '{name}'. " f"Available: {list(_MODEL_CONFIGS)}") from None


def get_config_adapter(name: str) -> Type[BaseModel]:
    try:
        return _ADAPTER_CONFIGS[name]
    except KeyError:
        raise KeyError(f"Unknown adapter config name '{name}'. " f"Available: {list(_ADAPTER_CONFIGS)}") from None
