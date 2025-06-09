from enum import Enum
from pathlib import Path
from typing import Literal, Union, Dict, Any


class ConfigOptions(Enum):
    MODEL_CONFIG = "model"
    ADAPTER = "adapter"


ConfigOptionT = Literal[ConfigOptions.MODEL_CONFIG, ConfigOptions.ADAPTER]


ConfigSourceT = Union[str, Path, Dict[str, Any]]
