from pydantic import Field
from rl_trainer.base.registry import register_config
from rl_trainer.base.types import ConfigOptions

from rl_trainer.base.conf.model_config import ModelConfig


@register_config(ConfigOptions.MODEL_CONFIG)
class CustomPPOConfig(ModelConfig):
    """ """

    name: str = Field("CustomPPO", alias="$name")

    cls: str = "rl_trainer.algorithms:CustomPPO"

    inputs: dict = Field(
        default_factory=lambda: {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "gamma": 0.99,
        }
    )
