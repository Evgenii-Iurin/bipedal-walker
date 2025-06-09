# rl_trainer/configs/my_ppo_config.py
from pydantic import Field
from rl_trainer.base.registry import register_config
from rl_trainer.base.types import ConfigOptions

from rl_trainer.base.conf.model_config import ModelConfig


@register_config(ConfigOptions.MODEL_CONFIG)
class VanillaPPOConfig(ModelConfig):
    """ """

    name: str = Field("VanillaPPO", alias="$name")

    cls: str = "stable_baselines3:PPO"

    inputs: dict = Field(
        default_factory=lambda: {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "gamma": 0.99,
        }
    )
