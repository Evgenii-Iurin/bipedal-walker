from pydantic import Field
from rl_trainer.algorithms.stable_baselines_config import StableBaselinesConfig
from rl_trainer.base.model.base_model import Model
from rl_trainer.base.registry import register_config
import gymnasium as gym

import logging

logging.basicConfig(level=logging.INFO)


@register_config
class PPOBipedalConfig(StableBaselinesConfig):
    """
    Configuration for the PPO BipedalWalker model.

    Attributes:
        cls (str): Fully qualified path to the PPO BipedalWalker model class.
        inputs (Any | None): Input data for the model, can be any type defined by the user.
        kwargs (dict[str, Any] | None): Additional keyword arguments for model initialization.
    """

    name: str = Field("PPOBipedalBaseline", alias="$name")

    def create(self, env: gym.Env) -> Model:

        model = self._create(env)

        if self.logger:
            loggers = self._setup_logger()
            logging.info(f"Setting up {len(loggers)} loggers for the model")
            for logger in loggers:
                model.set_logger(logger)

        if self.callbacks:
            callbacks = self._setup_callbacks()
            logging.info(f"Setting up {len(callbacks)} callbacks for the model")
            for callback in callbacks:
                model.set_callbacks(callback)

        return model
