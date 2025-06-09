from pydantic import Field
from typing import Any
from rl_trainer.base import BaseAdapter
from stable_baselines3.common.base_class import BaseAlgorithm
import logging

logging.basicConfig(level=logging.INFO)


class StableBaselinesAdapter(BaseAdapter):
    """
    Adapter for Stable-Baselines3 RL algorithms.

    This adapter implements the BaseAdapter interface to integrate Stable-Baselines3
    models into the rl_trainer pipeline. It handles the configuration of callbacks,
    loggers, and training execution while maintaining separation between the algorithm
    logic and pipeline infrastructure.

    The adapter follows the adapter pattern to decouple RL algorithms from pipeline
    concerns such as logging, progress tracking, and callback management.

    Attributes:
        name (str): Identifier for the adapter type, defaults to "StableBaselineAdapter"
        timesteps (int): Total number of training timesteps to execute
        progress_bar (bool): Whether to display training progress bar
        callbacks (list): List of callback functions to register with the model
        loggers (list): List of logger instances to register with the model
        model (BaseAlgorithm | None): The loaded Stable-Baselines3 model instance

    Example:
        >>> from stable_baselines3 import PPO
        >>> import gymnasium as gym
        >>>
        >>> # Create environment and model
        >>> env = gym.make("CartPole-v1")
        >>> model = PPO("MlpPolicy", env)
        >>>
        >>> # Create adapter with configuration
        >>> adapter = StableBaselinesAdapter(
        ...     timesteps=10000,
        ...     progress_bar=True,
        ...     callbacks=[checkpoint_callback],
        ...     loggers=[mlflow_logger]
        ... )
        >>>
        >>> # Load model and train
        >>> adapter.load(model).learn()

    Note:
        This adapter is specifically designed for Stable-Baselines3 models that
        implement the BaseAlgorithm interface. For other RL libraries, separate
        adapters should be implemented following the same BaseAdapter interface.
    """

    name: str = Field("StableBaselineAdapter", alias="$name")

    def __init__(
        self,
        *,
        timesteps: int,
        progress_bar: bool = True,
        callbacks: list | None = None,
        loggers: list | None = None,
    ):
        self.timesteps = timesteps
        self.progress_bar = progress_bar
        self.callbacks = callbacks or []
        self.loggers = loggers or []
        self.model: BaseAlgorithm | None = None

    def load(self, model: Any) -> "StableBaselinesAdapter":
        """Register loggers / callbacks and save teh model"""
        if hasattr(model, "set_logger"):
            for lg in self.loggers:
                model.set_logger(lg)
        if hasattr(model, "set_callbacks"):
            for cb in self.callbacks:
                model.set_callbacks(cb)

        self.model = model
        return self

    def learn(self):
        assert self.model is not None, "Load a model before calling learn()"

        logging.info(
            """
                     Starting training with Stable Baselines Adapter.
                     Model: %s
                     Timesteps: %d
                     Progress Bar: %s
                     Callbacks: %s
                     Loggers: %s
                     """,
            self.model.__class__.__name__,
            self.timesteps,
            self.progress_bar,
            [callback.__class__.__name__ for callback in self.callbacks],
            [logger.__class__.__name__ for logger in self.loggers],
        )

        return self.model.learn(total_timesteps=self.timesteps, callback=self.callbacks, progress_bar=self.progress_bar)
