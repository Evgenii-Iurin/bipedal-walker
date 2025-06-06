from typing import Optional, Type, Any

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from rl_trainer.base import Model


class StableBaselinesModels(Model):
    """Base class for StableBaselines3 model wrappers"""

    def __init__(
        self,
        model_cls: Type[BaseAlgorithm],
        progress_bar: bool = True,
        timesteps: int = 10000,
        callback: Optional[BaseCallback] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a StableBaselines model wrapper.

        Args:
            model_cls: StableBaselines3 model class
            progress_bar: Whether to show progress bar during training
            timesteps: Total timesteps for training
            callback: Optional callback for training
            **kwargs: Additional arguments passed to the model constructor
        """
        self.model = model_cls(**kwargs)
        self.progress_bar = progress_bar
        self.timesteps = timesteps
        self.callbacks = callback

    def learn(self) -> "StableBaselinesModels":
        """
        Train the model and track metrics with MLflow.

        Returns:
            self: The trained model instance
        """
        self.model.learn(total_timesteps=self.timesteps, callback=self.callbacks, progress_bar=self.progress_bar)
        return self

    def set_logger(self, logger: Any) -> None:
        """
        Set up the logger for the model.

        Args:
            logger: The logger instance to use for logging
        """
        self.model.set_logger(logger)

    def set_callbacks(self, callbacks: list[BaseCallback]) -> None:
        """
        Set the callbacks for the model.

        Args:
            callbacks: List of callback instances to use during training
        """
        self.callbacks = callbacks

    def save(self, path: str) -> None:
        """
        Save the model to the specified path.

        Args:
            path: The path where the model will be saved.
        """
        self.model.save(path)

    def load(self, path: str) -> "StableBaselinesModels":
        """
        Load a model from the specified path.

        Args:
            path: The path where the model is saved.

        Returns:
            self: The model instance with loaded weights
        """
        self.model = self.model.__class__.load(path)
        return self
