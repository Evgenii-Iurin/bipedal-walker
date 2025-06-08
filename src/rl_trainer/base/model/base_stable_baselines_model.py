from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from rl_trainer.base import Model


class StableBaselinesModels(Model):
    """
    Base class for StableBaselines3 model wrappers in the RL training framework.

    This class serves as an abstract base for wrapping StableBaselines3 algorithms,
    providing a common interface for reinforcement learning model operations such as
    training, saving, loading, and callback management. It standardizes the interaction
    with different SB3 algorithms while maintaining consistency across the training pipeline.

    The class handles:
    - Model lifecycle management (save/load operations)
    - Callback system integration for training monitoring and control
    - Logger configuration for training metrics and debugging
    - Unified interface for different SB3 algorithm implementations

    Attributes:
        callbacks (list[BaseCallback]): List of SB3 callbacks used during training
        model (BaseAlgorithm): The underlying SB3 algorithm instance

    Note:
        This is a base class and should be subclassed for specific SB3 algorithms
        (e.g., PPO, SAC, DQN). Concrete implementations should initialize the
        `self.model` attribute with the appropriate SB3 algorithm instance.

    Example:
        >>> class PPOModel(StableBaselinesModels):
        ...     def __init__(self, env, **kwargs):
        ...         super().__init__()
        ...         self.model = PPO("MlpPolicy", env, **kwargs)
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize a StableBaselines model wrapper.

        Args:
            callback: Optional callback for training
        """
        self.callbacks: list[BaseCallback] = []

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

    def save(self, path: Path) -> None:
        """
        Save the model to the specified path.

        Args:
            path: The path where the model will be saved.
        """
        self.model.save(path)

    def load(self, path: Path) -> "StableBaselinesModels":
        """
        Load a model from the specified path.

        Args:
            path: The path where the model is saved.

        Returns:
            self: The model instance with loaded weights
        """
        self.model = self.model.__class__.load(path)
        return self
