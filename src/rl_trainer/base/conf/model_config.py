import importlib
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from rl_trainer.base import Model
import gymnasium as gym
from abc import abstractmethod, ABC
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class ModelConfig(ABC, BaseModel):
    """
    Configuration class for dynamically creating model instances.

    This class provides a pydantic-based configuration system for defining and creating
    model instances through dynamic imports. It validates the model class path format
    and ensures type safety during model instantiation.

    Attributes:
        cls (str): Fully qualified path to the model class in the format
            'module_path:ClassName'. The class must be a subclass of the base Model class.
        inputs (Optional[Dict[str, Any]]): Input parameters for model initialization as
            a dictionary containing keyword arguments to pass to the model constructor.

    Example:
        >>> # Basic model configuration
        >>> config = ModelConfig(
        ...     cls="stable_baselines3:PPO",
        ...     inputs={"policy": "MlpPolicy", "learning_rate": 0.001}
        ... )
        >>> model = config._create(env=my_env)

        >>> # Configuration without inputs
        >>> config = ModelConfig(cls="custom_models:DQN")
        >>> model = config._create()

    Raises:
        ValueError: If cls field is None or has invalid format (missing ':' separator).

    Note:
        The class uses pydantic for validation and type checking. The cls field is
        validated to ensure it follows the 'module_path:ClassName' format.
    """

    cls: str = Field(
        ...,
        description=(
            """Fully qualified path to the model class
            in the format 'module_path:ClassName'.
            For example: 'stable_baselines3:PPO' or 'custom_models:MyModel'"""
        ),
    )

    inputs: Optional[Dict[str, Any]] = Field(
        default=None, description="Input parameters for the model as keyword arguments dictionary."
    )

    @abstractmethod
    def create(self, env: gym.Env | None = None) -> Model:
        """
        Create a model instance from the configuration.

        This method must be implemented by subclasses and should call the internal
        `_create()` method to instantiate the base model, then apply any additional
        customizations (callbacks, wrappers, etc.).

        Args:
            env (gym.Env | None, optional): Gymnasium environment instance.

        Returns:
            Model: Configured model instance with any additional customizations.

        Example:
            >>> def create(self, env=None):
            ...     model = self._create(env)  # Create base model
            ...     # Add callbacks, wrappers, or other customizations
            ...     model.add_callback(CustomCallback())
            ...     return model
        """
        pass

    @field_validator("cls", mode="before")
    @classmethod
    def _check_formats(cls, v):
        """Validates the format of the model"""
        if v is None:
            raise ValueError("cls field cannot be None")
        try:
            _, _ = v.split(":")
        except ValueError as exc:
            raise ValueError(f"Invalid class path format: '{v}'.\n" "Expected format 'module_path:ClassName'") from exc
        return v

    def _create(self, env: gym.Env | None = None) -> Model:
        """
        Create an instance of the model from the configuration.

        This method dynamically imports the specified model class and instantiates it
        with the provided inputs. The model class must be a subclass of the base Model class.

        Args:
            env (gym.Env | None, optional): Gymnasium environment instance. If provided
                and inputs exist, the environment will be added to the inputs dictionary
                under the 'env' key. Defaults to None.

        Returns:
            Model: An instantiated model object of the class specified in the configuration.

        Raises:
            AttributeError: If the specified class name is not found in the target module.
            TypeError: If the specified name is not a class or if the class is not a
                subclass of Model.
            ImportError: If the specified module path cannot be imported.

        Example:
            >>> config = ModelConfig(cls="stable_baselines3:PPO", inputs={"policy": "MlpPolicy"})
            >>> model = config._create(env=my_env)
            >>> isinstance(model, Model)
            True
        """

        module_path, class_name = self.cls.split(":")
        module = importlib.import_module(module_path)

        try:
            model_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'.") from exc

        if not isinstance(model_cls, type):
            raise TypeError(f"'{class_name}' in module '{module_path}' is not a class.")

        if not issubclass(model_cls, Model):
            raise TypeError(f"'{class_name}' in module '{module_path}' must be a subclass of Model.")

        if env is not None and self.inputs:
            self.inputs["env"] = env

        model = model_cls(**self.inputs or {})

        return model
