from typing import Any
from pydantic import Field, field_validator
from pydantic import BaseModel
from rl_trainer.base import BaseAdapter
from rl_trainer.base.types import ConfigOptions
from rl_trainer.base.registry import register_config
import importlib

import logging

logging.basicConfig(level=logging.INFO)


@register_config(ConfigOptions.ADAPTER)
class StableBaselinesAdapterConfig(BaseModel):
    """
    Configuration class for Stable Baselines3 reinforcement learning models.

    This class extends ModelConfig to provide specific configuration options for
    Stable Baselines3 algorithms, including logger and callback setup with dynamic
    class loading capabilities.

    Attributes:
        logger (list[dict[str, Any]] | None): List of logger configurations for the model.
            Each logger config should be a dictionary with a single key-value pair where:
            - Key: Fully qualified class path in format 'module_path:ClassName'
            - Value: Dictionary of parameters to pass to the logger constructor
            Example: [{'stable_baselines3.common.logger:TensorBoardOutputFormat': {'log_dir': './logs'}}]

        callbacks (list[dict[str, Any]] | None): List of callback configurations for the model.
            Each callback config should be a dictionary with a single key-value pair where:
            - Key: Fully qualified class path in format 'module_path:ClassName'
            - Value: Dictionary of parameters to pass to the callback constructor
            Example: [{'stable_baselines3.common.callbacks:CheckpointCallback': {'save_freq': 1000}}]

    Methods:
        _setup_logger(): Sets up logger instances from configuration
        _setup_callbacks(): Sets up callback instances from configuration
        load_class_from_path(class_path): Dynamically loads a class from module path

    Raises:
        ValueError: If logger or callback configurations are invalid
        ImportError: If specified modules cannot be imported
        AttributeError: If specified classes are not found in modules
        TypeError: If class instantiation fails due to parameter mismatch

    Example:
        ```python
        config = StableBaselinesAdapterConfig(
            logger=[
                {'stable_baselines3.common.logger:TensorBoardOutputFormat': {'log_dir': './logs'}}
            ],
            callbacks=[
                {'stable_baselines3.common.callbacks:CheckpointCallback': {'save_freq': 1000}}
            ]
        )
        ```
    """

    cls: str = "rl_trainer.adapters:StableBaselinesAdapter"

    name: str = Field("StableBaselineAdapter", alias="$name")

    loggers: list[dict[str, Any]] | None = Field(default=None, description="List of logger configurations for the model")

    callbacks: list[dict[str, Any]] = Field(default=None, description="List of callback configurations for the model")

    timesteps: int = Field(
        default=10000,
        description="Total number of timesteps for training the model. Default is 10000.",
    )
    progress_bar: bool = Field(
        default=True,
        description="Whether to display a progress bar during training. Default is True.",
    )

    def create(self):
        """ """

        module_path, class_name = self.cls.split(":")
        module = importlib.import_module(module_path)

        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'.") from exc

        if not isinstance(cls, type):
            raise TypeError(f"'{class_name}' in module '{module_path}' is not a class.")

        if not issubclass(cls, BaseAdapter):
            raise TypeError(f"'{class_name}' in module '{module_path}' is not a BaseAdapter.")

        adapter = cls(
            timesteps=self.timesteps,
            progress_bar=self.progress_bar,
            callbacks=self.setup_callback(),
            loggers=self.setup_logger(),
        )

        return adapter

    @field_validator("loggers", mode="before")
    @classmethod
    def _validate_logger(cls, v):
        """Validates the logger configuration format"""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("Logger configuration must be a list")

        for logger_config in v:
            if not isinstance(logger_config, dict):
                raise ValueError("Each logger configuration must be a dictionary")

            logger_cls_path, params = next(iter(logger_config.items()))

            try:
                _, _ = logger_cls_path.split(":")
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid logger class path format: '{logger_cls_path}'.\n" "Expected format 'module_path:ClassName'"
                ) from exc

            if params is not None and not isinstance(params, dict):
                raise ValueError("Logger parameters must be a dictionary")

        return v

    def setup_logger(self):
        """
        Setup loggers based on the logger configuration.

        Returns:
            List of logger instances or empty list if no logger config provided
        """
        if not self.loggers:
            return []

        logger_instances = []

        for logger_config in self.loggers:
            logger_cls_path, params = next(iter(logger_config.items()))

            try:
                logger_cls = self.__class__.load_class_from_path(logger_cls_path)

                if params:
                    for k, v in params.items():
                        if isinstance(v, list):
                            param_list = []
                            for item in v:
                                if isinstance(item, str) and ":" in item:
                                    cls = self.__class__.load_class_from_path(item)
                                    param_list.append(cls())
                                else:
                                    param_list.append(item)
                            params[k] = param_list
                        if isinstance(v, str) and ":" in v:
                            cls = self.__class__.load_class_from_path(v)
                            params[k] = cls()

                    logger_instance = logger_cls(**params)
                else:
                    logger_instance = logger_cls()

                logger_instances.append(logger_instance)

            except (ValueError, ImportError, AttributeError) as exc:
                raise ValueError(f"Failed to setup logger '{logger_cls_path}': {exc}") from exc
            except TypeError as exc:
                raise ValueError(
                    f"Failed to instantiate logger '{logger_cls_path}' with parameters {params}: {exc}"
                ) from exc
            except Exception as exc:
                raise ValueError(f"Unexpected error setting up logger '{logger_cls_path}': {exc}") from exc

        return logger_instances

    @field_validator("callbacks", mode="before")
    @classmethod
    def _validate_callbacks(cls, v):
        """Validates the callbacks configuration format"""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("Callbacks configuration must be a list")

        for callback_config in v:
            if not isinstance(callback_config, dict):
                raise ValueError("Each callback configuration must be a dictionary")

            if len(callback_config) != 1:
                raise ValueError("Each callback configuration must have exactly one key-value pair")

            callback_cls_path, params = next(iter(callback_config.items()))

            try:
                _, _ = callback_cls_path.split(":")
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid callback class path format: '{callback_cls_path}'.\n"
                    "Expected format 'module_path:ClassName'"
                ) from exc

            if params is not None and not isinstance(params, dict):
                raise ValueError("Callback parameters must be a dictionary")

        return v

    def setup_callback(self):
        """
        Setup callbacks based on the callbacks configuration.

        Returns:
            List of callback instances or empty list if no callbacks config provided
        """
        if not self.callbacks:
            return []

        callback_instances = []

        for callback_config in self.callbacks:
            callback_cls_path, params = next(iter(callback_config.items()))

            callback_cls_path = callback_cls_path.rstrip(":")
            callback_cls = self.__class__.load_class_from_path(callback_cls_path)

            if params:
                callback_instance = callback_cls(**params)
            else:
                callback_instance = callback_cls()

            callback_instances.append(callback_instance)

        return callback_instances

    @classmethod
    def load_class_from_path(cls, class_path: str):
        """
        Dynamically load a class from a module path string.

        Args:
            class_path (str): Fully qualified class path in format 'module_path:ClassName'

        Returns:
            The loaded class

        Raises:
            ValueError: If the class path format is invalid
            ImportError: If the module cannot be imported
            AttributeError: If the class is not found in the module
        """
        try:
            module_path, class_name = class_path.split(":")
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Invalid class path format: '{class_path}'. Expected format 'module_path:ClassName'"
            ) from exc

        try:
            module = importlib.import_module(module_path)
            loaded_class = getattr(module, class_name)
            return loaded_class
        except ImportError as exc:
            raise ImportError(f"Failed to import module '{module_path}': {exc}") from exc
        except AttributeError as exc:
            raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {exc}") from exc
