from typing import Any
from pydantic import Field, field_validator
from rl_trainer.base.model.base_model import Model
from rl_trainer.base.conf import ModelConfig
from rl_trainer.base.registry import register_config
import importlib
import gymnasium as gym

import logging

logging.basicConfig(level=logging.INFO)


@register_config
class PPOBipedalConfig(ModelConfig):
    """
    Configuration for the PPO BipedalWalker model.

    Attributes:
        cls (str): Fully qualified path to the PPO BipedalWalker model class.
        inputs (Any | None): Input data for the model, can be any type defined by the user.
        kwargs (dict[str, Any] | None): Additional keyword arguments for model initialization.
    """

    name: str = Field("PPOBipedalBaseline", alias="$name")

    logger: list[dict[str, Any]] | None = Field(default=None, description="List of logger configurations for the model")

    callbacks: list[dict[str, Any]] = Field(default=None, description="List of callback configurations for the model")

    @field_validator("logger", mode="before")
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

    def _setup_logger(self):
        """
        Setup loggers based on the logger configuration.

        Returns:
            List of logger instances or empty list if no logger config provided
        """
        if not self.logger:
            return []

        logger_instances = []

        for logger_config in self.logger:
            logger_cls_path, params = next(iter(logger_config.items()))

            try:
                module_path, class_name = logger_cls_path.split(":")

                module = importlib.import_module(module_path)
                logger_cls = getattr(module, class_name)

                if params:
                    for k, v in params.items():
                        if isinstance(v, list):
                            param_list = []
                            for item in v:
                                if isinstance(item, str) and ":" in item:
                                    module_path, class_name = item.split(":")
                                    module = importlib.import_module(module_path)
                                    cls = getattr(module, class_name)
                                    param_list.append(cls())
                                else:
                                    param_list.append(item)
                            params[k] = param_list
                        if isinstance(v, str) and ":" in v:
                            module_path, class_name = v.split(":")
                            module = importlib.import_module(module_path)
                            cls = getattr(module, class_name)
                            params[k] = cls()

                    logger_instance = logger_cls(**params)
                else:
                    logger_instance = logger_cls()

                logger_instances.append(logger_instance)

            except ImportError as exc:
                raise ValueError(f"Failed to import logger module '{module_path}': {exc}") from exc
            except AttributeError as exc:
                raise ValueError(f"Logger class '{class_name}' not found in module '{module_path}': {exc}") from exc
            except TypeError as exc:
                raise ValueError(f"Failed to instantiate logger '{class_name}' with parameters {params}: {exc}") from exc
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

    def _setup_callbacks(self):
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
            module_path, class_name = callback_cls_path.split(":")

            module = importlib.import_module(module_path)
            callback_cls = getattr(module, class_name)

            if params:
                callback_instance = callback_cls(**params)
            else:
                callback_instance = callback_cls()

            callback_instances.append(callback_instance)

        return callback_instances

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
