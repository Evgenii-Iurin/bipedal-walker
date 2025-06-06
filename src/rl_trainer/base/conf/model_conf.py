import importlib
from typing import TypeVar, Optional
from pydantic import BaseModel, Field, field_validator
from rl_trainer.base import Model
import gymnasium as gym
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

T = TypeVar("T")


class ModelConfig(BaseModel):
    """
    Base configuration class for models.
    """

    cls: str = Field(
        ...,
        description=(
            """Fully qualified path to the model class
            in the format 'module_path:ClassName'.
            For example: 'stable_baselines3:PPO' or 'custom_models:MyModel'"""
        ),
    )

    inputs: Optional[T] = Field(
        default=None, description="Input data for the model, can be any type defined by the user."
    )

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
        Create an instance of the model configuration from a DictConfig.
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
