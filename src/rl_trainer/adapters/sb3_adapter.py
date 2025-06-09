from pydantic import Field
from typing import Any
from rl_trainer.base import BaseAdapter
from stable_baselines3.common.base_class import BaseAlgorithm
import logging

logging.basicConfig(level=logging.INFO)


class StableBaselinesAdapter(BaseAdapter):

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
