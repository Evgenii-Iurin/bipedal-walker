from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """
    Abstract adapter / trainer, independent of any specific RL library.
    Defines the unified API exposed to notebooks and training scripts.
    """

    @abstractmethod
    def load(self, model: Any) -> "BaseAdapter":
        """
        Attach an algorithm instance to the adapter and configure any
        surrounding utilities (callbacks, loggers, schedulers, …).
        """
        ...

    @abstractmethod
    def learn(self) -> Any:
        """Start the training process according to the adapter’s settings."""
        ...

    def predict(self, *args, **kwargs):
        """Proxy to model.predict if a model has been loaded."""
        raise NotImplementedError

    def save_checkpoint(self, path: str):
        """Persist model weights and, if needed, adapter state."""
        raise NotImplementedError
