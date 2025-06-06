from abc import ABC, abstractmethod
from pathlib import Path


class Model(ABC):
    """
    Abstract base class for all models.
    All models should inherit from this class.
    """

    @abstractmethod
    def learn(self) -> None:
        """
        Main method to learn the policy.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save the model to the specified path.

        :param path: Path to save the model.
        """

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Load the model from the specified path.

        :param path: Path to load the model from.
        """
