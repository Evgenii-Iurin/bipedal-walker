from rl_trainer.algorithms.stable_baselines_model import StableBaselinesModels
from stable_baselines3 import PPO
from typing import Type, Any
from stable_baselines3.common.base_class import BaseAlgorithm


class PPOBipedal(StableBaselinesModels):
    """Implementation of PPO for the BipedalWalker environment"""

    def __init__(self, model_cls: Type[BaseAlgorithm] = PPO, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the PPOBipedal model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(model_cls, *args, **kwargs)
        self.name = "PPOBipedalBaseline"
