from rl_trainer.base.model.base_stable_baselines_model import StableBaselinesModels
from stable_baselines3 import PPO
from typing import Type
from stable_baselines3.common.base_class import BaseAlgorithm


class PPOBaselineModel(StableBaselinesModels):

    def __init__(
        self,
        *,
        algo_cls: Type[BaseAlgorithm] = PPO,
        timesteps: int = 10_000,
        progress_bar: bool = True,
        **algo_kwargs,
    ):
        super().__init__()
        self.model: BaseAlgorithm = algo_cls(**algo_kwargs)
        self.progress_bar = progress_bar
        self.timesteps = timesteps

    def learn(self) -> "PPOBaselineModel":
        """
        Train the model and track metrics with MLflow.

        Returns:
            self: The trained model instance
        """
        self.model.learn(total_timesteps=self.timesteps, callback=self.callbacks, progress_bar=self.progress_bar)
        return self
