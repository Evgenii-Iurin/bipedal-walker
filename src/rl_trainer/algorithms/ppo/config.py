from pydantic import Field
from rl_trainer.base.conf.model_stable_baselines_config import StableBaselinesConfig
from rl_trainer.base.model.base_model import Model
from rl_trainer.base.registry import register_config
import gymnasium as gym

import logging

logging.basicConfig(level=logging.INFO)


@register_config
class PPOBaselineConfig(StableBaselinesConfig):
    """
    Configuration class for Proximal Policy Optimization (PPO) algorithm applied to BipedalWalker environment.

    This class extends StableBaselinesConfig to provide specific configuration for training
    a PPO agent on gymnasium environment. It handles model creation,
    logger setup, and callback configuration for the reinforcement learning training process.

    Attributes:
        name (str): Model identifier, defaults to "PPOBaselineConfig".
                   Used for naming and registration purposes.

    Example:
        ```python
        config = PPOBaseline(
            name="my_ppo_model",
            cls="stable_baselines3:PPO",
            kwargs={
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64
            }
        )

        env = gym.make("BipedalWalker-v3")
        model = config.create(env)
        ```

    Note:
        This class is automatically registered in the configuration registry
        via the @register_config decorator, allowing it to be instantiated
        from configuration files or programmatically
    """

    name: str = Field("PPOBaseline", alias="$name")

    def create(self, env: gym.Env) -> Model:
        """
        Create and configure a PPO model instance for the given environment.

        This method orchestrates the complete model setup process including:
        1. Base model creation using inherited _create method
        2. Logger configuration and attachment if specified
        3. Callback setup and registration if specified

        Args:
            env (gym.Env): The gymnasium environment instance that the model will interact with.
                          Should be compatible with the PPO algorithm (typically BipedalWalker-v3).

        Returns:
            Model: Fully configured PPO model instance ready for training or evaluation.
                  The model includes all specified loggers and callbacks.

        Raises:
            ValueError: If the environment is incompatible with PPO algorithm.
            RuntimeError: If model creation fails due to configuration issues.

        Example:
            ```python
            env = gym.make("BipedalWalker-v3")
            config_dict = path/to/your/config.json # contains "PPOBaseline" name
            model_config = get_model_config(config_dict)
            model = model_config.create(env)
            model.learn()
            ```
        """
        model = self._create(env)

        if self.logger:
            loggers = self._setup_logger()
            logging.info("Setting up %d loggers for the model", len(loggers))
            for logger in loggers:
                model.set_logger(logger)

        if self.callbacks:
            callbacks = self._setup_callbacks()
            logging.info("Setting up %d callbacks for the model", len(callbacks))
            for callback in callbacks:
                model.set_callbacks(callback)

        return model
