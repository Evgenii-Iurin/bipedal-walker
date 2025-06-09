import gymnasium as gym
import hydra
from omegaconf import DictConfig
from rl_trainer.base.types import ConfigOptions
from rl_trainer.base.loader import get_config
from rl_trainer.base.mflow_setup import mlflow_run

import logging

logging.basicConfig(level=logging.INFO)


def build_env(cfg_env: DictConfig) -> gym.Env:
    """
    Build the environment based on the configuration.
    """
    env = gym.make(cfg_env.name)
    return env


def run_training_pipeline(cfg: DictConfig) -> None:
    """
    Main function to run the training pipeline.
    """
    env = build_env(cfg.setup.env)
    model_config = get_config(cfg.setup.algo.config, ConfigOptions.MODEL_CONFIG)
    adapter_config = get_config(cfg.setup.adapter.config, ConfigOptions.ADAPTER)

    model = model_config.create(env)
    adapter = adapter_config.create()

    trainer = adapter.load(model)

    with mlflow_run(cfg.mlflow):
        trainer.learn()


@hydra.main(version_base=None, config_path="configs", config_name="common")
def main(cfg: DictConfig) -> None:
    run_training_pipeline(cfg)


if __name__ == "__main__":
    main()
