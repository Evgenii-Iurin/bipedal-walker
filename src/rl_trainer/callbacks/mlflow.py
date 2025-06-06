import numpy as np
import mlflow
import os
from stable_baselines3.common.callbacks import BaseCallback

import logging

logging.basicConfig(level=logging.INFO)


class MLflowCallback(BaseCallback):
    def __init__(self, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq

    def _on_training_start(self) -> None:
        hp = {}
        for attr_name in dir(self.model):
            if not attr_name.startswith("_"):
                try:
                    attr_value = getattr(self.model, attr_name)
                    if np.isscalar(attr_value) and not callable(attr_value):
                        if isinstance(attr_value, (np.integer, np.floating)):
                            hp[attr_name] = attr_value.item()
                        elif isinstance(attr_value, (int, float, str, bool)):
                            hp[attr_name] = attr_value
                except (AttributeError, TypeError):
                    continue
        mlflow.log_params(hp)

    def _on_step(self) -> bool:
        if self.num_timesteps and self.num_timesteps % self.save_freq == 0:
            tmp = f"ckpt_{self.num_timesteps}.zip"
            self.model.save(tmp)
            logging.info("Saving model checkpoint to: %s", tmp)
            mlflow.log_artifact(tmp, artifact_path="checkpoints")
            os.remove(tmp)
        return True
