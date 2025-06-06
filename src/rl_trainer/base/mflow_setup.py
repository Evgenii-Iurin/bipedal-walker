from contextlib import contextmanager
from typing import Any
from omegaconf import DictConfig
import mlflow


@contextmanager
def mlflow_run(cfg: DictConfig) -> Any:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=cfg.run_name):
        if cfg.tags:
            mlflow.set_tags(cfg.tags)
        yield
