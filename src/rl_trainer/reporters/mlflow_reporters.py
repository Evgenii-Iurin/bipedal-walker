from typing import Any, Dict, Tuple, Union
import numpy as np
import mlflow
from stable_baselines3.common.logger import KVWriter

import logging

logging.basicConfig(level=logging.INFO)


class MLflowOutputFormat(KVWriter):
    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        for (k, v), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded and "mlflow" in excluded:
                continue
            if isinstance(v, np.ScalarType) and not isinstance(v, str):
                mlflow.log_metric(k, v, step)
