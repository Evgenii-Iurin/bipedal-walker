from pydantic import Field
from rl_trainer.base.registry import register_config
from rl_trainer.base.types import ConfigOptions

from rl_trainer.base.conf.model_config import ModelConfig


@register_config(ConfigOptions.MODEL_CONFIG)
class CustomPPOConfig(ModelConfig):
    """Configuration class for Custom PPO algorithm.

    The class is registered with the model configuration registry and can be
    automatically matched with YAML configuration files through the 'name' field.

    Attributes:
        name (str): Unique identifier for this configuration, used for automatic
            matching with YAML configs. Defaults to "CustomPPO".
        cls (str): Module path to the Custom PPO model class in the format
            "module:class". Points to the actual implementation.
        inputs (dict): Dictionary containing PPO algorithm parameters including:
            - policy (str): Policy network architecture type, defaults to "MlpPolicy"
            - learning_rate (float): Learning rate for optimization, defaults to 3e-4
            - gamma (float): Discount factor for future rewards, defaults to 0.99

    Example:
        YAML configuration file (custom_ppo.yaml):
        ```yaml
        $name: CustomPPO
        cls: rl_trainer.algorithms:CustomPPO
        inputs:
          policy: MlpPolicy
          learning_rate: 0.0003
          gamma: 0.99
          n_steps: 2048
          batch_size: 64
        ```

        Usage in pipeline:
        ```python
        # Configuration is automatically loaded and matched by name
        # The factory pattern creates the model using this config
        config = load_config("custom_ppo.yaml")
        model = config.create(env=environment)
        ```

    Note:
        This config is registered with ConfigOptions.MODEL_CONFIG, making it
        discoverable by the training pipeline. The 'cls' field must point to
        a valid Custom PPO implementation that follows the base model interface.
    """

    name: str = Field("CustomPPO", alias="$name")

    cls: str = "rl_trainer.algorithms:CustomPPO"

    inputs: dict = Field(
        default_factory=lambda: {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "gamma": 0.99,
        }
    )
