# End-to-end pipeline for training RL

Contains an end-to-end pipeline for training a RL algorithm in a bipedal environment

### Dependencies

- python: `^3.11`
- Linux

### Install Dependencies

1. Our highly recomendation to use `conda`
    - Install `conda`. Use [following instruction](https://www.anaconda.com/docs/getting-started/miniconda/install)
    - Create environment `conda create -n rl-trainer-env python=3.11`
    - Activate environment `conda activate rl-trainer-env`
    - We highly recommend to use bind `conda` + `poetry`, so let's set up poetry in a conda environment.
        ```
        conda env list # Check path to the environment, e.g. ~/miniconda3/envs
        export CONDA_ENV_PATH=<your path>
        poetry config virtualenvs.path $CONDA_ENV_PATH
        poetry config virtualenvs.create false
        ```
        ```
        pip install poetry
        ```
    - Check the path of the poetry installation: `which poetry`. It should be installed in your created conda environment

2. Install all dependencies
    `poetry install`
3. Activate pre-commit
    `pre-commit install`

### Run training

1. First run MLFlow server
    ```
    mlflow server --host 127.0.0.1 --port 8080
    ```
2. Run training
    ```
    python src/rl_trainer/train.py
    ```

### Model config

Each model config should contain two things:

* `cls` – path to the module in the format `path:class`
* `$name` – **unique name** for the given model. This name is used to automatically match the config with the provided model.

If your model requires additional parameters, like in this case `inputs`, `logger`, or `callbacks`, you can define them here as well. Remember to add a method into the class to process the given parameters.

Here is the example of config:

```yaml
cls: rl_trainer.algorithms.ppo.model:PPOBipedal

$name: PPOBipedalBaseline

inputs:
    policy: MlpPolicy
    seed: 0
    learning_rate: 0.0003
    gamma:         0.99
    gae_lambda:    0.95
    clip_range:    0.2
    n_steps:       1024
    batch_size:    32
    n_epochs:      1
    ent_coef:      0.0
    vf_coef:       0.5
    max_grad_norm: 0.5
    progress_bar: True

logger:
    - stable_baselines3.common.logger:Logger:
        folder: null
        output_formats:
            - rl_trainer.reporters.mlflow_reporters:MLflowOutputFormat

callbacks:
    - rl_trainer.callbacks.mlflow:MLflowCallback:
        save_freq: 5000
```

### MLFlow

To track all experiments, we are wrapping the training pipeline with MLflow. This allows us to track all metrics, model parameters, and artifacts

![MLFlow UI](assets/mlflow.png)


### Model creation strategy

![Register the Model](assets/pipeline_overview.png)


## Git Flow

```mermaid
gitGraph
   commit id: "init"

   branch base/dataset
   checkout base/dataset
   commit id: "data a"
   commit id: "data b"
   commit id: "data c"

   branch exp/dataset/exp_name
   checkout exp/dataset/exp_name
   commit id: "data 1"
   commit id: "data 2"
   checkout base/dataset
   merge exp/dataset/exp_name

   checkout main

   branch base/model
   checkout base/model
   commit id: "model a"
   commit id: "model b"
   commit id: "model c"

   branch exp/model/exp_name
   checkout exp/model/exp_name
   commit id: "model 1"
   commit id: "model 2"
   checkout base/model
   merge exp/model/exp_name

   checkout main

   branch dev
   checkout dev
   commit id: "dev 1"
   commit id: "dev 2"

   checkout base/dataset
   checkout dev
   merge base/dataset

   checkout base/model
   checkout dev
   merge base/model
   commit id: "dev 3"

   checkout main
   merge dev tag: "0.1.0"
```
