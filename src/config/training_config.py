########################### CONSTANTS #################################

import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

import gymnasium
import yaml

RANDOM_AGENT = "RANDOM"

VALIDATION_RESULTS_DIR = "Validation_Results"
TRAINING_RESULTS_DIR = "Training_Results"
MODELS_DIR = "models"
SEEDS_DIR = "seeds"

CHECKPOINT_PREFIX = "checkpoint_"
INTERMEDIATE_RESULTS_PREFIX = "eval_"

VALIDATION_RESULTS_FILE = "validation_results.csv"
TRAINING_RESULTS_FILE = "training_results.csv"

MEAN_FILE = "mean.csv"
SMA_MEAN_FILE = "sma_mean.csv"
STD_FILE = "std.csv"
SMA_STD_FILE = "sma_std.csv"


def arg_true(arg: str):
    return arg == "true" or arg == "True" or arg == "1"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if is_dataclass(obj):
            # Convert dataclass to a dictionary
            return asdict(obj)  # type: ignore
        elif isinstance(obj, Enum):
            # Serialize Enum as its value
            return obj.value

        return super().default(obj)


def make_gym(env_id: str, seed=42):
    env = gymnasium.make(env_id)
    env.reset(seed=seed)

    return env


class Algorithm(Enum):
    RANDOM = "RANDOM"
    DQN = "DQN"


class Environments(Enum):
    LUNAR_LANDER = "LunarLander-v3"


class EnsembleAggregation(Enum):
    AVG = "avg"
    MAJ_VOTE = "maj_vote"


@dataclass
class HyperParams:
    pass


@dataclass
class AlgoConfig:
    algorithm = Algorithm.RANDOM
    num_agents = 1
    num_q_networks = 1
    ensemble_aggregation = EnsembleAggregation.AVG

    ignore_hyper_params = False
    hyper_params: HyperParams = field(default_factory=HyperParams)


@dataclass
class TrainingConfig:
    env_id = Environments.LUNAR_LANDER
    steps = 100_000

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    train_env_seed = 2
    eval_env_seed = 3

    eval_phases = 20
    eval_episodes = 20
    algo_config: AlgoConfig = field(default_factory=AlgoConfig)


# TODO: check if works correctly
def save_training_config(config: TrainingConfig, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(f"{dir}/training_config.yaml", "w") as file:
        dict = asdict(config)

        # TODO:
        # params = dict["hyper_params"]

        yaml.safe_dump(dict, file, indent=4)


# TODO: check
def load_training_config(dir="src", training_args: dict[str, Any | None] | None = None):
    """Optionally takes args dict to override loaded config! Helpful when not being able to rely on loading a yaml within slurm!"""

    file_path = f"{dir}/training_config.yaml"
    train_config = None

    if not os.path.exists(file_path):
        train_config = create_new_training_config(dir, training_args=training_args)
    else:
        with open(file_path, "r") as file:
            config_data: dict = yaml.safe_load(file)

        train_config = TrainingConfig(**config_data)
        train_config = __overide_config(train_config, training_args)

    json_format = json.dumps(train_config, cls=CustomJSONEncoder, indent=4)
    print(f"Train config:\n {json_format}\n")

    return train_config


def create_new_training_config(
    dir: str, training_args: dict[str, Any | None] | None = None
):
    config = TrainingConfig()
    config = __overide_config(config, training_args)
    save_training_config(config, dir)
    return config


def __overide_config(
    train_config: TrainingConfig, training_args: dict[str, Any | None] | None
):
    # override training_config with launch args
    if training_args is None:
        return train_config

    for key, value in training_args.items():
        if value is None:
            continue

        # if hasattr(train_config.hyper_params, key):
        #     print(f"Setting {key} of Hyper Params to {value}")
        #     setattr(train_config.hyper_params, key, value)

    return train_config
