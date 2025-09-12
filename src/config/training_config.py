########################### CONSTANTS #################################

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
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


def serialize(obj):
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {k: serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj


def deserialize(cls, data):
    """Rebuild a dataclass (recursively) from a dict with Enum fields as strings."""
    kwargs = {}
    for field_entry in fields(cls):
        value = data[field_entry.name]
        ftype = field_entry.type

        if is_dataclass(ftype):
            kwargs[field_entry.name] = deserialize(ftype, value)
        elif isinstance(value, dict) and is_dataclass(ftype):
            kwargs[field_entry.name] = deserialize(ftype, value)
        elif isinstance(ftype, type) and issubclass(ftype, Enum):
            kwargs[field_entry.name] = ftype(value)
        else:
            kwargs[field_entry.name] = value
    return cls(**kwargs)


def make_gym(env_id: str, seed=42):
    env = gymnasium.make(env_id)
    env.reset(seed=seed)

    return env


class Algorithm(Enum):
    RANDOM = "RANDOM"
    CUSTOM_DQN = "Custom_DQN"


class Environments(Enum):
    LUNAR_LANDER = "LunarLander-v3"


class EnsembleAggregation(Enum):
    AVG = "avg"
    MAJ_VOTE = "maj_vote"


from dataclasses import dataclass


@dataclass
class HyperParams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch_size: int = 256
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    tau: float = 1.0
    max_epsilon: float = 1.0
    min_epsilon: float = 0.05
    epsilon_decay_steps: int = 250_000
    clip_grad_norm: float = 10.0
    double_q: bool = True
    dueling: bool = False
    n_q_heads: int = 5
    hidden_sizes: tuple[int, ...] = (256, 256)


@dataclass
class AlgoConfig:
    algorithm: Algorithm = Algorithm.RANDOM
    num_agents: int = 1
    num_q_networks: int = 1
    ensemble_aggregation: EnsembleAggregation = EnsembleAggregation.AVG

    ignore_hyper_params: bool = False
    hyper_params: HyperParams = field(default_factory=HyperParams)


@dataclass
class TrainingConfig:
    env_id: Environments = Environments.LUNAR_LANDER
    steps: int = 100_000

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    train_env_seed: int = 2
    eval_env_seed: int = 3

    eval_phases: int = 20
    eval_episodes: int = 20
    algo_config: AlgoConfig = field(default_factory=AlgoConfig)

    record_training: bool = True
    train_log_phases: int = 20


def save_training_config(config: TrainingConfig, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(f"{dir}/training_config.yaml", "w") as file:
        dict = serialize(config)

        yaml.safe_dump(dict, file, indent=4)


def load_training_config(dir="src", training_args: dict[str, Any | None] | None = None):
    """Optionally takes args dict to override loaded config! Helpful when not being able to rely on loading a yaml within slurm!"""

    file_path = f"{dir}/training_config.yaml"
    train_config = None

    if not os.path.exists(file_path):
        train_config = create_new_training_config(dir, training_args=training_args)
    else:
        with open(file_path, "r") as file:
            config_dict: dict = yaml.safe_load(file)

            train_config = deserialize(TrainingConfig, config_dict)
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
