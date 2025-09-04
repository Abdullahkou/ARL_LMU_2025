import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Optional

import torch
import yaml

from tuning.params import Params

########################### CONSTANTS #################################

RANDOM_AGENT = "RANDOM"

MEAN_FILE = "mean.csv"
SMA_MEAN_FILE = "sma_mean.csv"
STD_FILE = "std.csv"
SMA_STD_FILE = "sma_std.csv"

TRAIN_DIR = "train"
EVAL_DIR = "eval"

MODELS_DIR = "models"
SEEDS_DIR = "seeds"
SEED_PREFIX = "seed_"
TRAINING_RESULTS_FILE = "training_results.csv"
CHECKPOINT_PREFIX = "checkpoint_"
INTERMEDIATE_RESULTS_PREFIX = "eval_"

ENV_SEED = "env_seed"

GAMMA = "gamma"
LR = "lr"

USE_RENDERING = "use_rendering"


def arg_true(arg: str):
    return arg == "true" or arg == "True" or arg == "1"


def get_short_hyperparam_name(param: str):
    if param == GAMMA:
        return "g"
    if param == LR:
        return "lr"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            # Convert dataclass to a dictionary
            return asdict(obj)
        elif isinstance(obj, Enum):
            # Serialize Enum as its value
            return obj.value
        elif issubclass(obj, torch.nn.Module):
            # Serialize PyTorch module as its name
            return obj.__name__

        return super().default(obj)


@dataclass
class TrainingConfig:
    algorithm: str = ""
    steps: int = 100_000

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    use_fixed_env_seeds: bool = True
    fixed_env_seeds: tuple[int, int] = (2, 3)

    eval_phases: int = 20
    eval_episodes: int = 20

    hyper_params: Params = field(default_factory=Params)
    ignore_hyper_params: bool = False


def save_training_config(config: TrainingConfig, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(f"{dir}/training_config.yaml", "w") as file:
        dict = asdict(config)

        # TODO:
        # params = dict["hyper_params"]

        yaml.safe_dump(dict, file, indent=4)


def load_training_config(
    dir: str = "src", training_args: Optional[dict[str, Optional[Any]]] = None
):
    file_path = f"{dir}/training_config.yaml"
    train_config = None

    if not os.path.exists(file_path):
        train_config = create_new_training_config(dir, training_args=training_args)
    else:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

        if "hyper_params" not in config_data or config_data["hyper_params"] is None:
            config_data["hyper_params"] = {}
        train_config = TrainingConfig(**config_data)

        train_config = __overide_config(train_config, training_args)

    json_format = json.dumps(train_config, cls=CustomJSONEncoder, indent=4)
    print(f"Train config:\n {json_format}\n")

    return train_config


# def load_training_config(
#     dir: str = "src", training_args: Optional[dict[str, Optional[Any]]] = None
# ):
#     file_path = f"{dir}/training_config.yaml"
#     train_config = None
#     if not os.path.exists(file_path):
#         train_config = create_new_training_config(dir, training_args=training_args)
#     else:
#         config = None
#         with open(file_path, "r") as file:
#             config = yaml.safe_load(file)
#         try:
#             params_dict: dict = config["hyper_params"]

#             config["hyper_params"] = Params(**params_dict)
#         # TODO:
#         # params: Params = config["hyper_params"]

#         except Exception:
#             traceback.print_exc()
#             print(
#                 "Could not load hyperparams config --- might be old. Using default instead"
#             )
#             config["hyper_params"] = Params()

#         train_config = TrainingConfig(**config)
#         train_config = __overide_config(train_config, training_args)

#     json_format = json.dumps(train_config, cls=CustomJSONEncoder, indent=4)
#     print(f"Train config:\n {json_format}\n")

#     return train_config


def __overide_config(
    train_config: TrainingConfig, training_args: Optional[dict[str, Optional[Any]]]
):
    # override training_config with launch args
    if training_args is None:
        return train_config

    for key, value in training_args.items():
        if value is None:
            continue
        if key == ENV_SEED:
            print(f"Setting {key} to {value}")
            train_config.use_fixed_env_seeds = True
            train_config.fixed_env_seeds = (
                value,
                value + 1,
            )
            continue

        if hasattr(train_config.hyper_params, key):
            print(f"Setting {key} of Hyper Params to {value}")
            setattr(train_config.hyper_params, key, value)

    return train_config


def create_new_training_config(
    dir: str, training_args: Optional[dict[str, Optional[Any]]] = None
):
    config = TrainingConfig()
    config = __overide_config(config, training_args)
    save_training_config(config, dir)
    return config
