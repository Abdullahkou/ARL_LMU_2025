import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Optional

import torch as t
import yaml

#from agents.sac import BufferType
from tuning.params import NetworkType, Params, Schedule

ENV_SEED = "env_seed"

TRAINING_ITERATIONS = "training_iterations"
NETWORK_TYPE = "network_type"
GAMMA = "gamma"
TD_STEPS = "td_steps"
LEARNING_STARTS = "learning_starts"
TAU = "tau"
TARGET_UPDATE_INTERVAL = "target_update_interval"
BUFFER_SIZE = "buffer_size"
BUFFER_TYPE = "buffer_type"
PER_ALPHA = "per_alpha"
PER_BETA = "per_beta"
BATCH_SIZE = "batch_size"
ENTROPY_COEF = "entropy_coef"
VF_COEF = "vf_coef"
NORMALIZE_VALUES = "normalize_values"
MAX_GRAD_NORM = "max_grad_norm"
LR = "lr"
ACTIVATION = "activation"
HIDDEN_SIZES = "hidden_sizes"

def arg_true(arg: str):
    return arg == "true" or arg == "True" or arg == "1"



def get_short_hyperparam_name(param: str):
    if param == GAMMA:
        return "g"
    if param == TD_STEPS:
        return "td"
    if param == LEARNING_STARTS:
        return "ls"
    if param == TRAINING_ITERATIONS:
        return "ti"
    if param == TAU:
        return "t"
    if param == TARGET_UPDATE_INTERVAL:
        return "tu"
    if param == BUFFER_SIZE:
        return "bus"
    if param == PER_ALPHA:
        return "pa"
    if param == PER_BETA:
        return "pb"
    if param == BATCH_SIZE:
        return "b"
    if param == VF_COEF:
        return "vc"
    if param == NORMALIZE_VALUES:
        return "nv"
    if param == MAX_GRAD_NORM:
        return "mgn"
    if param == LR:
        return "lr"
    if param == HIDDEN_SIZES:
        return "hs"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            # Convert dataclass to a dictionary
            return asdict(obj)
        elif isinstance(obj, Enum):
            # Serialize Enum as its value
            return obj.value
        elif issubclass(obj, t.nn.Module):
            # Serialize PyTorch module as its name
            return obj.__name__

        return super().default(obj)



@dataclass
class TrainingConfig:
    steps: int = 1000000

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    use_fixed_env_seeds: bool = True
    fixed_env_seeds: tuple[int, int] = (2, 3)

    eval_phases: int = 20
    eval_episodes: int = 20

    hyper_params: Params = field(default_factory=Params)


def save_training_config(config: TrainingConfig, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(f"{dir}/training_config.yaml", "w") as file:
        dict = asdict(config)

        params = dict["hyper_params"]
        params["network_type"] = params["network_type"].name
        params["activation"] = params["activation"].__name__
        params["buffer_type"] = params["buffer_type"].name

        entropy_coef = config.hyper_params.entropy_coef
        params["entropy_coef_min"] = None

        if isinstance(config.hyper_params.entropy_coef, Schedule):
            params["entropy_coef"] = entropy_coef.start
            params["entropy_coef_min"] = entropy_coef.end
        else:
            params["entropy_coef"] = entropy_coef

        yaml.safe_dump(dict, file, indent=4)


def load_training_config(
    dir: str = "src", training_args: Optional[dict[str, Optional[Any]]] = None
):
    file_path = f"{dir}/training_config.yaml"
    train_config = None
    if not os.path.exists(file_path):
        train_config = create_new_training_config(dir, training_args=training_args)
    else:
        config = None
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        try:
            params_dict: dict = config["hyper_params"]
            entropy_coef = params_dict["entropy_coef"]
            entropy_coef_min = params_dict["entropy_coef_min"]

            if entropy_coef_min is not None:
                params_dict["entropy_coef"] = Schedule(start=entropy_coef, end=entropy_coef_min)
            params_dict.pop("entropy_coef_min")

            config["hyper_params"] = Params(**params_dict)
            params: Params = config["hyper_params"]
            params.network_type = getattr(NetworkType, params.network_type)
            params.activation = getattr(t.nn, params.activation)
            params.buffer_type = getattr(BUFFER_TYPE, params.buffer_type)
        except Exception:
            print("Could not load hyperparams config --- might be old. Using default instead")
            config["hyper_params"] = Params()

        train_config = TrainingConfig(**config)
        train_config = __overide_config(train_config, training_args)

    json_format = json.dumps(train_config, cls=CustomJSONEncoder, indent=4)
    print(f"Train config:\n {json_format}\n")

    return train_config


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
        if key == BUFFER_TYPE:
            print(f"Setting {key} to {value}")
            train_config.hyper_params.buffer_type = getattr(BUFFER_TYPE, value)
            continue

        if hasattr(train_config.hyper_params, key):
            print(f"Setting {key} of Hyper Params to {value}")
            setattr(train_config.hyper_params, key, value)

    return train_config


def create_new_training_config(dir: str, training_args: Optional[dict[str, Optional[Any]]] = None):
    config = TrainingConfig()
    config = __overide_config(config, training_args)
    save_training_config(config, dir)
    return config