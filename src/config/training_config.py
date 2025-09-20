########################### CONSTANTS #################################

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

import gymnasium
import yaml
from gymnasium import Env

from utils.env_wrapper import ObsArrayWrapper, SafeActionWrapper

RANDOM_AGENT = "RANDOM"

VALIDATION_RESULTS_DIR = "Validation_Results"
TRAINING_RESULTS_DIR = "Training_Results"
MODELS_DIR = "models"
SEEDS_DIR = "seeds"
HEADS_DIR = "head_results"

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


def make_gym(id: str, seed=42, render_mode: str | None = None) -> Env:
    env = gymnasium.make(id=id, render_mode=render_mode)
    env.reset(seed=seed)

    env = ObsArrayWrapper(env)
    env = SafeActionWrapper(env)

    return env


class Algorithm(Enum):
    RANDOM = "RANDOM"

    CUSTOM_DQN = "Custom_DQN"

    SB3_DQN = "SB3_DQN"
    SB3_A2C = "SB3_A2C"
    SB3_PPO = "SB3_PPO"
    SB3_TD3 = "SB3_TD3"


class Environments(Enum):
    LUNAR_LANDER = "LunarLander-v3"
    FROZEN_LAKE = "FrozenLake-v1"
    MOUNTAIN_CAR = "MountainCar-v0"
    TAXI = "Taxi-v3"
    CARTPOLE_V1 = "CartPole-v1"


class EnsembleAggregation(Enum):
    AVG = "avg"
    MAJ_VOTE = "maj_vote"


@dataclass
class HyperParams:
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 30_000
    learning_starts: int = 3_000
    train_freq: int = 1  # env steps per gradient step
    gradient_steps: int = 1  # how many SGD steps per train call
    target_update_interval: int = 1_000  # in env steps
    tau: float = 1.0  # hard update by default; set <1.0 for soft update
    max_epsilon: float = 0.5
    min_epsilon: float = 0.05
    epsilon_decay_steps: int = 250_000  # linear decay steps
    clip_grad_norm: float | None = 10.0
    double_q: bool = True  # enable Double DQN
    dueling: bool = False  # enable Dueling DQN
    n_q_heads: int = 1  # multiple Q heads for ensembles/uncertainty
    hidden_sizes: tuple[int, ...] = (256, 256)  # MLP
    use_ebql: bool = True  # Ensemble Bootstrapped Q-Learning
    independent_heads: bool = False  # shared encoder vs. completely independent
    ensemble_aggregation: EnsembleAggregation = EnsembleAggregation.AVG
    ebql_strict: bool = True
    boostrap_prob: str | float = "auto"


@dataclass
class AlgoConfig:
    algorithm: Algorithm = Algorithm.CUSTOM_DQN

    ignore_hyper_params: bool = False
    hyper_params: HyperParams = field(default_factory=HyperParams)


@dataclass
class TrainingConfig:
    env_id: Environments = Environments.LUNAR_LANDER
    steps: int = 1_000_000

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    train_env_seed: int = 2
    eval_env_seed: int = 3

    eval_phases: int = 20
    eval_episodes: int = 20
    algo_config: AlgoConfig = field(default_factory=AlgoConfig)

    record_training: bool = True
    train_log_phases: int = 40

    record_heads: bool = True


def save_training_config(config: TrainingConfig, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(f"{dir}/training_config.yaml", "w") as file:
        dict = serialize(config)

        yaml.safe_dump(dict, file, indent=4)


def load_training_config(dir: str = "src", training_args: dict[str, Any] | None = None):
    file_path = f"{dir}/training_config.yaml"

    if not os.path.exists(file_path):
        create_new_training_config(dir, training_args=training_args)

    # YAML als Dict laden
    with open(file_path, "r") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Nur existierende Keys überschreiben (strict)
    if training_args:
        cfg = apply_overrides(cfg, training_args)  # <--- HIER auf dem Dict

    json_format = json.dumps(cfg, cls=CustomJSONEncoder, indent=4)
    print(f"Train config:\n {json_format}\n")

    # Jetzt erst in die Dataclass
    return deserialize(TrainingConfig, cfg)


def apply_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    if not overrides:
        return config

    for dotted, raw in overrides.items():
        # 1) Wert aus String sinnvoll parsen (true/123/[...])
        try:
            value = yaml.safe_load(raw) if isinstance(raw, str) else raw
        except Exception:
            value = raw

        # Neu: leere Strings oder None ignorieren → Default bleibt
        if value is None or value == "":
            continue

        # 2) Strikt durch existierende Dict-Pfade navigieren
        parts = dotted.split(".")
        cur: dict[str, Any] = config
        for p in parts[:-1]:
            if p not in cur:
                raise KeyError(f"Unbekannter Key-Pfad: '{dotted}' (fehlend: '{p}')")
            if not isinstance(cur[p], dict):
                raise TypeError(
                    f"Pfad kollidiert mit Nicht-Dict bei '{p}' für Key '{dotted}'"
                )
            cur = cur[p]  # type: ignore[assignment]

        # 3) Letztes Segment MUSS existieren
        last = parts[-1]
        if last not in cur:
            raise KeyError(f"Unbekannter End-Key: '{dotted}'")
        cur[last] = value

    return config


# def parse_training_args() -> dict[str, str]:
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--set",
#         dest="overrides",
#         action="append",
#         default=[],
#         help="Config-Override, z. B. --set steps=200000 --set algo_config.algorithm=PPO",
#     )
#     args = parser.parse_args()

#     overrides: dict[str, str] = {}
#     for item in args.overrides:
#         if "=" not in item:
#             raise ValueError(f"--set erwartet key=value, bekommen: {item}")
#         k, v = item.split("=", 1)
#         overrides[k] = v
#     return overrides


def parse_training_args(extras: list[str]) -> dict[str, str]:
    overrides = {}
    for item in extras:
        if "=" not in item:
            raise ValueError(f"Config-Override erwartet key=value, bekommen: {item}")
        k, v = item.split("=", 1)
        overrides[k] = v
    return overrides


def create_new_training_config(
    dir: str, training_args: dict[str, Any | None] | None = None
):
    config = TrainingConfig()
    save_training_config(config, dir)
    return config
