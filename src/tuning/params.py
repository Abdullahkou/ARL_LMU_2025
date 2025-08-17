import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np
import torch as t


class NetworkType(Enum):
    MLP = 0
    SELF_ATTENTION = 1


class BufferType(Enum):
    ER = 0
    PER = 1
    SIL = 2

    def __repr__(self):
        return self.name


@dataclass
class Schedule:
    start: float
    end: float

    def post_init(self, total_steps: int):
        self.value = self.start
        self.current_step = 0
        self.decay_factor = (self.start - self.end) / total_steps

    def decay(self, current_step: int):
        step_reduction = np.abs(current_step - self.current_step)
        self.current_step = current_step
        self.value = np.maximum(self.value - self.decay_factor * step_reduction, self.end)


@dataclass
class Params:
    network_type: NetworkType = NetworkType.MLP

    gamma: float = 0.99
    # 1 SAC, 2048 PPO, 1 DDPG
    td_steps: int = 1
    learning_starts: int = 100
    # 1 SAC, 10 PPO, 1 DDPG
    training_iterations: int = 1

    tau: float = 0.005
    target_update_interval: int = 1
    buffer_size: int = 1_000_000
    buffer_type: BufferType = BufferType.ER
    per_alpha: float = 0.7
    per_beta: float = 0.5

    # 256 SAC, 64 PPO, 256 DDPG
    batch_size: int = 256
    # "auto" SAC, 0 PPO
    entropy_coef: Union[float, Schedule] = field(default_factory=lambda: Schedule(start=1, end=-1))
    vf_coef: float = 0.5
    normalize_values: bool = False
    max_grad_norm: float = 0.5

    lr: float = 3e-4
    # t.nn.ReLU SAC, t.nn.Tanh PPO, t.nn.ReLU + t.nn.Tanh DDPG
    activation: t.nn.Module = t.nn.Tanh

    # [256, 256] SAC, [64, 64] PPO, [400, 300] DDPG
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])


# def export_params_json(params: Params, dir: str):
#     file = f"{dir}/params.json"

#     params_dict = dataclasses.asdict(params)
#     params_dict["pi_activation"] = params.pi_activation.__name__
#     params_dict["val_activation"] = params.val_activation.__name__
#     json.dump(params_dict, open(file, "w"))

#     # print(" ")
#     # print(f"Exported params to {file}")


# def load_params_json(dir: str):
#     file = f"{dir}/params.json"
#     with open(file, "r") as f:
#         params_dict = json.load(f)

#     params_dict["pi_activation"] = getattr(t.nn, params_dict["pi_activation"])
#     params_dict["val_activation"] = getattr(t.nn, params_dict["val_activation"])

#     params = Params(**params_dict)

#     return params


# def export_params_text(params: Params, dir: str):
#     file = f"{dir}/params.txt"
#     with open(file, "w") as f:
#         for key, value in params.__dict__.items():
#             f.write(f"{key} = {value}\n")

#     print(" ")
#     print(f"Exported params to {file}")


def make_dir(base_dir: str):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dir = f"{base_dir}/run_{len(next(os.walk(base_dir))[1])}"

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir