import os
from dataclasses import dataclass


@dataclass
class Params:
    gamma: float = 0.99
    learning_rate: float = 3e-4


def make_dir(base_dir: str):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dir = f"{base_dir}/run_{len(next(os.walk(base_dir))[1])}"

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir