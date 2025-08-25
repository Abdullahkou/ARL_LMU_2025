import os
from dataclasses import dataclass


@dataclass
class Params:
    # TODO: handle optional parameters, if defaults for sb3, save those!
    pass


def make_dir(base_dir: str):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dir = f"{base_dir}/run_{len(next(os.walk(base_dir))[1])}"

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir
