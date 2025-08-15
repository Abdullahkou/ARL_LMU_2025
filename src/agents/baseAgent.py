from typing import Callable, Protocol

import numpy as np
from gymnasium import Env


class BaseAgent(Protocol):
    def __init__(self, config, env: Env) -> None:
        """config: TODO to be defined
        env: TODO to be defined -> gym"""
        super().__init__()

    def learn(self, total_steps: int, eval_fn: Callable | None = None) -> None:
        """total_steps: int, number of steps to learn
        eval_fn(optional): function, evaluation function to call during learning"""
        pass

    def predict(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        """obs: np.ndarray, observation to predict action for
        returns: np.ndarray, action to take"""
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        """internal loading"""
        pass
