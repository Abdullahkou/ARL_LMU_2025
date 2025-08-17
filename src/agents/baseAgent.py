from enum import Enum
from typing import Callable, Protocol

import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC


class Algo(Enum):
    PPO = PPO
    SAC = SAC
    A2C = A2C
    DQN = DQN
    # TODO: The other algorithms from stable baselines or elsewhere


class BaseAgent(Protocol):
    """Basic agent interface"""

    def __init__(
        self,
        algos: list[Algo],
        get_env: Callable,
        hyper_params: dict = {},
        seed: int = 69,
        device: str = "cpu",
    ) -> None:
        """
        Base ctor.
        """
        pass

    def learn(self, total_steps: int, eval_fn: Callable | None = None) -> None:
        """
        Learn for a certain number of steps optionally doing eval.

        :param total_steps: number of steps to learn
        :param (optional) eval_fn: evaluation fn to call during learning
        """
        pass

    def predict(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        """
        Predict the action to take given an observation.

        :param obs: observation to predict action for
        :param deterministic: whether to use deterministic action selection
        :return: action to take
        """
        pass

    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.

        :param path: path to save the agent's state to
        """
        pass

    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.

        :param path: path to load the agent's state from
        """
        pass
