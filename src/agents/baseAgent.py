from typing import Callable, Protocol

import numpy as np
from gymnasium import Env

from config.training_config import AlgoConfig
from utils.logging import TrainLogger


class IAgent(Protocol):
    """Basic agent interface"""

    def __init__(
        self,
        config: AlgoConfig,
        train_env_factory: Callable[[], Env],
        seed: int = 69,
        device: str = "cpu",
    ) -> None:
        """Remember to close your train env constructed from factory at the end of learn!"""
        pass

    def learn(
        self,
        total_steps: int,
        eval_fn: Callable[[int], None] | None = None,
        train_logger: TrainLogger | None = None,
        eval_schedule: int | None = None,
    ) -> None:
        """
        Learn for a certain number of steps optionally doing eval.

        :param total_steps: number of steps to learn
        :param (optional) train_logger: useful for recording training stats
        :param (optional) eval_fn: evaluation fn to call during learning, called at every step!
        :param (optional) eval_schedule: eval interval, useful for multi agent setting!
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
