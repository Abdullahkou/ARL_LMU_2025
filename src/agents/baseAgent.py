from typing import Callable, Protocol

import numpy as np
from gymnasium import Env

from agents.randomAgent import RandomAgent
from config.training_config import AlgoConfig, Algorithm
from utils.logging import TrainLogger

from agents.dqn_agent import DQNAgent

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

    def save(self, base_dir: str) -> None:
        """
        Save the agent's state to a file.

        :algo_name: name of the algorithm
        :param base_dir: base_dir to save the agent's state to
        """
        pass

    def load(self, base_dir: str) -> None:
        """
        Load the agent's state from a file.

        :param base_dir: base_dir to load the agent's state from
        """
        pass


class WrapperAgent:
    def __init__(self, config: AlgoConfig, train_env_factory: Callable[[], Env],
                 eval_step_intervals: list[int] | None = None, seed: int = 48, device: str = "cpu"):
        self.agent = None
        match config.algorithm:
            case Algorithm.RANDOM:
                self.agent = RandomAgent(
                    seed=seed,
                    train_env_factory=train_env_factory,
                    eval_schedule=eval_step_intervals,
                )
            case Algorithm.CUSTOM_DQN:
                self.agent = DQNAgent(
                    config=config,
                    train_env_factory=train_env_factory,
                    seed=seed,
                    device=device,
                )
            case _:
                raise NotImplementedError(f"Algorithm {config.algorithm} not supported yet.")

    def learn(
        self,
        total_steps: int,
        eval_fn: Callable | None = None,
        train_logger: TrainLogger | None = None,
    ):
        """
        Learn for a certain number of steps, optionally doing eval.

        :param total_steps: number of steps to learn
        :param (optional) eval_fn: evaluation fn to call during learning
        """

        self.agent.learn(
            total_steps=total_steps, eval_fn=eval_fn, train_logger=train_logger
        )

    def predict(self, obs: np.ndarray, deterministic=True):
        """
        Predict the action to take given an observation.

        :param obs: observation to predict action for
        :param deterministic: whether to use deterministic action selection
        :return: action to take
        """
        action, _ = self.agent.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        """
        Save the agent's state to a file.

        :param path: path to save the agent's state to
        """
        if isinstance(self.agent, RandomAgent):
            return

        self.agent.save(path)

    def load(self, path: str):
        """
        Load the agent's state from a file.

        :param path: path to load the agent's state from
        """
        if isinstance(self.agent, RandomAgent):
            return

        self.agent.load(path)
