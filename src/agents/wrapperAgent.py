from typing import Callable

import numpy as np
from gymnasium import Env

from agents.baseAgent import IAgent
from agents.dqn_agent import DQNAgent
from agents.randomAgent import RandomAgent
from agents.sb3Wrapper import SB3Wrapper
from config.training_config import AlgoConfig, Algorithm
from utils.logging import TrainLogger


class WrapperAgent:
    def __init__(
        self,
        config: AlgoConfig,
        train_env_factory: Callable[[], Env],
        eval_step_intervals: list[int] = None,
        seed: int = 69,
        device: str = "cpu",
    ):
        self.agent: IAgent = None
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
                self.agent = SB3Wrapper(
                    config=config,
                    train_env_factory=train_env_factory,
                    seed=seed,
                    device=device,
                )

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
        action = self.agent.predict(obs, deterministic=deterministic)
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
