from enum import Enum
from typing import Callable, Protocol

import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from agents.models import RandomAgent, SB3Callback
from tuning.training_config import RANDOM_AGENT


class Algo(Enum):
    PPO = PPO
    SAC = SAC
    A2C = A2C
    DQN = DQN
    DDPG = DDPG
    TD3 = TD3
    RANDOM = RandomAgent


ALGO_IMPLS = {
    "PPO": lambda env, seed, hyper_params={}: PPO(
        "MlpPolicy", env, seed=seed, **hyper_params
    ),
    "A2C": lambda env, seed, hyper_params={}: A2C(
        "MlpPolicy", env, seed=seed, **hyper_params
    ),
    "DQN": lambda env, seed, hyper_params={}: DQN(
        "MlpPolicy", env, seed=seed, **hyper_params
    ),
    "SAC": lambda env, seed, hyper_params={}: SAC(
        "MlpPolicy", env, seed=seed, **hyper_params
    ),
    "DDPG": lambda env, seed, hyper_params={}: DDPG(
        "MlpPolicy", env, seed=seed, **hyper_params
    ),
    "TD3": lambda env, seed, hyper_params={}: TD3(
        "MlpPolicy", env, seed=seed, **hyper_params
    ),
    RANDOM_AGENT: lambda env, seed, hyper_params={}: RandomAgent(
        action_space=env.action_space,
        seed=seed,
    ),
}


class IAgent(Protocol):
    """Basic agent interface"""

    def __init__(
        self,
        algos: list[Algo],
        train_env_id: str,
        hyper_params: dict = {},
        seed: int = 69,
        device: str = "cpu",
    ) -> None:
        pass

    def learn(
        self,
        total_steps: int,
        eval_fn: Callable[[int], None] | None = None,
        eval_schedule: int | None = None,
    ) -> None:
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


class BaseAgent(Protocol):
    """Basic agent interface"""

    def __init__(
        self,
        algos: list[Algo],
        train_env: Callable,
        hyper_params: dict = {},
        seed: int = 69,
        device: str = "cpu",
    ) -> None:
        """
        Base ctor.
        """
        # Train Umgebung seeden
        train_env.reset(seed=seed)

        self.agents = None

        builder = ALGO_IMPLS[algos[0]]
        self.agents = builder(train_env, seed, hyper_params)

    def learn(
        self,
        total_steps: int,
        eval_fn: Callable | None = None,
        eval_schedule_list: list[int] = None,
    ) -> None:
        """
        Learn for a certain number of steps optionally doing eval.

        :param total_steps: number of steps to learn
        :param (optional) eval_fn: evaluation fn to call during learning
        """
        if eval_fn is not None:
            if isinstance(self.agents, RandomAgent):
                self.agents.eval_model = eval_fn
                self.agents.learn(eval_schedule_list=eval_schedule_list)
            else:
                callback = SB3Callback(eval_model=eval_fn, total_timesteps=total_steps)
                return self.agents.learn(
                    total_steps, callback=callback, log_interval=None
                )

    def predict(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        """
        Predict the action to take given an observation.

        :param obs: observation to predict action for
        :param deterministic: whether to use deterministic action selection
        :return: action to take
        """
        if self.agents is not None:
            action, _ = self.agents.predict(obs, deterministic=True)
        return action

    def save(self, algo_name: str, path: str) -> None:
        """
        Save the agent's state to a file.

        :algo_name: name of the algorithm
        :param path: path to save the agent's state to
        """
        if hasattr(self.agents, "save"):  # should cover Random
            self.agents.save(path)

    def load(self, algo_name: str, path: str, env, action_space=None) -> None:
        """
        Load the agent's state from a file.

        :param path: path to load the agent's state from
        """
        if algo_name in ["PPO", "A2C", "SAC", "TD3", "DDPG", "DQN"]:
            # SB3-Loader
            AlgoClass = getattr(__import__("stable_baselines3"), algo_name)
            return AlgoClass.load(path + "_" + algo_name, env=env)
        elif algo_name == RANDOM_AGENT:
            return RandomAgent.load(path + "_" + algo_name, action_space)
        else:
            raise ValueError(f"Unbekannter Algorithmus: {algo_name}")
