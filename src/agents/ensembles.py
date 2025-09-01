from typing import Any, Callable

import gymnasium
import numpy as np
from scipy.stats import mode
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from agents.baseAgent import AlgoConfig, IEnsemble
from agents.models import RandomAgent
from tuning.training_config import RANDOM_AGENT, SEED_PREFIX

ALGO_LOADERS = {
    "PPO": lambda path, env: PPO.load(path, env=env),
    "A2C": lambda path, env: A2C.load(path, env=env),
    "DQN": lambda path, env: DQN.load(path, env=env),
    "SAC": lambda path, env: SAC.load(path, env=env),
    "TD3": lambda path, env: TD3.load(path, env=env),
    "DDPG": lambda path, env: DDPG.load(path, env=env),
    RANDOM_AGENT: lambda _, env: RandomAgent(action_space=env.action_space),
}


class EvalEnsemble(IEnsemble):
    def __init__(
        self,
        config: dict[str, Any | AlgoConfig | int | str],
    ) -> None:
        """Initialize the ensemble agent.

        :param config: configuration dictionary containing:
        {
            algos: [("algoname", "path/to/model"),...],
            env: gymnasium.Env,
            device: "cpu",
            seed: 42,
            is_discrete: False
        }
        """
        self.algos: list[tuple[str, str]] = config.get("algos", [])
        # TODO: fix this
        self.device = config.get("device", "cpu")
        self.seed = config.get("seed", 69)
        self.is_discrete = config.get("is_discrete", False)
        self.env = config.get("env", gymnasium.make("CartPole-v1"))

        self.ensemble: list[PPO | DDPG | DQN | SAC | TD3 | A2C | RandomAgent] = []
        self.load()

    def load(self, _: str = "") -> None:
        """
        Load the state of each algorithm in the ensemble from the specified directory.

        """

        # TODO: ensemble strategy (best seed?)
        for algo_name, algo_path in self.algos:
            file = f"{algo_path}/{SEED_PREFIX}0.zip"
            algo: PPO | DDPG | DQN | SAC | TD3 | A2C | RandomAgent = ALGO_LOADERS[
                algo_name
            ](file, env=self.env)
            algo.seed = self.seed
            self.ensemble.append(algo)

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Predict the action to take given an observation using majority voting for discrete
        actions or averaging for continuous actions.

        :param obs: observation to predict action for
        :param deterministic: whether to use deterministic action selection
         action to take
        """

        predictions = [algo.predict(obs, deterministic)[0] for algo in self.ensemble]

        if self.is_discrete:
            predictions = np.array(predictions)
            action, _ = mode(predictions, axis=0)
            return action.squeeze()
        else:
            predictions = np.array(predictions)
            return np.mean(predictions, axis=0)

    def learn(
        self,
        total_steps: int,
        eval_fn: Callable[[int], None] | None = None,
        eval_schedule: int | None = None,
    ) -> None:
        pass  # Not implemented

    def save(self, base_dir: str) -> None:
        pass  # Not implemented
