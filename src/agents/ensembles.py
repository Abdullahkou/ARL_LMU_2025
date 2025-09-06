from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

import gymnasium
import numpy as np
import pandas as pd
from scipy.stats import mode
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from agents.baseAgent import AlgoConfig, IEnsemble
from agents.models import RandomAgent
from tuning.training_config import RANDOM_AGENT

Algorithm: TypeAlias = PPO | DDPG | DQN | SAC | TD3 | A2C | RandomAgent

ALGO_LOADERS: dict[str, Callable[[str, gymnasium.Env], Algorithm]] = {
    "PPO": lambda path, env: PPO.load(path, env=env),
    "A2C": lambda path, env: A2C.load(path, env=env),
    "DQN": lambda path, env: DQN.load(path, env=env),
    "SAC": lambda path, env: SAC.load(path, env=env),
    "TD3": lambda path, env: TD3.load(path, env=env),
    "DDPG": lambda path, env: DDPG.load(path, env=env),
    RANDOM_AGENT: lambda _, env: RandomAgent(action_space=env.action_space),
}

base_dir = "results/test/Walker2d-v5/1M/train"


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

        self.ensemble: list[Algorithm] = []
        self.eval_csv = [
            ("PPO", f"{base_dir}/PPO/seeds/seed_0/training_results.csv"),
            ("SAC", f"{base_dir}/SAC/seeds/seed_0/training_results.csv"),
            ("TD3", f"{base_dir}/TD3/seeds/seed_0/training_results.csv"),
        ]

        self.perf_weights_dict = self.calculate_performance_weights(self.eval_csv)

        self.load()

    def load(self, _: str = "") -> None:
        """
        Load the state of each algorithm in the ensemble from the specified directory.

        """

        # TODO: ensemble strategy (best seed?)
        for algo_name, algo_path in self.algos:
            # file = f"{algo_path}/{SEED_PREFIX}0.zip"
            algo: Algorithm = ALGO_LOADERS[algo_name](algo_path, env=self.env)
            algo.set_random_seed(self.seed)
            self.ensemble.append(algo)

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        aggregation: str = "weighted",
    ) -> np.ndarray:
        """
        Predict the action to take given an observation using majority voting for discrete
        actions or averaging for continuous actions.

        :param obs: observation to predict action for
        :param deterministic: whether to use deterministic action selection
         action to take
        :param aggregation: method to aggregate actions ('mean' or 'weighted' or "performance")
        """

        predictions = [algo.predict(obs, deterministic)[0] for algo in self.ensemble]

        if self.is_discrete:
            predictions = np.array(predictions)
            action, _ = mode(predictions, axis=0)
            return action.squeeze()

        if len(predictions) < 2:
            return predictions[0]

        if aggregation == "mean":
            predictions = np.array(predictions)
            return np.mean(predictions, axis=0)

        elif aggregation == "weighted":
            mean_action = np.mean(predictions, axis=0)
            distances = np.linalg.norm(predictions - mean_action, axis=1)
            epsilon = 1e-8
            weights = 1.0 / (distances + epsilon)
            normalized_weights = weights / np.sum(weights)
            weighted_action = np.average(
                predictions, axis=0, weights=normalized_weights
            )
            return weighted_action

        elif aggregation == "performance":
            perf_weights = np.array(
                [self.perf_weights_dict[name] for name, _ in self.eval_csv]
            )

            weighted_action = np.average(predictions, axis=0, weights=perf_weights)
            return weighted_action

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def learn(
        self,
        total_steps: int,
        eval_fn: Callable[[int], None] | None = None,
        eval_schedule: int | None = None,
    ) -> None:
        pass  # Not implemented

    def save(self, base_dir: str) -> None:
        pass  # Not implemented

    def calculate_performance_weights(
        self, csv_file_paths: List[Tuple[str, str]], temperature: float = 100
    ) -> Optional[Dict[str, float]]:
        model_names = []
        performance_scores = []

        for name, csv_file_path in csv_file_paths:
            df = pd.read_csv(csv_file_path)
            if "Reward" not in df.columns:
                print(f"Fehler: Spalte 'Reward' in '{csv_file_path}' nicht gefunden.")
                return None

            last_reward = df["Reward"].iloc[-1]
            model_names.append(name)
            performance_scores.append(float(last_reward))

        scores = np.array(performance_scores)
        stable_scores = (scores - np.max(scores)) / temperature
        exp_scores = np.exp(stable_scores)
        weights = exp_scores / np.sum(exp_scores)

        name_to_weight_map = dict(zip(model_names, weights))
        return name_to_weight_map
