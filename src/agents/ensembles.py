from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

import gymnasium
import numpy as np
import pandas as pd
from scipy.stats import mode
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from agents.baseAgent import AlgoConfig, IEnsemble
from agents.models import RandomAgent
from tuning.training_config import RANDOM_AGENT
from utils.logging import mix_rbf_mmsd

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

        self.is_homogeneous = len(set(self.algos)) != len(self.algos)

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

    def collect_individual_rollouts(self, envs: list[gymnasium.Env], n_steps=1_000):
        states_per_model = list[np.ndarray]()
        for i, m in enumerate(self.ensemble):
            env = envs[i]
            s = collect_rollouts(m, env, n_steps=n_steps)

            states_per_model.append(s)

        return states_per_model

    def action_disagreement(
        self, states_per_model: list[np.ndarray], max_actions=5_000
    ):
        """
        Computes total action disagreement across models at given states.
        Limits MMDS calc to ``max_actions`` samples for efficiency.

        :param states: [(n_states, obs_dim),...] list of states per model
        :return:
            :dict with:
                {
                'total': mean across all results,
                'pairwise': { "model_a-model_b": distance, ... }
                }
        """
        stacked_states = np.vstack(states_per_model)

        actions_per_model = list[np.ndarray]()
        for m in self.ensemble:
            actions, _ = m.predict(
                stacked_states,
                deterministic=True,
            )

            actions_per_model.append(actions)

        subsampled_actions = actions_per_model
        if len(actions_per_model[0]) > max_actions:
            indices = np.random.choice(
                len(actions_per_model[0]), size=max_actions, replace=False
            )
            subsampled_actions = [a[indices] for a in actions_per_model]

        action_dis = self.__mmds_results(subsampled_actions)

        return action_dis

    def state_visit_divergence(
        self, states_per_model: list[np.ndarray], max_states=5_000
    ):
        """
        Computes total state divergence across models and given states.
        Limits MMDS calc to ``max_states`` samples for efficiency.

        :param states: [(n_states, obs_dim),...] list of states per model
        :return:
            :dict with:
                {
                'total': mean across all results,
                'pairwise': { "model_a-model_b": distance, ... }
                }
        """
        subsampled_states = states_per_model
        if len(states_per_model[0]) > max_states:
            indices = np.random.choice(
                len(states_per_model[0]), size=max_states, replace=False
            )
            subsampled_states = [s[indices] for s in states_per_model]

        state_dis = self.__mmds_results(subsampled_states)

        return state_dis

    def __mmds_results(self, arrays_per_model: list[np.ndarray]):
        """
        Computes pairwise MMD^2 between array results from models in the ensemble.

        :param arrays_per_model: list of arrays per model
        :return:
            :dict with:
                {
                'total': mean across all results,
                'pairwise': { "model_a-model_b": distance, ... }
                }
        """
        pairwise = {}
        for i, j in combinations(range(len(arrays_per_model)), 2):
            mmd = mix_rbf_mmsd(arrays_per_model[i], arrays_per_model[j])

            model_a = self.ensemble[i].__class__.__name__
            model_b = self.ensemble[j].__class__.__name__
            pairwise[f"{model_a}-{model_b}"] = mmd

        total_state_dis = np.mean(list(pairwise.values()))

        results: dict[str, np.floating | dict[str, np.floating]] = {
            "total": total_state_dis,
            "pairwise": pairwise,
        }

        return results


def collect_rollouts(model: Algorithm, env: gymnasium.Env, n_steps=1000):
    """Rolls out episodes to collect visited states."""
    states = []
    steps = 0
    while steps < n_steps:
        state: np.ndarray
        state, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            steps += 1
            states.append(state)

            if steps >= n_steps:
                break

    return np.array(states)
