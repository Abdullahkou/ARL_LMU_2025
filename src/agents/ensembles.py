from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

import gymnasium
import numpy as np
import pandas as pd
from scipy.stats import mode
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from agents.baseAgent import AlgoConfig, IEnsemble
from agents.models import RandomAgent
from tuning.training_config import RANDOM_AGENT
from utils.logging import mmd_sq

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

    def collect_individual_rollouts(
        self, env: gymnasium.Env, n_steps=1000, max_states: int = None
    ):
        states_per_model = list[np.ndarray]()
        for m in self.ensemble:
            s = collect_rollouts(m, env, n_steps=n_steps)

            # Subsample states if needed
            if max_states and len(s) > max_states:
                idx = np.random.choice(len(s), max_states, replace=False)
                s = s[idx]

            states_per_model.append(s)

        return states_per_model

    def action_disagreement(self, states: np.ndarray):
        """
        Computes mean std of actions across models at given states.
        models: list of RL policies
        states: (n_states, obs_dim)
        """
        all_actions = []
        for m in self.ensemble:
            acts = []
            for s in states:
                a, _ = m.predict(s)

                acts.append(a)
            all_actions.append(np.stack(acts))
        all_actions = np.stack(all_actions)  # shape: (n_models, n_states, act_dim)
        return float(np.std(all_actions, axis=0).mean())

    def state_visit_divergence(self, states_per_model: list[np.ndarray]):
        """
        Computes pairwise MMD^2 between states of models in the ensemble.

        :param states_per_model: list of (n_states, obs_dim) arrays per model
        :return: (mmd_mean, mmd_min, mmd_max)
        """
        num_models = len(states_per_model)
        mmd_vals = []
        for i in range(num_models):
            for j in range(i + 1, num_models):
                mmd_vals.append(mmd_sq(states_per_model[i], states_per_model[j]))

        mmd_mean = np.mean(mmd_vals)
        mmd_min: np.floating = np.min(mmd_vals)
        mmd_max: np.floating = np.max(mmd_vals)

        return mmd_mean, mmd_min, mmd_max


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
