from typing import Any, Callable, Optional, List, Tuple, Dict, TypeAlias

import gymnasium
import numpy as np
from scipy.stats import mode
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
import torch as th
import pandas as pd
from agents.baseAgent import AlgoConfig, IEnsemble
from agents.models import RandomAgent
from tuning.training_config import RANDOM_AGENT, SEED_PREFIX

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
        self.q_critics = []
        self.eval_csv = [
            ("PPO", f"{base_dir}/PPO/seeds/seed_0/training_results.csv"),
            ("SAC", f"{base_dir}/SAC/seeds/seed_0/training_results.csv"),
            ("TD3", f"{base_dir}/TD3/seeds/seed_0/training_results.csv"),
        ]

        self.perf_weights_dict = self.calculate_performance_weights(self.eval_csv)
        self.active_idx = None
        self.prev_action = None
        self.switch_margin = 0.10  # required advantage to switch (in Q units)
        self.inertia_lambda = 0.05  # penalize big action changes
        self.load()
        self.q_norm = {"SAC": {"mu": 0.0, "sigma": 1.0}, "TD3": {"mu": 0.0, "sigma": 1.0}}  # running scale
        self.norm_momentum = 0.01  # EW moving average for normalization

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

            if algo_name == "SAC":
                self.q_critics.append(self._sac_q)
            elif algo_name == "TD3":
                self.q_critics.append(self._td3_q)
            elif algo_name == "PPO":
                self.q_critics.append(self.ppo_logprob)

    def _make_q_scorer(self, model) -> Optional[Callable[[np.ndarray, np.ndarray], float]]:
        """
        Return a callable that scores (obs, act) with the model's Q-critics.
        For SAC/TD3/DDPG this tries, in order:
          - policy.qf1/qf2
          - policy.critic (returns (q1,q2) or q)
          - policy.q_net (single Q)
        Aggregation: return the MIN across all available Q heads (pessimistic).
        For PPO/A2C/DQN (no action-conditional Q), returns None.
        """
        policy = model.policy
        policy.eval()  # ensure eval mode
        has_q = False

        # Prepare the forwarders we find
        q_forwarders = []

        # Case 1: separate qf1/qf2 modules (common in SAC, sometimes TD3/DDPG)
        for attr in ("qf1", "qf2"):
            if hasattr(policy, attr):
                m = getattr(policy, attr)
                if callable(m):
                    has_q = True
                    q_forwarders.append(lambda obs, act, m=m: m(obs, act))

        # Case 2: a 'critic' module whose forward returns q or (q1, q2)
        if hasattr(policy, "critic"):
            m = policy.critic
            if callable(m):
                has_q = True

                def critic_forward(obs, act, m=m):
                    out = m(obs, act)
                    if isinstance(out, tuple) or isinstance(out, list):
                        return out  # (q1, q2, ...)
                    return (out,)  # single q

                q_forwarders.append(lambda obs, act, f=critic_forward: f(obs, act))

        # Case 3: a single q_net that takes concatenated [obs, act]
        if hasattr(policy, "q_net"):
            m = getattr(policy, "q_net")
            if callable(m):
                has_q = True

                def qnet_forward(obs, act, m=m):
                    x = th.cat([obs, act], dim=1)
                    return (m(x),)

                q_forwarders.append(qnet_forward)

        if not has_q:
            return None  # e.g., PPO/A2C/DQN

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        aggregation: str = "critic",
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

        elif aggregation == "critic":
            # collect models in the same order as predictions
            models = [m for m in self.ensemble]  # SAC, TD3, PPO (maybe)
            scores = []

            for model, a in zip(models, predictions):
                if isinstance(model, SAC):
                    raw = self._sac_q(model, obs, a);
                    self._update_norm("SAC", raw)
                    scores.append(self._z("SAC", raw))
                elif isinstance(model, TD3):
                    raw = self._td3_q(model, obs, a);
                    self._update_norm("TD3", raw)
                    scores.append(self._z("TD3", raw))
                else:
                    # PPO has no Q(s,a); either skip or give a tiny prior
                    scores.append(-np.inf)  # skip for now

            scores = np.array(scores, dtype=float)

            # Stabilizers (optional but recommended)
            if getattr(self, "prev_action", None) is not None:
                inertia_lambda = getattr(self, "inertia_lambda", 0.05)
                deltas = np.linalg.norm(predictions - self.prev_action, axis=1)
                scores -= inertia_lambda * deltas

            best_idx = int(np.argmax(scores))

            # hysteresis: only switch if clearly better
            margin = getattr(self, "switch_margin", 0.10)
            if getattr(self, "active_idx", None) is None:
                self.active_idx = best_idx
            else:
                if scores[best_idx] >= scores[self.active_idx] + margin:
                    self.active_idx = best_idx

            a = predictions[self.active_idx]
            self.prev_action = a
            return a

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

    def _sac_q(self, sac_model, obs, act) -> float:
        with th.inference_mode():
            obs_t, _ = sac_model.policy.obs_to_tensor(obs)
            act_t = th.as_tensor(act, device=obs_t.device).float()
            if act_t.ndim == 1: act_t = act_t.unsqueeze(0)
            q1, q2 = sac_model.policy.critic(obs_t, act_t)
            qmin = th.minimum(q1.view(q1.size(0), -1), q2.view(q2.size(0), -1))
            return float(qmin[0, 0].item())

    def _td3_q(self, td3_model, obs, act) -> float:
        with th.inference_mode():
            obs_t, _ = td3_model.policy.obs_to_tensor(obs)
            act_t = th.as_tensor(act, device=obs_t.device).float()
            if act_t.ndim == 1: act_t = act_t.unsqueeze(0)
            try:
                q1, q2 = td3_model.policy.critic(obs_t, act_t)
            except TypeError:
                q1 = td3_model.policy.critic.q1_forward(obs_t, act_t)
                q2 = td3_model.policy.critic.q2_forward(obs_t, act_t)
            qmin = th.minimum(q1.view(q1.size(0), -1), q2.view(q2.size(0), -1))
            return float(qmin[0, 0].item())

    def ppo_logprob(self, ppo, obs_np: np.ndarray, act_np: np.ndarray) -> float:
        with th.inference_mode():
            obs = th.as_tensor(obs_np, device=self.device, dtype=th.float32)
            act = th.as_tensor(act_np, device=self.device, dtype=th.float32)
            if obs.ndim == 1:  obs = obs.unsqueeze(0)
            if act.ndim == 1:  act = act.unsqueeze(0)
            # evaluate_actions returns (values, log_prob, entropy)
            values, log_prob, entropy = ppo.policy.evaluate_actions(obs, act)
            return float(log_prob.view(-1)[0].item())

    def _update_norm(self, key: str, x: float) -> None:
        mu = self.q_norm[key]["mu"];
        sg = self.q_norm[key]["sigma"]
        mu_new = (1 - self.norm_momentum) * mu + self.norm_momentum * x
        # simple EW std update around new mean
        sg_new = (1 - self.norm_momentum) * sg + self.norm_momentum * abs(x - mu_new)
        self.q_norm[key]["mu"], self.q_norm[key]["sigma"] = mu_new, max(sg_new, 1e-6)

    def _z(self, key: str, x: float) -> float:
        mu, sg = self.q_norm[key]["mu"], self.q_norm[key]["sigma"]
        return (x - mu) / sg
