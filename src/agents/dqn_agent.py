# agents/dqn_agent.py
from __future__ import annotations

import random
from typing import Callable, Dict, Optional
from warnings import warn_explicit

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env
from tqdm import tqdm

from agents.baseAgent import IAgent
from agents.components.q_network import IndependentQNetwork, QNetwork
from agents.components.replay_buffer import ReplayBuffer
from config.training_config import AlgoConfig
from utils.logging import TrainLogger


class DQNAgent(IAgent):
    def __init__(
        self,
        config: AlgoConfig,
        train_env_factory: Callable[[], Env],
        seed: int = 69,
        device: str = "cpu",
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device)

        self.env_factory = train_env_factory
        self.env: Env = self.env_factory()
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        if not hasattr(act_space, "n"):
            self.env.close()
            raise ValueError(
                "DQNAgent requires a discrete action space (gymnasium.spaces.Discrete)."
            )

        self.obs_shape = obs_space.shape
        self.n_actions = act_space.n

        self.hp = config.hyper_params

        if getattr(self.hp, "independent_heads", False):
            NetClass = IndependentQNetwork
        else:
            NetClass = QNetwork

        self.q_net = NetClass(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            hidden_sizes=self.hp.hidden_sizes,
            dueling=self.hp.dueling,
            n_heads=self.hp.n_q_heads,
        ).to(self.device)

        self.target_q_net = NetClass(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            hidden_sizes=self.hp.hidden_sizes,
            dueling=self.hp.dueling,
            n_heads=self.hp.n_q_heads,
        ).to(self.device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.hp.learning_rate)

        # ---- Bootstrapped replay masks (optional; paper notes deep impls often share data) ----
        bootstrap_prob = 0.5
        boostrap_prob_field = self.hp.bootstrap_prob
        if boostrap_prob_field == "auto":
            bootstrap_prob = 1 / max(1, self.hp.n_q_heads)
        else:
            assert isinstance(boostrap_prob_field, float)
            if self.hp.n_q_heads == 1 and boostrap_prob_field != 1:
                warn_explicit(
                    f"Using 1 head with bootstrap_prob {boostrap_prob_field} != 1"
                )
            bootstrap_prob = boostrap_prob_field

        self.replay = ReplayBuffer(
            capacity=self.hp.buffer_size,
            obs_shape=self.obs_shape,
            seed=self.seed,
            device=self.device,
            n_heads=self.hp.n_q_heads,
            bootstrap_prob=bootstrap_prob,
        )

        self.global_step = 0
        self.epsilon = self.hp.max_epsilon
        self.best_eval_metric = -float("inf")

        self.eval_heads_fn = None
        self.active_head = 0  # used for bootstrapped acting when not using EBQL

        # Sanity: EBQL requires K>=2
        if getattr(self.hp, "use_ebql", False):
            assert self.hp.n_q_heads >= 2, "EBQL needs at least 2 ensemble members."

    def set_indiviudal_head_eval_fn(
        self, eval_heads_fn: Callable[[int], None] | None = None
    ):
        self.eval_heads_fn = eval_heads_fn

    @torch.no_grad()
    def get_heads_predictions(self, obs: np.ndarray):
        num_heads = self.hp.n_q_heads
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q_values: torch.Tensor = self.q_net(obs_t)

        if q_values.ndim == 3:
            q_values_per_head = [q_values[i, i] for i in range(num_heads)]
        else:
            q_values_per_head = [q_values[i] for i in range(num_heads)]
        actions = [
            int(torch.argmax(q_head, dim=-1).item()) for q_head in q_values_per_head
        ]
        return actions

    def learn(
        self,
        total_steps: int,
        eval_fn: Optional[Callable[[int], None]] = None,
        train_logger: Optional[TrainLogger] = None,
        eval_schedule: Optional[int] = None,
    ) -> None:
        obs, _ = self.env.reset(seed=self.seed)
        self.active_head = self.rng.integers(0, self.hp.n_q_heads)
        ep_return = 0.0
        ep_len = 0
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        if self.eval_heads_fn is not None:
            self.eval_heads_fn(self.global_step)
        if eval_fn is not None:
            eval_fn(self.global_step)

        try:
            while self.global_step < total_steps:
                pbar.update(1)

                action = self._select_action(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.replay.add(obs, action, reward, next_obs, float(done))
                obs = next_obs

                ep_return += float(reward)
                ep_len += 1
                self.global_step += 1

                logs = None
                if self._should_learn():
                    for _ in range(self.hp.gradient_steps):
                        logs = self._sgd_step()

                if eval_fn is not None:
                    eval_fn(self.global_step)
                if self.eval_heads_fn is not None:
                    self.eval_heads_fn(self.global_step)

                if train_logger is not None:
                    error = (
                        logs.get("train/head_losses", np.nan)
                        if logs is not None
                        else np.nan
                    )
                    train_logger.log_step(reward=reward, value_error=error)

                if self.global_step % self.hp.target_update_interval == 0:
                    self._update_target()

                self._update_epsilon()

                if done:
                    obs, _ = self.env.reset()
                    self.active_head = self.rng.integers(0, self.hp.n_q_heads)
                    ep_return = 0.0
                    ep_len = 0

            pbar.close()
        finally:
            self.env.close()

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = False):
        """
        Returns (action, extras)
        EBQL: ε-greedy on ensemble-mean Q(s,·) (paper's acting rule).
        Otherwise: bootstrapped DQN uses current active head.
        """
        if obs.ndim == len(self.obs_shape):
            obs_batch = np.expand_dims(obs, axis=0)
        else:
            obs_batch = obs

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        q = self.q_net(obs_t)  # [B, H, A] or [B, A]

        if q.ndim == 3 and getattr(self.hp, "use_ebql", False):
            q_used = q.mean(dim=1)  # ensemble mean for acting (Algorithm 1)
        elif q.ndim == 3:
            q_used = q[:, self.active_head]  # bootstrapped acting
        else:
            q_used = q

        if deterministic:
            act = torch.argmax(q_used, dim=-1).cpu().numpy()
        else:
            if self.rng.random() < self.epsilon:
                act = self.rng.integers(0, self.n_actions, size=(obs_batch.shape[0],))
            else:
                act = torch.argmax(q_used, dim=-1).cpu().numpy()

        action = act if obs.ndim > len(self.obs_shape) else act[0]
        extras = {"q": q_used.detach().cpu().numpy(), "epsilon": float(self.epsilon)}
        return action, extras

    def save(self, path: str) -> None:
        payload = {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hp": self.hp.__dict__,
            "global_step": self.global_step,
            "epsilon": self.epsilon,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "seed": self.seed,
        }
        torch.save(payload, f"{path}.pt")

    def load(self, path: str) -> None:
        path = f"{path}.pt"
        payload: dict = torch.load(path, weights_only=False)

        if (
            tuple(payload["obs_shape"]) != tuple(self.obs_shape)
            or payload["n_actions"] != self.n_actions
        ):
            raise ValueError(
                "Loaded checkpoint shape/action mismatch with current environment."
            )

        self.q_net.load_state_dict(payload["q_net"])
        self.target_q_net.load_state_dict(payload["target_q_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
        self.epsilon = float(payload.get("epsilon", self.hp.max_epsilon))

    def _should_learn(self) -> bool:
        return (self.global_step >= self.hp.learning_starts) and (
            self.global_step % self.hp.train_freq == 0
        )

    def _update_epsilon(self):
        if self.global_step <= self.hp.learning_starts:
            self.epsilon = self.hp.max_epsilon
            return
        steps_into_decay = max(0, self.global_step - self.hp.learning_starts)
        frac = 1.0 - min(1.0, steps_into_decay / float(self.hp.epsilon_decay_steps))
        self.epsilon = self.hp.min_epsilon + (
            self.hp.max_epsilon - self.hp.min_epsilon
        ) * max(0.0, frac)

    def _update_target(self):
        if self.hp.tau >= 1.0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:
            with torch.no_grad():
                for p, tp in zip(
                    self.q_net.parameters(), self.target_q_net.parameters()
                ):
                    tp.mul_(1.0 - self.hp.tau)
                    tp.add_(self.hp.tau * p)

    def _sample_batch(self):
        batch = self.replay.sample(self.hp.batch_size)
        return {
            "obs": torch.as_tensor(
                batch["obs"], dtype=torch.float32, device=self.device
            ),
            "actions": torch.as_tensor(
                batch["actions"], dtype=torch.long, device=self.device
            ),
            "rewards": torch.as_tensor(
                batch["rewards"], dtype=torch.float32, device=self.device
            ),
            "next_obs": torch.as_tensor(
                batch["next_obs"], dtype=torch.float32, device=self.device
            ),
            "dones": torch.as_tensor(
                batch["dones"], dtype=torch.float32, device=self.device
            ),
            "masks": torch.as_tensor(
                batch["masks"], dtype=torch.float32, device=self.device
            ),
        }

    def _select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        action, _ = self.predict(obs, deterministic=deterministic)
        return int(action)

    def _sgd_step(self) -> Dict[str, float]:
        """
        EBQL (paper-correct) when hp.use_ebql=True:
          - sample k ~ U([K])
          - a* = argmax_a Q_k(s', a)  (selection)
          - target = r + γ * mean_{j≠k} Q_j_target(s', a*)  (evaluation; committee excludes k)
        Else:
          - fall back to standard (Double)DQN per head with bootstrap masks.
        """
        batch = self._sample_batch()
        obs, actions, rewards, next_obs, dones, masks = (
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs"],
            batch["dones"],
            batch["masks"],
        )

        if getattr(self.hp, "use_ebql", False):
            # --- EBQL update: update exactly one ensemble member ---
            K = self.hp.n_q_heads
            k = int(self.rng.integers(0, K))
            mask = masks[:, k]  # [B]
            if mask.sum() == 0:
                # No samples for this head in the batch – skip update gracefully
                return {"train/loss": 0.0, "train/head_losses": 0.0, "train/epsilon": float(self.epsilon)}

            q_k = self.q_net(obs)[:, k]                                 # [B, A]
            q_sa = q_k.gather(1, actions.view(-1, 1)).squeeze(1)        # [B]

            with torch.no_grad():
                # Selection: argmax using online net of head k at next state
                q_next_online_k = self.q_net(next_obs)[:, k]            # [B, A]
                next_actions = torch.argmax(q_next_online_k, dim=1)     # [B]

                # Evaluation: average over other heads (target net) at the chosen action
                q_next_targets = self.target_q_net(next_obs)            # [B, K, A]
                committee_vals = []
                for j in range(K):
                    if j == k:
                        continue
                    vj = q_next_targets[:, j].gather(1, next_actions.view(-1, 1)).squeeze(1)  # [B]
                    committee_vals.append(vj)
                q_next = torch.stack(committee_vals, dim=1).mean(dim=1)  # [B]

                target = rewards + (1.0 - dones) * self.hp.gamma * q_next  # [B]

            # Masked MSE only over samples visible to head k
            loss = ((q_sa - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.hp.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.q_net.parameters(), self.hp.clip_grad_norm)
            self.optimizer.step()

            return {
                "train/loss": float(loss.item()),
                "train/head_losses": float(loss.item()),
                "train/epsilon": float(self.epsilon),
            }

        # --------- Fallback: (Double)DQN with bootstrapped heads ----------
        loss_total: Optional[torch.Tensor] = None
        head_losses = []
        for h in range(self.hp.n_q_heads):
            mask = masks[:, h]
            if mask.sum() == 0:
                continue

            q_h = self.q_net(obs)[:, h]  # [B, A]
            q_sa = q_h.gather(1, actions.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                if getattr(self.hp, "double_q", True):
                    q_next_online = self.q_net(next_obs)[:, h]
                    next_actions = torch.argmax(q_next_online, dim=1)
                    q_next_target = self.target_q_net(next_obs)[:, h]
                    q_next = q_next_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
                else:
                    q_next_target = self.target_q_net(next_obs)[:, h]
                    q_next = q_next_target.max(dim=1).values

                target = rewards + (1.0 - dones) * self.hp.gamma * q_next

            loss_h = ((q_sa - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            loss_total = loss_h if loss_total is None else (loss_total + loss_h)
            head_losses.append(loss_h.item())

        if loss_total is None:
            return {"train/loss": 0.0, "train/head_losses": 0.0, "train/epsilon": float(self.epsilon)}

        self.optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        if self.hp.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.hp.clip_grad_norm)
        self.optimizer.step()

        return {
            "train/loss": float(loss_total.item()),
            "train/head_losses": float(np.mean(head_losses)) if head_losses else 0.0,
            "train/epsilon": float(self.epsilon),
        }
