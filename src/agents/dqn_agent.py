# agents/dqn_agent.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env
from tqdm import tqdm

from agents.baseAgent import IAgent
from config.training_config import AlgoConfig
from networks.q_network import IndependentQNetwork, QNetwork
from utils.logging import TrainLogger
from utils.replay_buffer import ReplayBuffer


@dataclass
class DQNHyperParams:
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 30_000
    learning_starts: int = 3_000
    train_freq: int = 1  # env steps per gradient step
    gradient_steps: int = 1  # how many SGD steps per train call
    target_update_interval: int = 1_000  # in env steps
    tau: float = 1.0  # hard update by default; set <1.0 for soft update
    max_epsilon: float = 0.5
    min_epsilon: float = 0.05
    epsilon_decay_steps: int = 250_000  # linear decay steps
    clip_grad_norm: Optional[float] = 10.0
    double_q: bool = True  # enable Double DQN
    dueling: bool = False  # enable Dueling DQN
    n_q_heads: int = 3  # multiple Q heads for ensembles/uncertainty
    hidden_sizes: Tuple[int, ...] = (256, 256)  # MLP
    use_ebql: bool = True  # Ensemble Bootstrapped Q-Learning
    independent_heads: bool = False  # shared encoder vs. completely independent


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

        # DQN requires discrete actions
        if not hasattr(act_space, "n"):
            self.env.close()
            raise ValueError(
                "DQNAgent requires a discrete action space (gymnasium.spaces.Discrete)."
            )

        self.obs_shape = obs_space.shape
        self.n_actions = act_space.n

        self.hp = config.hyper_params

        print(f"DQN using hyper-parameters: {self.hp}")
        print(f"Number of Q-heads: {self.hp.n_q_heads}")
        # networks

        if getattr(self.hp, "independent_heads", False):
            print("Using IndependentQNetwork with separate encoders per head.")
            NetClass = IndependentQNetwork
        else:
            print("Using QNetwork with shared encoder.")
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

        print(
            f"Q-Network has {sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)} trainable parameters."
        )
        print(
            f"Target Q-Network has {sum(p.numel() for p in self.target_q_net.parameters() if p.requires_grad)} trainable parameters."
        )

        print(self.q_net)
        print(self.target_q_net)

        # self.q_net = QNetwork(
        #     obs_shape=self.obs_shape,
        #     n_actions=self.n_actions,
        #     hidden_sizes=self.hp.hidden_sizes,
        #     dueling=self.hp.dueling,
        #     n_heads=self.hp.n_q_heads,
        # ).to(self.device)

        # self.target_q_net = QNetwork(
        #     obs_shape=self.obs_shape,
        #     n_actions=self.n_actions,
        #     hidden_sizes=self.hp.hidden_sizes,
        #     dueling=self.hp.dueling,
        #     n_heads=self.hp.n_q_heads,
        # ).to(self.device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.hp.learning_rate)

        self.replay = ReplayBuffer(
            capacity=self.hp.buffer_size,
            obs_shape=self.obs_shape,
            seed=self.seed,
            device=self.device,
            n_heads=self.hp.n_q_heads,
            bootstrap_prob=0.5,  # 1.0 = jede Transition sichtbar für jeden Head
        )

        self.global_step = 0
        self.epsilon = self.hp.max_epsilon
        self.best_eval_metric = -float("inf")

    def learn(
        self,
        total_steps: int,
        eval_fn: Optional[Callable[[int], None]] = None,
        train_logger: Optional[TrainLogger] = None,
        eval_schedule: Optional[int] = None,
    ) -> None:
        """
        Collect experience and train until total_steps. Closes env at the end.
        eval_fn: optional callback(eval_step) -> None (user evaluates externally)
        eval_schedule: run eval_fn every 'eval_schedule' env steps if given
        """
        obs, _ = self.env.reset(seed=self.seed)
        self.active_head = self.rng.integers(0, self.hp.n_q_heads)
        ep_return = 0.0
        ep_len = 0
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        #  evaluation --OPTIONAL--
        if eval_fn is not None:
            eval_fn(self.global_step)

        # t0 = time.time()
        try:
            while self.global_step < total_steps:
                pbar.update(1)

                action = self._select_action(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # store
                self.replay.add(obs, action, reward, next_obs, float(done))
                obs = next_obs

                ep_return += float(reward)
                ep_len += 1
                self.global_step += 1

                # training
                logs = None
                if self._should_learn():
                    for _ in range(self.hp.gradient_steps):
                        logs = self._sgd_step()

                #  evaluation --OPTIONAL--
                if eval_fn is not None:
                    eval_fn(self.global_step)

                # log training
                if train_logger is not None:
                    error = (
                        logs.get("train/head_losses", np.nan)
                        if logs is not None
                        else np.nan
                    )

                    train_logger.log_step(reward=reward, value_error=error)

                # target update
                if self.global_step % self.hp.target_update_interval == 0:
                    self._update_target()

                # epsilon scheduling
                self._update_epsilon()

                # episode end
                if done:
                    # if train_logger is not None:
                    #     if hasattr(train_logger, "log_dict"):
                    #         train_logger.log_dict(
                    #             {
                    #                 "train/episode_return": ep_return,
                    #                 "train/episode_len": ep_len,
                    #             },
                    #             step=self.global_step,
                    #         )
                    #     elif hasattr(train_logger, "log"):
                    #         try:
                    #             train_logger.log(
                    #                 "train/episode_return",
                    #                 ep_return,
                    #                 step=self.global_step,
                    #             )
                    #             train_logger.log(
                    #                 "train/episode_len", ep_len, step=self.global_step
                    #             )
                    #         except Exception:
                    #             pass

                    obs, _ = self.env.reset()
                    self.active_head = self.rng.integers(0, self.hp.n_q_heads)

                    ep_return = 0.0
                    ep_len = 0

            pbar.close()

            # TODO:
            # if train_logger is not None:
            #     wall = time.time() - t0
            #     if hasattr(train_logger, "log"):
            #         try:
            #             train_logger.log(
            #                 "train/wall_time_sec", wall, step=self.global_step
            #             )
            #         except Exception:
            #             pass
            #     elif hasattr(train_logger, "log_dict"):
            #         train_logger.log_dict(
            #             {"train/wall_time_sec": wall}, step=self.global_step
            #         )
        finally:
            self.env.close()

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = False):
        """
        Returns (action, extras)
        - Bootstrapped DQN: nutzt self.active_head
        - EBQL: mittelt Q-Werte über alle Heads
        """
        if obs.ndim == len(self.obs_shape):
            obs_batch = np.expand_dims(obs, axis=0)
        else:
            obs_batch = obs

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        q = self.q_net(
            obs_t
        )  # [B, n_heads, n_actions] oder [B, n_actions] janahch ob multi-head oder nicht

        if q.ndim == 3:
            if getattr(self.hp, "use_ebql", False):
                # EBQL: Mittelung über alle Heads
                q_used = q.mean(dim=1)  # [B, n_actions]
            else:
                # Bootstrapped DQN: nur aktiver Head
                q_used = q[:, self.active_head]
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

    def save(self, base_dir: str) -> None:
        os.makedirs(base_dir, exist_ok=True)
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
        torch.save(payload, os.path.join(base_dir, "dqn.pt"))

    def load(self, base_dir: str) -> None:
        path = os.path.join(base_dir, "dqn.pt")
        payload: dict = torch.load(path, map_location=self.device)

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

    def _q_select(self, q: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        NOT USED CURRENTLY
        q: [B, n_actions] or [B, n_heads, n_actions] -> select first head if multi-head
        returns: [B]
        """
        if q.ndim == 3:
            q = q[:, 0]  # head 0 for training; extend later for bootstrapped heads
        return q.gather(1, actions.view(-1, 1)).squeeze(1)

    def _q_max(self, q: torch.Tensor) -> torch.Tensor:
        """
        NOT USED CURRENTLY
        returns max over actions per sample; shape [B]
        """
        if q.ndim == 3:
            q = q[:, 0]
        return q.max(dim=1).values

    def _select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        action, _ = self.predict(obs, deterministic=deterministic)
        return int(action)

    # _sgd_step MIT EBQL
    def _sgd_step(self) -> Dict[str, float]:
        batch = self._sample_batch()
        obs, actions, rewards, next_obs, dones, masks = (
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs"],
            batch["dones"],
            batch["masks"],
        )

        loss_total = 0.0
        head_losses = []

        for h in range(self.hp.n_q_heads):
            mask = torch.as_tensor(masks[:, h], dtype=torch.float32, device=self.device)
            if mask.sum() == 0:  # kein Sample für diesen Head
                continue

            # Q(s,a) für diesen Head
            q = self.q_net(obs)[:, h]  # [B, n_actions]
            q_sa = q.gather(1, actions.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                if self.hp.double_q:
                    # Actionwahl immer noch Head-spezifisch
                    q_next_online = self.q_net(next_obs)[:, h]
                    next_actions = torch.argmax(q_next_online, dim=1)

                    if self.hp.use_ebql:
                        # Ensemble-Target: Mittelung über alle Heads
                        q_next_targets = self.target_q_net(
                            next_obs
                        )  # [B, n_heads, n_actions]
                        q_all = []
                        for u in range(self.hp.n_q_heads):
                            q_all.append(
                                q_next_targets[:, u]
                                .gather(1, next_actions.view(-1, 1))
                                .squeeze(1)
                            )
                        q_next = torch.stack(q_all, dim=1).mean(dim=1)  # [B]
                    else:
                        q_next_target = self.target_q_net(next_obs)[:, h]
                        q_next = q_next_target.gather(
                            1, next_actions.view(-1, 1)
                        ).squeeze(1)
                else:
                    if self.hp.use_ebql:
                        # Ensemble-Target: max über Actions und Mittelung über Heads
                        q_next_targets = self.target_q_net(
                            next_obs
                        )  # [B, n_heads, n_actions]
                        q_next = q_next_targets.max(dim=2).values.mean(dim=1)
                    else:
                        q_next_target = self.target_q_net(next_obs)[:, h]
                        q_next = q_next_target.max(dim=1).values

                target = rewards + (1.0 - dones) * self.hp.gamma * q_next

            # Maskierter Loss
            loss = ((q_sa - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            loss_total += loss
            head_losses.append(loss.item())

        # Backprop über Summe der Head-Losses
        self.optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        if self.hp.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.hp.clip_grad_norm)
        self.optimizer.step()

        head_losses = np.array(head_losses)

        return {
            "train/loss": float(loss_total.item()),
            "train/head_losses": head_losses.mean(),
            "train/epsilon": float(self.epsilon),
        }

    # region --- old predict with active_head and epsilon-greedy ---
    # @torch.no_grad()
    # def predict(self, obs: np.ndarray, deterministic: bool = False):
    #     """
    #     Returns (action, extras) to match WrapperAgent usage.
    #     extras: {"q": np.ndarray of Q-values (from head 0), "epsilon": float}
    #     """
    #     if obs.ndim == len(self.obs_shape):
    #         obs_batch = np.expand_dims(obs, axis=0)
    #     else:
    #         obs_batch = obs

    #     obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
    #     q = self.q_net(obs_t)  # shape: [B, n_heads, n_actions] or [B, n_actions]
    #     if q.ndim == 3:
    #         # mehrere Heads -> nur den aktiven Head nehmen
    #         q0 = q[:, self.active_head]
    #     else:
    #         q0 = q

    #     if deterministic:
    #         act = torch.argmax(q0, dim=-1).cpu().numpy()
    #     else:
    #         # epsilon-greedy for predict (optional)
    #         if self.rng.random() < self.epsilon:
    #             act = self.rng.integers(0, self.n_actions, size=(obs_batch.shape[0],))
    #         else:
    #             act = torch.argmax(q0, dim=-1).cpu().numpy()

    #     action = act if obs.ndim > len(self.obs_shape) else act[0]
    #     extras = {"q": q0.detach().cpu().numpy(), "epsilon": float(self.epsilon)}
    #     return action, extras
    # endregion


# region --- old _sgd_step with multiple heads and masks --- OHNE EBQL
# def _sgd_step(self) -> Dict[str, float]:
#     batch = self._sample_batch()
#     obs, actions, rewards, next_obs, dones, masks = (
#         batch["obs"],
#         batch["actions"],
#         batch["rewards"],
#         batch["next_obs"],
#         batch["dones"],
#         batch["masks"],
#     )

#     loss_total = 0.0
#     head_losses = []

#     for h in range(self.hp.n_q_heads):
#         mask = torch.as_tensor(masks[:, h], dtype=torch.float32, device=self.device)
#         if mask.sum() == 0:  # kein Sample für diesen Head
#             continue

#         q = self.q_net(obs)[:, h]  # [B, n_actions]
#         q_sa = q.gather(1, actions.view(-1, 1)).squeeze(1)

#         with torch.no_grad():
#             if self.hp.double_q:
#                 q_next_online = self.q_net(next_obs)[:, h]
#                 next_actions = torch.argmax(q_next_online, dim=1)
#                 q_next_target = self.target_q_net(next_obs)[:, h]
#                 q_next = q_next_target.gather(1, next_actions.view(-1, 1)).squeeze(
#                     1
#                 )
#             else:
#                 q_next_target = self.target_q_net(next_obs)[:, h]
#                 q_next = q_next_target.max(dim=1).values

#             target = rewards + (1.0 - dones) * self.hp.gamma * q_next

#         # Nur Samples mit Maske==1 gehen in den Loss
#         loss = ((q_sa - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
#         loss_total += loss
#         head_losses.append(loss.item())

#     # Backprop über Summe der Head-Losses
#     self.optimizer.zero_grad(set_to_none=True)
#     loss_total.backward()
#     if self.hp.clip_grad_norm is not None:
#         nn.utils.clip_grad_norm_(self.q_net.parameters(), self.hp.clip_grad_norm)
#     self.optimizer.step()

#     return {
#         "train/loss": float(loss_total.item()),
#         "train/head_losses": head_losses,
#         "train/epsilon": float(self.epsilon),
#     }

# endregion


# region --- IGNORE ---
# def _sgd_step(self) -> Dict[str, float]:
#     batch = self._sample_batch()
#     obs, actions, rewards, next_obs, dones = (
#         batch["obs"],
#         batch["actions"],
#         batch["rewards"],
#         batch["next_obs"],
#         batch["dones"],
#     )

#     # current Q(s, a)
#     q = self.q_net(obs)
#     q_sa = self._q_select(q, actions)  # [B]

#     with torch.no_grad():
#         # Double DQN action selection w.r.t. online net
#         if self.hp.double_q:
#             q_next_online = self.q_net(next_obs)
#             next_actions = torch.argmax(
#                 q_next_online[:, 0] if q_next_online.ndim == 3 else q_next_online,
#                 dim=1,
#             )
#             q_next_target = self.target_q_net(next_obs)
#             q_next = self._q_select(q_next_target, next_actions)  # [B]
#         else:
#             q_next_target = self.target_q_net(next_obs)
#             q_next = self._q_max(q_next_target)  # [B]

#         target = rewards + (1.0 - dones) * self.hp.gamma * q_next  # [B]

#     loss = nn.functional.smooth_l1_loss(q_sa, target)

#     self.optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     if self.hp.clip_grad_norm is not None:
#         nn.utils.clip_grad_norm_(self.q_net.parameters(), self.hp.clip_grad_norm)
#     self.optimizer.step()

#     with torch.no_grad():
#         avg_q = q_sa.mean().item()
#         td_err = (q_sa - target).abs().mean().item()

#     return {
#         "train/loss": float(loss.item()),
#         "train/avg_q": float(avg_q),
#         "train/td_abs_err": float(td_err),
#         "train/epsilon": float(self.epsilon),
#     }

# endregion
