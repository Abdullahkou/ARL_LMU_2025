# utils/replay_buffer.py
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(
        self, capacity, obs_shape, seed, device, n_heads=1, bootstrap_prob=0.5
    ):
        self.capacity = int(capacity)
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        # Masken für Bootstrappig
        self.masks = np.zeros((self.capacity, n_heads), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.n_heads = n_heads
        self.bootstrap_prob = bootstrap_prob

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = done

        # Bootstrap-Maske ziehen (Bernoulli für jeden Head)
        mask = self.rng.binomial(1, self.bootstrap_prob, size=self.n_heads)
        self.masks[self.idx] = mask

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        size = len(self)
        if size == 0:
            raise ValueError("ReplayBuffer is empty.")
        idxs = self.rng.integers(0, size, size=batch_size)
        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
            "masks": self.masks[idxs],
        }


# class ReplayBuffer:
#     def __init__(self, capacity: int, obs_shape: Tuple[int, ...], seed: int, device: torch.device):
#         self.capacity = int(capacity)
#         self.device = device
#         self.rng = np.random.default_rng(seed)

#         self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
#         self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
#         self.actions = np.zeros((self.capacity,), dtype=np.int64)
#         self.rewards = np.zeros((self.capacity,), dtype=np.float32)
#         self.dones = np.zeros((self.capacity,), dtype=np.float32)

#         self.idx = 0
#         self.full = False

#     def __len__(self) -> int:
#         return self.capacity if self.full else self.idx

#     def add(self, obs, action, reward, next_obs, done):
#         self.obs[self.idx] = obs
#         self.actions[self.idx] = action
#         self.rewards[self.idx] = reward
#         self.next_obs[self.idx] = next_obs
#         self.dones[self.idx] = done

#         self.idx = (self.idx + 1) % self.capacity
#         self.full = self.full or self.idx == 0

#     def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
#         size = len(self)
#         if size == 0:
#             raise ValueError("ReplayBuffer is empty.")
#         idxs = self.rng.integers(0, size, size=batch_size)
#         return {
#             "obs": self.obs[idxs],
#             "actions": self.actions[idxs],
#             "rewards": self.rewards[idxs],
#             "next_obs": self.next_obs[idxs],
#             "dones": self.dones[idxs],
#         }
