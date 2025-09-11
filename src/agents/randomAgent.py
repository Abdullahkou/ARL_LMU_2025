from typing import Callable

import numpy as np
from gymnasium import Space


class RandomAgent:
    def __init__(
        self,
        action_space: Space,
        eval_step_intervals: list[int] | None = None,
        seed: int = 42,
    ):
        self.eval_step_intervals = eval_step_intervals
        self.action_space = action_space
        self.__set_random_seed(seed=seed)

    def __set_random_seed(self, seed: int):
        self.seed = seed
        np.random.seed(self.seed)
        self.action_space.seed(self.seed)

    def predict(self, state: np.ndarray, deterministic=False):
        assert self.action_space is not None

        action = self.action_space.sample()

        return action, {}

    def learn(self, eval_fn: Callable[[int], None] | None = None):
        if eval_fn is None or self.eval_step_intervals is None:
            return

        for step_interval in self.eval_step_intervals:
            eval_fn(current_step=step_interval)
