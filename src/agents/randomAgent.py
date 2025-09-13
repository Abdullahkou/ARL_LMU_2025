from typing import Callable

import numpy as np
from gymnasium import Env

from agents.baseAgent import IAgent
from utils.logging import TrainLogger


class RandomAgent(IAgent):
    def __init__(
        self,
        train_env_factory: Callable[[], Env],
        eval_schedule: int | None = None,
        seed: int = 42,
    ):
        self.env = train_env_factory()
        self.action_space = self.env.action_space

        self.eval_schedule = eval_schedule
        self.__set_random_seed(seed=seed)

    def __set_random_seed(self, seed: int):
        self.seed = seed
        np.random.seed(self.seed)
        self.action_space.seed(self.seed)

    def predict(self, state: np.ndarray, deterministic=False):
        assert self.action_space is not None

        action = self.action_space.sample()

        return action, {}

    def learn(
        self,
        total_steps: int,
        eval_fn: Callable[[int], None] | None = None,
        train_logger: TrainLogger | None = None,
    ):
        if eval_fn is None or self.eval_schedule is None:
            return

        if train_logger is not None:
            step = 0
            while step <= total_steps:
                state, _ = self.env.reset()
                done = False
                while not done:
                    if step > total_steps:
                        break

                    eval_fn(current_step=step)

                    action, _ = self.predict(state, deterministic=False)
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    step += 1
                    train_logger.log_step(reward=reward, value_error=np.nan)
        else:
            step_intervals = [i * self.eval_schedule for i in range(total_steps + 1)]

            for step_interval in step_intervals:
                eval_fn(current_step=step_interval)

        self.env.close()
