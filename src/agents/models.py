from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RandomAgent:
    def __init__(
        self,
        action_space: gym.Space = None,
        eval_model: Optional[Callable[[int], None]] = None,
        seed: int = 42,
    ):
        self.action_space = action_space if action_space else None
        self.eval_model = eval_model

        self.set_random_seed(seed=seed)

    def set_random_seed(self, seed: int):
        self.seed = seed
        np.random.seed(self.seed)
        self.action_space.seed(self.seed) if self.action_space is not None else None

    def predict(self, state: Any, deterministic: bool = False):
        assert self.action_space is not None
        action = self.action_space.sample()
        return action, {}

    def learn(self, eval_schedule_list: list[int]):
        for i in eval_schedule_list:
            self.eval_model(current_step=i)

        return self


class SB3Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        verbose: int = 0,
        total_timesteps: int = 1000,
        eval_model: Optional[Callable[[int], None]] = None,
    ):
        super().__init__(verbose)
        self.eval_model = eval_model
        self.total_timesteps = total_timesteps

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        if self.eval_model is not None:  # eval right away
            self.eval_model(current_step=self.num_timesteps)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.num_timesteps > self.total_timesteps:
            return False

        if self.eval_model is not None:
            self.eval_model(current_step=self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
