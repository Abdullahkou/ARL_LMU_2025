from enum import Enum
from typing import Callable, Optional, Protocol

import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3


class Algo(Enum):
    PPO = PPO
    SAC = SAC
    A2C = A2C
    DQN = DQN
    DDPG = DDPG
    TD3 = TD3
    # TODO: The other algorithms from stable baselines or elsewhere


class BaseAgent(Protocol):
    """Basic agent interface"""

    def __init__(
        self,
        algos: list[Algo],
        train_env: Callable,
        hyper_params: dict = {},
        seed: int = 69,
        device: str = "cpu",
    ) -> None:
        """
        Base ctor.
        """
        # Train Umgebung seeden
        train_env.reset(seed=seed)

        self.agents = None

        if algos is PPO:
            self.agents = PPO("MlpPolicy", train_env, seed=seed)
        elif algos is [SAC]:
            self.agents = SAC("MlpPolicy", train_env, seed=seed)
        elif algos is [A2C]:
            self.agents = A2C("MlpPolicy", train_env, seed=seed)
        elif algos is [DQN]:
            self.agents = DQN("MlpPolicy", train_env, seed=seed)
        elif algos is [DDPG]:
            self.agents = DDPG("MlpPolicy", train_env, seed=seed)
        elif algos is [TD3]:
            self.agents = TD3("MlpPolicy", train_env, seed=seed)
        else:
            raise ValueError(f"Model {algos} not supported")

    def learn(self, total_steps: int, eval_fn: Callable | None = None) -> None:
        """
        Learn for a certain number of steps optionally doing eval.

        :param total_steps: number of steps to learn
        :param (optional) eval_fn: evaluation fn to call during learning
        """
        if self.agents is not None:
            if eval_fn is not None:
                callback = SB3Callback(eval_model=eval_fn, total_timesteps=total_steps)
                return self.agents.learn(total_steps, callback=callback, log_interval=None)
            else:
                self.agents.learn(total_steps, callback=callback, log_interval=None)

    def predict(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        """
        Predict the action to take given an observation.

        :param obs: observation to predict action for
        :param deterministic: whether to use deterministic action selection
        :return: action to take
        """
        action, _ = self.agents.predict(obs, deterministic=True)
        return action

    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.

        :param path: path to save the agent's state to
        """
        pass

    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.

        :param path: path to load the agent's state from
        """
        pass


from stable_baselines3.common.callbacks import BaseCallback


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
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
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
