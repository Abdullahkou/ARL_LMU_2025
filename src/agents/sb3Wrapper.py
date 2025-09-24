from typing import Callable

import numpy as np
from gymnasium import Env
from stable_baselines3 import A2C, DQN, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import ReLU

from agents.baseAgent import IAgent
from config.training_config import Algorithm, TrainingConfig
from utils.logging import TrainLogger


class SB3Wrapper(IAgent):
    """SB3Wrapper"""

    def __init__(
        self,
        train_config: TrainingConfig,
        train_env_factory: Callable[[], Env],
        seed: int = 69,
        device: str = "cpu",
    ) -> None:
        env = train_env_factory()

        config = train_config.algo_config
        hp = config.hyper_params

        # activation fn currently not part of hyperparams
        # but we use ReLU for our agent as well
        policy_kwargs = dict(activation_fn=ReLU, net_arch=hp.hidden_sizes)

        base_qlearn_dict = {
            "learning_rate": hp.learning_rate,
            "buffer_size": hp.buffer_size,
            "learning_starts": hp.learning_starts,
            "batch_size": hp.batch_size,
            "tau": hp.tau,
            "gamma": hp.gamma,
            "train_freq": hp.train_freq,
            "gradient_steps": hp.gradient_steps,
        }

        dqn_dict = {
            **base_qlearn_dict,
            "target_update_interval": hp.target_update_interval,
            "exploration_fraction": hp.epsilon_decay_steps / train_config.steps,
            "exploration_initial_eps": hp.max_epsilon,
            "exploration_final_eps": hp.min_epsilon,
            "max_grad_norm": hp.clip_grad_norm,
        }

        self.sb3Agent = None
        match config.algorithm:
            case Algorithm.SB3_DQN:
                self.sb3Agent = DQN(
                    policy="MlpPolicy",
                    env=env,
                    seed=seed,
                    device=device,
                    policy_kwargs=policy_kwargs,
                    **dqn_dict,
                )
            case Algorithm.SB3_TD3:
                self.sb3Agent = TD3(
                    policy="MlpPolicy",
                    env=env,
                    seed=seed,
                    device=device,
                    policy_kwargs=policy_kwargs,
                    **base_qlearn_dict,
                )
            case Algorithm.SB3_A2C:
                self.sb3Agent = A2C(
                    policy="MlpPolicy",
                    env=env,
                    seed=seed,
                    device=device,
                    policy_kwargs=policy_kwargs,
                    learning_rate=hp.learning_rate,
                    gamma=hp.gamma,
                    max_grad_norm=hp.clip_grad_norm,
                )
            case Algorithm.SB3_PPO:
                self.sb3Agent = PPO(
                    policy="MlpPolicy",
                    env=env,
                    seed=seed,
                    device=device,
                    policy_kwargs=policy_kwargs,
                    learning_rate=hp.learning_rate,
                    gamma=hp.gamma,
                    max_grad_norm=hp.clip_grad_norm,
                    batch_size=hp.batch_size,
                )
            case _:
                self.sb3Agent = DQN(
                    policy="MlpPolicy",
                    env=env,
                    seed=seed,
                    device=device,
                    policy_kwargs=policy_kwargs,
                    **dqn_dict,
                )

    def learn(
        self,
        total_steps: int,
        eval_fn: Callable[[int], None] | None = None,
        train_logger: TrainLogger | None = None,
        eval_schedule: int | None = None,
    ) -> None:
        """
        Learn for a certain number of steps optionally doing eval.

        :param total_steps: number of steps to learn
        :param (optional) train_logger: useful for recording training stats
        :param (optional) eval_fn: evaluation fn to call during learning, called at every step!
        :param (optional) eval_schedule: eval interval, useful for multi agent setting!
        """
        callback = SB3Callback(
            total_timesteps=total_steps, eval_fn=eval_fn, train_logger=train_logger
        )
        self.sb3Agent.learn(total_timesteps=total_steps, callback=callback)

    def predict(self, obs: np.ndarray, deterministic=False):
        """
        Predict the action to take given an observation.

        :param obs: observation to predict action for
        :param deterministic: whether to use deterministic action selection
        :return: action to take
        """
        return self.sb3Agent.predict(observation=obs, deterministic=deterministic)

    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.

        :algo_name: name of the algorithm
        :param path: path to save the agent's state to
        """
        self.sb3Agent.save(path=path)

    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.

        :param path: path to load the agent's state from
        """
        self.sb3Agent = self.sb3Agent.__class__.load(path=path, env=self.sb3Agent.env)


class SB3Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        verbose: int = 0,
        total_timesteps: int = 1000,
        eval_fn: Callable[[int], None] | None = None,
        train_logger: TrainLogger | None = None,
    ):
        super().__init__(verbose)
        self.eval_fn = eval_fn
        self.total_timesteps = total_timesteps
        self.train_logger = train_logger

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        if self.eval_fn is not None:  # eval right away
            self.eval_fn(current_step=self.num_timesteps)

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

        if self.eval_fn is not None:
            self.eval_fn(current_step=self.num_timesteps)

        if self.train_logger:
            reward = self.locals["rewards"][0]
            self.train_logger.log_step(reward=reward, value_error=np.nan)

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
