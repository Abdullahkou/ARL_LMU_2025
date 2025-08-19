import json
from enum import Enum
from typing import Any, Callable, Optional, Protocol

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3


class RandomAgent:
    """Can be used for onxx mystery model or Random agent, depending on whether you provide a path"""

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
    
    def save(self, path: str):
        data = {"seed": self.seed}
        with open(path + ".json", "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, action_space):
        with open(path + ".json", "r") as f:
            data = json.load(f)
        return cls(action_space=action_space, seed=data["seed"])



class Algo(Enum):
    PPO = PPO
    SAC = SAC
    A2C = A2C
    DQN = DQN
    DDPG = DDPG
    TD3 = TD3
    RANDOM = RandomAgent
    # TODO: The other algorithms from stable baselines or elsewhere

ALGO_IMPLS = {
    "PPO":    lambda env, seed: PPO("MlpPolicy", env, seed=seed),
    "A2C":    lambda env, seed: A2C("MlpPolicy", env, seed=seed),
    "DQN":    lambda env, seed: DQN("MlpPolicy", env, seed=seed),
    "SAC":    lambda env, seed: SAC("MlpPolicy", env, seed=seed),
    "DDPG":    lambda env, seed: DDPG("MlpPolicy", env, seed=seed),
    "TD3":    lambda env, seed: TD3("MlpPolicy", env, seed=seed),
    "RANDOM": lambda env, seed: RandomAgent(
        action_space=env.action_space,
        seed=seed,
    ),
}


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
        self.random_agent = None

        agents = []
        for algo_name in algos:
            builder = ALGO_IMPLS[algo_name]
            agent = builder(train_env, seed)
            agents.append(agent)

        # Teste für einzelene Algorithmen
        if len(agents) == 1:
            self.agents = agents[0]   


    def learn(self, total_steps: int, eval_fn: Callable | None = None, eval_schedule_list: list[int]=None) -> None:
        """
        Learn for a certain number of steps optionally doing eval.

        :param total_steps: number of steps to learn
        :param (optional) eval_fn: evaluation fn to call during learning
        """
        if eval_fn is not None:
            if isinstance(self.agents, RandomAgent):
                self.agents.eval_model = eval_fn
                self.agents.learn(eval_schedule_list=eval_schedule_list)
            else:
                callback = SB3Callback(eval_model=eval_fn, total_timesteps=total_steps)
                return self.agents.learn(total_steps, callback=callback, log_interval=None)

    def predict(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        """
        Predict the action to take given an observation.

        :param obs: observation to predict action for
        :param deterministic: whether to use deterministic action selection
        :return: action to take
        """
        if self.agents is not None:
            action, _ = self.agents.predict(obs, deterministic=True)
        elif self.random_agent is not None:
             action, _ = self.random_agent.predict(obs, deterministic=True)   
        return action

    def save(self, algo_name: str, path: str) -> None:
        """
        Save the agent's state to a file.

        :algo_name: name of the algorithm
        :param path: path to save the agent's state to
        """
        if hasattr(self.agents, "save"):
            self.agents.save(path + "_" + algo_name)
        else:
            raise TypeError(f"Agent {algo_name} hat keine save()-Methode")

    def load(self, algo_name: str, path: str, env, action_space=None) -> None:
        """
        Load the agent's state from a file.

        :param path: path to load the agent's state from
        """
        if algo_name in ["PPO", "A2C", "SAC", "TD3", "DDPG", "DQN"]:
            # SB3-Loader
            AlgoClass = getattr(__import__("stable_baselines3"), algo_name)
            return AlgoClass.load(path + "_" + algo_name, env=env)
        elif algo_name == "RANDOM":
            return RandomAgent.load(path + "_" + algo_name, action_space)
        else:
            raise ValueError(f"Unbekannter Algorithmus: {algo_name}")


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
