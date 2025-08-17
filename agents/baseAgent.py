from typing import Protocol
from stable_baselines3 import PPO, SAC, A2C, DQN
from enum import Enum

class Algo(Enum):
    PPO = PPO
    SAC = SAC
    A2C = A2C
    DQN = DQN
    # TODO: The other algorithms from stable baselines or elsewhere

class BaseAgent(Protocol):
    def __init__(self, algo: Algo, params: dict = {}, env = None) -> None:
        """config: TODO to be defined
        env: TODO to be defined -> gym"""
        self.algo = algo
        self.params = params
        super().__init__()

    def learn(self, total_steps, eval_fn=None) -> None:
        """total_steps: int, number of steps to learn
        eval_fn(optional): function, evaluation function to call during learning"""
        pass

    def predict(self, obs, deterministic=False):
        """obs: np.ndarray, observation to predict action for
        returns: np.ndarray, action to take"""
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
