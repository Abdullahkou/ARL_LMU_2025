from typing import Protocol


class BaseAgent(Protocol):
    def __init__(self, config, env) -> None:
        """config: TODO to be defined
        env: TODO to be defined -> gym"""
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
