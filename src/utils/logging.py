import csv
from collections import deque

import numpy as np
from pandas import DataFrame

Training_STEP_COL = "Training Step"
EVAL_EP_COL = "Evaluation Episode"

# ep col names
STEPS = "Steps"
REWARD = "Reward"

EVAL_COLS = [
    STEPS,
    REWARD,
]


class EpisodeLogger:
    def __init__(self, intermediate_results_file: str | None = None):
        self.eps = 0
        self.episode_results: list[tuple[float, float]] = []
        self.__reset()

        self.intermediate_results_file = intermediate_results_file

        if intermediate_results_file is not None:
            with open(intermediate_results_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([EVAL_EP_COL] + EVAL_COLS)

    def __reset(self):
        self.steps = 0
        self.reward = 0.0

    def log_step(self, reward: float):
        self.steps += 1
        self.reward += reward

    def log_episode(self):
        self.eps += 1
        result = (self.steps, self.reward)

        self.episode_results.append(result)

        if self.intermediate_results_file is not None:
            with open(self.intermediate_results_file, "a", newline="") as f:
                writer = csv.writer(f)
                row = (self.eps,) + result
                writer.writerow(row)

        self.__reset()

    def get_eps_mean(self):
        mean = np.mean(self.episode_results, axis=0)

        return tuple(mean)


def safe_rolling_avg(df: DataFrame, file_name: str, window_size=3, min_periods=1):
    rolling_avg = df.rolling(window=window_size, min_periods=min_periods).mean()
    rolling_avg.to_csv(file_name)


def safe_mean(arr: np.ndarray | list | deque):
    """-- Credit to sb3 --
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))  # type: ignore[arg-type]


def safe_min(arr: np.ndarray | list | deque):
    """-- Credit to sb3 --
    Compute the min of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else float(np.min(arr))  # type: ignore[arg-type]
