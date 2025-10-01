import csv
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from config.training_config import (
    HEADS_DIR,
    MEAN_FILE,
    SEEDS_DIR,
    SMA_MEAN_FILE,
    SMA_STD_FILE,
    STD_FILE,
    VALIDATION_RESULTS_FILE,
)

TRAINING_STEP_COL = "Training Step"
EVAL_EP_COL = "Evaluation Episode"

# ep col names
STEPS = "Steps"
REWARD = "Reward"
VALUE_ERROR = "Value Error"

EVAL_COLS = [
    STEPS,
    REWARD,
]

TRAINING_COLS = [REWARD, VALUE_ERROR]


class TrainLogger:
    def __init__(self, save_file: str, log_interval=100):
        self.steps = 0
        self.log_interval = log_interval
        self.episode_results: list[tuple[float, float, float]] = []
        self.__reset()

        self.save_file = save_file

        with open(save_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([TRAINING_STEP_COL] + TRAINING_COLS)

    def __reset(self):
        self.reward = 0.0
        self.value_error = 0.0

    def log_step(self, reward: float, value_error: float):
        self.reward += reward

        self.steps += 1

        if not np.isnan(value_error):
            self.value_error += value_error

        if self.steps % self.log_interval == 0:
            self.__save_bulk(self.steps)

    def __save_bulk(self, current_training_step: int):
        reward_per_step = self.reward / self.log_interval
        result = (reward_per_step, self.value_error)

        self.episode_results.append(result)

        with open(self.save_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = (current_training_step,) + result
            writer.writerow(row)

        self.__reset()

    def get_training_results(self):
        return self.episode_results


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


def aggregate_head_results(agent_dir: str, num_heads: int, seeds: list[int]):
    agent_path = Path(agent_dir)
    head_results_dir = agent_path / HEADS_DIR
    head_results_dir.mkdir(exist_ok=True)

    for h in range(num_heads):
        seeds_results = []
        step_intervals = None

        for s in seeds:
            csv_path = (
                agent_path
                / f"{SEEDS_DIR}/seed_{s}/{HEADS_DIR}/h{h}/{VALIDATION_RESULTS_FILE}"
            )
            if not csv_path.exists():
                print(f"Warning: missing {csv_path}")
                continue

            df = pd.read_csv(csv_path, index_col=TRAINING_STEP_COL)
            if step_intervals is None:
                step_intervals = df.index.to_list()
            seeds_results.append(df.values)

        if not seeds_results:
            print(f"No results for head {h}")
            continue

        out_dir = head_results_dir / f"h{h}"
        out_dir.mkdir(parents=True, exist_ok=True)

        save_seed_totals(
            seeds_results,
            dir=out_dir,
            step_intervals=step_intervals,
        )
        print(f"Saved head {h} aggregates → {out_dir}")


def save_seed_totals(
    seeds_results: list[np.ndarray],
    dir: str,
    step_intervals: list[int],
    save_rolling=True,
    columns: list[str] = EVAL_COLS,
    index_name=TRAINING_STEP_COL,
):
    if not os.path.exists(dir):
        os.makedirs(dir)

    mean = np.nanmean(seeds_results, axis=0)
    mean_df = pd.DataFrame(mean, columns=columns, index=step_intervals)
    mean_df.index.name = index_name
    mean_df.to_csv(f"{dir}/{MEAN_FILE}", index=True)

    if save_rolling:
        save_rolling_avg(
            mean_df, f"{dir}/{SMA_MEAN_FILE}", window_size=3, min_periods=1
        )

    std = np.nanstd(seeds_results, axis=0)
    std_df = pd.DataFrame(std, columns=columns, index=step_intervals)
    std_df.index.name = index_name
    std_df.to_csv(f"{dir}/{STD_FILE}", index=True)

    if save_rolling:
        save_rolling_avg(std_df, f"{dir}/{SMA_STD_FILE}", window_size=3, min_periods=1)


def save_rolling_avg(df: DataFrame, file_name: str, window_size=3, min_periods=1):
    rolling_avg = df.rolling(window=window_size, min_periods=min_periods).mean()
    rolling_avg.to_csv(file_name)


def mix_rbf_mmsd(X: np.ndarray, Y: np.ndarray, sigmas=(1,)):
    """
    Mixture RBF kernel Squared Maximum Mean Discrepancy between two sets of states.
    New Method of measuring the distance between two distributions.
    It 'can comprehensively reflect the mean and variance information of data samples in the reproducing kernel'.

    cf. https://github.com/liguge/Maximum-mean-square-discrepancy
    """
    K_XX, K_XY, K_YY = __mix_rbf_kernel(X, Y, sigmas)
    mmds = __mmsd(K_XX, K_XY, K_YY)

    return mmds


def __mix_rbf_kernel(X: np.ndarray, Y: np.ndarray, sigmas: tuple) -> tuple:
    """
    Mix RBF kernel between two sets of states.

    cf. https://github.com/liguge/Maximum-mean-square-discrepancy
    """
    wts = [1] * len(sigmas)

    # (n_states, n_states)
    # pairwise products of state vectors
    XX = np.matmul(X, X.T)
    XY = np.matmul(X, Y.T)
    YY = np.matmul(Y, Y.T)

    # (n_states,), left-right/top-down diagonal of matrices
    # axis1 and axis2 are the two last axes
    X_sqnorms = np.diagonal(XX, axis1=-2, axis2=-1)
    Y_sqnorms = np.diagonal(YY, axis1=-2, axis2=-1)

    K_XX, K_XY, K_YY = 0.0, 0.0, 0.0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * np.exp(-gamma * (-2 * XX + __c(X_sqnorms) + __r(X_sqnorms)))
        K_XY += wt * np.exp(-gamma * (-2 * XY + __c(X_sqnorms) + __r(Y_sqnorms)))
        K_YY += wt * np.exp(-gamma * (-2 * YY + __c(Y_sqnorms) + __r(Y_sqnorms)))

        return K_XX, K_XY, K_YY  # ??


def __c(x: np.ndarray):
    return np.expand_dims(x, 1)


def __r(x: np.ndarray):
    return np.expand_dims(x, 0)


def __mmsd(
    K_XX: np.ndarray,
    K_XY: np.ndarray,
    K_YY: np.ndarray,
):
    """
    Maximum Mean Square Discrepancy between two sets of states.

    cf. https://github.com/liguge/Maximum-mean-square-discrepancy
    """
    m = K_XX.shape[0]
    n = K_YY.shape[0]

    C_K_XX = np.pow(K_XX, 2)
    C_K_YY = np.pow(K_YY, 2)
    C_K_XY = np.pow(K_XY, 2)

    mmsd: np.floating = (
        np.sum(C_K_XX) / (m * m)
        + np.sum(C_K_YY) / (n * n)
        - 2 * np.sum(C_K_XY) / (m * n)
    )

    return mmsd


def parse_steps(total_steps: int):
    """Parse steps int into something like 1M or 1.5M or 0.5M..."""
    if total_steps < 1_000:
        return str(total_steps)
    elif total_steps < 1_000_000:
        return f"{total_steps / 1_000:.1f}K"
    else:
        return f"{total_steps / 1_000_000:.1f}M"


def save_mean(arr: np.ndarray | list | deque):
    """-- Credit to sb3 --
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))  # type: ignore[arg-type]


def save_min(arr: np.ndarray | list | deque):
    """-- Credit to sb3 --
    Compute the min of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else float(np.min(arr))  # type: ignore[arg-type]
