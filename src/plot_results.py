import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

from config.training_config import RANDOM_AGENT, SMA_MEAN_FILE, SMA_STD_FILE, HEADS_DIR
from utils.logging import REWARD, STEPS, VALUE_ERROR

VALIDATION = "Validation_Results"
TRAINING = "Training_Results"


def plot_results(
    results_to_plot,
    save_dir: str,
    is_training_result=False,
    interval_x_axis: int | None = 5,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x_vals = list(results_to_plot.values())[0][0].index
    x_label = f"{'Training' if is_training_result else 'Validation'} Steps"

    heads_present = any(re.search(r'h\d', s) for s in results_to_plot.keys())

    cols = [
        (REWARD, "Reward", f"{'training' if is_training_result else 'validation'}_reward{'_with_heads' if heads_present else ''}.png")
    ]
    if is_training_result:
        cols.append(
            (VALUE_ERROR, "Value Error", f"{'training' if is_training_result else 'validation'}_value_error.png")
        )
    else:
        cols.append(
            (STEPS, "Steps", f"{'training' if is_training_result else 'validation'}_steps{'_with_heads' if heads_present else ''}.png")
        )

    for col_name, title, file_name in cols:
        plt.rcParams["figure.figsize"] = (12, 7)
        plt.margins(x=0)
        plt.title(title)

        plt.xlabel(x_label)
        plt.ylabel(col_name)
        for idx, (agent_name, (means, stds)) in enumerate(results_to_plot.items()):
            if col_name not in means:
                continue
            mean = means[col_name].values.astype(float)
            if not np.isfinite(mean).any():
                continue
            std = stds[col_name].values if stds is not None else None
            # random agent represented by horizontal line due to no training progress
            if agent_name == RANDOM_AGENT:
                plt.axhline(
                    y=np.nanmean(mean, axis=0),
                    linestyle="--",
                    label=agent_name,
                    linewidth=2,
                    color=f"C{idx}",
                )
                continue

            plt.plot(x_vals, mean, label=agent_name, linewidth=2, color=f"C{idx}")

            if std is None:
                continue

            plt.fill_between(x_vals, mean - std, mean + std, alpha=0.3, color=f"C{idx}")

        plt.legend()

        # Adapt x axes labeling
        interval = x_vals[-1] // interval_x_axis
        plt.xticks(np.arange(0, x_vals[-1] + interval, interval))

        # Customize y-axis and x-axis to start at 0
        # plt.ylim(bottom=0)
        plt.xlim(left=0)

        plt.savefig(f"{save_dir}/{file_name}")
        plt.clf()


def read_csv(path_to_file):
    return pd.read_csv(path_to_file, index_col=0, dtype=str).apply(pd.to_numeric, errors="coerce")


def load_csvs(base_dir, algos_to_plot, load_training_csvs=False, load_head_results=False):
    results_to_plot = {}
    for algo_name in algos_to_plot:
        results_dir = f"{base_dir}/{algo_name}/{TRAINING if load_training_csvs else VALIDATION}/"
        path_to_mean = f"{results_dir}/{SMA_MEAN_FILE}"
        path_to_std = f"{results_dir}/{SMA_STD_FILE}"

        mean_df = read_csv(path_to_mean)
        std_df = read_csv(path_to_std)
        results_to_plot[algo_name] = (mean_df, std_df)

        algo_heads_dir = f"{base_dir}/{algo_name}/{HEADS_DIR}"
        if load_head_results and os.path.isdir(algo_heads_dir):
            for head in os.listdir(algo_heads_dir):
                path_to_head_mean = f"{algo_heads_dir}/{head}/{SMA_MEAN_FILE}"
                path_to_head_std = f"{algo_heads_dir}/{head}/{SMA_STD_FILE}"
                head_mean_df = read_csv(path_to_head_mean)
                head_std_df = read_csv(path_to_head_std)
                results_to_plot[f"{algo_name}_{head}"] = (head_mean_df, head_std_df)
    return results_to_plot


def load_csvs_heads(base_dir):
    pass


def main():
    base_dir = "results/LunarLander-v3/1.0M"
    algos_to_plot = ["CUSTOM_DQN_3qh"]#, "SB3_DQN", "SB3_PPO", "RANDOM"]
    save_dir = f"{base_dir}/_plots"

    plot_training_results = False  # set false to plot eval results
    results_to_plot = load_csvs(base_dir, algos_to_plot, load_training_csvs=plot_training_results, load_head_results=True)

    # If you need to plot results from outside of base_dir, use results_to_plot["my_result"] =  read_csv(path_to_result)
    plot_results(results_to_plot, save_dir=save_dir, is_training_result=plot_training_results)


if __name__ == "__main__":
    main()
