import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.training_config import HEADS_DIR, SMA_MEAN_FILE, SMA_STD_FILE
from utils.logging import REWARD, STEPS, VALUE_ERROR

VALIDATION = "Validation_Results"
TRAINING = "Training_Results"
RANDOM_AGENT = "RANDOM"


def plot_results(
    results_to_plot,
    save_dir: str,
    env_name="",
    is_training_result=False,
    interval_x_axis: int | None = 5,
    color_in_std=True,
    save_file_postfix=""
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x_vals = list(results_to_plot.values())[0][0].index
    x_label = "Training Steps"

    heads_present = any(re.search(r"h\d", s) for s in results_to_plot.keys())

    cols = [
        (
            REWARD,
            "Reward",
            f"{'training' if is_training_result else 'validation'}_reward{'_with_heads' if heads_present else ''}.png",
        )
    ]
    if is_training_result:
        cols.append(
            (
                VALUE_ERROR,
                "Value Error",
                f"{'training' if is_training_result else 'validation'}_value_error.png",
            )
        )
    else:
        cols.append(
            (
                STEPS,
                "Steps",
                f"{'training' if is_training_result else 'validation'}_steps{'_with_heads' if heads_present else ''}.png",
            )
        )

    for col_name, title, file_name in cols:
        plt.rcParams["figure.figsize"] = (12, 7)
        plt.margins(x=0)
        plt.title(
            f"{env_name} {'Training' if is_training_result else 'Validation'} {title}"
        )

        plt.xlabel(x_label)
        plt.ylabel(col_name)

        for idx, (agent_name, (means, stds)) in enumerate(results_to_plot.items()):
            if col_name not in means:
                continue
            if col_name == VALUE_ERROR and (
                "SB3" in agent_name.upper() or RANDOM_AGENT in agent_name.upper()
            ):
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

            if color_in_std:
                plt.fill_between(
                    x_vals, mean - std, mean + std, alpha=0.3, color=f"C{idx}"
                )

        plt.legend()

        # Adapt x axes labeling
        interval = x_vals[-1] // interval_x_axis
        plt.xticks(np.arange(0, x_vals[-1] + interval, interval))

        # Customize y-axis and x-axis to start at 0
        # plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.savefig(f"{save_dir}/{file_name}{save_file_postfix}")
        plt.clf()


def read_csv(path_to_file):
    return pd.read_csv(path_to_file, index_col=0, dtype=str).apply(
        pd.to_numeric, errors="coerce"
    )


def load_csvs(
    base_dir, algos_to_plot, load_training_csvs=False, load_head_results=False
):
    results_to_plot = {}
    if algos_to_plot is None:
        algos_to_plot = os.listdir(base_dir)
    for algo_name in algos_to_plot:
        results_dir = (
            f"{base_dir}/{algo_name}/{TRAINING if load_training_csvs else VALIDATION}/"
        )
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


def load_seeds(base_dir, algo_name, load_train_results=False, apply_rolling=True):
    seeds_dir = f"{base_dir}/{algo_name}/seeds"
    results_to_plot = dict[str, tuple[pd.DataFrame, pd.DataFrame]]()

    for seed in os.listdir(seeds_dir):
        path_to_seed_csv = f"{seeds_dir}/{seed}/{'training_results.csv' if load_train_results else 'validation_results.csv'}"
        mean_df = read_csv(path_to_seed_csv)

        if apply_rolling:
            mean_df = mean_df.rolling(window=3, min_periods=1).mean()

        # This is a hack but it works: create a second df where every col except training steps is 0
        # When we plot this using plot_results it wont have the highlighted interval
        std_df = mean_df.copy()
        std_df[mean_df.columns[0]] = 0
        std_df[mean_df.columns[1]] = 0
        results_to_plot[f"{algo_name}_{seed}"] = (mean_df, std_df)

    return results_to_plot


def plot_seeds():
    env_name = "LunarLander-v3"
    # env_name = "FrozenLake-v1"

    base_dir = f"results/{env_name}/1.0M"

    algo_name = "Custom_DQN_strict_ebql_20qh"

    seeds_save_dir = f"{base_dir}/{algo_name}/_plots"

    plot_training_results = False  # toggle to either to plot eval or train results

    results_to_plot = load_seeds(
        base_dir,
        algo_name,
        load_train_results=plot_training_results,
        apply_rolling=True,
    )

    plot_results(
        results_to_plot=results_to_plot,
        save_dir=seeds_save_dir,
        is_training_result=plot_training_results,
        env_name=env_name,
    )


def main_plots():
    env_name = "LunarLander-v3"
    # env_name = "FrozenLake-v1"

    base_dir = f"results/{env_name}/1.0M"
    algos_to_plot = [
        #"Custom_DQN_1qh",
        #"Custom_DQN_3qh",
        #"Custom_DQN_5qh",
        "Custom_DQN_10qh",
    ]

    save_dir = f"{base_dir}/_plots"
    # save_dir = f"{base_dir}/Custom_DQN_10qh/_plots"

    plot_training_results = False  # toggle to either to plot eval or train results
    load_head_results = True  # set True to see head plots

    results_to_plot = load_csvs(
        base_dir,
        algos_to_plot=algos_to_plot,
        load_training_csvs=plot_training_results,
        load_head_results=load_head_results,
    )
    # If you need to plot results from outside of base_dir, use results_to_plot["my_result"] =  read_csv(path_to_result)
    plot_results(
        results_to_plot,
        save_dir=save_dir,
        is_training_result=plot_training_results,
        env_name=env_name,
        color_in_std=True,
        save_file_postfix=""
    )


def main():
    main_plots()
    # plot_seeds()


if __name__ == "__main__":
    main()
