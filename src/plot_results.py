import os
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tuning.training_config import RANDOM_AGENT, SMA_MEAN_FILE, SMA_STD_FILE, TRAIN_DIR
from utils.logging import REWARD, STEPS


def plot_results(
    agents_to_plot: dict[str, tuple[str, str | None]],
    save_dir: str,
    interval_x_axis: Optional[int] = 5,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_to_plot = dict[str, tuple[pd.DataFrame, pd.DataFrame | None]]()

    x_vals = None  # common x basis among all dfs
    for agent_name, (mean_file, std_file) in agents_to_plot.items():
        mean_df = pd.read_csv(mean_file, index_col=0)
        mean = mean_df

        std = None
        if std_file is not None:
            std = pd.read_csv(std_file, index_col=0)

        results_to_plot[agent_name] = (mean, std)

        # init x_vals
        if x_vals is None:
            x_vals = mean_df.index.values

    x_label = "Training Steps"

    for col_name, title, file_name in [
        (STEPS, "Steps", "steps.png"),
        (REWARD, "Reward", "reward.png"),
    ]:
        plt.margins(x=0)
        plt.title(title)

        plt.xlabel(x_label)
        plt.ylabel(col_name)

        for idx, (agent_name, (means, stds)) in enumerate(results_to_plot.items()):
            mean = means[col_name].values
            std = stds[col_name].values if stds is not None else None
            # random agent represented by horizontal line due to no training progress
            if agent_name == RANDOM_AGENT:
                std = None  # no std

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


def main():
    base_dir = f"results/test/Walker2d-v5/1.0M/{TRAIN_DIR}"
    save_dir = f"{base_dir}/_plots"

    agents = {
        "PPO": (
            f"{base_dir}/PPO/{SMA_MEAN_FILE}",
            f"{base_dir}/PPO/{SMA_STD_FILE}",
        )
    }
    # agents = {
    #     RANDOM_AGENT: (
    #         f"{base_dir}/RANDOM/{SMA_MEAN_FILE}",
    #         f"{base_dir}/RANDOM/{SMA_STD_FILE}",
    #     ),
    #     "SAC": (
    #         f"{base_dir}/SAC/{SMA_MEAN_FILE}",
    #         f"{base_dir}/SAC/{SMA_STD_FILE}",
    #     ),
    #     "PPO": (
    #         f"{base_dir}/PPO/{SMA_MEAN_FILE}",
    #         f"{base_dir}/PPO/{SMA_STD_FILE}",
    #     ),
    #     "TD3": (
    #         f"{base_dir}/TD3/{SMA_MEAN_FILE}",
    #         f"{base_dir}/TD3/{SMA_STD_FILE}",
    #     ),
    #     "PPO_TD3 (weighted)": (
    #         f"results/test/Walker2d-v5/1.5M/eval/checkpoints_PPO_TD3_weighted/sma_mean.csv",
    #         f"results/test/Walker2d-v5/1.5M/eval/checkpoints_PPO_TD3_weighted/sma_std.csv"
    #     ),
    #     "PPO_SAC_TD3 (weighted)": (
    #         f"results/test/Walker2d-v5/1.5M/eval/checkpoints_PPO_SAC_TD3_weighted/sma_mean.csv",
    #         f"results/test/Walker2d-v5/1.5M/eval/checkpoints_PPO_SAC_TD3_weighted/sma_std.csv"
    #     )
    # }

    plot_results(agents, save_dir=save_dir)


if __name__ == "__main__":
    main()
