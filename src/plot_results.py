import os
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.logging import REWARD, STEPS


def plot_results(
    agents_to_plot: dict[str, tuple[str, str | None]],
    save_dir: str,
    interval_x_axis: Optional[int] = 5,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_to_plot = dict[str, tuple[np.ndarray, np.ndarray | None]]()

    x_vals = None  # common x basis among all dfs
    for agent_name, (mean_file, std_file) in agents_to_plot.items():
        mean_df = pd.read_csv(mean_file, index_col=0)
        mean = mean_df.values

        std = None
        if std_file is not None:
            std = pd.read_csv(std_file, index_col=0).values

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

        for idx, (agent_name, (mean, std)) in enumerate(results_to_plot.items()):
            # random agent represented by horizontal line due to no training progress
            # TODO: refactor magic str "Random" to imported constant
            if agent_name == "Random":
                assert std is None, "Random agent should not have std!"

                plt.axhline(
                    y=np.nanmean(mean, axis=0),
                    linestyle="--",
                    label=agent_name,
                    linewidth=2,
                    color=f"C{idx}",
                )
                continue

            plt.plot(x_vals, mean, label=agent_name, linewidth=2, color=f"C{idx}")

            if std is not None:
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
    base_dir = "results/test"
    save_dir = f"{base_dir}/plots"

    agents = {
        "Random": (f"{base_dir}/Random/sma_mean.csv", f"{base_dir}/Random/sma_std.csv"),
    }

    plot_results(agents, save_dir=save_dir)


if __name__ == "__main__":
    main()
