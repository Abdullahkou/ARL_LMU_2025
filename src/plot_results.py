import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.training_config import HEADS_DIR, SMA_MEAN_FILE, SMA_STD_FILE
from utils.logging import REWARD, STEPS, TRAINING_STEP_COL, VALUE_ERROR

VALIDATION = "Validation_Results"
TRAINING = "Training_Results"
MEAN_PER_SEED = "mean_per_seed.csv"
RANDOM_AGENT = "Random Agent"


def boxplot_results(
    results_to_plot,
    save_dir: str,
    env_name="",
    is_training_result=False,
    save_file_postfix="",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cols = [
        (
            REWARD,
            "Reward",
            f"{'training' if is_training_result else 'validation'}_reward_{save_file_postfix}.png",
        )
    ]
    if is_training_result:
        cols.append(
            (
                VALUE_ERROR,
                "Value Error",
                f"{'training' if is_training_result else 'validation'}_value_error_{save_file_postfix}.png",
            )
        )
    else:
        cols.append(
            (
                STEPS,
                "Steps",
                f"{'training' if is_training_result else 'validation'}_steps_{save_file_postfix}.png",
            )
        )

    for col_name, title, file_name in cols:
        plt.rcParams["figure.figsize"] = (12, 7)
        plt.margins(x=0)
        plt.title(
            f"{env_name.split('-')[0]} {'Training' if is_training_result else 'Validation'} {title}"
        )
        plt.xlabel("Agents")
        plt.ylabel(col_name)

        data_list = []
        agents_list = []
        for idx, (agent_name, means) in enumerate(results_to_plot.items()):
            if col_name not in means:
                continue
            if col_name == VALUE_ERROR and (
                "SB3" in agent_name.upper() or RANDOM_AGENT in agent_name.upper()
            ):
                continue
            data = means[col_name].values.astype(float)
            if not np.isfinite(data).any():
                continue

            # random agent represented by horizontal line due to no training progress
            if agent_name == RANDOM_AGENT:
                plt.axhline(
                    y=np.nanmean(data, axis=0),
                    linestyle="--",
                    label=agent_name,
                    linewidth=2,
                    color=f"C{idx}",
                )
                continue

            if "/" in agent_name:
                agent_name = agent_name.split("/")[-1]

            data_list.append(data)
            agents_list.append(agent_name)

        # Ein Boxplot pro Agent
        boxplot = plt.boxplot(data_list, patch_artist=True)

        # Farben anpassen
        for patch, color in zip(
            boxplot["boxes"], plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ):
            patch.set_facecolor(color)

        plt.xticks(
            range(1, len(agents_list) + 1), range(1, len(agents_list) + 1)
        )  # nur Indizes
        plt.legend(boxplot["boxes"], agents_list + ["Random"], loc="best")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{file_name}")
        plt.clf()


def plot_results(
    results_to_plot,
    save_dir: str,
    env_name="",
    is_training_result=False,
    interval_x_axis: int | None = 5,
    color_in_std=True,
    save_file_postfix="",
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
            f"{'training' if is_training_result else 'validation'}_reward{'_with_heads' if heads_present else ''}_{save_file_postfix}.png",
        )
    ]
    if is_training_result:
        cols.append(
            (
                VALUE_ERROR,
                "Value Error",
                f"{'training' if is_training_result else 'validation'}_value_error_{save_file_postfix}.png",
            )
        )
    else:
        cols.append(
            (
                STEPS,
                "Steps",
                f"{'training' if is_training_result else 'validation'}_steps{'_with_heads' if heads_present else ''}_{save_file_postfix}.png",
            )
        )

    for col_name, title, file_name in cols:
        plt.rcParams["figure.figsize"] = (12, 7)
        plt.margins(x=0)
        plt.title(
            f"{env_name.split('-')[0]} {'Training' if is_training_result else 'Validation'} {title}"
        )

        plt.xlabel(x_label)
        plt.ylabel(col_name)

        for idx, (agent_name, (means, stds)) in enumerate(results_to_plot.items()):
            if col_name not in means:
                continue
            if "/" in agent_name:
                agent_name = agent_name.split("/")[-1]
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
        plt.savefig(f"{save_dir}/{file_name}")
        plt.clf()


def read_csv(path_to_file):
    return pd.read_csv(path_to_file, index_col=0, dtype=str).apply(
        pd.to_numeric, errors="coerce"
    )


def load_csvs(
    base_dir, algos_to_plot, load_training_csvs=False, load_head_results=False
):
    results_to_plot = dict[str, tuple[str, str]]()
    if algos_to_plot is None:
        algos_to_plot = os.listdir(base_dir)
    for algo_dir, legend_label in algos_to_plot:
        results_dir = (
            f"{base_dir}/{algo_dir}/{TRAINING if load_training_csvs else VALIDATION}/"
        )
        path_to_mean = f"{results_dir}/{SMA_MEAN_FILE}"
        path_to_std = f"{results_dir}/{SMA_STD_FILE}"

        mean_df = read_csv(path_to_mean)
        std_df = read_csv(path_to_std)
        results_to_plot[legend_label] = (mean_df, std_df)

        algo_heads_dir = f"{base_dir}/{algo_dir}/{HEADS_DIR}"
        if load_head_results and os.path.isdir(algo_heads_dir):
            for head in os.listdir(algo_heads_dir):
                path_to_head_mean = f"{algo_heads_dir}/{head}/{SMA_MEAN_FILE}"
                path_to_head_std = f"{algo_heads_dir}/{head}/{SMA_STD_FILE}"
                head_mean_df = read_csv(path_to_head_mean)
                head_std_df = read_csv(path_to_head_std)
                results_to_plot[f"{algo_dir}_{head}"] = (head_mean_df, head_std_df)
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


def create_mean_per_seed(root_dir: str):
    seeds_dir = Path(root_dir) / "seeds"
    out_training = Path(root_dir) / TRAINING
    out_validation = Path(root_dir) / VALIDATION

    if not seeds_dir.is_dir():
        raise FileNotFoundError(f"Expected folder '{seeds_dir}' not found!")

    seed_dirs = sorted([p for p in seeds_dir.iterdir() if p.is_dir()])

    if not seed_dirs:
        raise FileNotFoundError(f"No seed folders '{seeds_dir}' found!")

    train_rows = []
    val_rows = []

    # Union der Spaltennamen (für konsistente Ausgabe über alle Seeds)
    train_cols_union = set()
    val_cols_union = set()

    for seed in seed_dirs:
        train_csv = seed / "training_results.csv"
        val_csv = seed / "validation_results.csv"

        if not train_csv.exists():
            raise FileNotFoundError(f"No '{train_csv}' found!")

        if not val_csv.exists():
            raise FileNotFoundError(f"No '{val_csv}' found!")

        # ---------- Training ----------
        df_training = pd.read_csv(train_csv)

        if TRAINING_STEP_COL in df_training.columns:
            df_training = df_training.drop(columns=[TRAINING_STEP_COL])

        num_training = df_training.select_dtypes(include=[np.number])
        training_means = num_training.mean(numeric_only=True)

        seed_id = int(seed.name[-1])
        row_training = {"seed": seed_id}
        row_training.update(training_means.to_dict())
        train_rows.append(row_training)
        train_cols_union.update(training_means.index.tolist())

        # ---------- Validation ----------
        df_validation = pd.read_csv(val_csv)

        if TRAINING_STEP_COL in df_validation.columns:
            df_validation = df_validation.drop(columns=[TRAINING_STEP_COL])

        num_validation = df_validation.select_dtypes(include=[np.number])
        validation_means = num_validation.mean(numeric_only=True)

        row_validation = {"seed": seed_id}
        row_validation.update(validation_means.to_dict())
        val_rows.append(row_validation)
        val_cols_union.update(validation_means.index.tolist())

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_df.to_csv(out_training / MEAN_PER_SEED, index=False)
    val_df.to_csv(out_validation / MEAN_PER_SEED, index=False)


def load_csv_for_boxplot(
    base_dir, algos_to_plot, load_training_csvs=False, load_head_results=False
):
    results_to_plot = {}
    if algos_to_plot is None:
        algos_to_plot = os.listdir(base_dir)
    for algo_dir, legend_label in algos_to_plot:
        results_dir = (
            f"{base_dir}/{algo_dir}/{TRAINING if load_training_csvs else VALIDATION}/"
        )
        path_to_mean_per_seed = f"{results_dir}/{MEAN_PER_SEED}"

        if not os.path.exists(path_to_mean_per_seed):
            create_mean_per_seed(root_dir=f"{base_dir}/{algo_dir}")

        mean_df = read_csv(path_to_mean_per_seed)
        results_to_plot[legend_label] = mean_df

    return results_to_plot


def boxplot():
    # env_name = "MountainCar-v0"
    env_name = "LunarLander-v3"

    base_dir = f"results/{env_name}/1.0M"

    # The second item will appear on the legend.
    algos_to_plot = [
        ("Custom_DQN_1qh_128_dep_bp0.5", "Custom DQN (1 head)"),
        ("Custom_DQN_5qh_128_dep_bp0.5", "Custom DQN (5 heads)"),
        ("Custom_DQN_10qh_128_dep_bp0.5", "Custom DQN (10 heads)"),
        ("RANDOM", "Random Agent"),
    ]

    save_dir = f"{base_dir}/_boxplots/128x128/"

    train_results_to_plot = load_csv_for_boxplot(
        base_dir=base_dir,
        algos_to_plot=algos_to_plot,
        load_training_csvs=True,
    )

    validation_results_to_plot = load_csv_for_boxplot(
        base_dir=base_dir,
        algos_to_plot=algos_to_plot,
        load_training_csvs=False,
    )

    boxplot_results(
        results_to_plot=train_results_to_plot,
        save_dir=save_dir,
        is_training_result=True,
        env_name=env_name,
        save_file_postfix="128",
    )

    boxplot_results(
        results_to_plot=validation_results_to_plot,
        save_dir=save_dir,
        is_training_result=False,
        env_name=env_name,
        save_file_postfix="128",
    )


def main_plots():
    # env_name = "MountainCar-v0"
    env_name = "Taxi-v3"

    base_dir = f"results/{env_name}/200.0K"

    # The second item will appear on the legend.
    algos_to_plot = [
        ("Custom_DQN_1qh_64x64_bp0.5.old", "EBQL (1 head bp0.5)"),
        # ("Custom_DQN_1qh_64x64_bp1.0.old", "EBQL (1 head bp1.0)"),
        ("Custom_DQN_5qh_s_64x64_bp0.5", "Strict EBQL (5 heads bp0.5)"),
        ("Custom_DQN_10qh_s_64x64_bp0.5", "Strict EBQL (10 heads bp0.5)"),
        ("RANDOM", "Random Agent"),
    ]

    save_dir = f"{base_dir}/_plots/strict_bp0.5"

    load_head_results = False  # set True to see head plots

    train_results_to_plot = load_csvs(
        base_dir,
        algos_to_plot=algos_to_plot,
        load_training_csvs=True,
        load_head_results=load_head_results,
    )

    validation_results_to_plot = load_csvs(
        base_dir,
        algos_to_plot=algos_to_plot,
        load_training_csvs=False,
        load_head_results=load_head_results,
    )

    # If you need to plot results from outside of base_dir, use results_to_plot["my_result"] =  read_csv(path_to_result)
    plot_results(
        train_results_to_plot,
        save_dir=save_dir,
        is_training_result=True,
        env_name=env_name,  # Appears on the plot title
        color_in_std=True,
        save_file_postfix="",
    )

    plot_results(
        validation_results_to_plot,
        save_dir=save_dir,
        is_training_result=False,
        env_name=env_name,  # Appears on the plot title
        color_in_std=True,
        save_file_postfix="",
    )


def main():
    # plot_training = False
    # boxplot()

    main_plots()
    # plot_seeds()


if __name__ == "__main__":
    main()
