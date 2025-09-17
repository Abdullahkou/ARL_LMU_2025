import os

import numpy as np
import pandas as pd

from config.training_config import (
    SEEDS_DIR,
    TRAINING_RESULTS_DIR,
    TRAINING_RESULTS_FILE,
    VALIDATION_RESULTS_DIR,
    VALIDATION_RESULTS_FILE,
    Algorithm,
    load_training_config,
    save_training_config,
)
from utils.logging import TRAINING_COLS, aggregate_head_results, save_seed_totals


def recalc_seed_means(model_dir: str, seeds_to_collect: list[int]):
    """
    Recalculate mean results across seeds from individuals seed results.
    """

    config = load_training_config(model_dir)
    config.seeds = seeds_to_collect
    save_training_config(config=config, dir=model_dir)

    seeds_dir = f"{model_dir}/{SEEDS_DIR}"

    # ---------- VALIDATION ----------
    validation_seed_results = list[np.ndarray]()
    validation_intervals = None
    for seed in seeds_to_collect:
        val_file = f"{seeds_dir}/seed_{seed}/{VALIDATION_RESULTS_FILE}"

        assert os.path.exists(val_file), (
            f"couldnt locate {val_file} for seeds {seeds_to_collect}"
        )

        df = pd.read_csv(val_file, index_col=0)

        if validation_intervals is None:
            validation_intervals = df.index.values

        validation_seed_results.append(df.values)

    if len(validation_seed_results) > 0:
        save_seed_totals(
            seeds_results=validation_seed_results,
            dir=f"{model_dir}/{VALIDATION_RESULTS_DIR}",
            step_intervals=validation_intervals,
        )

    # ---------- TRAINING ----------
    if config.record_training:
        training_seed_results = list[np.ndarray]()
        training_intervals = None
        for seed in seeds_to_collect:
            train_file = f"{seeds_dir}/seed_{seed}/{TRAINING_RESULTS_FILE}"
            if not os.path.exists(train_file):
                continue

            df = pd.read_csv(train_file, index_col=0)

            if training_intervals is None:
                training_intervals = df.index.values

            training_seed_results.append(df.values)

        if len(training_seed_results) > 0:
            save_seed_totals(
                seeds_results=training_seed_results,
                dir=f"{model_dir}/{TRAINING_RESULTS_DIR}",
                step_intervals=training_intervals,
                columns=TRAINING_COLS,
            )

    # ---------- HEADS ----------
    # only aggregate if the run actually used multi-head DQN
    if (
        config.record_heads
        and config.algo_config.algorithm.name == Algorithm.CUSTOM_DQN
        and config.algo_config.hyper_params.n_q_heads > 1
    ):
        aggregate_head_results(
            agent_dir=model_dir,
            num_heads=config.algo_config.hyper_params.n_q_heads,
            seeds=seeds_to_collect,
        )


def main():
    model_dir = "results/test/LunarLander-v3/1.0K/RANDOM_0"

    seeds_to_collect = [0, 1]
    recalc_seed_means(model_dir, seeds_to_collect=seeds_to_collect)


if __name__ == "__main__":
    main()
