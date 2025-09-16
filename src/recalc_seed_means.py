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
)
from utils.logging import TRAINING_COLS, aggregate_head_results, save_seed_totals


def recalc_seed_means(model_dir: str):
    """
    Recalculate mean results across seeds for both training and validation.
    """

    config = load_training_config(model_dir)
    seeds = config.seeds

    # ---------- VALIDATION ----------
    validation_seed_results = list[np.ndarray]()
    validation_intervals = None
    for seed in seeds:
        val_file = f"{model_dir}/{SEEDS_DIR}/seed_{seed}/{VALIDATION_RESULTS_FILE}"
        if not os.path.exists(val_file):
            continue

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
        for seed in seeds:
            train_file = f"{model_dir}/{SEEDS_DIR}/seed_{seed}/{TRAINING_RESULTS_FILE}"
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
            seeds=seeds,
        )


def main():
    model_dir = "results/test/Walker2d-v5/1.5M/RANDOM"
    recalc_seed_means(model_dir)


if __name__ == "__main__":
    main()
