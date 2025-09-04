import numpy as np
import pandas as pd

from tuning.training_config import (
    SEED_PREFIX,
    SEEDS_DIR,
    TRAINING_RESULTS_FILE,
    load_training_config,
)
from utils.logging import save_seed_totals


def recalc_seed_means(model_dir):
    config = load_training_config(model_dir)
    seeds = config.seeds

    eval_interval_steps = None
    seed_results: list[np.ndarray] = []
    for seed in seeds:
        train_results_file = (
            f"{model_dir}/{SEEDS_DIR}/{SEED_PREFIX}{seed}/{TRAINING_RESULTS_FILE}"
        )

        df = pd.read_csv(train_results_file, index_col=0)

        # init index column array
        if eval_interval_steps is None:
            eval_interval_steps = df.index.values

        vals = df.values
        seed_results.append(vals)

    assert eval_interval_steps is not None

    save_seed_totals(
        seeds_results=seed_results,
        model_dir=model_dir,
        interval_steps=eval_interval_steps,
    )


def main():
    model_dir = "results/test/Walker2d-v5/1.5M/eval/checkpoints_PPO_TD3_weighted"

    recalc_seed_means(model_dir)


if __name__ == "__main__":
    main()
