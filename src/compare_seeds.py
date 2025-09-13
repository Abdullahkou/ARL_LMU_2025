from config.training_config import (
    SEEDS_DIR,
    TRAINING_RESULTS_FILE,
    load_training_config,
)
from plot_results import plot_results


def plot_seeds(model_dir: str, save_dir: str):
    config = load_training_config(model_dir)

    seeds = config.seeds

    seeds_dict: dict[str, tuple[str, str | None]] = {}
    for seed in seeds:
        results_file = f"{model_dir}/{SEEDS_DIR}/seed_{seed}/{TRAINING_RESULTS_FILE}"
        seeds_dict[seed] = (results_file, None)

    plot_results(agents_to_plot=seeds_dict, save_dir=save_dir)


def main():
    model_dir = "results/test/Walker2d-v5/1M/RANDOM"
    save_dir = f"{model_dir}/seed_plots"

    plot_seeds(model_dir, save_dir)


if __name__ == "__main__":
    main()
