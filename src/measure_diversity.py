from utils.diversity import evaluate_ensemble_diversity, plot_diversity


def compute_div():
    base_dir = "results/test/Walker2d-v5/1.5M/train"
    env_id = "Walker2d-v5"

    algos = [
        ("PPO", f"{base_dir}/PPO"),
        ("SAC", f"{base_dir}/SAC"),
        ("TD3", f"{base_dir}/TD3"),
    ]
    model_name = "_".join([algo[0] for algo in algos])
    target_dir = f"results/test/Walker2d-v5/1.5M/diversity/{model_name}"

    evaluate_ensemble_diversity(
        config_dir=f"{base_dir}/PPO",
        target_dir=target_dir,
        algos=algos,
        env_id=env_id,
        num_rollout_steps=10_000,
    )


def compute_plots():
    model_dir = "results/test/Walker2d-v5/1.5M/diversity/PPO_SAC_TD3"
    save_dir = f"{model_dir}/_plots"

    plot_diversity(model_dir=model_dir, save_dir=save_dir)


def main():
    # compute_div()
    compute_plots()


if __name__ == "__main__":
    main()
