import os

import gymnasium
import numpy as np

from agents.ensembles import EvalEnsemble
from tuning.training_config import (
    MODELS_DIR,
    SEED_PREFIX,
    TRAINING_RESULTS_FILE,
    load_training_config,
)
from utils.eval_model import eval, eval_checkpoint
from utils.logging import save_seed_totals


def evaluate_ensemble(
    base_dir: str,
    env_id: str,
    algos: list[tuple[str, str]],
    save_location: str,
    num_episodes=20,
    seeds=[0, 1, 2],
    use_rendering: bool = False,
):
    config_dir = f"{base_dir}/{algos[0][0]}"
    config = load_training_config(dir=config_dir)
    eval_env_seed = config.fixed_env_seeds[1]

    seeds_results: list[np.ndarray] = []

    for seed in seeds:
        print(f"Starting seed {seed}")
        seed_dir = f"{save_location}/{SEED_PREFIX}{seed}"

        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)

        save_file = f"{seed_dir}/{TRAINING_RESULTS_FILE}"

        eval_env = gymnasium.make(env_id, render_mode="rgb_array")
        eval_env.reset(seed=eval_env_seed)

        ensemble = EvalEnsemble({"algos": algos, "env": eval_env, "seed": seed})

        results = eval(
            model=ensemble,
            eval_env=eval_env,
            num_episodes=num_episodes,
            save_file=save_file,
            use_rendering=use_rendering,
        )
        seeds_results.append([results])

    save_seed_totals(
        seeds_results=seeds_results,
        model_dir=save_location,
        interval_steps=[0],
        save_rolling=False,
    )


def evaluate_checkpoints():
    base_dir = "results/test/Walker2d-v5/1.5M/train"
    num_checkpoints = 21 # there are actually 21 checkpoints with 0
    config = load_training_config(dir=f"{base_dir}/PPO")
    eval_env_seed = config.fixed_env_seeds[1]
    seeds = [0, 1, 2]

    eval_env = gymnasium.make("Walker2d-v5", render_mode="rgb_array")

    for seed in seeds:
        seed_results: list[np.ndarray] = []
        for i in range(num_checkpoints):
            print(f"Evaluating ensemble for seed {seed} checkpoint {i}")
            eval_env.reset(seed=eval_env_seed)
            algos = [
                ("PPO", f"{base_dir}/PPO/seeds/seed_{seed}/checkpoint_{i}.zip"),
                ("SAC", f"{base_dir}/SAC/seeds/seed_{seed}/checkpoint_{i}.zip"),
                ("TD3", f"{base_dir}/TD3/seeds/seed_{seed}/checkpoint_{i}.zip"),
            ]
            ensemble = EvalEnsemble({"algos": algos, "env": eval_env, "seed": seed})
            results = eval(
                model=ensemble,
                eval_env=eval_env,
                num_episodes=20,
                save_file=f"results/test/Walker2d-v5/1.5M/eval/checkpoints_{'_'.join([algo_name for algo_name, _ in algos])}_critic/seeds/seed_{seed}/training_results.csv",
                use_rendering=False,
                evaluating_checkpoints=True,
                current_step=(config.steps // 20) * i,
            )
            seed_results.append([results])
        save_seed_totals(
            seeds_results=seed_results,
            model_dir=f"results/test/Walker2d-v5/1.5M/eval/checkpoints_{'_'.join([algo_name for algo_name, _ in algos])}_critic",
            interval_steps=[0],
            save_rolling=True,
        )


def main():
    evaluate_checkpoints()
    #base_dir = "results/test/Walker2d-v5/1M/train"
#
    #env_id = "Walker2d-v5"
    #algos = [
    #    ("PPO", f"{base_dir}/PPO/{MODELS_DIR}"),
    #    ("SAC", f"{base_dir}/SAC/{MODELS_DIR}"),
    #    ("TD3", f"{base_dir}/TD3/{MODELS_DIR}"),
    #]
#
    #model_name = "_".join([algo[0] for algo in algos])
#
    #save_location = f"results/test/Walker2d-v5/1M/eval/{model_name}"
    #num_episodes = 40
    #seeds = [0, 1, 2]
#
    #evaluate_ensemble(
    #    base_dir=base_dir,
    #    env_id=env_id,
    #    algos=algos,
    #    save_location=save_location,
    #    num_episodes=num_episodes,
    #    seeds=seeds,
    #    use_rendering=False,
    #)


if __name__ == "__main__":
    main()
