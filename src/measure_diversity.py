import csv
import os

import gymnasium
import numpy as np

from agents.ensembles import EvalEnsemble
from tuning.training_config import (
    CHECKPOINT_PREFIX,
    ENSEMBLE_DIVERSITY_FILE,
    SEED_PREFIX,
    SEEDS_DIR,
    load_training_config,
)
from utils.logging import DIVERSITY_COLS, TRAINING_STEP_COL, save_seed_totals


def __evaluate_ensemble_checkpoint(
    ensemble: EvalEnsemble,
    env: gymnasium.Env,
    num_rollout_steps=1000,
    max_states: int = None,
):
    """
    Evaluates diversity of an ensemble of models on given environment.
    Diversity metrics: action disagreement, state visitation divergence (pairwise MMD^2)

    :param models: list of RL algorithms
    :param env: environment to evaluate on
    :param best_model_index: index of the best model in the models list (for eval disagreement)
    :param n_eval_episodes: number of episodes to roll out per model
    :param max_states: maximum number of states to use per model (subsampled if needed)
    :return: (action disagreement, (mmd_mean, mmd_min, mmd_max))
    """
    # Collect rollouts per model
    states_per_model = ensemble.collect_individual_rollouts(
        env, n_steps=num_rollout_steps, max_states=max_states
    )

    ####### action disagreement ############
    total_states = np.vstack(states_per_model)
    act_dis = ensemble.action_disagreement(total_states)

    ####### state visitation divergence (pairwise Squared Maximum Mean Discrepancy) ############
    mmd_mean, mmd_min, mmd_max = ensemble.state_visit_divergence(states_per_model)

    return act_dis, (mmd_mean, mmd_min, mmd_max)


def evaluate_ensemble_diversity(
    config_dir: str,
    target_dir: str,
    algos: list[tuple[str, str]],
    env_id: str,
    num_rollout_steps=1000,
    max_states=None,
):
    """
    Evaluates an ensemble of models in a set of checkpoints.
    """

    config = load_training_config(dir=config_dir)

    seeds = config.seeds
    eval_env_seed = config.fixed_env_seeds[1]

    eval_env = gymnasium.make(env_id, render_mode="rgb_array")
    eval_env.reset(seed=eval_env_seed)

    eval_phases = config.eval_phases
    steps = config.steps
    eval_schedule = steps // eval_phases

    interval_steps = [i * eval_schedule for i in range(eval_phases + 1)]

    seeds_results: list[np.ndarray] = []
    for seed in seeds:
        seed_dir = f"{target_dir}/{SEEDS_DIR}/{SEED_PREFIX}{seed}"

        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)

        save_file = f"{seed_dir}/{ENSEMBLE_DIVERSITY_FILE}"

        with open(save_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([TRAINING_STEP_COL] + DIVERSITY_COLS)

        results = []
        for checkpoint in range(eval_phases + 1):
            print(f"Evaluating ensemble for seed {seed} checkpoint {checkpoint}")
            checkpoint_models = [
                (
                    name,
                    f"{dir}/{SEEDS_DIR}/{SEED_PREFIX}{seed}/{CHECKPOINT_PREFIX}{checkpoint}.zip",
                )
                for name, dir in algos
            ]

            ensemble = EvalEnsemble(
                {"algos": checkpoint_models, "env": eval_env, "seed": seed}
            )

            # Evaluate ensemble
            act_dis, (mmd_mean, mmd_min, mmd_max) = __evaluate_ensemble_checkpoint(
                ensemble,
                eval_env,
                num_rollout_steps=num_rollout_steps,
                max_states=max_states,
            )

            result = (act_dis, mmd_mean, mmd_min, mmd_max)

            with open(save_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result)

            results.append(result)

        seeds_results.append(results)

    save_seed_totals(
        seeds_results=seeds_results,
        model_dir=target_dir,
        interval_steps=interval_steps,
        columns=DIVERSITY_COLS,
    )


def main():
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


if __name__ == "__main__":
    main()
