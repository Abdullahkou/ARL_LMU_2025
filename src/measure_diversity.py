import csv
import os
from itertools import combinations

import gymnasium
import numpy as np

from agents.ensembles import EvalEnsemble
from tuning.training_config import (
    ACTION_DISAGREEMENT_FILE,
    CHECKPOINT_PREFIX,
    SEED_PREFIX,
    SEEDS_DIR,
    STATE_DISCREPANCY_FILE,
    load_training_config,
)
from utils.logging import (
    TOTAL_ACTION_DIS_COL,
    TOTAL_STATE_DIS_COL,
    TRAINING_STEP_COL,
    save_seed_totals,
)


def __evaluate_ensemble_checkpoint(
    ensemble: EvalEnsemble,
    envs: list[gymnasium.Env],
    num_rollout_steps=1000,
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
        envs=envs, n_steps=num_rollout_steps
    )

    ####### action disagreement ############
    act_dis = ensemble.action_disagreement(states_per_model)

    ####### state visitation divergence (pairwise Squared Maximum Mean Discrepancy) ############
    state_dis = ensemble.state_visit_divergence(states_per_model)

    return act_dis, state_dis


def evaluate_ensemble_diversity(
    config_dir: str,
    target_dir: str,
    algos: list[tuple[str, str]],
    env_id: str,
    num_rollout_steps=1000,
):
    """
    Evaluates an ensemble of models in a set of checkpoints.
    """

    config = load_training_config(dir=config_dir)

    seeds = config.seeds
    eval_env_seed = config.fixed_env_seeds[1]

    eval_phases = config.eval_phases
    steps = config.steps
    eval_schedule = steps // eval_phases

    interval_steps = [i * eval_schedule for i in range(eval_phases + 1)]

    PAIRWISE_COLS = [f"{a[0]}-{a[1]}" for a in combinations([a[0] for a in algos], 2)]

    ACTION_DIS_COLS = [TRAINING_STEP_COL, TOTAL_ACTION_DIS_COL] + PAIRWISE_COLS
    STATE_DIS_COLS = [TRAINING_STEP_COL, TOTAL_STATE_DIS_COL] + PAIRWISE_COLS

    seed_act_dis_results = list[np.ndarray]()
    seed_state_dis_results = list[np.ndarray]()
    for seed in seeds:
        seed_dir = f"{target_dir}/{SEEDS_DIR}/{SEED_PREFIX}{seed}"

        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)

        action_dis_file = f"{seed_dir}/{ACTION_DISAGREEMENT_FILE}"
        state_dis_file = f"{seed_dir}/{STATE_DISCREPANCY_FILE}"

        with open(action_dis_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ACTION_DIS_COLS)

        with open(state_dis_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(STATE_DIS_COLS)

        envs = list[gymnasium.Env]()
        for _ in algos:
            env = gymnasium.make(env_id)
            env.reset(seed=eval_env_seed)
            envs.append(env)

        action_dis_results = list[tuple[np.floating]]()
        state_dis_results = list[tuple[np.floating]]()
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
                {"algos": checkpoint_models, "env": envs[0], "seed": seed}
            )

            act_dis, state_dis = __evaluate_ensemble_checkpoint(
                ensemble,
                envs,
                num_rollout_steps=num_rollout_steps,
            )

            action_dis_result = (act_dis["total"],) + tuple(
                act_dis["pairwise"].values()
            )
            state_dis_result = (state_dis["total"],) + tuple(
                state_dis["pairwise"].values()
            )

            with open(action_dis_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow((interval_steps[checkpoint],) + action_dis_result)

            with open(state_dis_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow((interval_steps[checkpoint],) + state_dis_result)

            action_dis_results.append(action_dis_result)
            state_dis_results.append(state_dis_result)

        seed_act_dis_results.append(action_dis_results)
        seed_state_dis_results.append(state_dis_results)

    save_seed_totals(
        seeds_results=seed_act_dis_results,
        model_dir=f"{target_dir}/action_disagreement/",
        interval_steps=interval_steps,
        columns=ACTION_DIS_COLS,
    )
    save_seed_totals(
        seeds_results=seed_state_dis_results,
        model_dir=f"{target_dir}/state_discrepancy/",
        interval_steps=interval_steps,
        columns=STATE_DIS_COLS,
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
