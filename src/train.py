import argparse
import os
import traceback
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np

from agents.dqn_agent import DQNAgent
from agents.wrapperAgent import WrapperAgent
from config.training_config import (
    HEADS_DIR,
    MODELS_DIR,
    SEEDS_DIR,
    TRAINING_RESULTS_DIR,
    TRAINING_RESULTS_FILE,
    VALIDATION_RESULTS_DIR,
    VALIDATION_RESULTS_FILE,
    Algorithm,
    load_training_config,
    make_gym,
    parse_training_args,
    save_training_config,
)
from utils.eval_model import eval_checkpoint, eval_checkpoints_heads
from utils.logging import (
    TRAINING_COLS,
    TrainLogger,
    aggregate_head_results,
    parse_steps,
    save_seed_totals,
)


def train(
    base_dir: str,
    training_args: dict[str, Any]
    | None = None,  # To override any args from the train_config
    use_rendering=False,
    save_checkpoints=False,
    save_intermediate_results=False,
    device="cpu",
    save_postfix="",
):
    training_config = load_training_config(training_args=training_args)
    seeds = training_config.seeds
    total_steps = training_config.steps
    eval_phases = training_config.eval_phases
    num_episodes = training_config.eval_episodes

    should_record_heads = (
        training_config.record_heads
        and training_config.algo_config.algorithm == Algorithm.CUSTOM_DQN
        and training_config.algo_config.hyper_params.n_q_heads > 1
    )

    model_name = training_config.algo_config.algorithm.value
    env_id = training_config.env_id.value
    print(
        f"Model: {model_name} | Env: {env_id} | seed: {seeds}  | {total_steps} steps | {eval_phases} evals | {num_episodes} eval eps"
    )

    steps_dir = parse_steps(total_steps=total_steps)
    model_dir = f"{base_dir}/{env_id}/{steps_dir}/{model_name}{save_postfix}"
    save_training_config(training_config, model_dir)

    eval_schedule = total_steps // eval_phases

    train_log_phases = training_config.train_log_phases
    log_interval = None
    if training_config.record_training:
        log_interval = total_steps // train_log_phases

    # the first training step is always evaluated/included
    eval_step_intervals = [i * eval_schedule for i in range(eval_phases + 1)]

    seeds_validation_results = list[np.ndarray]()
    seeds_training_results = list[np.ndarray]()

    eval_env: gym.Env | None = None
    eval_env_per_head: gym.vector.VectorEnv | None = None
    try:
        for seed in seeds:
            print(f"\nStarting seed {seed}")

            train_env_factory = partial(
                make_gym,
                id=env_id,
                seed=training_config.train_env_seed,
            )

            eval_env = make_gym(
                env_id, render_mode="rgb_array" if use_rendering else None
            )
            eval_env.reset(seed=training_config.eval_env_seed)

            wrapperAgent = WrapperAgent(
                train_config=training_config,
                train_env_factory=train_env_factory,
                eval_step_intervals=eval_step_intervals,
                seed=seed,
                device=device,
            )

            seed_result_dir = f"{model_dir}/{SEEDS_DIR}/seed_{seed}"

            if not os.path.exists(seed_result_dir):
                os.makedirs(seed_result_dir)

            # record heads
            if should_record_heads:
                assert isinstance(wrapperAgent.agent, DQNAgent)

                dqn_agent = wrapperAgent.agent
                num_heads = dqn_agent.hp.n_q_heads

                eval_env_per_head: gym.vector.VectorEnv = gym.make_vec(
                    id=env_id, num_envs=num_heads
                )
                eval_env_per_head.reset(seed=seed)

                heads_dirs = [
                    f"{seed_result_dir}/{HEADS_DIR}/h{head_idx}"
                    for head_idx in range(num_heads)
                ]

                validation_results_files = [
                    f"{head_dir}/{VALIDATION_RESULTS_FILE}" for head_dir in heads_dirs
                ]

                eval_heads_fn = partial(
                    eval_checkpoints_heads,
                    dqnAgent=dqn_agent,
                    eval_env_per_head=eval_env_per_head,
                    num_episodes=num_episodes,
                    eval_schedule=eval_schedule,
                    save_files=validation_results_files,
                    intermediate_results_dirs=heads_dirs
                    if save_intermediate_results
                    else None,
                )

                dqn_agent.set_indiviudal_head_eval_fn(eval_heads_fn=eval_heads_fn)

            validation_results_files = f"{seed_result_dir}/{VALIDATION_RESULTS_FILE}"

            validation_results = list[tuple]()

            eval_fn = partial(
                eval_checkpoint,
                model=wrapperAgent,
                eval_env=eval_env,
                validation_results=validation_results,
                results_dir=seed_result_dir,
                num_episodes=num_episodes,
                eval_schedule=eval_schedule,
                save_file=validation_results_files,
                use_rendering=use_rendering,
                save_model=save_checkpoints,
                intermediate_results_dir=seed_result_dir
                if save_intermediate_results
                else None,
            )

            training_results_file = f"{seed_result_dir}/{TRAINING_RESULTS_FILE}"

            train_logger = None
            if training_config.record_training:
                log_interval = total_steps // train_log_phases
                train_logger = TrainLogger(
                    training_results_file, log_interval=log_interval
                )

            wrapperAgent.learn(
                total_steps=total_steps, eval_fn=eval_fn, train_logger=train_logger
            )

            agent_model_dir = f"{model_dir}/{MODELS_DIR}"
            if not os.path.exists(agent_model_dir):
                os.makedirs(agent_model_dir)

            wrapperAgent.save(path=f"{agent_model_dir}/seed_{seed}")

            if train_logger is not None:
                trainig_results = train_logger.get_training_results()
                seeds_training_results.append(trainig_results)

            seeds_validation_results.append(validation_results)

            eval_env.close()
            eval_env = None

            if eval_env_per_head is not None:
                eval_env_per_head.close()
                eval_env_per_head = None

        save_seed_totals(
            seeds_results=seeds_validation_results,
            dir=f"{model_dir}/{VALIDATION_RESULTS_DIR}",
            step_intervals=eval_step_intervals,
        )

        if len(seeds_training_results) > 0:
            training_step_intervals = [
                i * log_interval for i in range(1, train_log_phases + 1)
            ]

            save_seed_totals(
                seeds_results=seeds_training_results,
                dir=f"{model_dir}/{TRAINING_RESULTS_DIR}",
                step_intervals=training_step_intervals,
                columns=TRAINING_COLS,
            )

        # head results
        if should_record_heads:
            aggregate_head_results(
                agent_dir=model_dir,
                num_heads=training_config.algo_config.hyper_params.n_q_heads,
                seeds=seeds,
            )

    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
    if eval_env is not None:
        eval_env.close()
    if eval_env_per_head is not None:
        eval_env_per_head.close()


def main():
    device = "cpu"  # if network is simple MLP, CPU is generally preferred!
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_postfix", type=str)

    args, extras = parser.parse_known_args()

    save_postfix = args.save_postfix or ""
    if save_postfix != "":
        save_postfix = f"_{save_postfix}"

    training_args = parse_training_args(extras=extras)

    base_dir = "results"

    # automatically loads training_config from main directory!, creates if not exists
    train(
        base_dir=base_dir,
        training_args=training_args,
        use_rendering=False,
        save_checkpoints=False,
        save_intermediate_results=False,
        device=device,
        save_postfix=save_postfix,
    )


if __name__ == "__main__":
    main()
