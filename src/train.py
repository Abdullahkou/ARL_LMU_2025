import argparse
import csv
import os
import traceback
from functools import partial
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DDPG, PPO, SAC

from agents.baseAgent import BaseAgent
from tuning.training_config import load_training_config, save_training_config
from utils.eval_model import eval
from utils.logging import (EVAL_COLS, TRAINING_STEP_COL,  # moving average
                           save_rolling_avg)


def train(
    # model: Union[type, str],
    model: type,
    model_dir: str,
    env_id: str,
    device="cpu",
    worker_id=0,
    speed=16.0,
    training_args: dict[str, Optional[Any]] = None,
    record_training=False,
):
    training_config = load_training_config(training_args=training_args)

    seeds = training_config.seeds
    total_steps = training_config.steps
    eval_phases = training_config.eval_phases
    num_episodes = training_config.eval_episodes

    save_freq = training_config.save_freq if not isinstance(model, str) else 0
    params = training_config.hyper_params

    model_name = model.__name__ if isinstance(model, type) else model

    print(
        f"""Model: {model_name} | Env: {env_id} | seed: {seeds}  | {total_steps} steps | {eval_phases} evals | {num_episodes} eval eps | fixed env seeds: {training_config.use_fixed_env_seeds}"""
    )

    save_training_config(training_config, model_dir)

    eval_schedule = total_steps // eval_phases
    # save_schedule = 0
    # if save_freq != 0:
    #     save_phases = eval_phases // save_freq
    #     save_schedule = total_steps // save_phases if eval_phases > save_freq else eval_schedule

    result_cols = EVAL_COLS
    interval_steps = [
        i * eval_schedule for i in range(0, eval_phases + 1)
    ]  # the first training step is always evaluated/included    seeds_results: list[np.ndarray] = []

    seeds_results: list[np.ndarray] = []

    train_env = None
    eval_env = None
    try:
        for seed in seeds:
            print(f"Starting seed {seed}")
            env_seeds = (seed, seed + 1)

            if training_config.use_fixed_env_seeds:
                env_seeds = training_config.fixed_env_seeds

            train_env = gym.make(env_id)

            eval_env = gym.make(env_id)
            eval_env.reset(seed=env_seeds[1])

            agent = BaseAgent(algos=model, train_env=train_env, device=device, seed=env_seeds[0], hyper_params=params)
            seed_dir = f"{model_dir}/seed_{seed}"

            if not os.path.exists(seed_dir):
                os.makedirs(seed_dir)

            with open(f"{seed_dir}/training_results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([TRAINING_STEP_COL] + EVAL_COLS)

            training_results = []

            results_dir = seed_dir  # set None to disable

            training_fn = partial(
                eval,
                model=agent,
                eval_env=eval_env,
                steps_until_next_eval=eval_schedule,
                results_dir=results_dir,
                training_results=training_results,
                num_episodes=num_episodes,
                # intermediate_results_dir = 
            )

            train_logger = None
            # if record_training:
            #     train_logger = TrainLogger(
            #         f"{seed_dir}/training", tensorboard=True, log_interval=10
            #     )

            # agent.learn(
            #     total_timesteps=total_steps,
            #     eval_fn=training_fn,
            #     eval_schedule=eval_schedule,
            #     evals=eval_phases,
            #     train_logger=train_logger,
            # )
            agent.learn(
                total_steps=total_steps,
                eval_fn=training_fn
            )

            # if (
            #     results_dir is None
            # ):  # disable bc saving now at every eval iteration instead of at the end
            #     train_df = pd.DataFrame(training_results, columns=result_cols, index=interval_steps)
            #     train_df.index.name = "Training Step"
            #     train_df.to_csv(f"{seed_dir}/training_results.csv", index=True)

            # not saving for RandomAgent and MysteryModel
            # if isinstance(model, type):
            #     agent.save_policy(seed_dir)

            seeds_results.append(training_results)

            train_env.close()
            train_env = None

            eval_env.close()
            eval_env = None

        mean = np.nanmean(seeds_results, axis=0)
        mean_df = pd.DataFrame(mean, columns=result_cols, index=interval_steps)
        mean_df.index.name = TRAINING_STEP_COL
        mean_df.to_csv(f"{model_dir}/mean.csv", index=True)

        save_rolling_avg(mean_df, f"{model_dir}/mean_moving_avg.csv", window_size=3, min_periods=1)

        std = np.nanstd(seeds_results, axis=0)
        std_df = pd.DataFrame(std, columns=result_cols, index=interval_steps)
        std_df.index.name = TRAINING_STEP_COL
        std_df.to_csv(f"{model_dir}/std.csv", index=True)

        save_rolling_avg(std_df, f"{model_dir}/std_moving_avg.csv", window_size=3, min_periods=1)

    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
    if eval_env is not None:
        eval_env.close()
    if train_env is not None:
        train_env.close()


def main():
    # Create Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="PPO | SAC | A2C | DQN | TD3 | DDPG | Random",
        type=str,
        default="PPO",
        nargs="?",
    )
    parser.add_argument(
        "--worker_id", help="Set Worker-ID for Parallelism", type=int, default=0, nargs="?"
    )
    parser.add_argument("--save_postfix", help="save_postfix", type=str, default="", nargs="?")
    parser.add_argument(
        "--env_id",
        help="Name der Gymnasium-Umgebung, z. B. 'CartPole-v1' oder 'MountainCar-v0'",
        type=str,
        default="CartPole-v1",   # Standard-Umgebung
    )

    args = parser.parse_args()

    # game_path_name = "GAME_PATH"
    # base_dir = "python/results/sparse_reward/million_steps"
    base_dir = "src/test/metrics"

    # device = "cpu"  # if network is simple MLP, CPU is preferred for SAC!
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "DDPG":
        model = DDPG
    elif args.model == "SAC":  # cannot parse type directly
        model = SAC
    elif args.model == "PPO":  # cannot parse type directly
        model = PPO
    elif args.model == "Random":
        model = args.model
    else:
        raise ValueError(f"Model {args.model} not supported")

    save_postfix = args.save_postfix

    if save_postfix != "":
        save_postfix = f"_{save_postfix}"

    model_dir = f"{base_dir}/{args.model}{save_postfix}"

    # Überprüfung der Nutzereingabe 
    if args.env_id not in gym.registry:
        raise ValueError(f"Ungültige Umgebung: {args.env_id}. "
                         f"Verfügbare sind z. B.: {list(gym.registry.keys())[:10]}")
    env_id = args.env_id


    # automatically loads training_config from main directory!, creates if not exists
    train(
        model=model,
        model_dir=model_dir,
        env_id=env_id,
        device=device,
        worker_id=args.worker_id,
    )

    # plot_stats(model, base_dir=base_dir)

    # export_ONNX(
    #     model,
    #     model_dir=model_dir,
    #     policy_dir=f"{model_dir}/seed_1",  # change this if not applicable
    #     device=device,
    #     game_path_name=game_path_name,
    # )


if __name__ == "__main__":
    main()