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

from agents.baseAgent import Algo, BaseAgent
from tuning.training_config import load_training_config, save_training_config
from utils.eval_model import eval
from utils.logging import EVAL_COLS, TRAINING_STEP_COL, save_rolling_avg
from utils.validate_algos import check_compat, choose_effective_algos


def train(
    # model: Union[type, str],
    model: list[type],
    model_dir: str,
    env_id: str,
    device="cpu",
    training_args: dict[str, Optional[Any]] = None,
):
    training_config = load_training_config(training_args=training_args)

    seeds = training_config.seeds
    total_steps = training_config.steps
    eval_phases = training_config.eval_phases
    num_episodes = training_config.eval_episodes

    params = training_config.hyper_params

    model_name = f"{'_'.join([a for a in model])}"

    print(
        f"""Model: {model_name} | Env: {env_id} | seed: {seeds}  | {total_steps} steps | {eval_phases} evals | {num_episodes} eval eps | fixed env seeds: {training_config.use_fixed_env_seeds}"""
    )

    save_training_config(training_config, model_dir)

    eval_schedule = total_steps // eval_phases
    eval_schedule_list = [eval_schedule * i for i in range(eval_phases + 1)]   # für RandomAgent
    
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
            #model = check_compat(model, train_env.action_space)
            #print("Verwendete Algorithmen:", model)

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


            agent.learn(
                total_steps=total_steps,
                eval_fn=training_fn,
                eval_schedule_list=eval_schedule_list
            )

            # speichere alle Modelle
            agent_model_dir = f"{model_dir}/models"
            if not os.path.exists(agent_model_dir):
                os.makedirs(agent_model_dir)

            agent.save(algo_name=model_name, path=f"{agent_model_dir}/seed_{seed}")

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
        "--algos",
        help=f"Erlaubt: {', '.join([a.name for a in Algo])}",
        type=str,
        default=["Random"],
        nargs="+",
    )
    parser.add_argument("--save_postfix", help="save_postfix", type=str, default="", nargs="?")
    parser.add_argument(
        "--env_id",
        help="Name der Gymnasium-Umgebung, z. B. 'CartPole-v1' oder 'MountainCar-v0'",
        type=str,
        default="MountainCarContinuous-v0",   # Standard-Umgebung
    )

    args = parser.parse_args()

    base_dir = "src/test"

    # device = "cpu"  # if network is simple MLP, CPU is preferred for SAC!
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Validierung der Algorithmen
    valid_algos = []
    invalid = []
    for token in args.algos:
        token_upper = token.upper()
        if token_upper in Algo.__members__:
            valid_algos.append(token_upper)
        else:
            invalid.append(token)

    if invalid:
        raise SystemExit(
            f"Ungültige Algorithmen: {', '.join(invalid)}\n"
            f"Erlaubt: {', '.join([a.name for a in Algo])}"
        )
    

    save_postfix = args.save_postfix

    if save_postfix != "":
        save_postfix = f"_{save_postfix}"

    
    effective_algos = choose_effective_algos(valid_algos, args.env_id)  # wirft Fehler bei Single+inkompatibel
    model_dir = f"{base_dir}/{args.env_id}/{'_'.join([a for a in effective_algos])}{save_postfix}"

    # Überprüfung der Nutzereingabe 
    if args.env_id not in gym.registry:
        raise ValueError(f"Ungültige Umgebung: {args.env_id}. "
                         f"Verfügbare sind z. B.: {list(gym.registry.keys())[:10]}")
    env_id = args.env_id


    # automatically loads training_config from main directory!, creates if not exists
    train(
        model=effective_algos,
        model_dir=model_dir,
        env_id=env_id,
        device=device,
    )

    # plot_stats(model, base_dir=base_dir)


if __name__ == "__main__":
    main()