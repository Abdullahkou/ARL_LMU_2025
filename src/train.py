import argparse
import os
import traceback
from functools import partial
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from agents.baseAgent import Algo, BaseAgent
from tuning.training_config import (
    MEAN_FILE,
    MODELS_DIR,
    SEED_PREFIX,
    SEEDS_DIR,
    SMA_MEAN_FILE,
    SMA_STD_FILE,
    STD_FILE,
    TRAIN_DIR,
    TRAINING_RESULTS_FILE,
    load_training_config,
    save_training_config,
)
from utils.eval_model import eval_checkpoint
from utils.logging import EVAL_COLS, TRAINING_STEP_COL, save_rolling_avg
from utils.validate_algos import choose_effective_algos


def train(
    models: list[str],
    model_dir: str,
    env_id: str,
    device="cpu",
    training_args: dict[
        str, Optional[Any]
    ] = None,  # This is always null as far as I can tell
    use_rendering=False,
):
    training_config = load_training_config(training_args=training_args)
    seeds = training_config.seeds
    total_steps = training_config.steps
    eval_phases = training_config.eval_phases
    num_episodes = training_config.eval_episodes
    ignore_hyper = training_config.ignore_hyper_params

    # print(ignore_hyper)

    if not ignore_hyper:
        params = training_config.hyper_params
    else:
        params = {}

    model_name = f"{'_'.join([a for a in models])}"

    print(
        f"Model: {model_name} | Env: {env_id} | seed: {seeds}  | {total_steps} steps | {eval_phases} evals | {num_episodes} eval eps | fixed env seeds: {training_config.use_fixed_env_seeds} | Hyperparams: {params}"
    )
    training_config.algorithm = model_name

    save_training_config(training_config, model_dir)

    eval_schedule = total_steps // eval_phases
    eval_schedule_list = [
        eval_schedule * i for i in range(eval_phases + 1)
    ]  # für RandomAgent

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

            eval_env = gym.make(env_id, render_mode="rgb_array")
            eval_env.reset(seed=env_seeds[1])
            agent = BaseAgent(
                algos=models,
                train_env=train_env,
                device=device,
                seed=seed,
                hyper_params=params,
            )

            results_dir = f"{model_dir}/{SEEDS_DIR}/{SEED_PREFIX}{seed}"

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            training_results = []

            results_file = f"{results_dir}/{TRAINING_RESULTS_FILE}"

            training_fn = partial(
                eval_checkpoint,
                model=agent,
                eval_env=eval_env,
                steps_until_next_eval=eval_schedule,
                results_dir=results_dir,
                training_results=training_results,
                num_episodes=num_episodes,
                save_file=results_file,
                renderer=use_rendering,
                save_model=False,
                # intermediate_results_dir =
            )

            agent.learn(
                total_steps=total_steps,
                eval_fn=training_fn,
                eval_schedule_list=eval_schedule_list,
            )

            # speichere alle Modelle
            agent_model_dir = f"{model_dir}/{MODELS_DIR}"
            if not os.path.exists(agent_model_dir):
                os.makedirs(agent_model_dir)

            agent.save(
                algo_name=model_name, path=f"{agent_model_dir}/{SEED_PREFIX}{seed}"
            )

            seeds_results.append(training_results)

            train_env.close()
            train_env = None

            eval_env.close()
            eval_env = None

        mean = np.nanmean(seeds_results, axis=0)
        mean_df = pd.DataFrame(mean, columns=result_cols, index=interval_steps)
        mean_df.index.name = TRAINING_STEP_COL
        mean_df.to_csv(f"{model_dir}/{MEAN_FILE}", index=True)

        save_rolling_avg(
            mean_df, f"{model_dir}/{SMA_MEAN_FILE}", window_size=3, min_periods=1
        )

        std = np.nanstd(seeds_results, axis=0)
        std_df = pd.DataFrame(std, columns=result_cols, index=interval_steps)
        std_df.index.name = TRAINING_STEP_COL
        std_df.to_csv(f"{model_dir}/{STD_FILE}", index=True)

        save_rolling_avg(
            std_df, f"{model_dir}/{SMA_STD_FILE}", window_size=3, min_periods=1
        )

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
        default=["sac"],
        nargs="+",
    )
    parser.add_argument(
        "--save_postfix", help="save_postfix", type=str, default="", nargs="?"
    )
    parser.add_argument(
        "--env_id",
        help="Name der Gymnasium-Umgebung, z. B. 'Pendulum-v1' oder 'MountainCarContinuous-v0'",
        type=str,
        default="Pendulum-v1",  # Standard-Umgebung
    )

    args = parser.parse_args()

    device = "cpu"  # if network is simple MLP, CPU is preferred for SAC!
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # Überprüfung der Nutzereingabe
    if args.env_id not in gym.registry:
        raise SystemExit(
            f"Ungültige Umgebung: {args.env_id}.\n"
            f"Verfügbare sind z. B.: {list(gym.registry.keys())[:10]}"
        )
    env_id = args.env_id

    if len(args.algos) > 1:
        raise SystemExit("Bitte nur einen Algorithmus angeben!")

    # Validierung der Algorithmen
    model_name = args.algos[0].upper()
    valid_algos = []  # Currently, only one algo can be given but the validation assumes a list of algos with elaborate cases for single and multiple algos
    # Is this a preemptive optimization?

    if model_name not in Algo._member_names_:
        raise SystemExit(
            f"Ungültiger Algorithmus: {model_name}\n"
            f"Erlaubt: {', '.join([a.name for a in Algo])}"
        )
    else:
        valid_algos.append(model_name)

    save_postfix = args.save_postfix

    if save_postfix != "":
        save_postfix = f"_{save_postfix}"

    effective_algos = choose_effective_algos(
        valid_algos, args.env_id
    )  # wirft Fehler bei Single+inkompatibel

    base_dir = "results/test"
    model_dir = (
        f"{base_dir}/{args.env_id}/{TRAIN_DIR}/{effective_algos[0]}{save_postfix}"
    )

    # automatically loads training_config from main directory!, creates if not exists
    train(
        models=effective_algos,
        model_dir=model_dir,
        env_id=env_id,
        device=device,
    )

    # plot_results(model, base_dir=base_dir)


if __name__ == "__main__":
    main()
