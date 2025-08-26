import argparse
import csv
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from tuning.training_config import (EVAL_DIR, MEAN_FILE, MODELS_DIR,
                                    RANDOM_AGENT, SEED_PREFIX, SMA_MEAN_FILE,
                                    SMA_STD_FILE, STD_FILE, TRAIN_DIR,
                                    TRAINING_RESULTS_FILE,
                                    load_training_config)
from utils.evaluation_utils import (build_heterogeneous, build_homogeneous,
                                    evaluate_across_val_seeds)
from utils.logging import EVAL_COLS, TRAINING_STEP_COL
from utils.validate_algos import choose_effective_algos

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.baseAgent import Algo, RandomAgent


class HyperAgent:
    def __init__(self, members, action_space):
        self.members = members
        self.aspace = action_space

    def predict(self, obs, deterministic=True):
        acts = [m.predict(obs, deterministic=deterministic)[0] for m in self.members]
        if isinstance(self.aspace, gym.spaces.Discrete):
            counts = np.bincount(np.asarray(acts, int), minlength=self.aspace.n)
            return int(np.argmax(counts)), {}
        else:  # Box
            agg = np.mean(np.asarray(acts, np.float32), axis=0)
            low = np.where(np.isfinite(self.aspace.low), self.aspace.low, -np.inf)
            high = np.where(np.isfinite(self.aspace.high), self.aspace.high, np.inf)
            return np.clip(agg, low, high).astype(self.aspace.dtype), {}

def eval_policy(env_id, hyperagents_list, seeds, save_dir):

    base_dir = f"{save_dir}/seeds"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    seeds_results: list[np.ndarray] = []
    env = gym.make(env_id)
    
    for seed in seeds:
        seed_result: list[np.ndarray] = []
        for hyperagent in hyperagents_list:

            # TODO: this would be currently overriding the existing seed results with intermediate ones??
            obs, _ = env.reset(seed=seed)
            done = truncated = False
            rewards = 0
            steps = 0
            while not (done or truncated):
                action, _ = hyperagent.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                rewards += reward
                steps += 1

            seed_result.append((steps, rewards))
            env.close()
        
        # Speicher seeds
        save_file_name = f"{base_dir}/{SEED_PREFIX}{seed}.csv"
        df = pd.DataFrame(seed_result, columns=[TRAINING_STEP_COL, "return"])
        df.to_csv(save_file_name)

        seeds_results.append(seed_result)

    mean = np.nanmean(seeds_results, axis=0)
    mean_df = pd.DataFrame(mean, columns=EVAL_COLS, index=range(len(hyperagents_list)))
    mean_df.index.name = TRAINING_STEP_COL
    mean_df.to_csv(f"{save_dir}/{MEAN_FILE}", index=True)

    std = np.nanstd(seeds_results, axis=0)
    std_df = pd.DataFrame(std, columns=EVAL_COLS, index=range(len(hyperagents_list)))
    std_df.index.name = TRAINING_STEP_COL
    std_df.to_csv(f"{save_dir}/{STD_FILE}", index=True)

    print("Saved as CSV")


ALGO_LOADERS = {
    "PPO": lambda path, env: PPO.load(path, env=env),
    "A2C": lambda path, env: A2C.load(path, env=env),
    "DQN": lambda path, env: DQN.load(path, env=env),
    "SAC": lambda path, env: SAC.load(path, env=env),
    "TD3": lambda path, env: TD3.load(path, env=env),
    "DDPG": lambda path, env: DDPG.load(path, env=env),
    RANDOM_AGENT: lambda path, env: RandomAgent.load(
        path, action_space=env.action_space
    ),
}

def main():
    # Create Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algos",
        help=f"Erlaubt: {', '.join([a.name for a in Algo])}",
        type=list[str],
        default=["SAC", "PPO"],
        nargs="+",
    )
    parser.add_argument(
        "--env_id",
        help="Name der Gymnasium-Umgebung, z. B. 'CartPole-v1' oder 'MountainCarContinuous-v0'",
        type=str,
        default="MountainCarContinuous-v0",  # Standard-Umgebung
    )
    parser.add_argument(
        "--val_seeds",
        help="Anzahl an seeds für die Validierung",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--eval_seeds",
        help="Anzahl an seeds für die Evaluierung",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--top_k_homogeneous",
        help="Die am besten bewerteten k Seeds pro Modell",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    # Validierung der Algorithmen
    valid_algos = []
    invalid = []

    for token in args.algos:
        token_upper = token.upper()
        # nur den Teil vor dem ersten "_" nehmen
        if "_" in token_upper:
            base_name = token_upper.split("_", 1)[0]  # nur der Teil vor dem ersten "_"
        else:
            base_name = token_upper

        if base_name in Algo.__members__:
            valid_algos.append(base_name)
        else:
            invalid.append(token)

    if invalid:
        raise SystemExit(
            f"Ungültige Algorithmen: {', '.join(invalid)}\n"
            f"Erlaubt: {', '.join([a.name for a in Algo])}"
        )

    effective_algos = choose_effective_algos(
        valid_algos, args.env_id
    )  # wirft Fehler bei Inkompatibel

    num_heterogen_models = len(set(effective_algos)) # Anzahl einzigartiger Modelle
    top_k_homogeneous = args.top_k_homogeneous
    if num_heterogen_models > 1:
        if not top_k_homogeneous == num_heterogen_models:
            raise SystemExit(
            f"top_k_homogeneous muss mit der Anzahl an einzigartigen Modellen übereinstimmen, damit es fair bleibt!\n"
            )

    env = gym.make(args.env_id)

    # Seeds aus dem Training werden geladen
    training_config = load_training_config(training_args=None)
    training_seeds = training_config.seeds

    # eval_seeds_list erstellen für die Validierung
    if not training_seeds:
        raise SystemError("Training seeds konnten nicht geladen werden! Überprüfe, ob sie existieren!")
    
    if not (args.top_k_homogeneous <= len(training_seeds)):
        raise SystemError(f"Top k seeds: {args.args.top_k_homogeneous} > Training_seeds: {training_seeds}!")

    training_fix_env_seeds = training_config.fixed_env_seeds
    validate_seeds_list = list(range(training_fix_env_seeds[-1] + 1, training_fix_env_seeds[-1] + 1 + args.val_seeds))
    
    model_name = args.algos
    validate_models_list = []
    # Modelle laden
    for i, algo in enumerate(model_name):
        path = f"results/test/{args.env_id}/{TRAIN_DIR}/{algo}/{MODELS_DIR}"

        model_dir_path = Path(path)
        if not model_dir_path:
            sys.exit(f"Fehler: Ordner '{model_dir_path}' existiert nicht.")

        for seed in training_seeds:
            if effective_algos[i] == RANDOM_AGENT:
                model_path = f"{path}/{SEED_PREFIX}{seed}"
            else:
                model_path = f"{path}/{SEED_PREFIX}{seed}.zip"

            model_path = Path(model_path)
            if not model_path.exists():
                sys.exit(f"Fehler: Datei '{model_path}' existiert nicht.")
            

            model = ALGO_LOADERS[effective_algos[i]](model_path, env)
            model_name_with_seed = f"{algo}_{SEED_PREFIX}{seed}"
            validate_models_list.append((model_name_with_seed, model))

   
    # save directory erstellen
    save_dir = f"results/test/{args.env_id}/{EVAL_DIR}/{'_'.join(sorted(model_name))}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Bewertung der Modelle
    valid_df = evaluate_across_val_seeds(validate_models_list=validate_models_list, validate_seeds_list=validate_seeds_list, env_id=args.env_id, save_dir = save_dir)

    # top k homogene Modelle und die besten k heterogenen Modelle auswählen (falls unterschiedliche Modelle vorhanden)
    homogeneous_list = build_homogeneous(valid_df=valid_df, top_k=args.top_k_homogeneous)
    print("homogeneous_ensembles: ", homogeneous_list)

    heterogeneous_list = []
    if num_heterogen_models > 1:
        heterogeneous_list = build_heterogeneous(homogeneous_list=homogeneous_list)
        print("heterogeneous ensembles: ", heterogeneous_list)
    print("\n")

    loop_ensemble_list = homogeneous_list + [heterogeneous_list] if heterogeneous_list else homogeneous_list

    training_eval_phases = training_config.eval_phases + 1
    for ensemble in loop_ensemble_list:
        hyperagents_list = []

        for eval_phase in range(training_eval_phases):
            ensemble_list = []
            model_seeds = []
            model_name_list = []

            for model_name, seed, base_model in ensemble:
                base_model.upper()

                if model_name == RANDOM_AGENT:
                    model_path = f"results/test/{args.env_id}/{TRAIN_DIR}/{model_name}/seeds/{SEED_PREFIX}{seed}/checkpoint_{eval_phase}"
                else:
                    model_path = f"results/test/{args.env_id}/{TRAIN_DIR}/{model_name}/seeds/{SEED_PREFIX}{seed}/checkpoint_{eval_phase}.zip"

                if not os.path.isfile(model_path):
                    sys.exit(f"Fehler: Datei '{model_path}' existiert nicht.")

                model = ALGO_LOADERS[base_model](model_path, env)
                ensemble_list.append(model)
                model_name_list.append(model_name)
                model_seeds.append(seed)

            Ensemble = HyperAgent(
                ensemble_list, ensemble_list[0].get_env().action_space
            )  
            env.close()

            # füge Ensemble in die Liste ein
            hyperagents_list.append(Ensemble)

        # save directory erstellen
        unique_model_name_list = set(model_name_list)
        save_dir = f"results/test/{args.env_id}/{EVAL_DIR}/{'_'.join(sorted(unique_model_name_list))}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # die besten Seeds und deren dazugehörigen Modelle in einer txt speichern
        output_path = os.path.join(save_dir, "algos_seeds.txt")

        with open(output_path, "w") as f:
            for algo, seed in zip(model_name_list, model_seeds):
                f.write(f"{algo}: {seed}\n")

        eval_seeds_list = list(range(validate_seeds_list[-1] + 1, validate_seeds_list[-1] + 1 + args.eval_seeds))

        print(
            f"Model: {unique_model_name_list} | Env: {args.env_id} | seed: {model_seeds}  | eval_seeds: {eval_seeds_list}"
        )

        eval_policy(
            env_id=args.env_id,
            hyperagents_list=hyperagents_list,
            seeds=eval_seeds_list,
            save_dir=save_dir,
        )

if __name__ == "__main__":
    main()
