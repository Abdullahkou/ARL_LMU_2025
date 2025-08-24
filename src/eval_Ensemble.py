import argparse
import csv
import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from utils.logging import (EVAL_COLS, TRAINING_STEP_COL, EpisodeLogger,
                           save_rolling_avg)
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
            high= np.where(np.isfinite(self.aspace.high), self.aspace.high, np.inf)
            return np.clip(agg, low, high).astype(self.aspace.dtype), {}



def eval_policy(env_id, model, seeds, save_dir, episodes_per_seed=5):
    seed_dir = f"{save_dir}/seeds"
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)


    seeds_results: list[np.ndarray] = []

    env = gym.make(env_id) 
    for s in seeds:
        with open(f"{seed_dir}/seeds_{s}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([TRAINING_STEP_COL] + EVAL_COLS)
        
        ep_logger = EpisodeLogger(intermediate_results_file=f"{seed_dir}/seeds_{s}.csv")

        for _ in range(episodes_per_seed):
            obs, info = env.reset(seed=s)
            done = truncated = False
            while not (done or truncated):
                a, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(a)
                ep_logger.log_step(reward)

            ep_logger.log_episode()

        seeds_results.append(ep_logger.episode_results)

        env.close()

    mean = np.nanmean(seeds_results, axis=0)
    mean_df = pd.DataFrame(mean, columns=EVAL_COLS, index=seeds)
    mean_df.index.name = TRAINING_STEP_COL
    mean_df.to_csv(f"{save_dir}/mean.csv", index=True)

    save_rolling_avg(mean_df, f"{save_dir}/mean_moving_avg.csv", window_size=3, min_periods=1)

    std = np.nanstd(seeds_results, axis=0)
    std_df = pd.DataFrame(std, columns=EVAL_COLS, index=seeds)
    std_df.index.name = TRAINING_STEP_COL
    std_df.to_csv(f"{save_dir}/std.csv", index=True)

    save_rolling_avg(std_df, f"{save_dir}/std_moving_avg.csv", window_size=3, min_periods=1)

    print("Saved as CSV")



ALGO_LOADERS = {
    "PPO":  lambda path, env: PPO.load(path, env=env),
    "A2C":  lambda path, env: A2C.load(path, env=env),
    "DQN":  lambda path, env: DQN.load(path, env=env),
    "SAC":  lambda path, env: SAC.load(path, env=env),
    "TD3":  lambda path, env: TD3.load(path, env=env),
    "DDPG": lambda path, env: DDPG.load(path, env=env),
    "RANDOM": lambda path, env: RandomAgent.load(path, action_space=env.action_space),
}


def main():
    # Create Arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--algos",
        help=f"Erlaubt: {', '.join([a.name for a in Algo])}",
        type=list[str],
        default=["PPO", "SAC", "DDPG_lr_187"],
        nargs="+",
    )
    parser.add_argument(
        "--env_id",
        help="Name der Gymnasium-Umgebung, z. B. 'CartPole-v1' oder 'MountainCarContinuous-v0'",
        type=str,
        default="MountainCarContinuous-v0",   # Standard-Umgebung
    )

    parser.add_argument(
        "--seeds",
        help="Liste von Seeds, z. B.: --seeds 0 1 2",
        type=int,  
        default=[0, 2, 1],
        nargs="+"              
    )

    parser.add_argument(
        "--eval_seeds",
        help="Liste von Seeds, z. B.: --seeds 0 1 2",
        type=int,  
        default=[0,1,2,3,4],
        nargs="+"              
    )
    parser.add_argument(
        "--eval_episodes",
        help="Anzahl der Episoden, die pro Agent ausgewertet werden sollen",
        type=int,
        default=5,
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

    effective_algos = choose_effective_algos(valid_algos, args.env_id)  # wirft Fehler bei Inkompatibel

    env = gym.make(args.env_id)

    model_name = args.algos
    seeds = args.seeds

    ensemble_list = []
    # Modelle laden
    for i, algo in enumerate(model_name):
        if effective_algos[i] == "RANDOM":
            model_path = f"src/test/{args.env_id}/train/{algo}/models/seed_{seeds[i]}"
        else:
            model_path = f"src/test/{args.env_id}/train/{algo}/models/seed_{seeds[i]}.zip"


        if not os.path.isfile(model_path):
            sys.exit(f"Fehler: Datei '{model_path}' existiert nicht.")


        model = ALGO_LOADERS[effective_algos[i]](model_path, env)
        ensemble_list.append(model)
        

    Ensemble = HyperAgent(ensemble_list, ensemble_list[0].get_env().action_space)  # in echt: mehrere kompatible Mitglieder
    env.close()


    # save directory erstellen
    save_dir = f"src/test/{args.env_id}/eval/{'_'.join(sorted(model_name))}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # die besten Seeds und deren dazugehörigen Modelle in einer txt speichern
    output_path = os.path.join(save_dir, "algos_seeds.txt")

    with open(output_path, "w") as f:
        for algo, seed in zip(model_name, seeds):
            f.write(f"{algo}: {seed}\n")



    eval_seeds = args.eval_seeds
    eval_episodes = args.eval_episodes
    

    print(
        f"Model: {model_name} | Env: {args.env_id} | seed: {args.seeds}  | eval_seeds: {eval_seeds} | eval eps: {eval_episodes}"
    )

    eval_policy(env_id=args.env_id, model=Ensemble, seeds=eval_seeds, save_dir=save_dir,episodes_per_seed=eval_episodes)



if __name__ == "__main__":
    main()