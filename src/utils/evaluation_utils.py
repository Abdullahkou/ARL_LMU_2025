from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd


def evaluate_across_val_seeds(
    env_id: str,
    validate_models_list: list[(str, type)],
    validate_seeds_list: list[int],
    save_dir: str,
):
    env = gym.make(env_id)

    save_file_name = f"{save_dir}/models_record.csv"
    save_file_name = Path(save_file_name)

    if save_file_name.exists():
        df = pd.read_csv(save_file_name)
    else:
        df = pd.DataFrame(columns=["algo_name", "val_mean"])

    print("Starte Bewertung der Modelle!")
    for model_name, model in validate_models_list:
        seeds_result = []
        for seed in validate_seeds_list:
            obs, _ = env.reset(seed=seed)
            done = truncated = False
            rewards = 0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                rewards += reward

            seeds_result.append(rewards)
            env.close()

        val_mean = np.mean(seeds_result)

        # Update, falls Algo schon existiert
        if model_name in df["algo_name"].values:
            df.loc[df["algo_name"] == model_name, "val_mean"] = val_mean
        else:
            # df = pd.concat([df, pd.DataFrame([{"algo_name": model_name, "val_mean": val_mean}])],
            #             ignore_index=True)
            df.loc[len(df)] = {"algo_name": model_name, "val_mean": val_mean}

    # Überschreiben (immer aktuelle Version)
    df.to_csv(save_file_name, index=False)
    return df


def build_homogeneous(valid_df: type, top_k):
    # Algo normalisieren (alles vor dem ersten "_")
    valid_df["algo_base"] = valid_df["algo_name"].str.split("_").str[0]

    # pro Algo Top-k
    top_k = (
        valid_df.sort_values("val_mean", ascending=False)
        .groupby("algo_base", group_keys=False)
        .head(top_k)
    )

    print("\n############################################################")
    print("Die besten k seeds für jedes Modell:")
    print(top_k)
    print("############################################################\n")

    homogeneous_list = []
    for algo, group in top_k.groupby("algo_base"):
        entries = []
        for _, row in group.iterrows():
            parts = row["algo_name"].split("_seed_")
            algo_name = parts[0]
            seed = int(parts[1]) if len(parts) > 1 else None
            entries.append((algo_name, seed, algo))
        homogeneous_list.append(entries)
    return homogeneous_list


def build_heterogeneous(homogeneous_list: type):
    heterogeneous_list = []

    for group in homogeneous_list:
        if not group:
            continue
        heterogeneous_list.append(group[0])
    return heterogeneous_list
