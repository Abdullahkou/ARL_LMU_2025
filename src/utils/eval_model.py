import csv

from gymnasium import Env

from agents.baseAgent import BaseAgent
from utils.logging import EVAL_COLS, TRAINING_STEP_COL, EpisodeLogger

import os

def eval(
    current_step: int,
    training_results: list[tuple],
    results_dir: str,
    model: BaseAgent,
    eval_env: Env,
    steps_until_next_eval: int,
    num_episodes: int,
    save_file:str,
    intermediate_results_dir: str | None = None,
    save_model: bool = True,
):
    if current_step % steps_until_next_eval != 0:
        return

    eval_ep = current_step // steps_until_next_eval
    print(f"Starting evaluation {eval_ep}")

    if eval_ep == 0:
        with open(save_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([TRAINING_STEP_COL] + EVAL_COLS)

    intermediate_results_file = None
    if intermediate_results_dir is not None:
        intermediate_results_file = f"{intermediate_results_dir}/eval_{eval_ep}.csv"

    ep_logger = EpisodeLogger(intermediate_results_file=intermediate_results_file)

    for _ in range(num_episodes):
        state, _ = eval_env.reset()

        done = False
        while not done:
            action = model.predict(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

            ep_logger.log_step(reward)

        ep_logger.log_episode()

    results = ep_logger.get_eps_mean()
    row = (eval_ep,) + results

    with open(save_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    training_results.append(results)
    print(f"Evaluation {eval_ep} Finished")

    if save_model:
        model_path = os.path.join(results_dir, f"model_eval_{eval_ep}.pth")
        model.save(model_path)
        print(f"Model saved at {model_path}")
