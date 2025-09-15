import csv
import os
from pathlib import Path

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv

from agents.dqn_agent import DQNAgent
from agents.wrapperAgent import WrapperAgent
from config.training_config import CHECKPOINT_PREFIX, INTERMEDIATE_RESULTS_PREFIX
from utils.logging import EVAL_COLS, TRAINING_STEP_COL, EpisodeLogger
from utils.rendering import Renderer


def eval_checkpoints_heads(
    current_step: int,
    dqnAgent: DQNAgent,
    eval_env_per_head: VectorEnv,
    eval_schedule: int,
    num_episodes: int,
    save_files: list[str],
    intermediate_results_dirs: str | None = None,
):
    if current_step % eval_schedule != 0:
        return

    eval_ep = current_step // eval_schedule

    print(f"Starting evaluation {eval_ep}")

    num_heads = len(save_files)
    assert eval_env_per_head.num_envs == num_heads, (
        f"VectorEnv has {eval_env_per_head.num_envs} envs but {num_heads} save files were given"
    )

    if eval_ep == 0:
        for file in save_files:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            with open(file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([TRAINING_STEP_COL] + EVAL_COLS)

    intermediate_results_files = None
    if intermediate_results_dirs is not None:
        intermediate_results_files = (
            f"{dir}/{INTERMEDIATE_RESULTS_PREFIX}{eval_ep}.csv"
            for dir in intermediate_results_dirs
        )

    ep_loggers = list[EpisodeLogger]()
    for head_idx in range(num_heads):
        ep_logger = EpisodeLogger(
            intermediate_results_file=intermediate_results_files[head_idx]
            if intermediate_results_dirs is not None
            else None
        )
        ep_loggers.append(ep_logger)

    # Run episodes in parallel
    for ep in range(num_episodes):
        states, _ = eval_env_per_head.reset()
        dones = np.array([False] * num_heads)

        while not np.all(dones):
            actions = dqnAgent.get_heads_predictions(
                states
            )  # shape (num_heads, action_dim)
            states, rewards, terminated, truncated, _ = eval_env_per_head.step(actions)
            step_dones = np.logical_or(terminated, truncated)

            for head_idx in range(num_heads):
                if not dones[head_idx]:  # log ep end
                    ep_loggers[head_idx].log_step(rewards[head_idx])

            dones |= step_dones

        for head_idx in range(num_heads):
            ep_loggers[head_idx].log_episode()

    # Write results per head
    for head_idx in range(num_heads):
        results = ep_loggers[head_idx].get_eps_mean()
        row = (current_step,) + results
        with open(save_files[head_idx], "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    print(f"Evaluation {eval_ep} Finished")


def eval_checkpoint(
    current_step: int,
    validation_results: list[tuple],
    results_dir: str,
    model: WrapperAgent,
    eval_env: Env,
    eval_schedule: int,
    num_episodes: int,
    save_file: str,
    intermediate_results_dir: str | None = None,
    save_model=False,
    use_rendering=False,
):
    if current_step % eval_schedule != 0:
        return

    eval_ep = current_step // eval_schedule

    results = eval(
        model=model,
        eval_env=eval_env,
        num_episodes=num_episodes,
        save_file=save_file,
        intermediate_results_dir=intermediate_results_dir,
        use_rendering=use_rendering,
        eval_ep=eval_ep,
        current_step=current_step,
    )

    validation_results.append(results)

    if save_model:
        model_path = os.path.join(results_dir, f"{CHECKPOINT_PREFIX}{eval_ep}")
        model.save(model_path)


def eval(
    model: WrapperAgent,
    eval_env: Env,
    num_episodes: int,
    save_file: str,
    intermediate_results_dir: str | None = None,
    use_rendering=False,
    eval_ep: int = 0,
    current_step: int = 0,
    evaluating_checkpoints=False,
):
    print(f"Starting evaluation {eval_ep}")

    save_path = Path(save_file)
    if eval_ep == 0 and (
        not evaluating_checkpoints
        or (evaluating_checkpoints and not save_path.exists())
    ):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([TRAINING_STEP_COL] + EVAL_COLS)

    intermediate_results_file = None
    if intermediate_results_dir is not None:
        intermediate_results_file = (
            f"{intermediate_results_dir}/{INTERMEDIATE_RESULTS_PREFIX}{eval_ep}.csv"
        )

    ep_logger = EpisodeLogger(intermediate_results_file=intermediate_results_file)

    renderer = None
    if use_rendering:
        renderer = Renderer(env=eval_env)

    for i in range(num_episodes):
        state, _ = eval_env.reset()

        done = False
        while not done:
            action, _ = model.predict(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_logger.log_step(reward)

            if renderer is not None and renderer._enabled:
                renderer.add_reward(reward)
                esc, space = renderer.pump_events()
                if esc:
                    renderer.close()
                elif space:
                    renderer.toggle_speed()
                else:
                    renderer.render(episode=i, eval_ep=eval_ep)

        ep_logger.log_episode()

    results = ep_logger.get_eps_mean()
    row = (current_step,) + results

    with open(save_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"Evaluation {eval_ep} Finished")
    return results
