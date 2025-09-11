import csv
import os
from pathlib import Path

from gymnasium import Env

from agents.baseAgent import WrapperAgent
from config.training_config import CHECKPOINT_PREFIX, INTERMEDIATE_RESULTS_PREFIX
from utils.logging import EVAL_COLS, TRAINING_STEP_COL, EpisodeLogger
from utils.rendering import Renderer


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
        model.save(model.agent.__class__.__name__, model_path)


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
        print("created")
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
            action = model.predict(state)
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
