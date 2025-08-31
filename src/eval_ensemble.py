import os

import gymnasium

from agents.ensembles import EvalEnsemble
from tuning.training_config import TRAINING_RESULTS_FILE


def evaluate_ensemble(
    ensemble: EvalEnsemble,
    env_id: str,
    save_location: str,
    num_episodes: int = 20,
    use_rendering: bool = False,
):
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    save_file = f"{save_location}/{TRAINING_RESULTS_FILE}"
    eval_env = gymnasium.make(env_id, render_mode="rgb_array")

    eval(
        model=ensemble,
        eval_env=eval_env,
        num_episodes=num_episodes,
        save_file=save_file,
        use_rendering=use_rendering,
    )


def main():
    env_id = "Pendulum-v1"
    save_location = "results/test/Pendulum-v1/eval"
    config = {
        "algos": [
            ("SAC", "results/test/Pendulum-v1/train/SAC/models"),
            ("PPO", "results/test/Pendulum-v1/train/PPO/models"),
            ("TD3", "results/test/Pendulum-v1/train/TD3/models"),
        ],
        "device": "cpu",
        "seed": 69,
        "training_results_dir": "results/test/Pendulum-v1",
        "is_discrete": False,
    }
    ensemble = EvalEnsemble(config)
    evaluate_ensemble(
        ensemble, env_id, save_location, num_episodes=100, use_rendering=False
    )


if __name__ == "__main__":
    main()
