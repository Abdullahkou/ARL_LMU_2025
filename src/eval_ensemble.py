import os

import gymnasium

from agents.ensembles import EvalEnsemble
from tuning.training_config import (
    MODELS_DIR,
    TRAINING_RESULTS_FILE,
    load_training_config,
)
from utils.eval_model import eval


def evaluate_ensemble(
    base_dir: str,
    env_id: str,
    algos: list[tuple[str, str]],
    save_location: str,
    num_episodes: int = 20,
    use_rendering: bool = False,
):
    config_dir = f"{base_dir}/{algos[0][0]}"
    config = load_training_config(dir=config_dir)
    eval_env_seed = config.fixed_env_seeds[1]

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    save_file = f"{save_location}/{TRAINING_RESULTS_FILE}"
    eval_env = gymnasium.make(env_id, render_mode="rgb_array")
    eval_env.reset(seed=eval_env_seed)

    ensemble = EvalEnsemble({"algos": algos, "env": eval_env})

    eval(
        model=ensemble,
        eval_env=eval_env,
        num_episodes=num_episodes,
        save_file=save_file,
        use_rendering=use_rendering,
    )


def main():
    base_dir = "results/test/Walker2d-v5/train"

    env_id = "Walker2d-v5"
    algos = [
        ("PPO", f"{base_dir}/PPO/{MODELS_DIR}"),
        ("SAC", f"{base_dir}/SAC/{MODELS_DIR}"),
        ("TD3", f"{base_dir}/TD3/{MODELS_DIR}"),
    ]

    model_name = "_".join([algo[0] for algo in algos])

    save_location = f"results/test/Walker2d-v5/eval/{model_name}"

    evaluate_ensemble(
        base_dir=base_dir,
        env_id=env_id,
        algos=algos,
        save_location=save_location,
        num_episodes=100,
        use_rendering=False,
    )


if __name__ == "__main__":
    main()
