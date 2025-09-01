import gymnasium

from agents.ensembles import EvalEnsemble
from tuning.training_config import MODELS_DIR, load_training_config
from utils.rendering import Renderer


def render_eval(
    base_dir: str, env_id: str, algos: list[tuple[str, str]], num_episodes: int = 10
):
    config_dir = f"{base_dir}/{algos[0][0]}"
    config = load_training_config(dir=config_dir)
    eval_env_seed = config.fixed_env_seeds[1]

    render_env = gymnasium.make(env_id, render_mode="rgb_array")
    render_env.reset(seed=eval_env_seed)
    renderer = Renderer(env=render_env)

    model = EvalEnsemble({"algos": algos, "env": render_env})

    for i in range(num_episodes):
        state, _ = render_env.reset()

        done = False
        while not done:
            action = model.predict(state)
            state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated

            if renderer is not None and renderer._enabled:
                renderer.add_reward(reward)
                esc, space = renderer.pump_events()
                if esc:
                    renderer.close()
                elif space:
                    renderer.toggle_speed()
                else:
                    renderer.render(episode=i, eval_ep=0)


def main():
    base_dir = "results/test/Walker2d-v5/1M/train"

    env_id = "Walker2d-v5"
    algos = [
        ("PPO", f"{base_dir}/PPO/{MODELS_DIR}"),
        ("SAC", f"{base_dir}/SAC/{MODELS_DIR}"),
        ("TD3", f"{base_dir}/TD3/{MODELS_DIR}"),
    ]

    render_eval(base_dir=base_dir, env_id=env_id, algos=algos, num_episodes=20)


if __name__ == "__main__":
    main()
