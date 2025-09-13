from functools import partial

import gymnasium

from agents.wrapperAgent import WrapperAgent
from config.training_config import MODELS_DIR, load_training_config, make_gym
from utils.rendering import Renderer


def render_eval(config_dir: str, env_id: str, seed: int, num_episodes: int = 10):
    config = load_training_config(dir=config_dir)
    eval_env_seed = config.train_env_seed[1]

    render_env = gymnasium.make(env_id, render_mode="rgb_array")
    render_env.reset(seed=eval_env_seed)
    renderer = Renderer(env=render_env)

    env_factory = partial(make_gym, env_id=env_id, seed=config.eval_env_seed)

    model = WrapperAgent(config.algo_config, env_factory=env_factory, seed=seed)
    load_path = f"{config_dir}/{MODELS_DIR}/seed_{seed}"
    model.load(load_path)

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
    env_id = "Walker2d-v5"
    model_dir = "results/test/Walker2d-v5/1M/RANDOM"
    seed = 0

    render_eval(config_dir=model_dir, env_id=env_id, seed=seed, num_episodes=20)


if __name__ == "__main__":
    main()
