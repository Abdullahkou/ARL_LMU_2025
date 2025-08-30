from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from agents.models import SB3Callback


def single():
    tmp_path = "tmp/single/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv"])
    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=4)

    total_timesteps = 30_000
    callback = SB3Callback(total_timesteps=total_timesteps, eval_model=None)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    obs = vec_env.reset()
    total_rew = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        total_rew += rewards.mean()

        if dones.any():
            break

    with open("single.txt", "w") as f:
        f.write(str(total_rew))


def multi():
    tmp_path = "tmp/multi/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv"])
    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=4)

    total_timesteps = 10_000
    callback = SB3Callback(total_timesteps=30_000, eval_model=None)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.set_logger(new_logger)
    model.learn(
        callback=callback, total_timesteps=total_timesteps, reset_num_timesteps=False
    )
    model.learn(
        callback=callback, total_timesteps=total_timesteps, reset_num_timesteps=False
    )
    model.learn(
        callback=callback, total_timesteps=total_timesteps, reset_num_timesteps=False
    )

    obs = vec_env.reset()
    total_rew = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        total_rew += rewards.mean()

        if dones.any():
            break

    with open("multi.txt", "w") as f:
        f.write(str(total_rew))


if __name__ == "__main__":
    # single()
    multi()
