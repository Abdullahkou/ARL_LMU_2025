from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main():
    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=2)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=2500)

    max_steps = 1000

    steps = 0
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        steps += 1
        print(rewards)
        if steps >= max_steps or dones.any():
            print("\nEpisode finished")
            break


if __name__ == "__main__":
    main()
