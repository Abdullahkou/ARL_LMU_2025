import time
from typing import Dict, list

import gymnasium as gym
import numpy as np
from scipy.stats import mode
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3


class EnsembleRL:
    ALGOS = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "DQN": DQN, "TD3": TD3, "DDPG": DDPG}

    def __init__(
        self,
        env_id: str,
        algos: list[str],
        total_timesteps=10000,
        algo_params: Dict = None,
        render_interval=2000,
    ):
        self.env_id = env_id
        self.algos = algos
        self.total_timesteps = total_timesteps
        self.render_interval = render_interval
        self.algo_params = algo_params if algo_params else {}

        self.envs = []
        self.models = []
        self._init_models()

    def _init_models(self):
        for algo_name in self.algos:
            if algo_name not in self.ALGOS:
                raise ValueError(
                    f"Algorithm {algo_name} not supported. Available: {list(self.ALGOS.keys())}"
                )

            env = gym.make(self.env_id)
            model_class = self.ALGOS[algo_name]
            params = self.algo_params.get(algo_name, {})
            model = model_class("MlpPolicy", env, verbose=0, **params)

            self.envs.append(env)
            self.models.append(model)

    def train_WITHOUT_visualization(self):
        for algo_name, model in zip(self.algos, self.models):
            print(f"Training {algo_name}...")
            model.learn(total_timesteps=self.total_timesteps)
        print("Training abgeschlossen!")

    def train_with_visualization(self):
        """Trainiert jedes Modell für total_timesteps und zeigt zwischendurch Episoden im Render-Fenster."""
        for algo_name, model in zip(self.algos, self.models):
            steps_done = 0
            print(f"\nTraining {algo_name} für {self.total_timesteps} Schritte...")
            while steps_done < self.total_timesteps:
                model.learn(
                    total_timesteps=self.render_interval, reset_num_timesteps=False
                )
                steps_done += self.render_interval

                # Live-Demo Episode
                print(f"Zeige Episode für {algo_name}...")
                demo_env = gym.make(self.env_id, render_mode="human")
                obs, _ = demo_env.reset()
                done, truncated = False, False
                while not (done or truncated):
                    action, _ = model.predict(obs)
                    obs, reward, done, truncated, _ = demo_env.step(action)
                    time.sleep(0.02)  # langsamer machen für Sichtbarkeit
                demo_env.close()
            print(f"\nTraining für {algo_name} abgeschlossen!")
        print("\nGesamtes Training abgeschlossen!")

    def predict(self, obs):
        actions = {}
        for algo_name, model in zip(self.algos, self.models):
            action, _ = model.predict(obs)
            actions[algo_name] = action
        return actions

    def close(self):
        for env in self.envs:
            env.close()

    def demo_ensemble(self, episodes=1, sleep_time=0.02):
        """Zeigt eine Live-Demo des Ensemble-Modells nach dem Training."""
        print("\nStarte Ensemble-Demo...")
        # Prüfe Aktionsraum
        demo_env = gym.make(self.env_id)
        is_discrete = isinstance(demo_env.action_space, gym.spaces.Discrete)
        combination_method = "Majority Voting" if is_discrete else "Action Averaging"
        print(
            f"Verwende {combination_method} für Aktionskombination (Aktionsraum: {'diskret' if is_discrete else 'kontinuierlich'})"
        )
        demo_env.close()

        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            demo_env = gym.make(self.env_id, render_mode="human")
            obs, _ = demo_env.reset()
            done, truncated = False, False
            total_reward = 0
            while not (done or truncated):
                actions = self.predict(obs)
                action_values = list(actions.values())

                if is_discrete:
                    action, _ = mode(action_values, axis=0, keepdims=False)
                    action = int(action)
                else:
                    action = np.mean(action_values, axis=0)

                # Führe kombinierte Aktion aus
                obs, reward, done, truncated, _ = demo_env.step(action)
                total_reward += reward
                time.sleep(sleep_time)  # Für Sichtbarkeit
            print(f"Episode {episode + 1} beendet. Gesamtreward: {total_reward:.2f}")
            demo_env.close()
        print("\nEnsemble-Demo abgeschlossen!")


# if __name__ == "__main__":
#     ensemble = EnsembleRL(
#         env_id="LunarLander-v3",
#         algos=["PPO", "A2C", "DQN"],
#         total_timesteps=1000000,
#         render_interval=10000,
#         algo_params={
#             "PPO": {"learning_rate": 0.0003},
#             "A2C": {"learning_rate": 0.0005}
#         }
#     )

#     ensemble.train_with_visualization()
#     obs = ensemble.envs[0].reset()[0]
#     print("Aktionen:", ensemble.predict(obs))
#     ensemble.close()


if __name__ == "__main__":
    # # env_id="LunarLander-v3",
    # "BipedalWalker-v3"
    #
    ensemble = EnsembleRL(
        env_id="Humanoid-v5",
        algos=["PPO", "DQN"],
        total_timesteps=500,
        render_interval=100,
        algo_params={
            "PPO": {"learning_rate": 0.001},
            # "SAC": {"learning_rate": 0.0003},
            # "DDPG": {"learning_rate": 0.0003},
            # "TD3": {"learning_rate": 0.0003}
        },
    )

    ensemble.train_with_visualization()
    ensemble.demo_ensemble(episodes=2, sleep_time=0.02)
    # obs = ensemble.envs[0].reset()[0]
    # print("Aktionen:", ensemble.predict(obs))

    ensemble.close()
