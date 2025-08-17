import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C, DQN, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.data = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if 'episode' in info:
                self.data.append((self.num_timesteps, info['episode']['r']))
        return True

class EnsembleRL:
    ALGOS = {
        "PPO": PPO, "SAC": SAC, "A2C": A2C, "DQN": DQN, "TD3": TD3, "DDPG": DDPG
    }
    
    def __init__(self, env_id: str, algos: List[str], total_timesteps=10000, algo_params: Dict = None):
        self.env_id = env_id
        self.algos = algos
        self.total_timesteps = total_timesteps
        self.algo_params = algo_params if algo_params else {}
        self.models = {} 
        self.training_rewards = {}
        self._init_models()

    def _init_models(self):
        for algo_name in self.algos:
            env = gym.make(self.env_id)
            model_class = self.ALGOS[algo_name]
            params = self.algo_params.get(algo_name, {})
            model = model_class("MlpPolicy", env, verbose=0, **params)
            self.models[algo_name] = model

    def train_models(self):
        for algo_name, model in self.models.items():
            print(f"--- Training von {algo_name} für {self.total_timesteps} Timesteps ---")
            
            reward_callback = RewardCallback()
            
            model.learn(total_timesteps=self.total_timesteps, callback=reward_callback)
            
            self.training_rewards[algo_name] = reward_callback.data
            print(f"--- Training für {algo_name} abgeschlossen ---")
        
        print("\nAlle Trainingsläufe sind beendet!")

    def plot_training_rewards(self, window_size=25):
        plt.figure(figsize=(14, 8))
        
        for algo_name, data in self.training_rewards.items():
            if not data:
                continue
            
            timesteps, rewards = zip(*data)
            
            if len(rewards) >= window_size:
                moving_avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                moving_avg_timesteps = timesteps[window_size-1:]
                plt.plot(moving_avg_timesteps, moving_avg_rewards, label=f"{algo_name}")
            else:
                plt.plot(timesteps, rewards, label=f"{algo_name} (zu wenige Daten)", alpha=0.5)

        plt.title("Fairer Trainings-Vergleich: Reward vs. Timesteps")
        plt.xlabel("Anzahl der Timesteps")
        plt.ylabel(f"Durchschnittlicher Reward (Gleitender Avg. w={window_size})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def predict(self, obs):
        actions = {}
        for algo_name, model in self.models.items():
            action, _ = model.predict(obs, deterministic=True)
            actions[algo_name] = action
        return actions

    def demo_single_agent(self, algo_name: str, episodes=3):
        """Zeigt eine Demo eines einzelnen, trainierten Agenten."""
        print(f"\n--- Demo für {algo_name} ---")
        if algo_name not in self.models:
            print(f"Modell {algo_name} nicht gefunden.")
            return

        model = self.models[algo_name]
        demo_env = gym.make(self.env_id, render_mode="human")
        
        for ep in range(episodes):
            obs, _ = demo_env.reset()
            done, truncated = False, False
            total_reward = 0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = demo_env.step(action)
                total_reward += reward
                time.sleep(0.02)
            print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")
        demo_env.close()

    def demo_ensemble(self, episodes=1, sleep_time=0.02):
            """Live-Demo des Ensemble-Modells"""
            print("\nStarte Ensemble-Demo...")
            # Prüfe Aktionsraum
            demo_env = gym.make(self.env_id)
            is_discrete = isinstance(demo_env.action_space, gym.spaces.Discrete)
            combination_method = "Majority Voting" if is_discrete else "Action Averaging"
            print(f"Verwende {combination_method} für Aktionskombination (Aktionsraum: {'diskret' if is_discrete else 'kontinuierlich'})")
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
                    
                    obs, reward, done, truncated, _ = demo_env.step(action)
                    total_reward += reward
                    time.sleep(sleep_time)
                print(f"Episode {episode + 1} beendet. Gesamtreward: {total_reward:.2f}")
                demo_env.close()
            print("\nEnsemble-Demo abgeschlossen!")






if __name__ == "__main__":


    # env_id= "LunarLander-v3",
    env_id= "BipedalWalker-v3"
    # env_id= "Humanoid-v5",

    env_for_shape = gym.make(env_id)
    n_actions = env_for_shape.action_space.shape[-1]
    ddpg_action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    ensemble = EnsembleRL(
        env_id=env_id,
        algos=["PPO", "SAC", "DDPG"],
        total_timesteps=150000,
        algo_params={
            "PPO": {"learning_rate": 0.0003},
            "A2C": {"learning_rate": 0.0007},
            "DDPG": {"learning_rate": 0.001, "buffer_size": 10000, "learning_starts": 2500, "action_noise": ddpg_action_noise}
        }
    )

    ensemble.train_models()
    
    print("\nErstelle den finalen Vergleichs-Plot...")
    ensemble.plot_training_rewards()

    ensemble.demo_ensemble(episodes=2, sleep_time=0.02)


    #obs = ensemble.envs[0].reset()[0]
    #print("Aktionen:", ensemble.predict(obs))

    #ensemble.close()

    #Eine Demo  von einem Agenten anzeigen
    #ensemble.demo_single_agent("PPO", episodes=1)



"""
algo_params = {
    "PPO": {
        # --- On-Policy Algorithmus (Actor-Critic) ---
        "learning_rate": 0.0003,    # Wie stark die Gewichte bei jedem Update angepasst werden.
        "n_steps": 2048,            # Anzahl der Schritte, die pro Agent und pro Update gesammelt werden.
        "batch_size": 64,           # Größe der Mini-Batches, in die die gesammelten Daten aufgeteilt werden.
        "n_epochs": 10,             # Wie oft die gesammelten Daten während eines Updates durchlaufen werden.
        "gamma": 0.99,              # Discount-Faktor, gewichtet zukünftige Belohnungen.
        "gae_lambda": 0.95,         # Faktor für die "Generalized Advantage Estimation".
        "clip_range": 0.2,          # Begrenzt, wie stark sich die Policy pro Update ändern darf. Wichtig für Stabilität.
    },
    "A2C": {
        # --- On-Policy Algorithmus (Actor-Critic), der Vorgänger von PPO ---
        "learning_rate": 0.0007,    # A2C verwendet oft eine etwas höhere Lernrate.
        "n_steps": 5,               # Sammelt standardmäßig viel weniger Daten pro Update als PPO.
        "gamma": 0.99,
        "gae_lambda": 1.0,          # Verwendet standardmäßig keine GAE, sondern einfache Advantage-Werte.
        "vf_coef": 0.5,             # Gewicht des Value-Function-Verlusts im Gesamtverlust.
    },
    "SAC": {
        # --- Off-Policy Algorithmus (Actor-Critic) mit maximaler Entropie ---
        "learning_rate": 0.0003,
        "buffer_size": 1000000,     # Größe des Replay-Buffers (Gedächtnis). 1 Mio. ist Standard.
        "learning_starts": 100,     # Aufwärmphase (in Schritten), bevor das Lernen beginnt.
        "batch_size": 256,          # Anzahl der Erfahrungen, die pro Trainings-Update aus dem Buffer gezogen werden.
        "tau": 0.005,               # "Soft-Update"-Faktor für die Ziel-Netzwerke.
        "gamma": 0.99,
        "train_freq": (1, "step"),  # Wie oft trainiert wird (hier: nach jedem einzelnen Schritt).
    },
    "TD3": {
        # --- Off-Policy Algorithmus (Actor-Critic), eine Verbesserung von DDPG ---
        "learning_rate": 0.001,
        "buffer_size": 1000000,
        "learning_starts": 2500,    # TD3/DDPG benötigen oft eine längere Aufwärmphase.
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "action_noise": td3_action_noise, # Wichtig für die Erkundung in kontinuierlichen Umgebungen.
    },
    "DDPG": {
        # --- Off-Policy Algorithmus (Actor-Critic), der Vorgänger von TD3 ---
        "learning_rate": 0.001,
        "buffer_size": 1000000,
        "learning_starts": 2500,
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "action_noise": ddpg_action_noise, # DDPG ist stark auf Action-Noise für die Erkundung angewiesen.
    },
    "DQN": {
        # --- Off-Policy, Value-Based Algorithmus (nur für diskrete Aktionen) ---
        "learning_rate": 0.001,
        "buffer_size": 100000,      # Ein kleinerer Puffer ist hier oft ausreichend.
        "learning_starts": 50000,   # Benötigt eine lange Aufwärmphase für stabile Q-Werte.
        "batch_size": 32,
        "tau": 1.0,                 # Standard-DQN verwendet "harte" Updates, daher tau=1.0.
        "gamma": 0.99,
        "train_freq": 4,            # Trainiert standardmäßig alle 4 Schritte.
        "exploration_fraction": 0.1, # Anteil der Trainingszeit, in der Epsilon von 1.0 auf den Endwert sinkt.
        "exploration_final_eps": 0.05, # Minimaler Wert für Epsilon (Wahrscheinlichkeit für zufällige Aktion).
    }
}
"""