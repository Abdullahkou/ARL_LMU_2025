import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C, DQN
from typing import List, Dict

class EnsembleRL:
    ALGOS = {
        "PPO": PPO,
        "SAC": SAC,
        "A2C": A2C,
        "DQN": DQN
        # TODO: The other algorithms from stable baselines or elsewhere
    }
    
    def __init__(self, env_id: str, algos: List[str], total_timesteps=10000, algo_params: Dict = None, render_interval=2000):
        self.env_id = env_id
        self.algos = algos
        self.total_timesteps = total_timesteps
        self.render_interval = render_interval  # Schritte zwischen den Live-Demos
        self.algo_params = algo_params if algo_params else {}
        
        self.envs = []
        self.models = []
        self._init_models()

    def _init_models(self):
        for algo_name in self.algos:
            if algo_name not in self.ALGOS:
                raise ValueError(f"Algorithm {algo_name} not supported. Available: {list(self.ALGOS.keys())}")
            
            env = gym.make(self.env_id)
            model_class = self.ALGOS[algo_name]
            params = self.algo_params.get(algo_name, {})
            model = model_class("MlpPolicy", env, verbose=0, **params)
            
            self.envs.append(env)
            self.models.append(model)

    def train_with_visualization(self):
        """Trainiert und zeigt zwischendurch Episoden im Render-Fenster."""
        steps_done = 0
        while steps_done < self.total_timesteps:
            for algo_name, model in zip(self.algos, self.models):
                print(f"\n📚 Training {algo_name} für {self.render_interval} Schritte...")
                model.learn(total_timesteps=self.render_interval, reset_num_timesteps=False)
                steps_done += self.render_interval

                # Live-Demo Episode
                print(f"🎬 Zeige Episode für {algo_name}...")
                demo_env = gym.make(self.env_id, render_mode="human")
                obs, _ = demo_env.reset()
                done, truncated = False, False
                while not (done or truncated):
                    action, _ = model.predict(obs)
                    obs, reward, done, truncated, _ = demo_env.step(action)
                    time.sleep(0.02)  # langsamer machen für Sichtbarkeit
                demo_env.close()
        print("\n✅ Training abgeschlossen!")

    def predict(self, obs):
        actions = {}
        for algo_name, model in zip(self.algos, self.models):
            action, _ = model.predict(obs)
            actions[algo_name] = action
        return actions

    def close(self):
        for env in self.envs:
            env.close()
    
    def decision(self):
        # TODO
        pass

    def plot_training(self):
        pass


if __name__ == "__main__":
    ensemble = EnsembleRL(
        env_id="CartPole-v1",
        algos=["PPO", "A2C"],
        total_timesteps=5000,
        render_interval=1000,  # alle 1000 Schritte eine Live-Demo
        algo_params={
            "PPO": {"learning_rate": 0.0003},
            "A2C": {"learning_rate": 0.0005}
        }
    )
    
    ensemble.train_with_visualization()
    obs = ensemble.envs[0].reset()[0]
    print("Aktionen:", ensemble.predict(obs))
    ensemble.close()
