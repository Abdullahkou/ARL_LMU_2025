import gym
from stable_baselines3 import PPO, SAC, A2C, DQN
from typing import List, Dict

class EnsembleRL:
    ALGOS = {
        "PPO": PPO,
        "SAC": SAC,
        "A2C": A2C,
        "DQN": DQN
    }
    
    def __init__(self, env_id: str, algos: List[str], total_timesteps=10000, algo_params: Dict = None):
        self.env_id = env_id
        self.algos = algos
        self.total_timesteps = total_timesteps
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
            params = self.algo_params.get(algo_name, {})  # individuelle Hyperparams pro Algo
            model = model_class("MlpPolicy", env, verbose=0, **params)
            
            self.envs.append(env)
            self.models.append(model)

    def train(self):
        for algo_name, model in zip(self.algos, self.models):
            print(f"Training {algo_name}...")
            model.learn(total_timesteps=self.total_timesteps)
        print("Training abgeschlossen!")

    def predict(self, obs):
        """Gibt die Aktion jedes Agenten zurück"""
        actions = {}
        for algo_name, model in zip(self.algos, self.models):
            action, _ = model.predict(obs)
            actions[algo_name] = action
        return actions

    def close(self):
        for env in self.envs:
            env.close()


# Beispiel-Nutzung:
if __name__ == "__main__":
    ensemble = EnsembleRL(
        env_id="CartPole-v1",
        algos=["PPO", "SAC"],
        total_timesteps=5000,
        algo_params={
            "PPO": {"learning_rate": 0.0003},
            "SAC": {"learning_rate": 0.0005}
        }
    )
    
    ensemble.train()
    obs = ensemble.envs[0].reset()[0]
    print("Aktionen:", ensemble.predict(obs))
    ensemble.close()
