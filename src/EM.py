import time
import gymnasium as gym
from typing import List
from agents.baseAgent import BaseAgent, Algo

class EnsembleRL:
    def __init__(self, env_id: str, learners: List[BaseAgent], total_timesteps=10000, render_interval=2000):
        self.env_id = env_id
        self.learners = learners
        self.total_timesteps = total_timesteps
        self.render_interval = render_interval  # Schritte zwischen den Live-Demos

        self.envs = []
        self.models = []
        self._init_models()

    def _init_models(self):
        for learner in self.learners:
            env = gym.make(self.env_id)
            model_class = learner.algo.value
            model = model_class("MlpPolicy", env, verbose=0, **learner.params)
            
            self.envs.append(env)
            self.models.append(model)

    def train_with_visualization(self):
        """Trainiert und zeigt zwischendurch Episoden im Render-Fenster."""
        steps_done = 0
        while steps_done < self.total_timesteps:
            for learner, model in zip(self.learners, self.models):
                print(f"\n📚 Training {learner.algo.name} für {self.render_interval} Schritte...")
                model.learn(total_timesteps=self.render_interval, reset_num_timesteps=False)
                steps_done += self.render_interval

                # Live-Demo Episode
                print(f"🎬 Zeige Episode für {learner.algo.name}...")
                demo_env = gym.make(self.env_id, render_mode="human")
                obs, _ = demo_env.reset()
                done, truncated = False, False
                while not (done or truncated):
                    action, _ = model.predict(obs)
                    obs, reward, done, truncated, _ = demo_env.step(action)
                    learner.history.append(reward)
                    time.sleep(0.02)  # langsamer machen für Sichtbarkeit
                demo_env.close()
        print("\n✅ Training abgeschlossen!")

    def predict(self, obs):
        actions = {}
        for learner, model in zip(self.learners, self.models):
            action, _ = model.predict(obs)
            actions[learner.algo.name] = action
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
    learners = [
        BaseAgent(algo=Algo.PPO, params={"learning_rate": 0.0003}),
        BaseAgent(algo=Algo.A2C, params={"learning_rate": 0.0005})
    ]

    ensemble = EnsembleRL(
        env_id="CartPole-v1",
        learners=learners,
        total_timesteps=5000,
        render_interval=1000,  # alle 1000 Schritte eine Live-Demo
    )
    
    ensemble.train_with_visualization()
    obs = ensemble.envs[0].reset()[0]
    print("Aktionen:", ensemble.predict(obs))
    ensemble.close()
