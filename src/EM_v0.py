import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C, DQN, TD3, DDPG 
from typing import List, Dict
import numpy as np
from scipy.stats import mode
from collections.abc import Callable
from enum import Enum
from dataclasses import dataclass, field

#from gym.wrappers.monitoring.video_recorder import VideoRecorder

class Algo(Enum):
    PPO = PPO
    SAC = SAC
    A2C = A2C
    DQN = DQN
    TD3 = TD3,
    DDPG = DDPG
    # TODO: The other algorithms from stable baselines or elsewhere
@dataclass
class Learner:
    algo: Algo
    params: Dict # TODO: maybe refine type later
    training_history: List = field(default_factory=list)



class CombinationMethod:
    def __init__(self, combinationFn: Callable, is_discrete: bool):
        self.combinationFn = combinationFn
        self.is_discrete = is_discrete

    def __call__(self, actionValues: List[int]):
        return self.combinationFn(actionValues)

majority_vote = CombinationMethod(
    combinationFn = lambda action_values: mode(action_values, axis=0, keepdims=False)[0],
    is_discrete = True,
)

action_average = CombinationMethod(
    combinationFn = lambda action_values: np.mean(action_values, axis=0),
    is_discrete=False,
)

class EnsembleRL:
    def __init__(self, env_id: str, learners: List[Learner], combination_method: CombinationMethod, total_timesteps=10000, render_interval=2000):
        self.env_id = env_id
        self.learners = learners
        self.total_timesteps = total_timesteps
        self.render_interval = render_interval  # Schritte zwischen den Live-Demos
        self.combination_method = combination_method
        
        self.envs = []
        self.models = []
        self._init_models()

    def _init_models(self):
        for learner in self.learners:     
            env = gym.make(self.env_id)

            is_discrete_env = isinstance(env.action_space, gym.spaces.Discrete)
            if (is_discrete_env ^ self.combination_method.is_discrete):
                raise ValueError(f"Environment is {"discrete" if is_discrete_env else "continuous"} but the combination method is {"discrete" if self.combination_method.is_discrete else "continuous"}")

            model_class = learner.algo.value
            model = model_class("MlpPolicy", env, verbose=0, **learner.params)
            
            self.envs.append(env)
            self.models.append(model)

    def train_WITHOUT_visualization(self):
        for algo, model in zip(self.algos, self.models):
            print(f"Training {algo.name}...")
            model.learn(total_timesteps=self.total_timesteps)
        print("Training abgeschlossen!")


    def train_with_visualization(self):
        """Trainiert jedes Modell für total_timesteps und zeigt zwischendurch Episoden im Render-Fenster."""
        for learner, model in zip(self.learners, self.models):
            steps_done = 0
            print(f"\nTraining {learner.algo.name} für {self.total_timesteps} Schritte...")
            while steps_done < self.total_timesteps:
                model.learn(total_timesteps=self.render_interval, reset_num_timesteps=False)
                steps_done += self.render_interval

                # Live-Demo Episode
                print(f"Zeige Episode für {learner.algo.name}...")
                demo_env = gym.make(self.env_id, render_mode="human")
                obs, _ = demo_env.reset()
                done, truncated = False, False
                while not (done or truncated):
                    action, _ = model.predict(obs)
                    obs, reward, done, truncated, _ = demo_env.step(action)
                    learner.training_history.append(reward)
                    time.sleep(0.02)  # langsamer machen für Sichtbarkeit
                demo_env.close()
            print(f"\nTraining für {learner.algo.name} abgeschlossen!")
        print("\nGesamtes Training abgeschlossen!")

    def predict(self, obs):
        actions = {}
        for learner, model in zip(self.learners, self.models):
            action, _ = model.predict(obs)
            actions[learner.algo.name] = action
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
                    
                    action = self.combination_method(action_values)
                    
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
    learners = [
        Learner(algo=Algo.PPO, params={"learning_rate": 0.001}),
        Learner(algo=Algo.SAC, params={"learning_rate": 0.0003}),
    ]
    ensemble = EnsembleRL(
        env_id="Humanoid-v5",
        learners=learners,
        total_timesteps=500,
        render_interval=100,  
        combination_method=action_average
    )

    ensemble.train_with_visualization()
    ensemble.demo_ensemble(episodes=2, sleep_time=0.02)
    #obs = ensemble.envs[0].reset()[0]
    #print("Aktionen:", ensemble.predict(obs))

    ensemble.close()
