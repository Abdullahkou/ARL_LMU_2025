from agents.baseAgent import BaseAgent, Algo
from src.EM import EnsembleRL

def main():
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


if __name__ == "__main__":
    main()
