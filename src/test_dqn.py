import gymnasium as gym
import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from config.training_config import Algorithm
from utils.rendering import Renderer

env_id = "LunarLander-v3"


def make_env():
    return gym.make(env_id)


agent = DQNAgent(config=Algorithm(), train_env_factory=make_env, device="cpu")


agent.learn(total_steps=100_000)
agent.save(path="independent_dqn_agent/dqn")

# Evaluation
n_eval_episodes = 5

# Teste jeden Head einzeln
for head in range(agent.hp.n_q_heads):
    returns = []
    for i in range(n_eval_episodes):
        eval_env = gym.make(env_id, render_mode="rgb_array")
        renderer = Renderer(eval_env)
        obs, _ = eval_env.reset()
        done = False
        episode_return = 0
        while not done:
            # explizit nur diesen Head nutzen
            q_values = agent.q_net(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            )
            q_head = q_values[:, head] if q_values.ndim == 3 else q_values
            action = int(torch.argmax(q_head, dim=-1).item())
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_return += reward
            done = terminated or truncated
            if renderer is not None and renderer._enabled:
                renderer.add_reward(reward)
                esc, space = renderer.pump_events()
                if esc:
                    renderer.close()
                elif space:
                    renderer.toggle_speed()
                else:
                    renderer.render()

        eval_env.close()
        returns.append(episode_return)
    avg_return = np.mean(returns)
    print(f"Head {head} average return over {n_eval_episodes} episodes: {avg_return}")


# Ensemble Voting (alle Heads mitteln)
returns = []
for i in range(n_eval_episodes):
    eval_env = gym.make(env_id, render_mode="rgb_array")
    renderer = Renderer(eval_env)
    obs, _ = eval_env.reset()
    done = False
    episode_return = 0
    while not done:
        q_values = agent.q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        if q_values.ndim == 3:  # [1, n_heads, n_actions]
            q_mean = q_values.mean(dim=1)  # [1, n_actions]
        else:
            q_mean = q_values
        action = int(torch.argmax(q_mean, dim=-1).item())
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        episode_return += reward
        done = terminated or truncated
        if renderer is not None and renderer._enabled:
            renderer.add_reward(reward)
            esc, space = renderer.pump_events()
            if esc:
                renderer.close()
            elif space:
                renderer.toggle_speed()
            else:
                renderer.render()
    eval_env.close()
    returns.append(episode_return)
print(f"Ensemble-Voting average return: {np.mean(returns)}")


# 2) Teste "Random Head pro Episode"
# returns = []
# for i in range(n_eval_episodes):
#     eval_env = gym.make(env_id, render_mode="human")
#     obs, _ = eval_env.reset()
#     done = False
#     episode_return = 0
#     head = np.random.randint(agent.hp.n_q_heads)
#     while not done:
#         q_values = agent.q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
#         q_head = q_values[:, head] if q_values.ndim == 3 else q_values
#         action = int(torch.argmax(q_head, dim=-1).item())
#         obs, reward, terminated, truncated, _ = eval_env.step(action)
#         episode_return += reward
#         done = terminated or truncated
#     eval_env.close()
#     returns.append(episode_return)
# print(f"Random-Head (Deep Exploration) average return: {np.mean(returns)}")
