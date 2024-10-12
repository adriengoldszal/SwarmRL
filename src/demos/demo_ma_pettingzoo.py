import sys

sys.path.append("../")

import gymnasium as gym
import swarm_env.single_env

# import supersuit as ss
from swarm_env.multi_env.multi_agent_pettingzoo import MultiSwarmEnv
import sys
import time
from stable_baselines3 import PPO
import imageio


env = MultiSwarmEnv(render_mode="human", n_agents=2, n_targets=2, map_name="Easy")
model = "models/ma/12-10/tn32lihj/model.zip" # 2 agents and 2 targets on easy map
model = PPO.load(model)
total_episodes = 10
images = []
for i in range(total_episodes):
    
    print(f'Episode {i} of {total_episodes}')
    obs, info = env.reset()

    score = 0
    count = 0
    while True:
        actions = env.sample_action()
        if model:
            actions = {}
            for agent in env.possible_agents:
                single_obs = obs[agent]
                action, _ = model.predict(single_obs)
                actions[agent] = action
        obs, reward, ter, trunc, info = env.step(actions)
        count += 1
        score += reward["agent_0"]
        if trunc["agent_0"] or ter["agent_0"]:
            print(f"Truc {trunc}, ter: {ter}, return: {score}, steps: {count}")
            images.extend(info["ep_frames"])
            break
    imageio.mimsave("./demo_marl.gif", images, fps=25)
    env.close()
