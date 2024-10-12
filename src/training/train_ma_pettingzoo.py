import sys

sys.path.append("../")
import wandb
from wandb.integration.sb3 import WandbCallback
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C
import gymnasium as gym
from swarm_env.multi_env.multi_agent_pettingzoo import MultiSwarmEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import numpy as np
import torch.nn as nn
from sb3_contrib import RecurrentPPO
import supersuit as ss
from datetime import datetime
from training.utils import DummyRun

use_wandb = True

config = {
    "algo": "PPO",
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 50_000,
    "num_envs": 4,
}

env_config = {
    "map_name": "Easy",
    "max_episode_steps": 100,
    "continuous_action": True,
    "n_agents": 2,
    "n_targets": 2,
}

kwargs_PPO = {
    "policy": "MultiInputPolicy",
    "learning_rate": 7e-4,
    "n_steps": 512,
    "ent_coef": 2e-3,
    "policy_kwargs": {
        "net_arch": {"pi": [16, 32, 64, 64, 16], "vf": [16, 32, 64, 64, 16]},
        "activation_fn": nn.ReLU,
        "ortho_init": True,
    },
    "verbose": 1,
}


env = MultiSwarmEnv(**env_config)
env = ss.pettingzoo_env_to_vec_env_v1(env)
envs = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3")

if use_wandb:
    run = wandb.init(
        project="multi-agent",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
else:
    run = DummyRun()

date = datetime.now()
formatted_date = date.strftime("%d-%m")

wandbcallback = (
    WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"models/ma/{formatted_date}/{run.id}",
        verbose=2,
    )
    if use_wandb
    else None
)

checkpoint_callback = CheckpointCallback(
    save_freq=5_000,
    save_path=f"./checkpoints/ma/{formatted_date}/{run.id}",
    name_prefix=f"model_{run.id}",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

callbacks = [checkpoint_callback]
if wandbcallback:
    callbacks.append(wandbcallback)


algo_map = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "R_PPO": RecurrentPPO}
model = algo_map[config["algo"]](
    env=envs, tensorboard_log=f"runs/ma/{formatted_date}/{run.id}", **kwargs_PPO
)

print("ALGO ", config["algo"])
print(model.policy)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=CallbackList(callbacks),
    progress_bar=True,
)

env.close()
run.finish()
