from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from tqdm import trange


def evaluate(
    agent: nn.Module,
    env: gym.Env,
    seed: int,
    num_episodes: int,
    terminate_on_success: bool = False,
) -> Dict[str, float]:
    stats = {
        "return": [],
        "length": [],
        "success": [],
        "near_object": [],
        "grasp_success": [],
        "grasp_reward": [],
        "in_place_reward": [],
        "obj_to_target": [],
        "unscaled_reward": [],
    }

    env.seed(seed)
    for _ in trange(num_episodes, desc="evaluation", leave=False):
        observation, info = env.reset()
        done = False
        ep_steps = 0
        ep_rewards = 0
        while not done:
            ep_steps += 1
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, done, _, info = env.step(action)
            ep_rewards += reward
            if ep_steps == env.max_path_length:
                done = True
        stats["return"] = ep_rewards
        stats["length"] = ep_steps
        for k in stats.keys():
            if k != "return" and k != "length":
                # print(k)
                # print(info[k])
                stats[k].append(info[k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
