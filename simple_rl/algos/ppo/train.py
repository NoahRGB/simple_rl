


from simple_rl.core.env import VecEnv
from simple_rl.envs.gym import GymEnv
from simple_rl.core.driver import Driver

import numpy as np


def train(cfg, run):
    env = VecEnv([lambda i=i: GymEnv("CartPole-v1", seed=i) for i in range(4)])
    driver = Driver(env)

    driver.on_episode(lambda ep, i: print(f"env {i}: return {ep['reward'].sum():.1f}, length {len(ep['reward'])}"))

    random_policy = lambda row: {"action": np.array([env.action_space.sample() for _ in range(env.num_envs)])}
    driver.run(random_policy, episodes=10)