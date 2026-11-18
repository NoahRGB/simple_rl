# NoahRGB 11/2026
#

#
#



import numpy as np

from simple_rl.envs.gym import GymEnv
from simple_rl.core.logger import logger as log
from simple_rl.core.timer import timer
from simple_rl.core.env import VecEnv, make_env
from simple_rl.core.driver import Driver
from simple_rl.core.checkpoint import Checkpoint



def train(cfg, run):
    env = VecEnv([lambda i=i: make_env(cfg.env, seed=cfg.seed + i) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)

    driver.on_episode(driver.log_episode)

    random_policy = lambda states, is_first: {"action": np.stack([env.action_space.sample() for _ in range(env.num_envs)])}
    driver.run(random_policy, steps=cfg.algo.total_timesteps)

    return None, env
