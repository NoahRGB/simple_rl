


from simple_rl.envs.gym import GymEnv
from simple_rl.core.logger import logger as log
from simple_rl.core.env import VecEnv
from simple_rl.core.driver import Driver, log_episode

import numpy as np


def train(cfg, run):
    env = VecEnv([lambda i=i: GymEnv(cfg.env.id, seed=cfg.seed + i) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)
    driver.on_episode(log_episode)

    random_policy = lambda row: {"action": np.array([env.action_space.sample() for _ in range(env.num_envs)])}

    while driver.total_steps < cfg.algo.steps:
        driver.run(random_policy, steps=cfg.algo.log_every)
        log.write(driver.total_steps)