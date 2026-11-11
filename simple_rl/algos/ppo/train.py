


from simple_rl.envs.gym import GymEnv
from simple_rl.core.logger import logger as log
from simple_rl.core.timer import timer
from simple_rl.core.env import VecEnv, get_action_size
from simple_rl.core.driver import Driver, log_episode

from simple_rl.algos.ppo.networks import ActorCriticNetwork

import numpy as np
import torch


def train(cfg, run):
    env = VecEnv([lambda i=i: GymEnv(cfg.env.id, seed=cfg.seed + i) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)

    action_dim = get_action_size(env.action_space)
    network = ActorCriticNetwork(cfg.algo.architecture, env.obs_space.shape, env.action_space)
    optim = torch.optim.Adam(network.parameters(), lr=cfg.lr)



    # driver.on_episode(log_episode)

    while driver.total_steps < cfg.algo.timesteps:
        with timer.section("collect"):
            driver.run(0, steps=cfg.algo.rollout_len * cfg.algo.num_envs)

        log.add(timer.stats(), prefix="timer")
        log.write(driver.total_steps)