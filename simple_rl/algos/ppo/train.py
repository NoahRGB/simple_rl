from simple_rl.envs.gym import GymEnv
from simple_rl.core.logger import logger as log
from simple_rl.core.timer import timer
from simple_rl.core.env import VecEnv
from simple_rl.core.driver import Driver

from simple_rl.algos.ppo.agent import PPOAgent
from simple_rl.algos.ppo.buffer import RolloutBuffer


def train(cfg, run):
    env = VecEnv([lambda i=i: GymEnv(cfg.env.id, seed=cfg.seed + i) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)

    agent = PPOAgent(cfg.algo, env.obs_space, env.action_space)
    buffer = RolloutBuffer(cfg.algo.rollout_len, cfg.algo.num_envs, env.obs_space.shape, env.action_space.shape, env.obs_space.dtype)

    driver.on_step(buffer.add)
    driver.on_episode(driver.log_episode)

    while driver.total_steps < cfg.algo.timesteps:
        with timer.section("collect"):
            driver.run(agent.policy, steps=cfg.algo.rollout_len * cfg.algo.num_envs)

        with timer.section("update"):
            metrics = agent.update(buffer)

        buffer.reset()

        log.add(metrics, prefix="ppo")
        log.add(timer.stats(), prefix="timer")
        log.write(driver.total_steps)
