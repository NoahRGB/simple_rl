# NoahRGB 11/2026
#
# the train loop for the ppo agent in algos/ppo/agent.py
# responsible for the timing, logging, rollout, etc.
#
#
#




from simple_rl.envs.gym import GymEnv
from simple_rl.core.logger import logger as log
from simple_rl.core.timer import timer
from simple_rl.core.env import VecEnv
from simple_rl.core.driver import Driver
from simple_rl.core.checkpoint import Checkpoint

from simple_rl.algos.ppo.agent import PPOAgent
from simple_rl.algos.ppo.buffer import RolloutBuffer


def train(cfg, run):
    env = VecEnv([lambda i=i: GymEnv(cfg.env.id, seed=cfg.seed + i) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)

    agent = PPOAgent(cfg.algo, env.obs_space, env.action_space)
    buffer = RolloutBuffer(cfg.algo.rollout_len, cfg.algo.num_envs, env.obs_space.shape, env.action_space.shape, env.obs_space.dtype)

    checkpoint = Checkpoint(run.dir, every=cfg.save_every, network=agent.network, optim=agent.optim)
    if cfg.load_dir is not None:
        checkpoint.load(cfg.load_dir)

    driver.on_step(buffer.add)
    driver.on_episode(driver.log_episode)

    while driver.total_steps < cfg.algo.timesteps:

        
        with timer.section("collect"):
            # run each sub env for rollout_len timesteps
            driver.run(agent.policy, steps=cfg.algo.rollout_len * cfg.algo.num_envs)

        with timer.section("update"):
            metrics = agent.update(buffer)

        buffer.reset()

        log.add(metrics, prefix="ppo")
        log.add(timer.stats(), prefix="timer")
        log.write(driver.total_steps)
        checkpoint.maybe_save(driver.total_steps)

    checkpoint.save()
