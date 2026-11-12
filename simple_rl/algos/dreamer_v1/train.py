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

from simple_rl.algos.dreamer_v1.agent import DreamerV1Agent 
from simple_rl.algos.dreamer_v1.buffer import EpisodicReplayMemory 


def train(cfg, run):
    env = VecEnv([lambda i=i: make_env(cfg.env, seed=cfg.seed + i, scale_actions=True) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)

    agent = DreamerV1Agent(cfg.algo, env.obs_space, env.action_space, run.device)
    buffer = EpisodicReplayMemory(cfg.algo.replay_size, env.obs_space.shape, env.action_space.shape, env.obs_space.dtype)

    checkpoint = Checkpoint(
        run.dir, every=cfg.save_every,
        world_model=agent.world_model, actor=agent.actor, value_model=agent.value_model,
        world_model_optim=agent.world_model_optim, actor_optim=agent.actor_optim, value_model_optim=agent.value_model_optim,
    )

    if cfg.load_dir is not None:
        values = checkpoint.load(cfg.load_dir)
        agent.gradient_steps = values.get("gradient_steps", 0)

    driver.on_episode(driver.log_episode)
    driver.on_episode(lambda ep, env_idx: buffer.add(ep["state"], ep["action"], ep["reward"], ep["next_state"], ep["is_term"]))

    # execute S random warmup episodes to initialise dataset
    random_policy = lambda states, is_first: {"action": np.stack([env.action_space.sample() for _ in range(env.num_envs)])}
    driver.run(random_policy, episodes=cfg.algo.warmup_episodes)

    last_log = driver.total_steps # the timestep of the last log.write()

    while driver.total_steps < cfg.algo.total_timesteps:

        
        with timer.section("collect"):
            # do train_every real environment steps 
            driver.run(agent.policy, steps=cfg.algo.train_every)


        with timer.section("update", sync=True):
            # do gradient_updates gradient updates
            metrics = agent.update(buffer)


        # metrics are added every update and averaged over the log_every interval when written
        log.add(metrics, prefix="dreamer_v1")

        if driver.total_steps - last_log >= cfg.algo.log_every:
            log.add(timer.stats(), prefix="timer")
            log.write(driver.total_steps)
            last_log = driver.total_steps

        if cfg.algo.save_checkpoints:
            checkpoint.maybe_save(driver.total_steps, gradient_steps=agent.gradient_steps)

    if cfg.algo.save_checkpoints:
        checkpoint.save(gradient_steps=agent.gradient_steps)

