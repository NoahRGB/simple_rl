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

from simple_rl.algos.inverse.agent import InverseAgent
from simple_rl.algos.inverse.buffer import EpisodicReplayMemory, RolloutBuffer
from simple_rl.core.normaliser import Normaliser



def train(cfg, run):
    env = VecEnv([lambda i=i: make_env(cfg.env, seed=cfg.seed + i, scale_actions=True) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)
    assert cfg.algo.num_envs == 1, "only works with 1 env running"

    agent = InverseAgent(cfg.algo, env.obs_space, env.action_space, run.device)
    dynamics_buffer = EpisodicReplayMemory(cfg.algo.replay_size, env.obs_space.shape, env.action_space.shape, env.obs_space.dtype, cfg.algo.sequence_length)
    ppo_buffer = RolloutBuffer(cfg.algo.ppo_rollout_len, cfg.algo.num_envs, env.obs_space.shape, env.action_space.shape, env.obs_space.dtype)

    intrinsic_normaliser = Normaliser((1,), use_mean=False)
    running_intrinsic_returns = np.zeros(cfg.algo.num_envs)

    def ppo_add(transitions):

        nonlocal running_intrinsic_returns

        intr = agent.get_intrinsic_reward(transitions["next_state"], transitions["action"])
        transitions["reward"] = np.array([intr], dtype=np.float32)

        running_intrinsic_returns = running_intrinsic_returns * cfg.algo.gamma + transitions["reward"] 
        intrinsic_normaliser.add_batch(running_intrinsic_returns)
        ppo_buffer.add(transitions)

    def on_ep(ep, env_idx):
        dynamics_buffer.add(ep["state"], ep["action"], ep["reward"], ep["next_state"], ep["is_term"])

    checkpoint = Checkpoint(run.dir, every=cfg.save_every, world_model=agent.world_model, world_model_optim=agent.world_model_optim)

    if cfg.load_dir is not None:
        checkpoint.load(cfg.load_dir)

    driver.on_episode(driver.log_episode)

    driver.on_episode(on_ep)

    # execute S random warmup episodes to initialise DYNAMICS dataset
    random_policy = lambda states, is_first: {"action": np.stack([env.action_space.sample() for _ in range(env.num_envs)])}
    driver.run(random_policy, episodes=cfg.algo.warmup_episodes)
    timer.reset() # dont time the warmup

    if cfg.algo.learn_ppo: driver.on_batch(ppo_add)

    last_log = driver.total_steps

    next_dynamics_update = driver.total_steps + cfg.algo.train_every

    while driver.total_steps < cfg.algo.total_timesteps:

        timesteps_to_ppo_update = cfg.algo.ppo_rollout_len - ppo_buffer.current_timestep
        timesteps_to_dynamics_update = max(1, np.ceil((next_dynamics_update - driver.total_steps) / cfg.algo.num_envs))
        this_chunk = min(timesteps_to_ppo_update, timesteps_to_dynamics_update)
        
        with timer.section("collect"):
            driver.run(agent.policy, steps=this_chunk * cfg.algo.num_envs)

        if driver.total_steps >= next_dynamics_update:
            if cfg.algo.learn_wm:
                with timer.section("dynamics_update", sync=True):
                    metrics = agent.dynamics_update(dynamics_buffer)
                    log.add(metrics, prefix="dynamics")
            next_dynamics_update += cfg.algo.train_every

        if ppo_buffer.current_timestep == cfg.algo.ppo_rollout_len:
            if cfg.algo.learn_ppo:
                with timer.section("ppo_update", sync=True):
                    metrics = agent.policy_update(ppo_buffer, intrinsic_normaliser)
                    log.add(metrics, prefix="ppo")
                ppo_buffer.reset()

        if driver.total_steps - last_log >= cfg.algo.log_every:
            log.add(timer.stats(), prefix="timer")
            log.write(driver.total_steps)
            last_log = driver.total_steps

        if cfg.algo.save_checkpoints:
            checkpoint.maybe_save(driver.total_steps)

    if cfg.algo.save_checkpoints:
        checkpoint.save()

    # returned so callers (e.g. experiment scripts) can inspect them after training
    return agent, env
