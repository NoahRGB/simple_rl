# NoahRGB 11/2026
#
#
#


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from simple_rl.envs.gym import GymEnv
from simple_rl.core.logger import logger as log
from simple_rl.core.timer import timer
from simple_rl.core.env import VecEnv, make_env
from simple_rl.core.driver import Driver
from simple_rl.core.checkpoint import Checkpoint

from simple_rl.algos.disagreement.agent import DisagreementAgent 
from simple_rl.algos.disagreement.buffer import EpisodicReplayMemory 


def plot_heatmap(heatmap, walls, title=""):
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgrey") # unvisited = grey

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.ma.masked_invalid(heatmap), cmap=cmap, vmin=0.0, vmax=0.00070) # show colour coded image
    ax.imshow(np.ma.masked_where(~walls, walls), cmap=ListedColormap(["black"])) # make walls black
    fig.colorbar(im, ax=ax, label="disagreement")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return fig

def train(cfg, run):
    env = VecEnv([lambda i=i: make_env(cfg.env, seed=cfg.seed + i, scale_actions=True) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)

    grid_width, grid_height = env.envs[0].env.unwrapped.width, env.envs[0].env.unwrapped.height
    grid_walls = env.envs[0].env.unwrapped.walls

    agent = DisagreementAgent(cfg.algo, env.obs_space, env.action_space, run.device, grid_width, grid_height, grid_walls)
    buffer = EpisodicReplayMemory(cfg.algo.replay_size, env.obs_space.shape, env.action_space.shape, env.obs_space.dtype, cfg.algo.sequence_length)

    policy = lambda states, is_first: agent.policy(states, is_first, positions=[env.info["pos"] for env in env.envs])

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
    timer.reset() # dont time the warmup

    last_log = driver.total_steps # the timestep of the last log.write()

    while driver.total_steps < cfg.algo.total_timesteps:

        
        with timer.section("collect"):
            # do train_every real environment steps 
            driver.run(policy, steps=cfg.algo.train_every)


        with timer.section("update", sync=True):
            # do gradient_updates gradient updates
            metrics = agent.update(buffer)


        # metrics are added every update and averaged over the log_every interval when written
        log.add(metrics, prefix="disagreement")

        if driver.total_steps - last_log >= cfg.algo.log_every:
            heatmap = plot_heatmap(agent.disagreement_heatmap(), grid_walls, f"step {driver.total_steps}")
            log.figure("heatmap", heatmap, driver.total_steps, prefix="disagreement")
            log.add(timer.stats(), prefix="timer")
            log.write(driver.total_steps)
            last_log = driver.total_steps

        if cfg.algo.save_checkpoints:
            checkpoint.maybe_save(driver.total_steps, gradient_steps=agent.gradient_steps)

    if cfg.algo.save_checkpoints:
        checkpoint.save(gradient_steps=agent.gradient_steps)

