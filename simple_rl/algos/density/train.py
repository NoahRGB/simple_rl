# NoahRGB 11/2026
#
#
#


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from sklearn.neighbors import KernelDensity

from simple_rl.envs.gym import GymEnv
from simple_rl.core.logger import logger as log
from simple_rl.core.timer import timer
from simple_rl.core.env import VecEnv, make_env
from simple_rl.core.driver import Driver
from simple_rl.core.checkpoint import Checkpoint

from simple_rl.algos.density.agent import DensityAgent 
from simple_rl.algos.density.buffer import EpisodicReplayMemory


@torch.no_grad()
def encode_positions(agent, gridworld, positions):
    # render each (row, col) position and encode it with the current world model encoder
    obs = np.array([np.transpose(gridworld._render_obs(pos), (2, 0, 1)) for pos in positions])
    return agent.world_model.encoder(agent._preprocess(obs)).cpu().numpy()

def novelty_heatmap(agent, gridworld, bandwidth=1.0):
    # fit a KDE to the latents of every visited state, then score the latent of every free cell
    walls = np.asarray(gridworld.walls, dtype=bool)
    free_cells = list(zip(*np.nonzero(~walls))) # row-major order, same as heat[~walls]

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(encode_positions(agent, gridworld, agent.pos_hist))
    log_p = kde.score_samples(encode_positions(agent, gridworld, free_cells))
    log_count = log_p + np.log(len(agent.pos_hist))

    heat = np.full(walls.shape, np.nan)
    heat[~walls] = -0.5 * log_count
    return heat

def true_novelty_heatmap(gridworld, pos_hist):
    # ground-truth novelty from exact visit counts, to sanity-check the KDE estimate against
    walls = np.asarray(gridworld.walls, dtype=bool)
    counts = np.zeros(walls.shape)
    np.add.at(counts, tuple(np.asarray(pos_hist, dtype=int).T), 1)
    return np.where(walls, np.nan, -0.5 * np.log(counts + 1))

def distance_heatmap(agent, gridworld, ref_cell):
    # euclidean distance in embedding space from ref_cell (row, col) to every free cell
    walls = np.asarray(gridworld.walls, dtype=bool)
    free_cells = list(zip(*np.nonzero(~walls)))

    zs = encode_positions(agent, gridworld, free_cells)
    ref_z = encode_positions(agent, gridworld, [ref_cell])[0]

    heat = np.full(walls.shape, np.nan)
    heat[~walls] = np.linalg.norm(zs - ref_z, axis=-1)
    return heat

def plot_heatmap(heatmap, title="", label="", vmin=None, vmax=None, mark=None):
    # fixed vmin/vmax keep the colour scale comparable across steps (None = autoscale to this frame)
    # mark: optional (row, col) to highlight with a red cross
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("black") # walls (NaN) = black

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.ma.masked_invalid(heatmap), cmap=cmap, vmin=vmin, vmax=vmax)
    if mark is not None:
        ax.plot(mark[1], mark[0], "rx", markersize=10, markeredgewidth=2) # imshow x = col, y = row
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return fig


def train(cfg, run):
    env = VecEnv([lambda i=i: make_env(cfg.env, seed=cfg.seed + i, scale_actions=True) for i in range(cfg.algo.num_envs)])
    driver = Driver(env)


    gridworld = env.unwrapped[0]
    grid_width, grid_height = gridworld.width, gridworld.height
    grid_walls = gridworld.walls

    agent = DensityAgent(cfg.algo, env.obs_space, env.action_space, run.device, grid_width, grid_height, grid_walls)
    buffer = EpisodicReplayMemory(cfg.algo.replay_size, env.obs_space.shape, env.action_space.shape, env.obs_space.dtype, cfg.algo.sequence_length)

    policy = lambda states, is_first: agent.policy(states, is_first, positions=[env.info["pos"] for env in env.envs])


    driver.on_episode(driver.log_episode)
    driver.on_episode(lambda ep, env_idx: buffer.add(ep["state"], ep["action"], ep["reward"], ep["next_state"], ep["is_term"]))

    # execute S random warmup episodes to initialise dataset
    random_policy = lambda states, is_first: {"action": np.stack([env.action_space.sample() for _ in range(env.num_envs)])}
    driver.run(random_policy, episodes=cfg.algo.warmup_episodes)
    timer.reset() # dont time the warmup

    last_log = driver.total_steps # the timestep of the last log.write()

    # fixed reference cell for the embedding distance heatmap (default: first free cell, row-major)
    ref_cell = cfg.algo.distance_ref_cell
    if ref_cell is None:
        ref_cell = tuple(int(i) for i in np.argwhere(~np.asarray(grid_walls, dtype=bool))[0])
    ref_cell = tuple(ref_cell)
    assert not grid_walls[ref_cell], f"distance_ref_cell {ref_cell} is a wall"

    while driver.total_steps < cfg.algo.total_timesteps:

        
        with timer.section("collect"):
            # do train_every real environment steps 
            driver.run(policy, steps=cfg.algo.train_every)


        with timer.section("update", sync=True):
            # do gradient_updates gradient updates
            metrics = agent.update(buffer)


        # metrics are added every update and averaged over the log_every interval when written
        log.add(metrics, prefix="density")

        if driver.total_steps - last_log >= cfg.algo.log_every:
            novelty = novelty_heatmap(agent, gridworld)

            heatmap = plot_heatmap(novelty, f"step {driver.total_steps}", label="novelty",
                                   vmin=cfg.algo.heatmap_vmin, vmax=cfg.algo.heatmap_vmax)
            log.figure("heatmap", heatmap, driver.total_steps, prefix="density")

            true_heatmap = plot_heatmap(true_novelty_heatmap(gridworld, agent.pos_hist), f"step {driver.total_steps}", label="true novelty")
            log.figure("true_heatmap", true_heatmap, driver.total_steps, prefix="density")

            dist_heatmap = plot_heatmap(distance_heatmap(agent, gridworld, ref_cell), f"step {driver.total_steps}",
                                        label=f"embedding distance from {ref_cell}", mark=ref_cell)
            log.figure("distance_heatmap", dist_heatmap, driver.total_steps, prefix="density")
            log.add(timer.stats(), prefix="timer")
            log.write(driver.total_steps)
            last_log = driver.total_steps

    # returned so callers (e.g. experiment scripts) can inspect them after training
    return agent, env
