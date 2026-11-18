# NoahRGB 11/2026
#
# runs the inverse agent's policy in the (onehot) gridworld and shows
# where the world model *imagines* the agent is, next to where it really is
#
# left panel:  the real gridworld
# right panel: the decoder's imagined next state (argmax cell in red, the rest
#              of the decoded probability mass in blue, real position outlined in green)
#
# usage: python simple_rl/algos/inverse/tester.py <run_dir>
#
# keys: SPACE pause, P toggle prior (imagination) / posterior (filtering), ESC quit
#


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3])) # lets simple_rl be imported

import torch
import numpy as np
import pygame
from omegaconf import OmegaConf

from simple_rl.core.env import make_env
from simple_rl.core.checkpoint import Checkpoint
from simple_rl.algos.inverse.agent import InverseAgent
from simple_rl.envs.gridworld import WALL, FLOOR, PLAYER


run_dir = sys.argv[1]
device = torch.device("cpu")
seed = 1
fps = 10
PANEL_PX = 640 # target size of each panel, cells are scaled to fit
HEAT = np.array([40, 90, 230], dtype=np.float32)
REAL_OUTLINE = (30, 200, 60)

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

env = make_env(cfg.env, seed=seed, scale_actions=True)
grid = env.unwrapped # the underlying GridWorld
assert grid.obs_mode == "onehot", "the inverse agent's decoder expects onehot gridworld observations"

agent = InverseAgent(cfg.algo, env.obs_space, env.action_space, device)
world_model = agent.world_model
Checkpoint(run_dir, world_model=world_model).load(run_dir)
world_model.eval()


def index_to_pos(index):
    # onehot obs are row-major, index = row * width + col
    return int(index // grid.width), int(index % grid.width)


CELL = max(1, PANEL_PX // max(grid.height, grid.width)) # screen pixels per grid cell
panel_w, panel_h = grid.width * CELL, grid.height * CELL


def grid_frame(pos):
    # one pixel per grid cell, independent of the env's img_size
    frame = np.where(grid.walls[..., None], WALL, FLOOR)
    frame[pos] = PLAYER
    return frame


def imagined_frame(probs, imagined_pos):
    # gridworld with the player at the imagined position and the
    # decoder's probability over all other floor cells blended in as a heatmap
    frame = grid_frame(imagined_pos).astype(np.float32)

    heat = probs.reshape(grid.height, grid.width).copy()
    heat[grid.walls] = 0.0
    heat[imagined_pos] = 0.0
    heat = np.sqrt(heat / max(probs.max(), 1e-8))[..., None] # sqrt so small probabilities are still visible

    frame = frame * (1 - heat) + HEAT * heat
    return frame.astype(np.uint8)


def to_surface(frame):
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return pygame.transform.scale(surface, (panel_w, panel_h))


def cell_rect(pos, x_offset):
    return pygame.Rect(x_offset + pos[1] * CELL, pos[0] * CELL, CELL, CELL)


pygame.init()
screen = pygame.display.set_mode((panel_w * 2, panel_h))
clock = pygame.time.Clock()


def reset():
    state = env.reset()
    hidden_state = torch.zeros(1, cfg.algo.rnn_size).to(device)
    latent = torch.zeros(1, cfg.algo.latent_dim).to(device)
    return state, hidden_state, latent


state, hidden_state, latent = reset()
is_first = True
use_posterior = False
paused = False
step, episode, correct = 0, 0, 0

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            if event.key == pygame.K_SPACE: paused = not paused
            if event.key == pygame.K_p: use_posterior = not use_posterior

    if paused:
        clock.tick(fps)
        continue

    # the inverse agent picks an action from the real state
    action = agent.policy(np.stack([state]), np.array([is_first]))["action"]
    outcome = env.step(action[0])
    next_state = outcome["next_state"]

    with torch.no_grad():
        action_onehot = torch.nn.functional.one_hot(torch.as_tensor(action).long(), num_classes=agent.action_size).float().to(device)
        hidden_state = world_model.rnn(torch.cat([latent, action_onehot], dim=-1), hidden_state)

        if use_posterior:
            # filtering: the model also sees the real next state
            encoded = world_model.encoder(torch.as_tensor(np.stack([next_state]), dtype=torch.float32).to(device))
            latent = world_model.posterior(torch.cat([hidden_state, encoded], dim=-1)).sample()
        else:
            # imagination: next latent comes only from the learned prior
            latent = world_model.prior(hidden_state).sample()

        logits = world_model.decoder(torch.cat([latent, hidden_state], dim=-1))
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    imagined_pos = index_to_pos(probs.argmax())
    real_pos = grid.pos
    step += 1
    correct += int(imagined_pos == real_pos)

    screen.blit(to_surface(grid_frame(real_pos)), (0, 0))
    screen.blit(to_surface(imagined_frame(probs, imagined_pos)), (panel_w, 0))
    pygame.draw.rect(screen, REAL_OUTLINE, cell_rect(real_pos, panel_w), width=2)
    pygame.draw.line(screen, (0, 0, 0), (panel_w, 0), (panel_w, panel_h), width=2)
    pygame.display.flip()

    mode = "posterior" if use_posterior else "prior"
    pygame.display.set_caption(
        f"inverse tester [{mode}] | ep {episode} step {step} | real {real_pos} imagined {imagined_pos} "
        f"(p={probs.max():.2f}) | match {correct / step:.0%}"
    )

    state = next_state
    is_first = False

    if outcome["done"]:
        print(f"episode {episode} done, imagined position matched real position {correct}/{step} steps ({correct / step:.0%})")
        state, hidden_state, latent = reset()
        is_first = True
        episode += 1
        step, correct = 0, 0

    clock.tick(fps)

pygame.quit()
