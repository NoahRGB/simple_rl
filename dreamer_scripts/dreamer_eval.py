# NoahRGB 11/2026
#
#
#
#


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # lets simple_rl be imported

import torch
import numpy as np
import pygame
from omegaconf import OmegaConf

from simple_rl.core.env import make_env
from simple_rl.core.checkpoint import Checkpoint
from simple_rl.algos.dreamer_v1.agent import DreamerV1Agent

run_dir = sys.argv[1]
device = torch.device("cpu")
seed = None

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

is_dmc = cfg.env.id.startswith("dm_control/")

# DMC environments with pixel observations dont work with render_mode="human"
if not is_dmc:
    if cfg.env.get("kwargs") is None:
        cfg.env.kwargs = {}
    cfg.env.kwargs.render_mode = "human"

env = make_env(cfg.env, seed=seed, scale_actions=True)

# DMC envs with pixel obs dont use render_mode="human"
# so a pygame screen is made manually instead
if is_dmc:
    view_size = 480
    camera_id = (cfg.env.get("kwargs") or {}).get("render_kwargs", {}).get("camera_id", 0)
    physics = env.env.unwrapped._env.physics
    pygame.init()
    screen = pygame.display.set_mode((view_size * 2, view_size))
    pygame.display.set_caption("dreamer eval")
    clock = pygame.time.Clock()

def show_dmc_frame(obs):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # manually render the mujoco physics instead of uisng render_mode="human"
    frame = physics.render(height=view_size, width=view_size, camera_id=camera_id)
    screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))

    # also add the agent's 64x64 view next to it
    agent_obs = np.asarray(obs[0])
    if agent_obs.ndim == 3 and agent_obs.shape[0] in (1, 3):
        agent_obs = agent_obs.transpose(1, 2, 0)
    if agent_obs.ndim == 2 or agent_obs.shape[-1] == 1:
        agent_obs = np.repeat(agent_obs.reshape(agent_obs.shape[0], agent_obs.shape[1], 1), 3, axis=-1)
    agent_surf = pygame.surfarray.make_surface(agent_obs.astype(np.uint8).swapaxes(0, 1))
    screen.blit(pygame.transform.scale(agent_surf, (view_size, view_size)), (view_size, 0))

    pygame.display.flip()
    clock.tick(30)


agent = DreamerV1Agent(cfg.algo, env.obs_space, env.action_space, device)
world_model, actor = agent.world_model, agent.actor
is_discrete, action_size = agent.is_discrete, agent.action_size

Checkpoint(run_dir, world_model=world_model, actor=actor).load(run_dir)

states = np.stack([env.reset()])
episode_done = False
episode_reward = 0.0
env_hidden_state = torch.zeros(1, cfg.algo.rnn_size).to(device)
env_latent = torch.zeros(1, cfg.algo.latent_dim).to(device)
previous_action = torch.zeros(1, action_size).to(device)
epsilon = 0.05

while not episode_done:

    with torch.no_grad():

        env_hidden_state = world_model.rnn(torch.cat([env_latent, previous_action], dim=-1), env_hidden_state)
        env_latent = world_model.posterior(torch.cat([env_hidden_state, world_model.encoder(agent._preprocess(states))], dim=-1)).sample()

        action_dist = actor(torch.cat([env_latent, env_hidden_state], dim=-1))

        if is_discrete:
            if np.random.rand() < epsilon:
                action = torch.as_tensor([env.action_space.sample()]).to(device)
            else:
                action = torch.argmax(action_dist.logits, dim=-1)
            previous_action = torch.nn.functional.one_hot(action, num_classes=action_size).float()
        else:
            action = agent._get_tanh_dist_mode(action_dist).clamp(-1, 1)
            previous_action = action

    # the env rescales [-1, 1] actions itself
    outcome = env.step(action.cpu().numpy()[0])
    episode_reward += outcome["reward"]
    episode_done = outcome["done"]

    states = np.stack([outcome["next_state"]])

    if is_dmc:
        show_dmc_frame(states)

print(f"episode completed, reward {episode_reward}")
