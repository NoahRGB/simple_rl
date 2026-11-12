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
import imageio
from omegaconf import OmegaConf

from simple_rl.core.env import make_env
from simple_rl.core.checkpoint import Checkpoint
from simple_rl.algos.dreamer_v1.agent import DreamerV1Agent


run_dir = sys.argv[1]
device = torch.device("cpu")
seed = 1
imagination_steps = 200

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

env = make_env(cfg.env, seed=seed, scale_actions=True)
agent = DreamerV1Agent(cfg.algo, env.obs_space, env.action_space, device)
world_model, actor = agent.world_model, agent.actor

Checkpoint(run_dir, world_model=world_model, actor=actor).load(run_dir)

# add a batch dimension (1, *state_dim)
start_state = np.stack([env.reset()])

with torch.no_grad():

    # starts with a real env state turned into latent using posterior
    # then imagines using prior
    imagination_latents = torch.zeros(1, cfg.algo.latent_dim).to(device)
    imagination_hidden_states = torch.zeros(1, cfg.algo.rnn_size).to(device)
    previous_action = torch.zeros(1, agent.action_size).to(device)

    imagination_hidden_states = world_model.rnn(torch.cat([imagination_latents, previous_action], dim=-1), imagination_hidden_states)
    imagination_latents = world_model.posterior(torch.cat([imagination_hidden_states, world_model.encoder(agent._preprocess(start_state))], dim=-1)).sample()

    decoded_observations = [world_model.decoder(torch.cat([imagination_latents, imagination_hidden_states], dim=-1))]
    for imagination_step in range(imagination_steps):
        # imagination comes from the learned prior

        actions = actor(torch.cat([imagination_latents, imagination_hidden_states], dim=-1)).rsample()

        imagination_hidden_states = world_model.rnn(torch.cat([imagination_latents, actions], dim=-1), imagination_hidden_states)
        imagination_latents = world_model.prior(imagination_hidden_states).rsample()

        decoded_observations.append(world_model.decoder(torch.cat([imagination_latents, imagination_hidden_states], dim=-1)))


# create a gif using the imagined frames
frames = []
for decoded in decoded_observations:

    # undo the preprocessing / scaling on states
    frame = (decoded[0].permute(1, 2, 0).numpy() + 0.5).clip(0, 1)
    frame = np.kron(frame, np.ones((4, 4, 1)))
    frames.append((frame * 255).astype(np.uint8))

imageio.mimsave(f"{run_dir}/imagination.gif", frames, duration=1000/15, loop=0)
print(f"saved {run_dir}/imagination.gif ({len(frames)} frames)")
