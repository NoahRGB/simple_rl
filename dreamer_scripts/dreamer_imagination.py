from noah_rl.algorithms.world_models.dreamer_v2 import WorldModel, Actor, preprocess, scale_action
import torch
from noah_rl.utils.gym import GymEnv
from omegaconf import OmegaConf
import  numpy as np
import sys, imageio
import gymnasium as gym

run_dir = sys.argv[1]
device = "cpu"
seed = 1
imagination_steps = 200

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

env = GymEnv(cfg.env_name, 1, seed=seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
is_discrete = isinstance(env.single_action_space, gym.spaces.Discrete)
action_dim = env.single_action_space.n if is_discrete else env.single_action_space.shape[0]

latent_dim = cfg.latent_variables * cfg.latent_classes

world_model = WorldModel(env.state_dim, action_dim, cfg.latent_variables, cfg.latent_classes, cfg.hidden_dim, cfg.hidden_layer_size).to(device)
actor = Actor(latent_dim, cfg.hidden_dim, action_dim, is_discrete, cfg.min_actor_stdev, cfg.init_actor_std, cfg.hidden_layer_size).to(device)

checkpoint = torch.load(f"{run_dir}/network.pt", map_location=device, weights_only=False)

world_model.load_state_dict(checkpoint["world_model"])
actor.load_state_dict(checkpoint["actor"])

start_state, _ = env.reset()

with torch.no_grad():

    # starts with a real env state turned into latent using posterior
    # then imagines using prior
    imagination_latents = torch.zeros(1, latent_dim).to(device)
    imagination_hidden_states = torch.zeros(1, cfg.hidden_dim).to(device)
    previous_action = torch.zeros(1, action_dim).to(device)

    imagination_hidden_states = world_model.rnn(torch.cat([imagination_latents, previous_action], dim=-1), imagination_hidden_states)
    imagination_latents = world_model.posterior(torch.cat([imagination_hidden_states, world_model.encoder(preprocess(start_state, device))], dim=-1)).sample().flatten(-2)

    decoded_observations = [world_model.decoder(torch.cat([imagination_latents, imagination_hidden_states], dim=-1))]
    for imagination_step in range(imagination_steps):
        # imagination comes from the learned prior

        actions = actor(torch.cat([imagination_latents, imagination_hidden_states], dim=-1)).rsample()

        imagination_hidden_states = world_model.rnn(torch.cat([imagination_latents, actions], dim=-1), imagination_hidden_states)
        imagination_latents = world_model.prior(imagination_hidden_states).rsample().flatten(-2)

        decoded_observations.append(world_model.decoder(torch.cat([imagination_latents, imagination_hidden_states], dim=-1)))


# create a gif using the imagined frames
frames = []
for decoded in decoded_observations:

    # undo the preprocessing / scaling on states 
    frame = (decoded[0].permute(1, 2, 0).numpy() + 0.5).clip(0, 1)
    frame = np.kron(frame, np.ones((4, 4, 1)))
    frames.append((frame * 255).astype(np.uint8))

imageio.mimsave(f"{run_dir}/imagination.gif", frames, duration=1000/15, loop=0)
