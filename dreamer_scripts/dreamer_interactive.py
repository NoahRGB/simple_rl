from noah_rl.algorithms.world_models.dreamer_v1 import WorldModel, preprocess
import torch
from noah_rl.utils.gym import GymEnv
from omegaconf import OmegaConf
import numpy as np
import sys
import gymnasium as gym
import pygame

def keys_to_action():
    # action numbers for ALE/Boxing-v5
    k = pygame.key.get_pressed()
    up, down, left, right, fire = k[pygame.K_UP], k[pygame.K_DOWN], k[pygame.K_LEFT], k[pygame.K_RIGHT], k[pygame.K_SPACE]
    if up and right and fire: return 14
    if up and left and fire: return 15
    if down and right and fire: return 16
    if down and left and fire: return 17
    if up and fire: return 10
    if right and fire: return 11
    if left and fire: return 12
    if down and fire: return 13
    if up and right: return 6
    if up and left: return 7
    if down and right: return 8
    if down and left: return 9
    if fire: return 1
    if up: return 2
    if right: return 3
    if left: return 4
    if down: return 5
    return 0

def keys_to_continuous_action():
    # keys for DMC cartpole balance
    # A = push fully left, D = push fully right
    k = pygame.key.get_pressed()
    return float(k[pygame.K_d]) - float(k[pygame.K_a])

run_dir = sys.argv[1]
device = "cpu"
seed = 1

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

env = GymEnv(cfg.env_name, 1, seed=seed, wrappers=cfg.wrappers, **cfg.env_kwargs)
is_discrete = isinstance(env.single_action_space, gym.spaces.Discrete)
action_dim = env.single_action_space.n if is_discrete else env.single_action_space.shape[0]

world_model = WorldModel(env.state_dim, action_dim, cfg.latent_dim, cfg.hidden_dim, cfg.get("hidden_layer_size", 300)).to(device)

checkpoint = torch.load(f"{run_dir}/network.pt", map_location=device, weights_only=False)

world_model.load_state_dict(checkpoint["world_model"])

start_state, _ = env.reset()
SCALE = 8
screen = pygame.display.set_mode((64 * SCALE, 64 * SCALE))
clock = pygame.time.Clock()

with torch.no_grad():

    # starts with a real env state turned into latent using posterior
    # then imagines using prior
    imagination_latents = torch.zeros(1, cfg.latent_dim).to(device)
    imagination_hidden_states = torch.zeros(1, cfg.hidden_dim).to(device)
    previous_action = torch.zeros(1, action_dim).to(device)

    imagination_hidden_states = world_model.rnn(torch.cat([imagination_latents, previous_action], dim=-1), imagination_hidden_states)
    imagination_latents = world_model.posterior(torch.cat([imagination_hidden_states, world_model.encoder(preprocess(start_state, device))], dim=-1)).sample()

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        if pygame.key.get_pressed()[pygame.K_ESCAPE]: running = False

        # undo the preprocessing / scaling on states
        decoded_obs = world_model.decoder(torch.cat([imagination_latents, imagination_hidden_states], dim=-1))
        frame = ((decoded_obs[0].permute(2, 1, 0).numpy() + 0.5).clip(0, 1) * 255).astype(np.uint8)
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(frame), screen.get_size()), (0, 0))
        pygame.display.flip()

        # imagination comes from the learned prior
        if is_discrete:
            action = torch.tensor([keys_to_action()]).to(device)
            actions = torch.nn.functional.one_hot(action, num_classes=action_dim).float()
        else:
            actions = torch.tensor([[keys_to_continuous_action()]], dtype=torch.float32).to(device)

        imagination_hidden_states = world_model.rnn(torch.cat([imagination_latents, actions], dim=-1), imagination_hidden_states)
        imagination_latents = world_model.prior(imagination_hidden_states).rsample()

        clock.tick(15)

