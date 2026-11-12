from noah_rl.algorithms.world_models.dreamer_v2 import WorldModel, Actor, preprocess, scale_action
import torch
from noah_rl.utils.gym import GymEnv
from omegaconf import OmegaConf
import imageio, numpy as np
import pygame
import sys
import gymnasium as gym

run_dir = sys.argv[1]
device = "cpu"
seed = None 

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

is_dmc = cfg.env_name.startswith("dm_control/")

# DMC environments with pixel observations dont work with render_mode="human"
if not is_dmc:
    cfg.env_kwargs.render_mode = "human"

env = GymEnv(cfg.env_name, 1, seed=seed, wrappers=cfg.wrappers, **cfg.env_kwargs)

# DMC envs with pixel obs dont use render_mode="human"
# so a pygame screen is made manually instead
if is_dmc:
    view_size = 480
    camera_id = cfg.env_kwargs.get("render_kwargs", {}).get("camera_id", 0)
    physics = env.env.envs[0].unwrapped._env.physics
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


is_discrete = isinstance(env.single_action_space, gym.spaces.Discrete)
action_dim = env.single_action_space.n if is_discrete else env.single_action_space.shape[0]

latent_dim = cfg.latent_variables * cfg.latent_classes

world_model = WorldModel(env.state_dim, action_dim, cfg.latent_variables, cfg.latent_classes, cfg.hidden_dim, cfg.hidden_layer_size).to(device)
actor = Actor(latent_dim, cfg.hidden_dim, action_dim, is_discrete, cfg.min_actor_stdev, cfg.init_actor_std, cfg.hidden_layer_size).to(device)

checkpoint = torch.load(f"{run_dir}/network.pt", map_location=device, weights_only=False)

world_model.load_state_dict(checkpoint["world_model"])
actor.load_state_dict(checkpoint["actor"])

states, _ = env.reset()
episode_done = False
env_hidden_state = torch.zeros(1, cfg.hidden_dim).to(device)
env_latent = torch.zeros(1, latent_dim).to(device)
previous_action = torch.zeros(1, action_dim).to(device)
epsilon = 0.05 

while not episode_done:

    with torch.no_grad():

        env_hidden_state = world_model.rnn(torch.cat([env_latent, previous_action], dim=-1), env_hidden_state)
        env_latent = world_model.posterior(torch.cat([env_hidden_state, world_model.encoder(preprocess(states, device))], dim=-1)).sample().flatten(-2)

        if is_discrete:
            if np.random.rand() < epsilon:
                action = torch.as_tensor(env.action_space.sample()).to(device)
            else:
                action = torch.argmax(actor(torch.cat([env_latent, env_hidden_state], dim=-1)).logits, dim=-1)
            previous_action = torch.nn.functional.one_hot(action, num_classes=action_dim).float() 
        else:
            action_dist = actor(torch.cat([env_latent, env_hidden_state], dim=-1))
            action = (action_dist.mean).clamp(-1, 1) 
            previous_action = action

    action_to_send = action.cpu().numpy() if is_discrete else scale_action(action.cpu().numpy(), env.single_action_space)
    sprimes, rewards, is_terms, is_truncs, info = env.step(action_to_send)

    if is_dmc:
        show_dmc_frame(sprimes)

    if "episode" in info:
        episode_done = True

    states = env.reset_finished_envs(sprimes, is_terms, is_truncs)