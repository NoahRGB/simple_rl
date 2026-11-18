# NoahRGB 11/2026

import torch
import gymnasium as gym

from simple_rl.nn.nets import build_net, detect_head

class DynamicsModel(torch.nn.Module):

    def __init__(self, state_dim: tuple, action_dim: int, rnn_size, hidden_size, latent_dim):
        super().__init__()

        self.rnn_size = rnn_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim


        self.encoder_out = 32

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim[0], 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )


        self.posterior_net = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_out+self.rnn_size, self.hidden_size),
            torch.nn.ELU(),
        )
        self.posterior_mu_head = torch.nn.Linear(self.hidden_size, self.latent_dim)
        self.posterior_sigma_head = torch.nn.Linear(self.hidden_size, self.latent_dim)

        self.rnn = torch.nn.GRUCell(self.latent_dim+action_dim, self.rnn_size)

        self.prior_net = torch.nn.Sequential(
            torch.nn.Linear(self.rnn_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
        )
        self.prior_mu_head = torch.nn.Linear(hidden_size, self.latent_dim)
        self.prior_sigma_head = torch.nn.Linear(hidden_size, self.latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.rnn_size+self.latent_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, state_dim[0]),
        )

        self.inverse_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_out * 2, self.hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(self.hidden_size, action_dim)
        )
    
    def posterior(self, inp):
        net_out = self.posterior_net(inp)
        mu = self.posterior_mu_head(net_out)
        sigma = torch.nn.functional.softplus(self.posterior_sigma_head(net_out)) + 0.1
        dist = torch.distributions.Normal(mu, sigma)
        return dist

    def prior(self, h):
        net_out = self.prior_net(h)
        mu = self.prior_mu_head(net_out)
        sigma = torch.nn.functional.softplus(self.prior_sigma_head(net_out)) + 0.1
        dist = torch.distributions.Normal(mu, sigma)
        return dist

    def inverse(self, inp):
        return self.inverse_head(inp)

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, architecture: dict, state_dim: tuple, action_space: gym.Space):
        super(ActorCriticNetwork, self).__init__()

        if "shared_enc" in architecture:
            self.using_shared_enc = True
            self.shared_enc, self.actor_enc_output = build_net(architecture["shared_enc"], input_shape=state_dim)
            self.value_enc_output = self.actor_enc_output
        else:
            self.using_shared_enc = False
            self.actor_enc, self.actor_enc_output = build_net(architecture["actor_enc"], input_shape=state_dim)
            self.value_enc, self.value_enc_output = build_net(architecture["value_enc"], input_shape=state_dim)
            
        self.actor_head = detect_head(action_space=action_space, input_size=self.actor_enc_output[0])
        self.value_head = torch.nn.Linear(self.value_enc_output[0], 1)

    def get_actor(self, inp):
        if self.using_shared_enc:
            enc_out = self.shared_enc(inp)
        else:
            enc_out = self.actor_enc(inp)
        return self.actor_head(enc_out)
    
    def get_value(self, inp):
        if self.using_shared_enc:
            enc_out = self.shared_enc(inp)
        else:
            enc_out = self.value_enc(inp)
        return self.value_head(enc_out)    