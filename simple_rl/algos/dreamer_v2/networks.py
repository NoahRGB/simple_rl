# NoahRGB 11/2026
#
# contains the world model, actor and critic classes
# for the dreamerv2 agent
#

import torch

from simple_rl.algos.dreamer_v2.dist import TruncatedNormalDistribution



class WorldModel(torch.nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int, latent_vars: int, latent_classes: int, hidden_dim: int, hidden_size: int):
        super(WorldModel, self).__init__()

        self.hidden_dim = hidden_dim

        # the latent is latent_vars categorical variables with latent_classes classes each
        self.latent_vars = latent_vars
        self.latent_classes = latent_classes
        self.latent_dim = self.latent_vars * self.latent_classes

        # the ENCODER part of the autoencoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(state_dim[0], 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(), # 1024 for 64x64, 9216 for 128x128
        )

        # compute flattened encoder output size from the input shape
        with torch.no_grad():
            embed_dim = self.encoder(torch.zeros(1, *state_dim)).shape[-1]

        self.posterior_net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim+self.hidden_dim, hidden_size),
            torch.nn.ELU(),
        )
        self.posterior_logits_head = torch.nn.Linear(hidden_size, self.latent_dim)

        # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        self.rnn = torch.nn.GRUCell(self.latent_dim+action_dim, self.hidden_dim)

        self.reward_model = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim+self.hidden_dim, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, 1),
        )

        self.discount_model = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim+self.hidden_dim, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, 1),
        )

        self.prior_net = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
        )
        self.prior_logits_head = torch.nn.Linear(hidden_size, self.latent_dim)

        # the DECODER part of the autoencoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim+self.hidden_dim, 1024),
            torch.nn.Unflatten(-1, unflattened_size=(1024, 1, 1)),
            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
        )

    def posterior(self, inp):
        net_out = self.posterior_net(inp)
        logits = self.posterior_logits_head(net_out)
        logits = logits.view(*logits.shape[:-1], self.latent_vars, self.latent_classes)
        dist = torch.distributions.OneHotCategoricalStraightThrough(logits=logits)
        return dist

    def prior(self, h):
        net_out = self.prior_net(h)
        logits = self.prior_logits_head(net_out)
        logits = logits.view(*logits.shape[:-1], self.latent_vars, self.latent_classes)
        dist = torch.distributions.OneHotCategoricalStraightThrough(logits=logits)
        return dist


class Actor(torch.nn.Module):
    # predicts actions that solve the imagination environment
    # given a latent, output an action
    # "the action model outputs a tanh-transformed Gaussian ... that allows for reparameterised sampling"

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int, is_discrete: bool, init_actor_std: float, min_actor_stdev: float, hidden_size: int):
        super(Actor, self).__init__()

        self.init_actor_std = init_actor_std
        self.min_actor_stdev = min_actor_stdev
        self.is_discrete = is_discrete

        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+hidden_dim, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
        )
        self.mu_head = torch.nn.Linear(hidden_size, action_dim)
        self.std_head = torch.nn.Linear(hidden_size, action_dim)

    def forward(self, inp):
        network_out = self.network(inp)

        if self.is_discrete:
            logits = self.mu_head(network_out)
            return torch.distributions.OneHotCategoricalStraightThrough(logits=logits)

        else:

            mu_out = self.mu_head(network_out)
            std_out = self.std_head(network_out)

            mean = torch.tanh(mu_out) 
            # std = torch.nn.functional.softplus(std_out) + self.min_actor_stdev
            std = 2 * torch.sigmoid((std_out  + self.init_actor_std) / 2) + self.min_actor_stdev

            return TruncatedNormalDistribution(mean, std) 

class ValueModel(torch.nn.Module):
    # estimates the expected imagined rewards that the action model achievs from state s_T
    def __init__(self, latent_dim: int, hidden_dim: int, hidden_size: int):
        super(ValueModel, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+hidden_dim, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, inp):
        return self.network(inp).squeeze(-1)