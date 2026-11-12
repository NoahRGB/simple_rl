# NoahRGB 11/2026
#
# contains the world model, actor and critic classes
# for the dreamerv1 agent
#

import torch



class WorldModel(torch.nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int, latent_dim: int, hidden_dim: int, hidden_size: int):
        super(WorldModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim 

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
            torch.nn.Flatten(), # 1024
        )

        # compute flattened encoder output size from the input shape
        with torch.no_grad():
            embed_dim = self.encoder(torch.zeros(1, *state_dim)).shape[-1]


        self.posterior_net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim+self.hidden_dim, hidden_size),
            torch.nn.ELU(),
        )
        self.posterior_mu_head = torch.nn.Linear(hidden_size, self.latent_dim)
        self.posterior_sigma_head = torch.nn.Linear(hidden_size, self.latent_dim)

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
        self.prior_mu_head = torch.nn.Linear(hidden_size, self.latent_dim)
        self.prior_sigma_head = torch.nn.Linear(hidden_size, self.latent_dim)

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

class Actor(torch.nn.Module):
    # predicts actions that solve the imagination environment
    # given a latent, output an action
    # "the action model outputs a tanh-transformed Gaussian ... that allows for reparameterised sampling"

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int, is_discrete: bool, init_stdev: float, min_actor_stdev: float, hidden_size: int):
        super(Actor, self).__init__()

        self.init_stdev = init_stdev
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

            mean = 5.0 * torch.tanh(mu_out / 5.0) 
            std = torch.nn.functional.softplus(std_out + self.init_stdev) + self.min_actor_stdev

            # the actor outputs a tanh transformed Gaussian
            # a TransformedDistribution with TanhTransform will automatically
            # apply tanh() to any sample() from the distribution
            base = torch.distributions.Normal(mean, std)
            dist = torch.distributions.TransformedDistribution(base, [torch.distributions.TanhTransform(cache_size=1)])
            return torch.distributions.Independent(dist, 1)

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