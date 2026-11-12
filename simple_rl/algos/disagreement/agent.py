# NoahRGB 11/2026
#
#
#
#

import torch
import numpy as np

from simple_rl.core.utils import is_discrete, get_action_size
from simple_rl.algos.disagreement.networks import WorldModel, Actor, ValueModel, Ensemble

class DisagreementAgent:

    def __init__(self, cfg, obs_space, action_space, device, grid_width, grid_height, grid_walls):
        self.cfg = cfg 
        self.device = device

        self.action_space = action_space
        self.action_size = get_action_size(action_space) 

        self.is_discrete = is_discrete(action_space)
        self.world_model = WorldModel(obs_space.shape, self.action_size, cfg.latent_dim, cfg.rnn_size, cfg.hidden_layer_size).to(device)
        self.actor = Actor(cfg.latent_dim, cfg.rnn_size, self.action_size, self.is_discrete, cfg.init_actor_std, cfg.min_actor_stdev, cfg.hidden_layer_size).to(device)
        self.value_model = ValueModel(cfg.latent_dim, cfg.rnn_size, cfg.hidden_layer_size).to(device)

        self.ensemble = Ensemble(cfg.num_predictors, cfg.rnn_size, self.action_size, cfg.hidden_layer_size, self.world_model.embed_dim).to(device)
        self.ensemble_optim = torch.optim.Adam(self.ensemble.parameters(), lr=cfg.ensemble_lr)

        self.world_model_optim = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_model_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.action_model_lr)
        self.value_model_optim = torch.optim.Adam(self.value_model.parameters(), lr=cfg.value_model_lr)
        self.gradient_steps = 0

        self.grid_width, self.grid_height, self.grid_walls = grid_width, grid_height, grid_walls
        self.disagreement_sum = np.zeros((grid_height, grid_width))
        self.disagreement_count = np.zeros((grid_height, grid_width))

        # used for keeping track of the real env interaction
        self.hidden_state, self.latent, self.previous_action = None, None, None

    def _preprocess(self, obs):
        # convert to torch tensor and scale
        tens = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        return (tens / 255.0) - 0.5

    def _get_tanh_dist_mode(self, dist, num_samples=100):
        # since a tanh-transformed Gausssian has no easy mode,
        # the Dreamer code draws N samples from the distribution and
        # uses the most common one as an estimation of the "mode"
        with torch.no_grad():
            samples = dist.sample((num_samples,))
            probs = dist.log_prob(samples)
            idxs = probs.argmax(0)
            return samples[idxs, torch.arange(samples.shape[1])]

    def policy(self, states, is_first, positions=None):
        num_envs = states.shape[0]

        if self.hidden_state is None:
            self.hidden_state = torch.zeros(num_envs, self.cfg.rnn_size, device=self.device)
            self.latent = torch.zeros(num_envs, self.cfg.latent_dim, device=self.device)
            self.previous_action = torch.zeros(num_envs, self.action_size, device=self.device)

        with torch.no_grad():
            # zero the running state of every env that just started a new episode
            keep = 1.0 - torch.as_tensor(is_first, dtype=torch.float32, device=self.device).unsqueeze(-1)
            self.hidden_state = self.hidden_state * keep
            self.latent = self.latent * keep
            self.previous_action = self.previous_action * keep

            encoded = self.world_model.encoder(self._preprocess(states))
            self.hidden_state = self.world_model.rnn(torch.cat([self.latent, self.previous_action], dim=-1), self.hidden_state)
            self.latent = self.world_model.posterior(torch.cat([self.hidden_state, encoded], dim=-1)).sample()

            action_dist = self.actor(torch.cat([self.latent, self.hidden_state], dim=-1))

            if self.is_discrete:
                rand_actions = np.stack([self.action_space.sample() for _ in range(num_envs)]) 
                action = torch.as_tensor(rand_actions).to(self.device)
            else:
                action = action_dist.rsample()
                action = (action + self.cfg.continuous_action_noise_stdev * torch.randn_like(action)).clamp(-1, 1)

            # check disagreement
            if positions is not None:

                ens_inp = torch.cat([self.hidden_state, self.previous_action], dim=-1)
                ens = self.ensemble(ens_inp) # (num_predictors, 1, latent_dim)
                disagreement = ens.var(dim=0).mean(-1).cpu().numpy()

                for (row, col), dis in zip(positions, disagreement):

                    self.disagreement_count[row, col] += 1
                    self.disagreement_sum[row, col] += dis


            self.previous_action = torch.nn.functional.one_hot(action, num_classes=self.action_size).float() if self.is_discrete else action 

        return {"action": action.cpu().numpy()}

    def disagreement_heatmap(self):
        return np.where(self.disagreement_count > 0, self.disagreement_sum / np.maximum(self.disagreement_count, 1), np.nan)

    def update(self, buffer):
        metric_names = ["world_model_loss", "reconstruction_loss", "reconstruction_mse", "kl", "kl_loss", "reward_loss",
                        "prior_std", "posterior_std", "actor_loss", "value_model_loss", "discount_loss"]
        metrics = {name: [] for name in metric_names}

        for update_step in range(self.cfg.gradient_updates):

            # ================================
            # dynamics / world model updates
            # ================================

            windows, sample_size = buffer.sample(self.cfg.batch_size, self.device, self.cfg.prioritise_ends)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_terms = windows # (B, L, ..., )

            batch_next_states = self._preprocess(batch_next_states)
            if self.is_discrete:
                batch_actions = torch.nn.functional.one_hot(batch_actions.long(), num_classes=self.action_size).float()
            else:
                batch_actions = batch_actions.float()
            batch_rewards = batch_rewards.float()

            encoded = self.world_model.encoder(batch_next_states.flatten(0, 1)).unflatten(0, (self.cfg.batch_size, self.cfg.sequence_length))
            hidden_state = torch.zeros(self.cfg.batch_size, self.cfg.rnn_size).to(self.device)
            latent = torch.zeros(self.cfg.batch_size, self.cfg.latent_dim).to(self.device)

            latents, hidden_states, prior_mus, prior_stddevs, posterior_mus, posterior_stddevs = [], [], [], [], [], []
            for t in range(self.cfg.sequence_length):
                hidden_state = self.world_model.rnn(torch.cat([latent, batch_actions[:, t]], dim=-1), hidden_state)
                prior = self.world_model.prior(hidden_state)
                posterior = self.world_model.posterior(torch.cat([hidden_state, encoded[:, t]], dim=-1))
                latent = posterior.rsample()

                hidden_states.append(hidden_state); latents.append(latent)
                prior_mus.append(prior.mean); prior_stddevs.append(prior.stddev)
                posterior_mus.append(posterior.mean); posterior_stddevs.append(posterior.stddev)

            hidden_states = torch.stack(hidden_states, 1)
            latents = torch.stack(latents, 1)
            priors = torch.distributions.Normal(torch.stack(prior_mus, 1), torch.stack(prior_stddevs, 1))
            posteriors = torch.distributions.Normal(torch.stack(posterior_mus, 1), torch.stack(posterior_stddevs, 1))



            # use h_{t+1} and a_{t+1} as input to each one-step ensemble predictor
            ensemble_input = torch.cat([hidden_states[:, :-1], batch_actions[:, 1:]], dim=-1).detach()
            ensemble_predictions = self.ensemble(ensemble_input) # (num_predictors, B, L, hidden_dim)
            ensemble_targets = encoded[:, 1:].detach()
            ensemble_loss = ((ensemble_predictions - ensemble_targets)**2).mean()

            self.ensemble_optim.zero_grad()
            ensemble_loss.backward()
            self.ensemble_optim.step()




            kl = torch.distributions.kl_divergence(posteriors, priors).sum(-1)
            kl_loss = torch.clamp(kl, min=self.cfg.kl_free_nats).mean()

            latent_hidden_features = torch.cat([latents, hidden_states], dim=-1) # (B, L, ...)

            recon_mean = self.world_model.decoder(latent_hidden_features.flatten(0, 1))
            recons = torch.distributions.Normal(recon_mean, self.cfg.recon_std)
            recon_loss = -recons.log_prob(batch_next_states.flatten(0, 1)).sum((1, 2, 3)).mean()

            reward_mean = self.world_model.reward_model(latent_hidden_features).squeeze(-1) # (B, L)
            reward_predictions = torch.distributions.Normal(reward_mean, 1.0)
            reward_loss = -reward_predictions.log_prob(batch_rewards).mean()

            discount_target = self.cfg.gamma * (1.0 - batch_terms.float()) # 0.0 if terminated, otherwise 1.0
            discount_logits = self.world_model.discount_model(latent_hidden_features).squeeze(-1)
            discount_loss = torch.nn.functional.binary_cross_entropy_with_logits(discount_logits, discount_target)

            world_model_loss = recon_loss + self.cfg.kl_beta * kl_loss + reward_loss + discount_loss
                
            self.world_model_optim.zero_grad()
            world_model_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.cgn)
            self.world_model_optim.step()


            self.gradient_steps += 1

            with torch.no_grad():
                metrics["world_model_loss"].append(world_model_loss.item())
                metrics["reconstruction_loss"].append(recon_loss.item())
                metrics["reconstruction_mse"].append(((recons.mean - batch_next_states.flatten(0, 1)) ** 2).mean().item())
                metrics["kl"].append(kl.mean().item())
                metrics["kl_loss"].append(kl_loss.item())
                metrics["reward_loss"].append(reward_loss.item())
                metrics["prior_std"].append(priors.stddev.mean().item())
                metrics["posterior_std"].append(posteriors.stddev.mean().item())
                metrics["discount_loss"].append(discount_loss.item())

        # average each metric over the gradient updates
        return {name: float(np.mean(values)) for name, values in metrics.items()}
