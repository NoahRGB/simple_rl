# NoahRGB 11/2026
#
#
#
#

import torch
import numpy as np

from simple_rl.core.utils import is_discrete, get_action_size
from simple_rl.algos.density.networks import WorldModel, Actor, ValueModel
from simple_rl.core.timer import timer

class DensityAgent:

    def __init__(self, cfg, obs_space, action_space, device, grid_width, grid_height, grid_walls):
        self.cfg = cfg 
        self.device = device

        self.action_space = action_space
        self.action_size = get_action_size(action_space) 

        self.is_discrete = is_discrete(action_space)
        self.world_model = WorldModel(obs_space.shape, self.action_size, cfg.latent_dim, cfg.rnn_size, cfg.hidden_layer_size).to(device)

        self.z_hist = []
        self.pos_hist = []

        self.world_model_optim = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_model_lr)
        self.gradient_steps = 0

    def _preprocess(self, obs):
        # convert to torch tensor and scale
        tens = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        return (tens / 255.0) - 0.5

    def policy(self, states, is_first, positions=None):
        num_envs = states.shape[0]

        with torch.no_grad():
            rand_actions = np.stack([self.action_space.sample() for _ in range(num_envs)]) 
            action = torch.as_tensor(rand_actions).to(self.device)

            # s = self._preprocess(states[0]).unsqueeze(0)
            # enc = self.world_model.encoder(s).squeeze(0)
            # self.z_hist.append(enc.cpu().numpy())
            self.pos_hist.append(positions[0])


        return {"action": action.cpu().numpy()}


    def update(self, buffer):
        metric_names = ["world_model_loss", "reconstruction_loss", "reconstruction_mse", "kl", "kl_loss", "reward_loss",
                        "prior_std", "posterior_std", "discount_loss"]
        metrics = {name: [] for name in metric_names}

        for update_step in range(self.cfg.gradient_updates):

            # ================================
            # dynamics / world model updates
            # ================================

            windows, sample_size = buffer.sample(self.cfg.batch_size, self.device, self.cfg.prioritise_ends)
            if windows and len(windows[0]) == self.cfg.batch_size:
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
