# NoahRGB 11/2026
#
# contains the logic for the dreamer v1 agent
# gradient updates, policy, etc.
#
#
#

import copy
import torch
import numpy as np

from simple_rl.core.utils import is_discrete, get_action_size
from simple_rl.algos.dreamer_v2.networks import WorldModel, Actor, ValueModel

class DreamerV2Agent:

    def __init__(self, cfg, obs_space, action_space, device):
        self.cfg = cfg # configs/algo/dreamer_v2.yaml
        self.device = device

        self.action_space = action_space
        self.action_size = get_action_size(action_space) 
        self.is_discrete = is_discrete(action_space)

        self.latent_dim = cfg.latent_variables * cfg.latent_classes

        self.world_model = WorldModel(obs_space.shape, self.action_size, self.latent_dim, cfg.rnn_size, cfg.hidden_layer_size).to(device)
        self.actor = Actor(self.latent_dim, cfg.rnn_size, self.action_size, self.is_discrete, cfg.min_actor_stdev, cfg.hidden_layer_size).to(device)
        self.value_model = ValueModel(self.latent_dim, cfg.rnn_size, cfg.hidden_layer_size).to(device)
        self.target_value_model = copy.deepcopy(self.value_model) 
        self.target_value_model.requires_grad_(False)

        self.world_model_optim = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_model_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.action_model_lr)
        self.value_model_optim = torch.optim.Adam(self.value_model.parameters(), lr=cfg.value_model_lr)
        self.gradient_steps = 0

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

    def policy(self, states, is_first):
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
                action = action_dist.sample()
                self.previous_action = action
            else:
                action = action_dist.rsample() 
                self.previous_action = action

        return {"action": action.cpu().numpy()}


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
            if is_discrete:
                batch_actions = torch.nn.functional.one_hot(batch_actions.long(), num_classes=self.action_size).float()
            else:
                batch_actions = batch_actions.float()
            batch_rewards = batch_rewards.float()

            encoded = self.world_model.encoder(batch_next_states.flatten(0, 1)).unflatten(0, (self.cfg.batch_size, self.cfg.sequence_length))
            hidden_state = torch.zeros(self.cfg.batch_size, self.cfg.rnn_size).to(self.device)
            latent = torch.zeros(self.cfg.batch_size, self.latent_dim).to(self.device)

            latents, hidden_states, prior_logits, posterior_logits = [], [], [], []
            for t in range(self.cfg.sequence_length):
                hidden_state = self.world_model.rnn(torch.cat([latent, batch_actions[:, t]], dim=-1), hidden_state)
                prior = self.world_model.prior(hidden_state)
                posterior = self.world_model.posterior(torch.cat([hidden_state, encoded[:, t]], dim=-1))
                latent = posterior.rsample().flatten(-2)

                hidden_states.append(hidden_state); latents.append(latent)
                prior_logits.append(prior.logits); posterior_logits.append(posterior.logits)

            hidden_states = torch.stack(hidden_states, 1)
            latents = torch.stack(latents, 1)
            prior_logits = torch.stack(prior_logits, 1)
            posterior_logits = torch.stack(posterior_logits, 1)

            priors = torch.distributions.OneHotCategoricalStraightThrough(logits=prior_logits)
            posteriors = torch.distributions.OneHotCategoricalStraightThrough(logits=posterior_logits)

            prior_kl = torch.distributions.kl_divergence(
                torch.distributions.OneHotCategoricalStraightThrough(logits=posterior_logits.detach()), priors
            ).sum(-1)

            posterior_kl = torch.distributions.kl_divergence(
                posteriors, 
                torch.distributions.OneHotCategoricalStraightThrough(logits=prior_logits.detach())
            ).sum(-1) 

            kl_loss = (self.cfg.alpha * prior_kl.mean().clamp(min=self.cfg.kl_free_nats) +
                        (1 - self.cfg.alpha) * posterior_kl.mean().clamp(min=self.cfg.kl_free_nats))

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



            # ================================
            # behvaiour updates
            # ================================

            # imagine trajectories from each s_t
            imagination_latents = latents.detach().flatten(0, 1) # (B*L, latent_dim)
            imagination_hidden_states = hidden_states.detach().flatten(0, 1) # (B*L, hidden_dim)

            imagined_latents, imagined_actions, imagined_hidden_states = [imagination_latents], [], [imagination_hidden_states]
            for imagination_step in range(self.cfg.imagination_horizon):
                # imagination comes from the learned prior distribution
   
                action_dist = self.actor(torch.cat([imagination_latents, imagination_hidden_states], dim=-1))
                actions = action_dist.rsample()

                imagination_hidden_states = self.world_model.rnn(torch.cat([imagination_latents, actions], dim=-1), imagination_hidden_states)
                imagination_latents = self.world_model.prior(imagination_hidden_states).rsample().flatten(-2)

                imagined_latents.append(imagination_latents)
                imagined_actions.append(actions)
                imagined_hidden_states.append(imagination_hidden_states)

            imagined_latents = torch.stack(imagined_latents) # (H+1, B*L, latent_dim)
            imagined_hidden_states = torch.stack(imagined_hidden_states) # (H+1, B*L, hidden_dim)
            imagined_actions = torch.stack(imagined_actions) # (H, B*L, action_dim)

            imagination_features = torch.cat([imagined_latents, imagined_hidden_states], dim=-1) # (H+1, B*L, latent_dim+hidden_dim)
            imagined_rewards = self.world_model.reward_model(imagination_features).squeeze(-1) # (H+1, B*L)
            imagined_target_state_values = self.target_value_model(imagination_features)

            # recompute so dynamics grads dont carry through
            policy_dist = self.actor(imagination_features[:-1].detach())
            action_log_probs = policy_dist.log_prob(imagined_actions.detach())
            action_entropy = policy_dist.entropy()

            # transforms logit into [0, 1] so it can be used as a discoun factor
            imagination_discounts = torch.sigmoid(self.world_model.discount_model(imagination_features).squeeze(-1)) # (H+1, B*L)

            value_estimates = [imagined_target_state_values[-1]]
            for t in reversed(range(self.cfg.imagination_horizon)):
                value_estimates.append(
                    imagined_rewards[t+1] + imagination_discounts[t+1] * ((1-self.cfg.lam) * imagined_target_state_values[t+1] + self.cfg.lam * value_estimates[-1])
                )

            # the last timestep is just used for a target
            value_estimates = torch.stack(value_estimates[::-1][:-1]) # (H, B*L)

            # Dreamer paper says:
            # "we weigh the loss terms of the actor and critic by the cumulative predicted
            # discount factors to softly account for the possibility of episode ends"
            weights = torch.cumprod(
                torch.cat([torch.ones_like(imagination_discounts[:1]), imagination_discounts[1:-1]], 0), 0
            ).detach() # (H, B*L)


            reinforce_loss = -(action_log_probs * (value_estimates - imagined_target_state_values[:-1]).detach())
            dynamics_loss = -value_estimates
            entropy_regularisation = -self.cfg.eta * action_entropy

            actor_loss = (weights * (self.cfg.rho * reinforce_loss + (1 - self.cfg.rho) * dynamics_loss + entropy_regularisation)).mean()

            value_model_preds = self.value_model(imagination_features[:-1].detach()) # value model shouldd not use the graph built up from imagintion
            value_model_loss = (weights * 0.5 * (value_model_preds - value_estimates.detach())**2).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.cgn)
            self.actor_optim.step()
            
            self.value_model_optim.zero_grad()
            value_model_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.cfg.cgn)
            self.value_model_optim.step()

            if self.gradient_steps % self.cfg.target_update_interval == 0:
                self.target_value_model.load_state_dict(self.value_model.state_dict())

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
                metrics["actor_loss"].append(actor_loss.item())
                metrics["value_model_loss"].append(value_model_loss.item())
                metrics["discount_loss"].append(discount_loss.item())

        # average each metric over the gradient updates
        return {name: float(np.mean(values)) for name, values in metrics.items()}
