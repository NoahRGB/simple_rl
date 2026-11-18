
import torch
import numpy as np

from simple_rl.core.utils import is_discrete, get_action_size
from simple_rl.algos.inverse.networks import DynamicsModel, ActorCriticNetwork


class InverseAgent:

    def __init__(self, cfg, obs_space, action_space, device):
        self.cfg = cfg; self.device = device

        self.action_space = action_space
        self.action_size = get_action_size(action_space)
        self.is_discrete = is_discrete(action_space)

        self.world_model = DynamicsModel(obs_space.shape, self.action_size, cfg.rnn_size, cfg.hidden_layer_size, cfg.latent_dim).to(device)
        self.world_model_optim = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_model_lr)

        self.ppo_net = ActorCriticNetwork(cfg.architecture, obs_space.shape, action_space).to(device)
        self.ppo_net_optim = torch.optim.Adam(self.ppo_net.parameters(), lr=cfg.ppo_lr)

    def policy(self, states, is_first):

        if self.cfg.learn_ppo:
            # return action/log probs given a batch of states (num_envs, *state_dim)
            with torch.no_grad():
                action_dist = self.ppo_net.get_actor(torch.as_tensor(states).to(self.device).float())
                actions = action_dist.sample()
                log_probs = action_dist.log_prob(actions)

            return {"action": actions.cpu().numpy(), "log_prob": log_probs.cpu().numpy()}

        return {"action": np.stack([self.action_space.sample() for _ in range(states.shape[0])])}

    def dynamics_update(self, buffer):

        metric_names = ["world_model_loss", "reconstruction_loss", "reconstruction_mse", "kl", "kl_loss", 
                        "prior_std", "posterior_std"]
        metrics = {name: [] for name in metric_names}

        for update_step in range(self.cfg.gradient_updates):

            # ================================
            # dynamics / world model updates
            # ================================

            windows, sample_size = buffer.sample(self.cfg.batch_size, self.device, self.cfg.prioritise_ends)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_terms = windows # (B, L, ..., )

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

            recon_logits = self.world_model.decoder(latent_hidden_features.flatten(0, 1))
            recons = torch.distributions.OneHotCategorical(logits=recon_logits)
            recon_loss = -recons.log_prob(batch_next_states.flatten(0, 1)).mean()

            world_model_loss = recon_loss + self.cfg.kl_beta * kl_loss
                
            self.world_model_optim.zero_grad()
            world_model_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.cgn)
            self.world_model_optim.step()

            with torch.no_grad():
                metrics["world_model_loss"].append(world_model_loss.item())
                metrics["reconstruction_loss"].append(recon_loss.item())
                metrics["reconstruction_mse"].append(((recons.mean - batch_next_states.flatten(0, 1)) ** 2).mean().item())
                metrics["kl"].append(kl.mean().item())
                metrics["kl_loss"].append(kl_loss.item())
                metrics["prior_std"].append(priors.stddev.mean().item())
                metrics["posterior_std"].append(posteriors.stddev.mean().item())

        # average each metric over the gradient updates
        return {name: float(np.mean(values)) for name, values in metrics.items()}

    def policy_update(self, buffer, intrinsic_normaliser):
        cfg = self.cfg
        T, N = buffer.rollout_len, buffer.num_envs

        states = buffer.states_buffer.float().to(self.device)
        next_states = buffer.next_states_buffer.float().to(self.device)
        actions, rewards, log_probs = buffer.actions_buffer.to(self.device), buffer.rewards_buffer.to(self.device), buffer.logprobs_buffer.to(self.device)
        is_terms, dones = buffer.is_terms_buffer.to(self.device), buffer.dones_buffer.to(self.device)

        rewards = torch.from_numpy(intrinsic_normaliser.normalise(rewards.cpu().numpy())).float().to(self.device)

        with torch.no_grad():
            # get state values across the rollout
            values = self.ppo_net.get_value(states.reshape(T*N, *buffer.state_dim)).reshape(T, N)
            next_values = self.ppo_net.get_value(next_states.reshape(T*N, *buffer.state_dim)).reshape(T, N)

        # calculate advantages+returns (using GAE)
        gae = 0.0
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(T)):
            delta = rewards[t] + cfg.gamma * next_values[t] * (1 - is_terms[t]) - values[t]
            gae = delta + cfg.gamma * cfg.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # once advantages have been calculated, the time dimension is no longer
        # needed and everything can be flattened
        states = states.reshape(T*N, *buffer.state_dim)
        actions = actions.reshape(T*N, *buffer.action_dim)
        log_probs = log_probs.reshape(T*N)
        advantages = advantages.reshape(T*N)
        returns = returns.reshape(T*N)

        metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

        for epoch in range(cfg.epochs):
            batch_size = T*N
            all_indices = np.arange(batch_size)
            np.random.shuffle(all_indices)
            for mb_start in range(0, batch_size, cfg.minibatch_size):
                mb_indices = all_indices[mb_start:mb_start+cfg.minibatch_size]
                mb_advantages = advantages[mb_indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_values = self.ppo_net.get_value(states[mb_indices]).squeeze(-1)
                new_dist = self.ppo_net.get_actor(states[mb_indices])
                new_log_prob = new_dist.log_prob(actions[mb_indices])

                log_ratios = new_log_prob - log_probs[mb_indices]
                ratios = torch.exp(log_ratios)
                clipped_ratios = torch.clamp(ratios, 1 - cfg.epsilon, 1 + cfg.epsilon)
                surrogate_obj = ratios * mb_advantages
                clipped_surrogate_obj = clipped_ratios * mb_advantages

                policy_loss = -torch.min(surrogate_obj, clipped_surrogate_obj).mean()
                entropy = new_dist.entropy().mean()
                value_loss = torch.nn.functional.mse_loss(new_values, returns[mb_indices])
                combined_loss = policy_loss - cfg.entropy_weight * entropy + cfg.value_weight * value_loss

                self.ppo_net_optim.zero_grad()
                combined_loss.backward()
                if cfg.ppo_cgn is not None:
                    torch.nn.utils.clip_grad_norm_(self.ppo_net.parameters(), cfg.ppo_cgn)
                self.ppo_net_optim.step()

                with torch.no_grad():
                    # how far the policy has moved from the one that collected the rollout
                    approx_kl = ((ratios - 1) - log_ratios).mean()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl.item())

        return {key: float(np.mean(values)) for key, values in metrics.items()}

