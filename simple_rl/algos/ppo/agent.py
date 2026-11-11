# NoahRGB 11/2026
#
# the PPO agent. owns the network + optimiser and contains
# all of the PPO maths (policy, GAE, clipped update)
#
#

import numpy as np
import torch

from simple_rl.algos.ppo.networks import ActorCriticNetwork


class PPOAgent:

    def __init__(self, cfg, obs_space, action_space):
        self.cfg = cfg # configs/algo/ppo.yaml
        self.network = ActorCriticNetwork(cfg.architecture, obs_space.shape, action_space)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=cfg.lr)

    def policy(self, states):
        # return action/log probs given a batch of states (num_envs, *state_dim)
        with torch.no_grad():
            action_dist = self.network.get_actor(torch.as_tensor(states).float())
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)

        return {"action": actions.cpu().numpy(), "log_prob": log_probs.cpu().numpy()}

    def update(self, buffer):
        cfg = self.cfg
        T, N = buffer.rollout_len, buffer.num_envs

        states = buffer.states_buffer.float()
        next_states = buffer.next_states_buffer.float()
        actions, rewards, log_probs = buffer.actions_buffer, buffer.rewards_buffer, buffer.logprobs_buffer
        is_terms, dones = buffer.is_terms_buffer, buffer.dones_buffer

        with torch.no_grad():
            # get state values across the rollout
            values = self.network.get_value(states.reshape(T*N, *buffer.state_dim)).reshape(T, N)
            next_values = self.network.get_value(next_states.reshape(T*N, *buffer.state_dim)).reshape(T, N)

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

                new_values = self.network.get_value(states[mb_indices]).squeeze(-1)
                new_dist = self.network.get_actor(states[mb_indices])
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

                self.optim.zero_grad()
                combined_loss.backward()
                if cfg.cgn is not None:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.cgn)
                self.optim.step()

                with torch.no_grad():
                    # how far the policy has moved from the one that collected the rollout
                    approx_kl = ((ratios - 1) - log_ratios).mean()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl.item())

        return {key: float(np.mean(values)) for key, values in metrics.items()}
