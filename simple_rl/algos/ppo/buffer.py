# NoahRGB 11/2026
#
# the on policy rollout buffer used by the ppo agent
# in algos/ppo/agent,.py
#
#

import numpy as np
import torch

class RolloutBuffer:

    def __init__(self, rollout_len, num_envs, state_dim, action_dim, state_dtype):
        self.rollout_len = rollout_len
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim

        if not isinstance(state_dtype, torch.dtype):
            self.state_dtype = torch.from_numpy(np.empty(0, dtype=state_dtype)).dtype 
        else:
            self.state_dtype = state_dtype

        self.states_buffer = torch.empty((self.rollout_len, self.num_envs, *self.state_dim), dtype=self.state_dtype)
        self.actions_buffer = torch.empty((self.rollout_len, self.num_envs, *self.action_dim), dtype=torch.float32)
        self.rewards_buffer = torch.empty((self.rollout_len, self.num_envs), dtype=torch.float32)
        self.next_states_buffer = torch.empty((self.rollout_len, self.num_envs, *self.state_dim), dtype=self.state_dtype)
        self.is_terms_buffer = torch.empty((self.rollout_len, self.num_envs), dtype=torch.float32)
        self.dones_buffer = torch.empty((self.rollout_len, self.num_envs), dtype=torch.float32)
        self.logprobs_buffer = torch.empty((self.rollout_len, self.num_envs), dtype=torch.float32)

        self.reset()

    def add(self, transitions):
        t = self.current_timestep
        assert t < self.rollout_len, "rollout buffer is full, call reset()"

        self.states_buffer[t] = torch.as_tensor(transitions["state"])
        self.actions_buffer[t] = torch.as_tensor(transitions["action"])
        self.rewards_buffer[t] = torch.as_tensor(transitions["reward"])
        self.next_states_buffer[t] = torch.as_tensor(transitions["next_state"])
        self.is_terms_buffer[t] = torch.as_tensor(transitions["is_term"])
        self.dones_buffer[t] = torch.as_tensor(transitions["done"])
        self.logprobs_buffer[t] = torch.as_tensor(transitions["log_prob"])

        self.current_timestep += 1

    def reset(self):
        self.current_timestep = 0