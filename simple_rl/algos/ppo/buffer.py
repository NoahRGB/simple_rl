
import torch

class RolloutBuffer:

    def __init__(self, rollout_len, num_envs, state_dim, action_dim, state_dtype):
        self.rollout_len = rollout_len
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dtype = state_dtype
        self.reset()

    def add(self, transition, env_idx):
        state, 


    def reset(self):
        states_buffer = torch.empty((self.rollout_len, self.num_envs, *self.state_dim), dtype=self.state_dtype)
        actions_buffer = torch.empty((self.rollout_len, self.num_envs, *self.action_dim), dtype=torch.float32)
        rewards_buffer = torch.empty((self.rollout_len, self.num_envs), dtype=torch.float32)
        next_states_buffer = torch.empty((self.rollout_len, self.num_envs, *self.state_dim), dtype=self.state_dtype)
        is_terms_buffer = torch.empty((self.rollout_len, self.num_envs), dtype=torch.float32)
        dones_buffer = torch.empty((self.rollout_len, self.num_envs), dtype=torch.float32)
        logprobs_buffer = torch.empty((self.rollout_len, self.num_envs), dtype=torch.float32)