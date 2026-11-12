# NoahRGB 11/2026
#
# contains the replay memory class used by the
# dreamer v1 agent
#
#



from collections import deque
import numpy as np
import torch

class EpisodicReplayMemory:
    # stores whole episodes and samples windows within the stored
    # episodes
    def __init__(self, size: int, state_dim: tuple, action_dim: tuple, state_dtype):
        self.max_size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dtype = state_dtype

        self.episodes = deque()
        self.total_stored_timesteps = 0

    def add(self, states, actions, rewards, next_states, term):
        new_episode_length = states.shape[0]
        while self.episodes and self.total_stored_timesteps + new_episode_length > self.max_size:
            removed_ep = self.episodes.popleft()
            self.total_stored_timesteps -= removed_ep[0].shape[0]

        self.total_stored_timesteps += new_episode_length
        self.episodes.append((states, actions, rewards, next_states, term))

    def sample(self, count, window_size, device):
        lengths = np.array([ep[0].shape[0] - window_size + 1 for ep in self.episodes])
        valid = np.clip(lengths, 0, None)

        if valid.sum() == 0:
            return tuple(), 0

        probs = valid / valid.sum()
        idx = np.random.choice(len(self.episodes), size=count, p=probs)

        windows = []
        for i in idx: 
            episode = self.episodes[i]
            episode_len = episode[0].shape[0]
            start_idx = np.random.randint(0, valid[i])
            windows.append(tuple([arr[start_idx:start_idx+window_size] for arr in episode]))

        return tuple(torch.from_numpy(np.stack(field)).to(device) for field in zip(*windows)), count 

