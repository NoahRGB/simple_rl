# NoahRGB 11/2026
#
# contains the replay memory class used by the
# dreamer v2 agent
#
#



from collections import deque
import numpy as np
import torch

class EpisodicReplayMemory:
    # stores whole episodes and samples windows within the stored
    # episodes
    def __init__(self, size: int, state_dim: tuple, action_dim: tuple, state_dtype, window_size: int):
        self.max_size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dtype = state_dtype
        self.window_size = window_size

        self.episodes = deque()
        self.total_stored_timesteps = 0

    def add(self, states, actions, rewards, next_states, term):
        new_episode_length = states.shape[0]
        if new_episode_length > self.window_size:
            while self.episodes and self.total_stored_timesteps + new_episode_length > self.max_size:
                removed_ep = self.episodes.popleft()
                self.total_stored_timesteps -= removed_ep[0].shape[0]

            self.total_stored_timesteps += new_episode_length
            self.episodes.append((states, actions, rewards, next_states, term))

    def sample(self, count, device, prioritise_ends=False):
        lengths = np.array([ep[0].shape[0] - self.window_size + 1 for ep in self.episodes])
        valid = np.clip(lengths, 0, None)

        if valid.sum() == 0:
            return tuple(), 0

        probs = valid / valid.sum()
        idx = np.random.choice(len(self.episodes), size=count, p=probs)

        windows = []
        for i in idx: 
            episode = self.episodes[i]

            # add extra points to sample from that are past the end and then clip to end of last window
            upper = valid[i] + (self.window_size if prioritise_ends else 0)
            start_idx = min(np.random.randint(0, upper), valid[i] - 1)
            windows.append(tuple([arr[start_idx:start_idx+self.window_size] for arr in episode]))

        return tuple(torch.from_numpy(np.stack(field)).to(device) for field in zip(*windows)), count 

