# NoahRGB 11/2026
#
#
#
#
#

from abc import ABC, abstractmethod
import numpy as np


# transitions are stored as:
# (s_t, r_t, is_first, is_last, is_terminal, a_t)
#
# r_t is the reward for landing in s_t
# a_t is the action chosen to apply in s_t
# is_first determines if this is the first transition of the episode
# is_last determines if the episode ended
# is_terminal determines if the episode terminated (instead of truncated)
#



class Env(ABC):

    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space

    @abstractmethod
    def reset(self) -> dict:
        ...

    @abstractmethod
    def step(self, action) -> dict:
        ...


class VecEnv:

    def __init__(self, makers):
        assert len(makers) > 0
        self.envs = [make() for make in makers] # listof Env instnaces
        self.num_envs = len(self.envs)
        self.obs_space = self.envs[0].obs_space
        self.action_space = self.envs[0].action_space

        self._needs_reset = np.ones(self.num_envs, dtype=bool) # tracks which subenv is waiting for a reset

    def reset(self) -> dict:
        self._needs_reset[:] = True
        return self.step(None)

    def step(self, actions) -> dict:
        transitions = []
        for env_idx, env in enumerate(self.envs):
            if self._needs_reset[env_idx]:
                transition = env.reset()
            else:
                transition = env.step(actions[env_idx])

            self._needs_reset[env_idx] = transition["is_last"]
            transitions.append(transition)
        return {key: np.stack([transition[key] for transition in transitions]) for key in transitions[0]}