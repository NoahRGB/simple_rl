# NoahRGB 11/2026
#
# defines the classes/methods used by environments
#
#

from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


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
    # a class that allows multiple Env classes to be created and 
    # handled appropriately
    # 
    # see Gymnasium SyncVectorEnv (https://gymnasium.farama.org/api/vector/sync_vector_env/)

    def __init__(self, makers):
        assert len(makers) > 0
        self.envs = [make() for make in makers] # listof Env instnaces
        self.num_envs = len(self.envs)
        self.obs_space = self.envs[0].obs_space
        self.action_space = self.envs[0].action_space

    def reset(self) -> dict:
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions) -> dict:
        outcomes, new_states = [], []
        # outcomes will store (next_state, reward, done, is_term)
        # new_state will store the true next state, which may involve resetting the env

        for env, action in zip(self.envs, actions):
            outcome = env.step(action)
            outcomes.append(outcome)
            new_states.append(env.reset() if outcome["done"] else outcome["next_state"])

        outcome = {key: np.stack([o[key] for o in outcomes]) for key in outcomes[0]}
        return outcome, np.stack(new_states)


def get_action_size(space) -> tuple:

    if isinstance(space, gym.spaces.Discrete):
        return (space.n,)

    if isinstance(space, gym.spaces.Box):
        return (space.shape[0],)

    raise ValueError(f"unsupported action space {space}")