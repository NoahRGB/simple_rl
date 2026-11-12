# NoahRGB 11/2026
#
# defines the classes/methods used by environments
#
#

from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

def make_env(env_cfg, seed=None, **overrides):
    # convert the env type in cfg.env into the corresponding env class

    # env_cfg is cfg.env; overrides are options the algorithm sets, e.g. scale_actions=True
    kwargs = {**(env_cfg.get("kwargs") or {}), **overrides}

    if env_cfg.type == "gym":
        from simple_rl.envs.gym import GymEnv
        return GymEnv(env_cfg.id, seed=seed, wrappers=env_cfg.get("wrappers"), **kwargs)

    # if env_cfg.type == "gridworld":
    #     from simple_rl.envs.gridworld import GridWorld
    #     return GridWorld(seed=seed, **kwargs)

    raise ValueError(f"unknown env type {env_cfg.type}")


class Env(ABC):

    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self, action):
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

    def reset(self):
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions):
        outcomes, new_states = [], []
        # outcomes will store (next_state, reward, done, is_term)
        # new_state will store the true next state, which may involve resetting the env

        for env, action in zip(self.envs, actions):
            outcome = env.step(action)
            outcomes.append(outcome)
            new_states.append(env.reset() if outcome["done"] else outcome["next_state"])

        outcome = {key: np.stack([o[key] for o in outcomes]) for key in outcomes[0]}
        return outcome, np.stack(new_states)
