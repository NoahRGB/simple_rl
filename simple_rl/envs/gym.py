# NoahRGB 11/2026
#
# wraps Gymnasium envs in a core/env.py Env object
# (these can be used in a core/env.py VecEnv instance)
#

import gymnasium as gym
import numpy as np
import ale_py
import shimmy

from simple_rl.envs.wrappers import WRAPPERS
from simple_rl.core.env import Env
from simple_rl.envs.gridworld import GridWorld

try:
    import minigrid
except:
    print("minigrid not available")

try:
    gym.register(
        id="ThrustEnv",
        entry_point="src.thrust_gym.thrust_gymenv:ThrustEnv",
    )
except:
    print("thrust not available")


gym.register(id="MyGridWorld", entry_point=GridWorld)
gym.register_envs(shimmy)

class GymEnv(Env):

    def __init__(self, env_id, seed=None, scale_actions=False, wrappers=None, **kwargs):
        env = gym.make(env_id, **kwargs)
        for name, wrapper_kwargs in (wrappers or {}).items():
            env = WRAPPERS[name](env, **(wrapper_kwargs or {}))

        action_space = env.action_space

        # only continuous action spaces can be rescaled
        if scale_actions and isinstance(action_space, gym.spaces.Box):
            assert np.isfinite(action_space.low).all() and np.isfinite(action_space.high).all(), \
                f"action bounds must be finite to rescale, got {action_space}"

            # agent operates in [-1, 1] and RescaleAction maps that onto the env [low, high]
            env = gym.wrappers.RescaleAction(env, min_action=np.float32(-1.0), max_action=np.float32(1.0))
            action_space = env.action_space # box [-1, 1]

            # clips anything outside [-1, 1] before it gets rescaled
            env = gym.wrappers.ClipAction(env)

        self.env = env
        self.seed = seed

        super().__init__(self.env.observation_space, action_space)

    def reset(self):
        states, info = self.env.reset(seed=self.seed)
        self.seed = None # only seed the first reset
        return states

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {"next_state": obs, "reward": reward, "done": terminated or truncated, "is_term": terminated}
