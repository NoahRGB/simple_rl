# NoahRGB 11/2026
#
# wraps Gymnasium envs in a core/env.py Env object
# (these can be used in a core/env.py VecEnv instance)
#

import gymnasium as gym

from simple_rl.core.env import Env

class GymEnv(Env):

    def __init__(self, env_id, seed=None, **kwargs):
        self.env = gym.make(env_id, **kwargs)
        self.seed = seed
        super().__init__(self.env.observation_space, self.env.action_space)

    def reset(self):
        states, info = self.env.reset(seed=self.seed)
        self.seed = None # only seed the first reset
        return states

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {"next_state": obs, "reward": reward, "done": terminated or truncated, "is_term": terminated} 