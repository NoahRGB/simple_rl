# NoahRGB 11/2026
#
# wraps Gymnasium envs in a core/env.py Env object
#
#
#
#
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

        # (s_t, r_t, is_first, is_last, is_terminal, a_t)
        return {"obs": states, "reward": 0.0, "is_first": True, "is_last": False, "is_terminal": False} 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {"obs": obs, "reward": reward, "is_first": False, "is_last": terminated or truncated, "is_terminal": terminated} 