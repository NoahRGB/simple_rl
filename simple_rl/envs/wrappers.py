# NoahRGB 11/2026
#
#


import gymnasium as gym
import numpy as np

class ActionRepeat(gym.Wrapper):
    # repeat action N times and sum rewards over those repeats
    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.transpose(old.low, (2, 0, 1)),
            high=np.transpose(old.high, (2, 0, 1)),
            dtype=old.dtype,
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))

def remove_reward(env):
    return gym.wrappers.TransformReward(env, lambda r: 0.0)

def tanh_reward(env):
    return gym.wrappers.TransformReward(env, lambda r: np.tanh(r))

def resize_wrapper(env, shape):
    return gym.wrappers.ResizeObservation(env, tuple(shape))

def minigrid_image(env):
    import minigrid
    env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
    return minigrid.wrappers.ImgObsWrapper(env)

# supported wrappers and their classes
WRAPPERS = {
    "normalise_obs": gym.wrappers.NormalizeObservation,
    "normalise_reward": gym.wrappers.NormalizeReward,
    "frame_stack": gym.wrappers.FrameStackObservation,
    "atari": gym.wrappers.AtariPreprocessing,
    "greyscale": gym.wrappers.GrayscaleObservation,
    "resize": resize_wrapper,
    "channel_first": ChannelFirst,
    "tanh_reward": tanh_reward,
    "sticky_actions": gym.wrappers.StickyAction,
    "render_observation": gym.wrappers.AddRenderObservation,
    "action_repeat": ActionRepeat,
    "minigrid": minigrid_image,
    "remove_reward": remove_reward,
}