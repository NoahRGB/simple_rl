# NoahRGB 11/2026
#
# miscelaneous functions used by different algs
#

import gymnasium as gym

def is_discrete(space):
    return isinstance(space, gym.spaces.Discrete)


def get_action_size(space) -> int:

    if isinstance(space, gym.spaces.Discrete):
        return space.n

    if isinstance(space, gym.spaces.Box):
        return space.shape[0]

    raise ValueError(f"unsupported action space {space}")