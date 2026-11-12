# NoahRGB 11/2026
#
# a simple grid world / maze gymnasium environment
#
#


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


FOUR_ROOMS = """
#########
#S..#...#
#...#...#
#.......#
##.###.##
#...#...#
#.......#
#########
"""

FOUR_ROOMS_LARGE = """
#################
#S......#.......#
#.......#.......#
#...............#
#.......#.......#
#.......#.......#
#.......#.......#
#.......#.......#
#####.#####.#####
#.......#.......#
#.......#.......#
#.......#.......#
#...............#
#.......#.......#
#.......#.......#
#.......#.......#
#################
"""


MOVES = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

# colours
WALL = np.array([40, 40, 40], dtype=np.uint8)
FLOOR = np.array([220, 220, 220], dtype=np.uint8)
PLAYER = np.array([200, 30, 30], dtype=np.uint8)


class GridWorld(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, layout=FOUR_ROOMS_LARGE, img_size=64, max_steps=500, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.img_size = img_size
        self.max_steps = max_steps

        lines = layout.strip("\n").split("\n")
        grid = np.array([list(line) for line in lines])
        self.walls = grid == "#"
        self.height, self.width = self.walls.shape
        self.cell_px = self.img_size // max(self.height, self.width)

        starts = np.argwhere(grid == "S")
        assert len(starts) == 1, "only one start pos (S) allowed"
        self.start = tuple(int(v) for v in starts[0])

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

        self.pos = None
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        self.t = 0
        obs = self._render_obs(self.pos)
        return obs, self._info()

    def step(self, action):

        new_row = self.pos[0] + MOVES[action][0]
        new_col = self.pos[1] + MOVES[action][1]
        if not self.walls[new_row, new_col]:
            self.pos = (int(new_row), int(new_col)) 

        self.t += 1

        obs = self._render_obs(self.pos)
        reward = 0.0
        terminated = False # never terminated
        truncated = self.t >= self.max_steps
        return obs, reward, terminated, truncated, self._info()

    def render(self):
        return self._render_obs(self.pos)

    def close(self):
        pass

    def render_at(self, pos):
        return self._render_obs(pos)

    def _render_obs(self, pos):

        # make small image of wall/floor then repeat to self.cell_px size
        small_img = np.where(self.walls[..., None], WALL, FLOOR)
        img = small_img.repeat(self.cell_px, axis=0).repeat(self.cell_px, axis=1) # (H*C, W*C, 3)

        # set the player position
        row, col = pos
        img[row * self.cell_px : (row + 1) * self.cell_px, col * self.cell_px : (col + 1) * self.cell_px] = PLAYER

        # pad to be size (self.img_size, self.img_size), e.g. 64x64
        obs = np.empty((self.img_size, self.img_size, 3), dtype=np.uint8)
        obs[:] = WALL
        img_height, img_width = img.shape[:2]
        obs[:img_height, :img_width] = img

        # plt.imshow(obs)
        # plt.show()

        return obs

    def _info(self):
        # keep the true pos in the info logs
        return {"pos": tuple(self.pos) if self.pos is not None else None, "t": self.t}


if __name__ == "__main__":
    env = GridWorld()
    env.reset(seed=0)
    env.step(0)
    assert env.pos == env.start
    env.step(3)
    assert env.pos == env.start
    env.step(1)
    assert env.pos == (1, 2)

    from gymnasium.utils.env_checker import check_env
    check_env(GridWorld())