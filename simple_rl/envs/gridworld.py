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

# 41x41, 4x4 rooms of 9x9 connected like a maze (with a few loops)
SIXTEEN_ROOMS = """
#########################################
#S........#.........#.........#.........#
#...................#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.............................#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
###########################.###.#########
#.........#.........#...................#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#...................#.........#.........#
#.........#...................#.........#
#.........#.........#.........#.........#
######.####.##########################.##
#.........#.........#.........#.........#
#.........#.........#...................#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#...................#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
###.#########.#######################.###
#...................#.........#.........#
#.........#.........#.........#.........#
#.........#.........#...................#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#.........#.........#.........#
#.........#...................#.........#
#.........#.........#.........#.........#
#########################################
"""

# layouts can be chosen by name in the env config, e.g. kwargs: {layout: sixteen_rooms}
LAYOUTS = {
    "four_rooms": FOUR_ROOMS,
    "four_rooms_large": FOUR_ROOMS_LARGE,
    "sixteen_rooms": SIXTEEN_ROOMS,
}

MOVES = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

# colours
WALL = np.array([40, 40, 40], dtype=np.uint8)
FLOOR = np.array([220, 220, 220], dtype=np.uint8)
PLAYER = np.array([200, 30, 30], dtype=np.uint8)


class GridWorld(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, layout=SIXTEEN_ROOMS, img_size=64, max_steps=500, render_mode="rgb_array", obs_mode="rgb", reward_mode="none"):
        super().__init__()
        assert obs_mode in ("rgb", "onehot"), f"unknown obs_mode {obs_mode}"
        assert reward_mode in ("none", "count"), f"unknown reward_mode {reward_mode}"
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.img_size = img_size
        self.max_steps = max_steps

        layout = LAYOUTS.get(layout, layout) # a layout name or a raw layout string
        lines = layout.strip("\n").split("\n")
        grid = np.array([list(line) for line in lines])
        self.walls = grid == "#"
        self.height, self.width = self.walls.shape
        self.cell_px = self.img_size // max(self.height, self.width)
        assert self.cell_px >= 1, f"{self.height}x{self.width} layout is too big for img_size {self.img_size}"

        starts = np.argwhere(grid == "S")
        assert len(starts) == 1, "only one start pos (S) allowed"
        self.start = tuple(int(v) for v in starts[0])

        if obs_mode == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8)
        else:
            # one entry per grid cell (row-major, index = row * width + col)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.height * self.width,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.pos = None
        self.t = 0

        # lifelong visit counts (never reset between episodes), used by reward_mode="count"
        self.visit_counts = np.zeros((self.height, self.width))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        self.t = 0
        obs = self.obs_at(self.pos)
        return obs, self._info()

    def step(self, action):

        new_row = self.pos[0] + MOVES[action][0]
        new_col = self.pos[1] + MOVES[action][1]
        if not self.walls[new_row, new_col]:
            self.pos = (int(new_row), int(new_col)) 
        self.visit_counts[self.pos] += 1

        self.t += 1

        obs = self.obs_at(self.pos)
        reward = 1 / np.sqrt(self.visit_counts[self.pos]) 
        terminated = False # never terminated
        truncated = self.t >= self.max_steps
        return obs, reward, terminated, truncated, self._info()

    def render(self):
        return self._render_obs(self.pos)

    def close(self):
        pass

    def render_at(self, pos):
        return self._render_obs(pos)

    def obs_at(self, pos):
        # the observation the agent would see at pos (depends on obs_mode)
        if self.obs_mode == "rgb":
            return self._render_obs(pos)
        return self._onehot_obs(pos)

    def _onehot_obs(self, pos):
        obs = np.zeros(self.height * self.width, dtype=np.float32)
        row, col = pos
        obs[row * self.width + col] = 1.0
        return obs

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

    env = GridWorld(obs_mode="onehot")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.height * env.width,) and obs.sum() == 1.0
    assert obs[env.start[0] * env.width + env.start[1]] == 1.0
    check_env(GridWorld(obs_mode="onehot"))

    env = GridWorld(reward_mode="count")
    env.reset(seed=0)
    _, r, *_ = env.step(0) # bump wall at start: first visit counted, no div by zero
    assert r == 1.0
    _, r, *_ = env.step(0)
    assert np.isclose(r, 1 / np.sqrt(2))
    _, r, *_ = env.step(1) # new cell
    assert r == 1.0
    check_env(GridWorld(reward_mode="count"))