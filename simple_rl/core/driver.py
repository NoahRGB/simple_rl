# NoahRGB 11/2026
#
# a class responsible for managing the interaction between
# environments and agents over different timeframes
#
# e.g. run an agent in an environment for 10 timesteps or 10 episodes
#
#

import numpy as np

from simple_rl.core import logger as log
from simple_rl.core.timer import timer


def log_episode(episode, env_idx, quiet=False):

    reward = episode["reward"].sum()
    log.add("episode/avg_reward", reward, agg="mean")
    log.add("episode/avg_length", len(episode["reward"]) - 1, agg="mean")
    if not quiet:
        print(f"episode completed, reward {reward}")

class Driver:

    def __init__(self, env):
        self.env = env

        # functions to call per timestep or per episode
        self.step_functions = []
        self.episode_functions = []

        self.current_transition = None
        self.episodes = [[] for _ in range(env.num_envs)] # the transitions that make up each subenvs current episode 
        self.total_steps = 0
        self.total_episodes = 0

    def on_step(self, function):
        self.step_functions.append(function)

    def on_episode(self, function):
        self.episode_functions.append(function)

    def run(self, agent, steps=0, episodes=0):
        if self.current_transition is None:
            self.current_transition = self.env.reset()

        # loop until the step or episode limit has beeen met
        steps_completed = episodes_completed = 0

        while steps_completed < steps or episodes_completed < episodes:

            this_transition = self.current_transition # (s, r, is_first, is_last, is_term)

            with timer.section("policy"):
                # get an action from the agent given the current transition
                agent_out = agent(this_transition)

            # zero out actions for envs that have just terminated
            action = agent_out["action"].copy()
            action[this_transition["is_last"]] = 0
            agent_out["action"] = action

            this_transition = {**this_transition, **agent_out}

            for env_idx in range(self.env.num_envs):
                # individually call each on_step and on_episode for each subenv
                env_transition = {key: value[env_idx] for key, value in this_transition.items()}

                for func in self.step_functions:
                    func(env_transition, env_idx)

                self.episodes[env_idx].append(env_transition)

                if env_transition["is_last"]:
                    completed_episode = {key: np.stack([transition[key] for transition in self.episodes[env_idx]]) for key in env_transition}

                    self.episodes[env_idx] = []
                    for func in self.episode_functions:
                        func(completed_episode, env_idx)

                    episodes_completed += 1
                    self.total_episodes += 1

            steps_completed += self.env.num_envs
            self.total_steps += self.env.num_envs

            with timer.section("env_step"):
                self.current_transition = self.env.step(this_transition["action"])