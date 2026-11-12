# NoahRGB 11/2026
#
# a class responsible for managing the interaction between
# environments and agents over different timeframes
#
# e.g. run an agent in an environment for 10 timesteps or 10 episodes
#
#

from collections import deque
import numpy as np

from simple_rl.core import logger as log
from simple_rl.core.timer import timer


class Driver:

    def __init__(self, env):
        self.env = env

        # functions to call per timestep or per episode
        self.step_functions = []
        self.episode_functions = []
        self.batch_functions = []

        self.current_states = None
        self.is_first = None # keeps track of whether each subenv is in its first step of an episode
        self.episodes = [[] for _ in range(env.num_envs)] # the transitions that make up each subenvs current episode 
        self.total_steps = 0
        self.total_episodes = 0
        self.recent_rewards = deque(maxlen=100) # hold the last 100 rewards

    def on_step(self, function):
        self.step_functions.append(function)

    def on_episode(self, function):
        self.episode_functions.append(function)

    def on_batch(self, function):
        self.batch_functions.append(function)

    def log_episode(self, episode, env_idx, quiet=False):
        # an on_episode function: driver.on_episode(driver.log_episode)
        reward = episode["reward"].sum()
        self.recent_rewards.append(reward)
        log.write_now("episode/reward", reward, self.total_steps)
        log.write_now("episode/mean_reward_100", np.mean(self.recent_rewards), self.total_steps)
        if not quiet:
            print(f"episode {self.total_episodes} completed, timesteps {self.total_steps}, reward {reward}")

    def run(self, agent, steps=0, episodes=0):
        if self.current_states is None:
            self.current_states = self.env.reset()
            self.is_first = np.ones(self.env.num_envs, dtype=bool) # every subenv is starting a new episode

        # loop until the step or episode limit has beeen met
        steps_completed = episodes_completed = 0

        while steps_completed < steps or episodes_completed < episodes:

            with timer.section("policy"):
                # get an action from the agent given the current transition
                agent_out = agent(self.current_states, self.is_first)


            with timer.section("env_step"):
                # outcome is {next_state, reward, is_term, done}
                # new_state is the true next_state, which may have been reset if the subenv finished
                outcome, new_state = self.env.step(agent_out["action"])

            transitions = {"state": self.current_states, **agent_out, **outcome}

            for func in self.batch_functions:
                func(transitions)

            self.current_states = new_state
            self.is_first = outcome["done"] # subenvs that are done will be on their first step of a new episode

            for env_idx in range(self.env.num_envs):
                # individually call each on_step and on_episode for each subenv
                transition = {key: value[env_idx] for key, value in transitions.items()}


                for func in self.step_functions:
                    func(transition, env_idx)

                self.episodes[env_idx].append(transition)

                if transition["done"]:
                    completed_episode = {key: np.stack([transition[key] for transition in self.episodes[env_idx]]) for key in transition}

                    self.episodes[env_idx] = []
                    # count the episode first so that on_episode functions see its number (starting from 1)
                    episodes_completed += 1
                    self.total_episodes += 1

                    for func in self.episode_functions:
                        func(completed_episode, env_idx)

            steps_completed += self.env.num_envs
            self.total_steps += self.env.num_envs
