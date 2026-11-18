
A repo for running RL algorithms on a range of environments

<br><br>

Contains functionality for stepping through environments, logging metrics, timing
- Algorithms are in `simple_rl/algos/` with associated configs in `simple_rl/configs/algo/`
- Environments are in `simple_rl/envs/` with associated configs in `simple_rl/configs/env/`
- Various shared classes (e.g. run, timer, logger, etc.) are in `simple_rl/core/`
- Shared PyTorch neural network helpers are in `simple_rl/nn/`

<br><br>


Algorithms can be run as:

`python run.py algo=ppo env=gym tite="testing_ppo" notes="trying out ppo on gym"`

using the parameters listed in the relevant config files

<br>

Parameters can be overwritten:

`python run.py algo=ppo gamma=0.99 lr=0.01 env=gym ...`

<br><br>

Inspired by other popular RL repos, as well as my [previous repo](https://github.com/NoahRGB/reinforcement-learning)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) 
- [SB](https://github.com/dlr-rm/stable-baselines3)
- [Embodied](https://github.com/danijar/embodied)