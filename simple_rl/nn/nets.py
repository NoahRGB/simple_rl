# NoahRGB 11/2026
#
#
#
#


import torch
from torch import nn
import gymnasium as gym

ACTIVATION_FUNCTIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


def build_net(architecture, input_shape):
    layers, x = [], torch.zeros(1, *input_shape)

    # for every layer, create the torch instance and activation func
    for layer in architecture:

        kind = layer["type"]
        if kind == "linear":
            mod = nn.Linear(x.shape[-1], layer["out"])
        elif kind == "conv":
            mod = nn.Conv2d(x.shape[1], layer["out"], layer["kernel"], layer.get("stride", 1))
        elif kind == "flatten":
            mod = nn.Flatten()
        elif kind == "layernorm":
            mod = nn.LayerNorm(x.shape[-1])
        else:
            raise ValueError(f"unknown layer type {kind}")

        layers.append(mod)
        x = mod(x) # keep track of sizes for next layer

        if "act" in layer:
            act = ACTIVATION_FUNCTIONS[layer["act"]]()
            layers.append(act)
            x = act(x)

    return nn.Sequential(*layers), tuple(x.shape[1:])

class CategoricalHead(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(CategoricalHead, self).__init__()

        self.head = torch.nn.Linear(input_size, output_size)

    def forward(self, inp):
        logits = self.head(inp)
        return torch.distributions.Categorical(logits=logits)

class GaussianHead(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(GaussianHead, self).__init__()

        self.mean_head = torch.nn.Linear(input_size, output_size)
        self.log_std_head = torch.nn.Linear(input_size, output_size)

    def forward(self, inp):
        mean = self.mean_head(inp)
        stdev = self.log_std_head(inp).exp()
        distribution = torch.distributions.Independent(torch.distributions.Normal(mean, stdev), 1)
        return distribution 



def detect_head(action_space, input_size):

    if isinstance(action_space, gym.spaces.Discrete):
        return CategoricalHead(input_size=input_size, output_size=action_space.n)

    elif isinstance(action_space, gym.spaces.Box):
        return GaussianHead(input_size=input_size, output_size=action_space.shape[0])

    raise ValueError(f"no head for action space {action_space}")
