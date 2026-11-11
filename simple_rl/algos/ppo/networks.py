# NoahRGB 11/2026
#
# contains the torch actor critic network used by the
# ppo agent in algos/ppo/agent.py 
#
#


import torch
import gymnasium as gym

from simple_rl.nn.nets import build_net, detect_head

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, architecture: dict, state_dim: tuple, action_space: gym.Space):
        super(ActorCriticNetwork, self).__init__()

        if "shared_enc" in architecture:
            self.using_shared_enc = True
            self.shared_enc, self.actor_enc_output = build_net(architecture["shared_enc"], input_shape=state_dim)
            self.value_enc_output = self.actor_enc_output
        else:
            self.using_shared_enc = False
            self.actor_enc, self.actor_enc_output = build_net(architecture["actor_enc"], input_shape=state_dim)
            self.value_enc, self.value_enc_output = build_net(architecture["value_enc"], input_shape=state_dim)
            
        self.actor_head = detect_head(action_space=action_space, input_size=self.actor_enc_output[0])
        self.value_head = torch.nn.Linear(self.value_enc_output[0], 1)

    def get_actor(self, inp):
        if self.using_shared_enc:
            enc_out = self.shared_enc(inp)
        else:
            enc_out = self.actor_enc(inp)
        return self.actor_head(enc_out)
    
    def get_value(self, inp):
        if self.using_shared_enc:
            enc_out = self.shared_enc(inp)
        else:
            enc_out = self.value_enc(inp)
        return self.value_head(enc_out)    