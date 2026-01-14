# This module defines neural network architectures for function approximation in RL algorithms
__all__ = [
    "ActionValue",
    "StateValue",
    "LyapunovValue",
    "ActionValueDistri",
    "StochaPolicy",
    "DetermPolicy",
]

import torch
import torch.nn as nn
import warnings
from RL.utils.common_utils import get_activation_func
from RL.utils.act_distribution_cls import Action_Distribution_Cls
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Create MLP network.
    sizes: Layer dimensions (e.g., [obs_dim+act_dim, 64, 64, 1])
    activation: Hidden layer activation
    output_activation: Output layer activation (default: Identity)
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]

    model = nn.Sequential(*layers)


    return model


#################### Value Functions ####################
class ActionValue(nn.Module):
    """Action value function (Q-function): obs+act → Q-value."""
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)
    

class StateValue(nn.Module):
    """State value function (V-function): obs → V-value."""
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)
    

class LyapunovValue(nn.Module):
    """Lyapunov function (positive output): input → squared sum of output vector."""
    def __init__(self, **kwargs):
        super().__init__()
        input_dim = kwargs["input_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        output_dim = kwargs["output_dim"]

        self.lya = mlp(
            [input_dim] + list(hidden_sizes) + [output_dim],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, input_obs):
        lya = torch.pow(self.lya(input_obs), 2).sum(dim=-1, keepdim=True)
        return torch.squeeze(lya, -1)
    

class ActionValueDistri(nn.Module):
    """Continuous Q-value distribution: obs+act → mean + std of Q-value."""
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_std = torch.nn.functional.softplus(value_std)
        return torch.cat((value_mean, value_std), dim=-1)
    
#################### Policy Functions ####################
class StochaPolicy(nn.Module, Action_Distribution_Cls):
    """Stochastic policy: obs → action mean + std (clamped log std)."""
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        self.policy = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim * 2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]

        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        logits = self.policy(obs)
        action_mean, action_log_std = torch.chunk(logits, chunks=2, dim=-1)
        action_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class DetermPolicy(nn.Module, Action_Distribution_Cls):
    """Deterministic policy: obs → action (clamped to action space via tanh)."""
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        self.pi = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        action = (
            (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.pi(obs)) 
            + (self.act_high_lim + self.act_low_lim) / 2
        )
        return action