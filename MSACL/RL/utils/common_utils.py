import numpy as np
from typing import Optional
import random
import torch
import sys
import torch.nn as nn

# Import action distribution classes (GaussDistribution, TanhGaussDistribution, etc.)
from RL.utils.act_distribution_cls import *


def seed_everything(seed: Optional[int] = None) -> int:
    """Set global random seed for reproducibility"""
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)
    elif not isinstance(seed, int):
        seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

def change_type(obj):
    """Recursively convert numpy types to native Python types (for JSON serialization)"""
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, type):
        return str(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = change_type(v)
        return obj
    elif isinstance(obj, list):
        for i, o in enumerate(obj):
            obj[i] = change_type(o)
        return obj
    else:
        return obj
    

def get_apprfunc_dict(key: str, **kwargs):
    """Build config dict for approx functions (value/policy networks)"""
    var = dict()
    var["apprfunc"] = kwargs[key + "_func_type"]  # Only "MLP" supported
    var["name"] = kwargs[key + "_func_name"]

    var["obs_dim"] = kwargs["obs_dim"]
    var["min_log_std"] = kwargs.get(key + "_min_log_std", float("-20"))
    var["max_log_std"] = kwargs.get(key + "_max_log_std", float("1.0"))

    # Lyapunov network input/output dim config
    if key == "lyapunov":
        var["input_dim"] = 1 if kwargs[key + "_single_input_dim"] else kwargs["obs_dim"]
        var["output_dim"] = kwargs[key + "_output_dim"]

    # Default output activation: linear
    if key + "_output_activation" not in kwargs.keys():
        kwargs[key + "_output_activation"] = "linear"

    # MLP config
    apprfunc_type = kwargs[key + "_func_type"]
    if apprfunc_type == "MLP":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
    else:
        raise NotImplementedError
    
    # Continuous action space only
    if kwargs["action_type"] == "continu":
        var["act_dim"] = kwargs["act_dim"]
        var["act_high_lim"] = np.array(kwargs["action_high_limit"])
        var["act_low_lim"] = np.array(kwargs["action_low_limit"])
    else:
        print("Only continuous action space is supported")
        raise NotImplementedError
    
    # Action distribution config (for stochastic policies)
    if kwargs["policy_act_distribution"] == "default":
        if kwargs["policy_func_name"] == "StochaPolicy":
            var["action_distribution_cls"] = GaussDistribution
        elif kwargs["policy_func_name"] == "DetermPolicy":
            var["action_distribution_cls"] = DiracDistribution
    else:
        var["action_distribution_cls"] = getattr(
            sys.modules[__name__], kwargs["policy_act_distribution"]
        )

    return var


def get_activation_func(key: str):
    """Get PyTorch activation function class from string"""
    assert isinstance(key, str)

    activation_func = None
    if key == "relu":
        activation_func = nn.ReLU
    elif key == "elu":
        activation_func = nn.ELU
    elif key == "gelu":
        activation_func = nn.GELU
    elif key == "selu":
        activation_func = nn.SELU
    elif key == "sigmoid":
        activation_func = nn.Sigmoid
    elif key == "tanh":
        activation_func = nn.Tanh
    elif key == "linear":
        activation_func = nn.Identity

    if activation_func is None:
        print("Can not identify activation name:" + key)
        raise RuntimeError

    return activation_func


class ModuleOnDevice:
    """Context manager to temporarily move module to target device (restore after use)"""
    def __init__(self, module, device):
        self.module = module
        self.prev_device = next(module.parameters()).device.type
        self.new_device = device
        self.different_device = self.prev_device != self.new_device

    def __enter__(self):
        if self.different_device:
            self.module.to(self.new_device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.different_device:
            self.module.to(self.prev_device)
