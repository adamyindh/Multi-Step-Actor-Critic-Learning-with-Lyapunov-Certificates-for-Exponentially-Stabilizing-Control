import torch
import warnings
import gymnasium as gym
import sys
import os
import datetime
import json
import copy
from RL.utils.common_utils import seed_everything, change_type

def init_args(envs, **args):
    """Initialize training arguments from env and config"""
    # Set PyTorch CPU threads (4 for serial trainer, 1 otherwise)
    num_threads_main = args.get("num_threads_main", None)
    if num_threads_main is None:
        num_threads_main = 4 if "serial" in args["trainer"] else 1
    torch.set_num_threads(num_threads_main)

    # CUDA config
    if args["enable_cuda"]:
        if torch.cuda.is_available():
            args["use_gpu"] = True
        else:
            warnings.warn("cuda is not available, use CPU instead")
            args["use_gpu"] = False
    else:
        args["use_gpu"] = False

    # Sampling batch size per sampler
    args["batch_size_per_sampler"] = args["sample_batch_size"]

    # Env observation dimension
    if len(envs.single_observation_space.shape) == 1:
        args["obs_dim"] = envs.single_observation_space.shape[0]
    else:
        args["obs_dim"] = envs.single_observation_space.shape

    # Continuous action space config (only supported)
    if isinstance(envs.single_action_space, gym.spaces.Box):
        args["action_type"] = "continu"
        args["act_dim"] = envs.single_action_space.shape[0] if len(envs.single_action_space.shape) == 1 else envs.single_action_space.shape
        args["action_high_limit"] = envs.single_action_space.high.astype("float32")
        args["action_low_limit"] = envs.single_action_space.low.astype("float32")
    elif isinstance(envs.single_action_space, gym.spaces.Discrete):
        print("Only continuous action space is supported")
        sys.exit(1)

    # Create save directory (auto-generate if not specified)
    if args["save_folder"] is None:
        dir_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if args["algorithm"] == "msacl":
            args["save_folder"] = os.path.join(
                dir_path + "/results/", args["env_name"], 
                "n_step_results",
                args["algorithm"] + '_' + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
                + '_' + str(args["lya_eta"]) + '_n=' + str(args["n_step"])
            )
        else:
            args["save_folder"] = os.path.join(
                dir_path + "/results/", args["env_name"], 
                args["algorithm"] + '_' + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            )

    os.makedirs(args["save_folder"], exist_ok=True)
    os.makedirs(args["save_folder"] + "/apprfunc", exist_ok=True)

    # Set global random seed
    seed = args.get("seed", None)
    args["seed"] = seed_everything(seed)
    print("Set the global seed: {}".format(args["seed"]))

    # Save config to JSON (for reproducibility)
    with open(args["save_folder"] + "/config.json", "w", encoding="utf-8") as f:
        json.dump(change_type(copy.deepcopy(args)), f, ensure_ascii=False, indent=4)

    return args