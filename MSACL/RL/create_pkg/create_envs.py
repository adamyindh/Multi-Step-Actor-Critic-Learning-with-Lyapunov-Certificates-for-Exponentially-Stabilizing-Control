# Create environment(s) from input args
import gymnasium as gym
from datetime import datetime
import numpy as np

from RL.env.make_env import make_env


def create_envs(**args):
    """Create environment(s)."""
    env_name = args.get("env_name")


    env_id = env_name
    print(f"Creating environment with env_id: {env_id}")

    env_seed = args.get("env_seed")
    capture_video = args.get("capture_video")
    env_num = args.get("env_num")
    run_name = f"{env_id}_{env_seed}_{datetime.now().strftime('%Y/%m/%d/%H:%M')}"


    # Create SyncVectorEnv for environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(
            env_id, 
            env_seed + i, 
            i, 
            capture_video, 
            run_name,
        ) for i in range(env_num)]
    )
    print(env_id, "environment created successfully!")

    return envs