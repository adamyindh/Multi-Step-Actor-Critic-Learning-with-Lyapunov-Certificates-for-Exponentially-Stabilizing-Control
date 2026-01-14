"""Design custom rewards and costs for different environments"""

import numpy as np
from typing import Tuple

def rew_plus_cost(
    env_id: str,
    obs: np.ndarray,
    real_next_obs: np.ndarray,
    target_value: float,
    cost_scale: float,
    reward_scale: float,
    original_rewards: np.ndarray,
    terminations: bool,
) -> Tuple[np.ndarray, np.ndarray]:

    # Reward: scaled original rewards
    rewards = original_rewards * reward_scale
    # Cost: sum of squared next_obs * cost_scale
    costs = (real_next_obs ** 2).sum(axis=1)
    costs = costs * cost_scale

    return rewards, costs