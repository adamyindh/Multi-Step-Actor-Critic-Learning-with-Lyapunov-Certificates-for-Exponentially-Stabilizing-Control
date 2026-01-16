"""
Derived from replay_buffer.py, used to store n-step data collected during interaction with the environment.
Reconstructed replay buffer to accommodate the different shape of n-step data compared to 1-step data.
"""
"""
Replay buffer for storing sampled data from environment interaction.
It is a component of the trainer, which consists of sampler (data collection) and buffer (data storage).
"""

import numpy as np
import sys
import torch

from collections import deque

__all__ = ["NstepReplayBuffer"]

def combined_shape(length: int, shape=None):
    """
    Create a tuple representing the combined array shape.
    
    Args:
        length: Size of the first dimension (e.g., batch size, buffer length).
        shape: Shape of remaining dimensions (int, tuple, or None).
    
    Returns:
        (length,) if shape is None;
        (length, shape) if shape is scalar;
        (length, *shape) if shape is a tuple.
    
    Examples:
        Input (100, 4) --> Output (100, 4)
        Input (100, (4, 84, 84)) --> Output (10000, 4, 84, 84)
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class NstepReplayBuffer:
    """Replay buffer for storing n-step interaction data with the environment."""
    
    def __init__(self, **kwargs):
        self.obsv_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        self.max_size = kwargs["buffer_max_size"]
        # Distinguished from standard replay buffer: n-step length of the data
        self.n_step = kwargs["n_step"]

        # Data storage dict (experience objects contain: s,a,r,s',done,log_prob)
        # Construct NumPy arrays for data storage
        self.n_step_buf = {
            "obs": np.zeros(
                combined_shape(self.max_size, (self.n_step, self.obsv_dim)), dtype=np.float32
            ),
            "act": np.zeros(
                combined_shape(self.max_size, (self.n_step, self.act_dim)), dtype=np.float32
            ),
            "rew": np.zeros((self.max_size, self.n_step), dtype=np.float32),
            "cost": np.zeros((self.max_size, self.n_step), dtype=np.float32),
            # "gain": np.zeros((self.max_size, self.n_step), dtype=np.float32),
            "obs2": np.zeros(
                combined_shape(self.max_size, (self.n_step, self.obsv_dim)), dtype=np.float32
            ),
            "done": np.zeros((self.max_size, self.n_step), dtype=np.float32),
            "logp": np.zeros((self.max_size, self.n_step), dtype=np.float32),
        }

        # Track stored data size and position pointer in the buffer
        self.ptr, self.size, = (0, 0,)

    def __len__(self):
        # Get current number of stored samples in the replay buffer
        return self.size

    def __get_RAM__(self):
        """
        Calculate the actual memory usage of stored valid data (in MB)
        """
        total_bytes = 0
        if self.size == 0:
            return 0.0
        for key in self.n_step_buf.keys():
            arr = self.n_step_buf[key]
            if isinstance(arr, np.ndarray):
                total_bytes += arr[:self.size].nbytes     
        used_mb = total_bytes / (1024 * 1024)
        return round(used_mb, 2)
    

    def store(
        self,
        obs: np.ndarray,  # shape=[n_step, obs_dim]
        act: np.ndarray,  # shape=[n_step, act_dim]
        rew: float,  # shape=[n_step,]
        cost: float,  # shape=[n_step,]
        # gain: float,  # shape=[n_step,]
        next_obs: np.ndarray,  # shape=[n_step, obs_dim]
        done: bool,  # shape=[n_step,]
        logp: np.ndarray,  # shape=[n_step,]
    ) -> None:
        """
        Store a single n-step sample into the buffer.
        Input data has no batch dimension (self.ptr defines the batch index).
        """
        self.n_step_buf["obs"][self.ptr] = obs
        self.n_step_buf["act"][self.ptr] = act
        self.n_step_buf["rew"][self.ptr] = rew
        self.n_step_buf["cost"][self.ptr] = cost
        # self.n_step_buf["gain"][self.ptr] = gain
        self.n_step_buf["obs2"][self.ptr] = next_obs
        self.n_step_buf["done"][self.ptr] = done
        self.n_step_buf["logp"][self.ptr] = logp

        # Cyclically update pointer (capped at max_size)
        self.ptr = (self.ptr + 1) % self.max_size

        # Update buffer size (capped at max_size)
        self.size = min(self.size + 1, self.max_size)


    def add_batch(self, samples: list) -> None:
        # Add list of nStepExperience objects to buffer one by one
        for sample in samples:
            self.store(*sample)


    def sample_batch(self, batch_size: int) -> dict:
        """
        Sample a batch of data for training.
        
        Returns:
            Dict of PyTorch tensors with shape:
            - batch["obs"]: (batch_size, n_step, obsv_dim)
            - batch["rew"]: (batch_size, n_step)
        """
        # Randomly select batch_size indices from valid data range
        idxes = np.random.randint(0, self.size, size=batch_size)

        batch = {}

        # Convert sampled numpy arrays to PyTorch tensors
        for k, v in self.n_step_buf.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.as_tensor(v[idxes], dtype=torch.float32)
            else:
                # Handle non-numpy array data types
                batch[k] = v[idxes].array2tensor()

        return batch
