"""
Create replay buffer for storing sampled data from environment interaction.
Component of trainer, which consists of sampler (data collection) and buffer (data storage).
"""

import numpy as np
import sys
import torch

__all__ = ["ReplayBuffer"]

def combined_shape(length: int, shape=None):
    """
    Create a tuple for combined array shape.
    
    Args:
        length: Size of the first dimension (e.g., batch size, buffer length).
        shape: Shape of remaining dimensions (int, tuple, or None).
    
    Returns:
        (length,) if shape is None;
        (length, shape) if shape is scalar;
        (length, *shape) if shape is tuple.
    
    Examples:
        Input (100, 4) --> Output (100, 4)
        Input (100, (4, 84, 84)) --> Output (10000, 4, 84, 84)
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """Uniform sampling of samples from replay buffer."""
    
    def __init__(self, **kwargs):
        self.obs_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        self.max_size = kwargs["buffer_max_size"]

        # Data storage dict (experience objects: s,a,r,s',done,log_prob)
        # Create NumPy arrays for data storage
        self.buf = {
            "obs": np.zeros(
                combined_shape(self.max_size, self.obs_dim), dtype=np.float32
            ),
            "act": np.zeros(
                combined_shape(self.max_size, self.act_dim), dtype=np.float32
            ),
            "rew": np.zeros(self.max_size, dtype=np.float32),
            "cost": np.zeros(self.max_size, dtype=np.float32),
            "obs2": np.zeros(
                combined_shape(self.max_size, self.obs_dim), dtype=np.float32
            ),
            "done": np.zeros(self.max_size, dtype=np.float32),
            "logp": np.zeros(self.max_size, dtype=np.float32),
        }

        # Position pointer and current buffer size
        self.ptr, self.size, = (0, 0,)

    def __len__(self):
        # Get current number of stored samples in replay buffer
        return self.size

    def __get_RAM__(self):
        """
        Calculate the actual memory usage of Tensor storage (in MB)
        """
        total_bytes = 0
        for key in self.buf.keys():
            arr = self.buf[key]
            if isinstance(arr, np.ndarray):
                total_bytes += arr.nbytes     
        used_bytes = total_bytes * (self.size / self.max_size) if self.max_size > 0 else 0
        used_mb = used_bytes / (1024 * 1024)
        return round(used_mb, 2)
    
    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        cost: float,
        next_obs: np.ndarray,
        done: bool,
        logp: np.ndarray,
    ) -> None:
        """
        Store a single sample into the buffer.
        Input data has no batch dimension (self.ptr defines the index).
        - obs: shape=[obs_dim] (1D vector)
        - rew: scalar float value (e.g., 1.0)
        """
        self.buf["obs"][self.ptr] = obs
        self.buf["act"][self.ptr] = act
        self.buf["rew"][self.ptr] = rew
        self.buf["cost"][self.ptr] = cost
        self.buf["obs2"][self.ptr] = next_obs
        self.buf["done"][self.ptr] = done
        self.buf["logp"][self.ptr] = logp

        # Cyclically update pointer (capped at max_size)
        self.ptr = (self.ptr + 1) % self.max_size

        # Update buffer size (capped at max_size)
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, samples: list) -> None:
        """Add a batch of Experience objects to buffer one by one."""
        for sample in samples:
            self.store(*sample)


    def sample_batch(self, batch_size: int) -> dict:
        """
        Sample a batch of data for training (uniform random sampling).
        
        Returns:
            Dict with same keys as self.buf, values are PyTorch tensors of sampled data.
        """
        # Randomly select batch_size indices from valid data range
        idxes = np.random.randint(0, self.size, size=batch_size)

        batch = {}

        # Convert sampled numpy arrays to PyTorch tensors
        for k, v in self.buf.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.as_tensor(v[idxes], dtype=torch.float32)
            else:
                batch[k] = v[idxes].array2tensor()

        return batch