"""
Implementation of the Van der Pol model
Reference:
B. van der Pol, "A theory of the amplitude of free and forced triode vibrations"

This model is a classic representation of nonlinear damped self-sustaining oscillatory circuits. 
The core equations and parameter definitions are all consistent with the above reference.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class vanderpol(gym.Env):
    """
    Van der Pol oscillator control environment based on Gymnasium
    State: [x, dx/dt] (position & velocity)
    Control input: [u] (scalar control force)
    """


    def __init__(self, dt=0.01):
        super().__init__()

        # System parameters
        self.mu = 1.0       # Nonlinearity strength (higher = stronger nonlinearity)

        # Simulation tracking
        self.current_step = 0  # Current simulation step

        # Observation space: [x, dx/dt]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0]),
            high=np.array([10.0, 10.0]),
            dtype=np.float32
        )
        self.reset_noise = 5  # Initial state perturbation range
        self.action_space = spaces.Box(
            low=np.array([-5.0]),
            high=np.array([5.0]),
            dtype=np.float32
        )


        self.obs = None  # Current observation (initialized in reset)

        # Random seed range
        self.min_seed_value = np.iinfo(np.uint32).min
        self.max_seed_value = np.iinfo(np.uint32).max

        # State/action dimensions
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]
        
        # Reward function weights (position > velocity)
        self.Q = np.array([2.0, 1.0], dtype=np.float32)  # State error weights
        self.R = np.array([0.1], dtype=np.float32)        # Control cost weight

        # Convergence threshold (origin radius)
        self.origin_radius = 1e-2  

        # Simulation parameters
        self.dt = 0.01          # Time step
        self.control_step = 5   # Control update interval
        self.max_step = 1000    # Maximum steps per episode


    def reset(self, seed=None, options=None):
        """Reset environment to random initial state (within reset_noise range)"""
        # Set random seed
        if seed == None:
            seed = random.randint(self.min_seed_value, self.max_seed_value)
        elif not isinstance(seed, int):
            seed = int(seed)
        super().reset(seed=seed)

        
        self.obs = self.np_random.uniform(
            low=-self.reset_noise * np.ones(self.obs_dim, dtype=np.float32),
            high=self.reset_noise * np.ones(self.obs_dim, dtype=np.float32)
        ).astype(np.float32)

        self.current_step = 0  # Reset step counter

        return self.obs.copy(), {}
    
    
    def _dynamics(self, obs, action):
        """Compute Van der Pol oscillator state derivative dx/dt"""
        x, dx = obs
        u = action[0]  # Scalar control input

        # Van der Pol dynamics: d²x/dt² = μ(1-x²)dx/dt - x + u
        d_dx = self.mu * (1 - x**2) * dx - x + u

        return np.array([dx, d_dx])
    

    def step(self, action):
        """Interact with environment (single step)"""
        # Update state with Euler integration (multiple control steps)
        for _ in range(self.control_step):
            control_dynamics = self._dynamics(self.obs, action)
            self.obs += control_dynamics * self.dt

        # Calculate reward (negative cost)
        obs_cost = np.sum(self.Q * self.obs ** 2)
        control_cost = np.sum(self.R * action ** 2)
        cost = obs_cost + control_cost
        reward = -cost

        # Bonus reward for convergence to origin
        if np.all(np.abs(self.obs) <= self.origin_radius):
            reward += 1

        # Check out-of-bounds termination
        out_of_bound = np.any(
            (self.obs < self.observation_space.low) |
            (self.obs > self.observation_space.high)
        )
        terminated = out_of_bound
        self.current_step += 1
        truncated = self.current_step >= self.max_step

        info = {}
        next_obs = self.obs.copy()

        # Return step results (comply with Gymnasium interface)
        return next_obs, reward, terminated, truncated, info