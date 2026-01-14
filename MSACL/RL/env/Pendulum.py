"""
Implementation of the Pendulum model
Reference: 
OpenAI Gym Pendulum Environment 
(https://www.gymlibrary.dev/environments/classic_control/pendulum/)

The model is a classic rigid-body system with a pivot point constrained to a vertical plane.
The core equations and parameter definitions are all consistent with the above reference.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class pendulum(gym.Env):
    """
    Inverted Pendulum environment inherited from gym
    Truncation/termination triggered when respective flags are True
    """

    def __init__(self, dt=0.01):
        super().__init__()

        # Physical parameters
        self.g = 9.81   # Gravitational acceleration
        self.L = 0.5    # Pendulum length
        self.m = 0.15   # Pendulum mass
        self.b = 0.1    # Damping ratio
        self.current_step = 0  # Current simulation step

        # Observation space: [theta, theta_dot]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -10.0]),
            high=np.array([np.pi, 10.0]),
            dtype=np.float32
        )

        # Action space (1D control input)
        self.action_space = spaces.Box(
            low=np.array([-5]),
            high=np.array([5]),
            dtype=np.float32
        )

        self.obs = None  # Current observation (initialized in reset)

        # Random seed range
        self.min_seed_value = np.iinfo(np.uint32).min
        self.max_seed_value = np.iinfo(np.uint32).max

        # State dimensions and reset perturbation
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]
        self.reset_noise = 2
        
        # Weights for reward function (performance functional matrices)
        self.Q = np.array([2.0, 1.0], dtype=np.float32)
        self.R = np.array([0.1], dtype=np.float32)

        # Threshold for convergence to origin
        self.origin_radius = 1e-2

        # Simulation parameters
        self.dt = 0.01          # Time step
        self.control_step = 5   # Control update interval
        self.max_step = 1000    # Maximum steps per episode


    def reset(self, seed=None, options=None):
        """
        Reset environment to random initial state in observation space
        """
        if seed == None:
            seed = random.randint(self.min_seed_value, self.max_seed_value)
        elif not isinstance(seed, int):
            seed = int(seed)

        # Initialize gym's random number generator
        super().reset(seed=seed)

        # Random initial state (full observation space range)
        self.obs = self.np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high,
        ).astype(np.float32)

        self.current_step = 0  # Reset step counter

        return self.obs.copy(), {}
    

    def _dynamics(self, obs, action):
        """Calculate pendulum system dynamics"""
        u = action[0]  # 1D control input

        # Extract state variables
        theta, theta_dot = obs

        # Compute dynamics
        d_theta = theta_dot
        d_theta_dot = (self.m * self.g * self.L * np.sin(theta) - self.b * theta_dot + u) / (self.m * self.L ** 2)

        return np.array([d_theta, d_theta_dot])

    
    def step(self, action):
        """Interact with environment (single step)"""
        # Update state with Euler integration
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

        # Check out-of-bounds condition
        out_of_bound = bool(
            (self.obs[0] < self.observation_space.low[0]) or (self.obs[0] > self.observation_space.high[0]) or
            (self.obs[1] < self.observation_space.low[1]) or (self.obs[1] > self.observation_space.high[1])
        )
        terminated = out_of_bound  # Terminate if out of bounds
        self.current_step += 1     # Increment step counter
        truncated = self.current_step >= self.max_step  # Truncate at max steps

        info = {}
        next_obs = self.obs.copy()

        # Return step results (comply with Gymnasium interface)
        return next_obs, reward, terminated, truncated, info
