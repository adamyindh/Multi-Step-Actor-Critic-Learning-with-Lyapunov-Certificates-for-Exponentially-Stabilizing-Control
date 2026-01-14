"""
Implementation of the Ductedfan model
Reference:
Neural Lyapunov Control
(https://github.com/YaChienChang/Neural-Lyapunov-Control)

The DuctedFan system models the coupled translational and rotational dynamics of a ducted-fan aircraft under the influence of gravitational and control forces.
The core equations are consistent with the above reference.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class ductedfan(gym.Env):
    """
    Gym-based environment for aircraft hover control (experimental environment in NLC paper)
    State variables: [x, y, θ, ẋ, ẏ, θ̇ ]
    Control inputs: [u₁, u₂] (two force inputs)
    """


    def __init__(self, dt=0.01):
        super().__init__()

        # Caltech ducted fan parameters (average values from reference)
        self.m = 8.5        # Mass
        self.g = 9.81       # Gravitational acceleration
        self.r = 0.26       # Moment arm
        self.d = 0.95       # Damping coefficient
        self.J = 0.048      # Moment of inertia

        self.current_step = 0  # Current simulation step

        # Observation space bounds
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -5.0, -np.pi/2, -5.0, -5.0, -5.0]),
            high=np.array([5.0, 5.0, np.pi/2, 5.0, 5.0, 5.0]),
            dtype=np.float32
        )

        # Action space bounds
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0]),
            high=np.array([5.0, 5.0]),
            dtype=np.float32
        )

        self.obs = None  # Current observation (initialized in reset)

        # Random seed range
        self.min_seed_value = np.iinfo(np.uint32).min
        self.max_seed_value = np.iinfo(np.uint32).max

        # Observation/action dimensions
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

        # Initial state perturbation (small neighborhood around origin)
        self.reset_noise = 0.5

        # Weights for reward function (performance functional matrix parameters)
        self.Q = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.R = np.array([0.1, 0.1], dtype=np.float32)

        # Threshold for origin convergence
        self.origin_radius = 1e-2  

        # Simulation parameters
        self.dt = 0.01          # Time step
        self.control_step = 5   # Control update interval
        self.max_step = 1000    # Maximum simulation steps per episode


    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state (random in small neighborhood of origin)
        """
        if seed == None:
            seed = random.randint(self.min_seed_value, self.max_seed_value)
        elif not isinstance(seed, int):
            seed = int(seed)

        # Initialize gym's random number generator
        super().reset(seed=seed)

        # Random initial state (small perturbation around origin)
        self.obs = self.np_random.uniform(
            low=-self.reset_noise * np.ones(self.obs_dim, dtype=np.float32),
            high=self.reset_noise * np.ones(self.obs_dim, dtype=np.float32)
        ).astype(np.float32)

        self.current_step = 0  # Reset step counter

        return self.obs.copy(), {}
    
    
    def _dynamics(self, obs, action):
        """
        Calculate system dynamics (required for RK4 integration)
        """
        x, y, theta, dx, dy, dtheta = obs
        u1, u2 = action

        # Calculate accelerations
        d_dx = (- self.m * self.g * np.sin(theta) - self.d * dx + u1 * np.cos(theta) - u2 * np.sin(theta)) / self.m
        d_dy = (self.m * self.g * (np.cos(theta)-1) - self.d * dy + u1 * np.sin(theta) + u2 * np.cos(theta)) / self.m
        d_dtheta = (self.r * u1) / self.J

        # Return 6D dynamics array
        return np.array([dx, dy, dtheta, d_dx, d_dy, d_dtheta])
    

    def step(self, action):
        """
        Interact with environment (single step)
        """
        # Apply same action for multiple control steps
        for _ in range(self.control_step):
            control_dynamics = self._dynamics(self.obs, action)
            self.obs += control_dynamics * self.dt

        # Calculate cost/reward
        obs_cost = np.sum(self.Q * self.obs ** 2)
        control_cost = np.sum(self.R * action ** 2)
        cost = obs_cost + control_cost
        reward = -cost  # Negative cost as reward

        # Bonus reward for convergence to origin
        if np.all(np.abs(self.obs) <= self.origin_radius):
            reward += 1

        # Check out-of-bounds condition
        out_of_bound = np.any(
            (self.obs < self.observation_space.low) |
            (self.obs > self.observation_space.high)
        )
        terminated = out_of_bound  # Terminate if out of bounds
        self.current_step += 1     # Increment step counter
        truncated = self.current_step >= self.max_step  # Truncate at max steps

        info = {}
        next_obs = self.obs.copy()

        # Return step results (comply with Gymnasium interface)
        return next_obs, reward, terminated, truncated, info