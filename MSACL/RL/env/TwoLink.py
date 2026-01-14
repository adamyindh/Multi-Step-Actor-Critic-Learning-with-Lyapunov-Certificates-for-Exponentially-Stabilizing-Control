"""
Implementation of the Twolink model
Reference:
M. W. Spong, S. Hutchinson, and M. Vidyasagar, Robot Modeling and Control

The model captures the rigid-body rotational dynamics of a serial two-link planar structure subject to gravitational effects and joint torque inputs.
The core equations are consistent with the above reference.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class twolink(gym.Env):
    """
    2-Link manipulator stabilization control environment based on Gymnasium
    State: [theta1, theta2, dtheta1, dtheta2] (joint angles & angular velocities)
    Control input: [tau1, tau2] (joint torques)
    """

    def __init__(self, dt=0.01, update_manner="Euler"):
        super().__init__()

        # 2-Link physical parameters (stabilization-hardened configuration)
        self.l1 = 1.0  # Length of link 1 [m]
        self.l2 = 1.0  # Length of link 2 [m]
        self.m1 = 1.0  # Mass of link 1 [kg]
        self.m2 = 1.0  # Mass of link 2 [kg]

        self.lc1 = self.l1 / 2  # COM of link 1 from joint 1 [m]
        self.lc2 = self.l2 / 2  # COM of link 2 from joint 2 [m]
        self.I1 = (1/12) * self.m1 * self.l1**2  # Moment of inertia of link 1 about COM [kg·m²]
        self.I2 = (1/12) * self.m2 * self.l2**2  # Moment of inertia of link 2 about COM [kg·m²]
        self.g = 9.81  # Gravitational acceleration [m/s²]

        # Simulation tracking
        self.current_step = 0  # Current simulation step

        # Observation space: [theta1, theta2, dtheta1, dtheta2]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi/2, -np.pi/2, -20.0, -20.0]),
            high=np.array([np.pi/2, np.pi/2, 20.0, 20.0]),
            dtype=np.float32
        )

        # Action space: joint torques [tau1, tau2]
        self.action_space = spaces.Box(
            low=np.array([-20.0, -20.0]),
            high=np.array([20.0, 20.0]),
            dtype=np.float32
        )

        # Initial state configuration
        self.reset_noise = 0.5  # Initial perturbation (small neighborhood of origin)
        self.obs = None  # Current observation (initialized in reset)

        # Random seed range
        self.min_seed_value = np.iinfo(np.uint32).min
        self.max_seed_value = np.iinfo(np.uint32).max

        # State/action dimensions
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

        # Reward function weights (position > velocity)
        self.Q = np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32)  # State error weights
        self.R = np.array([0.1, 0.1], dtype=np.float32)            # Control cost weights

        # Convergence threshold (origin radius)
        self.origin_radius = 1e-2

        # Simulation parameters
        self.dt = 0.01          # Time step
        self.control_step = 5   # Control update interval
        self.max_step = 1000    # Maximum steps per episode


    def reset(self, seed=None, options=None):
        """Reset environment to random initial state (small perturbation around origin)"""
        # Set random seed
        if seed == None:
            seed = random.randint(self.min_seed_value, self.max_seed_value)
        elif not isinstance(seed, int):
            seed = int(seed)
        super().reset(seed=seed)

        # Random initial state (small neighborhood of origin)
        self.obs = self.np_random.uniform(
            low=-self.reset_noise * np.ones(self.obs_dim, dtype=np.float32),
            high=self.reset_noise * np.ones(self.obs_dim, dtype=np.float32)
        ).astype(np.float32)

        self.current_step = 0  # Reset step counter

        return self.obs.copy(), {}
    

    # --- Dynamics Helper Functions ---
    def mass_matrix(self, q):
        """Compute 2x2 inertia matrix M(q) for 2-link system"""
        theta1, theta2 = q
        c2 = np.cos(theta2)
        M11 = self.I1 + self.I2 + self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*c2)
        M12 = self.I2 + self.m2*(self.lc2**2 + self.l1*self.lc2*c2)  # Symmetric mass matrix
        M22 = self.I2 + self.m2*self.lc2**2
        return np.array([[M11, M12], [M12, M22]])

    def coriolis_matrix(self, q, dq):
        """Compute Coriolis/centrifugal matrix C(q, dq)"""
        theta1, theta2 = q
        dtheta1, dtheta2 = dq
        s2 = np.sin(theta2)
        h = -self.m2 * self.l1 * self.lc2 * s2
        C11 = h * dtheta2
        C12 = h * dtheta2 + h * dtheta1
        C21 = -h * dtheta1
        C22 = 0
        return np.array([[C11, C12], [C21, C22]])

    def gravity_vector(self, q):
        """Compute gravitational force vector G(q)"""
        theta1, theta2 = q
        G1 = (
            -(self.m1*self.lc1 + self.m2*self.l1) * self.g * np.sin(theta1)
            - self.m2*self.lc2 * self.g * np.sin(theta1 + theta2)
        )
        G2 = - self.m2*self.lc2 * self.g * np.sin(theta1 + theta2)
        return np.array([G1, G2])
    
    def _dynamics(self, x, u):
        """Compute state derivative dx/dt using Lagrangian dynamics"""
        q = x[:2]       # Joint angles [theta1, theta2]
        dq = x[2:]      # Joint velocities [dtheta1, dtheta2]

        # Lagrangian dynamics components
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        G = self.gravity_vector(q)
        
        # M*ddq = u - C*dq - G → ddq = M⁻¹(u - C*dq - G)
        ddq = np.linalg.solve(M, u - C @ dq - G)
        dxdt = np.concatenate([dq, ddq])
        return dxdt
    

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