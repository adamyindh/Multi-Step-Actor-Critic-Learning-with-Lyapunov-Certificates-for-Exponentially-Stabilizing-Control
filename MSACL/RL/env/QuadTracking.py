"""
Implementation of the Twolink model
Reference:
Geometric tracking control of a quadrotor UAV on SE(3)
(https://github.com/tonyddg/uav_geometry_controller)

The model evaluates the 6-degree-of-freedom (6-DoF) rigid-body dynamics of a quadrotor UAV.
The core equations are consistent with the above reference.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
from typing import Callable
import random


class quadtracking(gym.Env):
    """
    Quadrotor UAV trajectory tracking environment
    Implemented based on "Geometric Tracking Control of a Quadrotor UAV on SE(3)"
    State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    Control input: [propeller1 thrust, propeller2 thrust, propeller3 thrust, propeller4 thrust]
    """


    def _default_trajectory(self):
        """Default trajectory: 3D sine/cosine curve with lower angular velocity"""
        return UAVTrajectory(
            xd=lambda t: np.array([0.4 * t, 0.4 * np.sin(t), 0.6 * np.cos(t)]),
            b1d=lambda t: np.array([np.cos(t), np.sin(t), 0.0]),
            vd=lambda t: np.array([0.4, 0.4 * np.cos(t), -0.6 * np.sin(t)]),
            ad=lambda t: np.array([0, -0.4 * np.sin(t), -0.6 * np.cos(t)])
        )

    def __init__(self, dt=0.01, traj=None):
        super().__init__()

        
        # UAV model and control parameters
        self.model = UAVModel()
        self.param = UAVControlParameter()
        
        # Trajectory (use default if not provided)
        self.traj = traj if traj is not None else self._default_trajectory()
        
        # Time/step tracking
        self.current_time = 0.0
        self.current_step = 0
        self.max_step = 1000 

        # Simulation parameters
        self.dt = 0.01          # Time step
        self.control_step = 4   # Control update interval
        
        # Observation space scaling factor
        self.obs_coef = 1.0

        # Initial state perturbation
        self.reset_noise = 0.01         # Position/velocity perturbation
        self.rotation_sigma = 0.01      # Rotation matrix perturbation (≈0.5 degrees)

        # Reward configuration
        self.max_reward = 10.0
        self.choose_reward_type = 1     # 0: fixed bonus | 1: linear | 2: tiered
        self.origin_radius = 0.1        # Convergence threshold
        self.reward_level_num = 50      # Number of tiers (for reward type 2)
        self.level_radii = np.linspace(0, self.origin_radius, self.reward_level_num + 1)[1:]

        # State variables (initialized in reset)
        self.x = None        # Position [x,y,z]
        self.v = None        # Velocity [vx,vy,vz]
        self.R = None        # Rotation matrix
        self.Omega = None    # Angular velocity [wx,wy,wz]
        self.t_last = np.zeros(2)       # Time history [current-1, current-2]
        self.Rd_last = None             # Last desired rotation matrix
        self.Omega_d_last = None        # Last desired angular velocity
        
        # Observation/action space initialization
        self.obs = None

        # Observation space (12D error: pos/vel/att/omega)
        obs_high = np.array([
            10.0, 10.0, 10.0,     # Position error
            10.0, 10.0, 10.0,     # Velocity error
            10.0, 10.0, 10.0,     # Attitude error
            10.0, 10.0, 10.0,     # Angular velocity error
        ], dtype=np.float32) * self.obs_coef
        obs_low = -obs_high
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        # Action space (thrust + 3 torques)
        gravity_value = self.model.m * g[2]  # Gravitational force
        self.action_space = spaces.Box(
            low=np.array([0.0 * gravity_value, -10.0, -10.0, -10.0], dtype=np.float32),
            high=np.array([2.0 * gravity_value, 10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )

        # Dimension calculation
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]
        
        # Reward weights
        self.Q_pos = np.array([1.0, 1.0, 1.0], dtype=np.float32)    # Position error weight
        self.Q_vel = np.array([1.0, 1.0, 1.0], dtype=np.float32)    # Velocity error weight
        self.Q_att = np.array([1.0, 1.0, 1.0], dtype=np.float32)    # Attitude error weight
        self.Q_omega = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # Angular velocity error weight
        self.R_act = np.array([0.0001, 0.01, 0.01, 0.01], dtype=np.float32)  # Control cost weight

        # Random seed range
        self.min_seed_value = np.iinfo(np.uint32).min
        self.max_seed_value = np.iinfo(np.uint32).max
        
    
    def _get_desired_states(self, t):
        """Get desired states (position/velocity/attitude/angular velocity) at time t"""
        xd = self.traj.xd(t)
        vd = self.traj.get_vd(t, self.t_last[0])
        ad = self.traj.get_ad(t, self.t_last[0], self.t_last[1])
        
        # Calculate desired thrust vector
        fd = -(-self.param.kx * cal_ex(self.x, xd) - self.param.kv * cal_ev(self.v, vd) 
              - self.model.m * g + self.model.m * ad)
        fd_norm = np.linalg.norm(fd)
        b3d = fd / fd_norm

        # Calculate desired attitude frame
        b1d = self.traj.b1d(t)
        cross_b3d_b1d = np.cross(b3d, b1d)
        b2d_norm = np.linalg.norm(cross_b3d_b1d)
        b2d = cross_b3d_b1d / b2d_norm
        b1d = np.cross(b2d, b3d)
        Rd = np.vstack([b1d, b2d, b3d]).T
        
        # Calculate desired angular velocity
        if isinstance(self.Rd_last, np.ndarray) and isinstance(self.Omega_d_last, np.ndarray):
            Rd_dot = RDerive(Rd, self.Rd_last, t, self.t_last[0])
            Omega_d = getOmega(Rd, Rd_dot)
        else:
            Omega_d = np.zeros(3)
            
        return xd, vd, Rd, Omega_d
    

    def reset(self, seed=None, options=None):
        """Reset environment to initial state (small perturbation around origin)"""
        # Set random seed
        if seed == None:
            seed = random.randint(self.min_seed_value, self.max_seed_value)
        elif not isinstance(seed, int):
            seed = int(seed)
        super().reset(seed=seed)
        
        # Reset time/state history
        self.current_time = 0.0
        self.current_step = 0
        self.t_last = np.zeros(2)
        self.Rd_last = None
        self.Omega_d_last = None

        # Initialize state with small perturbations
        self.x = self.np_random.uniform(
            low=-self.reset_noise * np.ones(3, dtype=np.float32),
            high=self.reset_noise * np.ones(3, dtype=np.float32)
        ).astype(np.float32)

        self.v = self.np_random.uniform(
            low=-self.reset_noise * np.ones(3, dtype=np.float32),
            high=self.reset_noise * np.ones(3, dtype=np.float32)
        ).astype(np.float32)

        # Random rotation matrix initialization
        random_rot_vec = np.random.randn(3) * self.rotation_sigma
        self.R = R.from_rotvec(random_rot_vec).as_matrix().astype(np.float32)

        self.Omega  = self.np_random.uniform(
            low=-self.reset_noise * np.ones(3, dtype=np.float32),
            high=self.reset_noise * np.ones(3, dtype=np.float32)
        ).astype(np.float32)

        # Get initial desired states
        xd, vd, Rd, Omega_d = self._get_desired_states(t=0)
        self.Rd_last = Rd.copy()
        self.Omega_d_last = Omega_d.copy()

        # Calculate initial observation (state errors)
        e_x = cal_ex(self.x, xd)
        e_v = cal_ev(self.v, vd)
        e_R = cal_eR(self.R, Rd)
        e_Omega = cal_eOmega(self.R, Rd, self.Omega, Omega_d)

        initial_obs = np.hstack([e_x, e_v, e_R, e_Omega]).astype(np.float32)
        self.obs = initial_obs.copy()

        return self.obs, {}
    

    def step(self, action):
        """Interact with environment (single step)"""
        # Extract control input (thrust + torques)
        force = action[0]  # Thrust
        M = action[1:]     # Torques [Mx, My, Mz]
        
        # Update state with Euler integration
        for _ in range(self.control_step):
            # Compute dynamics
            dx = self.v
            dv = g - force * self.R[:, 2] / self.model.m
            dR = self.R @ VecToSo3(self.Omega)
            dOmega = np.linalg.inv(self.model.J) @ (M - np.cross(self.Omega, self.model.J @ self.Omega))

            # Update state variables
            self.x += dx * self.dt
            self.v += dv * self.dt
            self.R += dR * self.dt
            self.Omega += dOmega * self.dt

            # Normalize rotation matrix (ensure valid SO(3))
            self.R = NormalizeOrientMatrix(self.R)

        # Update time and get desired states
        self.current_time += self.dt * self.control_step
        xd, vd, Rd, Omega_d = self._get_desired_states(t=self.current_time)
        
        # Save desired states for next derivative calculation
        self.Rd_last = Rd.copy()
        self.Omega_d_last = Omega_d.copy()

        # Update time history
        self.t_last[1] = self.t_last[0]
        self.t_last[0] = self.current_time

        # Calculate observation (state tracking errors)
        e_x = cal_ex(self.x, xd)
        e_v = cal_ev(self.v, vd)
        e_R = cal_eR(self.R, Rd)
        e_Omega = cal_eOmega(self.R, Rd, self.Omega, Omega_d)

        next_obs = np.hstack([e_x, e_v, e_R, e_Omega]).astype(np.float32)
        self.obs = next_obs.copy()

        # Calculate reward (negative cost)
        reward = - (
            np.sum(self.Q_pos * e_x**2) +
            np.sum(self.Q_vel * e_v**2) +
            np.sum(self.Q_att * e_R**2) +
            np.sum(self.Q_omega * e_Omega**2) +
            np.sum(self.R_act * action**2)
        )

        # Apply reward bonus based on convergence
        if self.choose_reward_type == 0:
            # Fixed bonus for convergence
            if np.all(np.abs(self.obs) <= self.origin_radius):
                reward += self.max_reward
        elif self.choose_reward_type == 1:
            # Linear reward (closer = higher bonus)
            dist = np.max(np.abs(self.obs))
            if dist <= self.origin_radius:
                reward += self.max_reward * (1 - dist / self.origin_radius)
        else:
            # Tiered reward
            dist = np.max(np.abs(self.obs))
            if dist <= self.origin_radius:
                reward_level = np.searchsorted(self.level_radii, dist, side='left')
                reward += (self.reward_level_num - reward_level) * (self.max_reward/self.reward_level_num)
        
        # Check termination/truncation conditions
        out_of_bound = np.any(
            (self.obs < self.observation_space.low) |
            (self.obs > self.observation_space.high)
        )
        terminated = out_of_bound
        self.current_step += 1
        truncated = self.current_step >= self.max_step

        info = {}
        return self.obs, reward, terminated, truncated, info


# Math utilities
###########################################################################
"""Mathematical utility functions"""
# Gravitational acceleration
g = np.array([0, 0, 9.8])

def So3ToVec(R) -> np.ndarray:
    '''Convert so3 (skew-symmetric matrix) to rotation vector (row vector)'''
    res=np.array([R[2, 1], R[0, 2], R[1, 0]])
    return res.astype(np.float32)

def VecToSo3(so3) -> np.ndarray:
    '''Convert rotation vector (row vector) to so3 (skew-symmetric matrix)'''
    res=np.array([
        [0, -so3[2], so3[1]],
        [so3[2], 0, -so3[0]],
        [-so3[1], so3[0], 0]
    ])
    return res.astype(np.float32)

def NormalizeOrientMatrix(mat) -> np.ndarray:
    '''Normalize rotation matrix (ensure valid SO(3) with det=1)'''
    U, s, Vh = np.linalg.svd(mat)
    R = U @ Vh
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh
    return R.astype(np.float32)

# Error functions
def cal_ex(x, xd) -> np.ndarray:
    '''Position error (x - xd)'''
    res = x - xd
    return res.astype(np.float32)

def cal_ev(v, vd) -> np.ndarray:
    '''Velocity error (v - vd)'''
    res = v - vd
    return res.astype(np.float32)

def cal_eR(R, Rd) -> np.ndarray:
    '''Attitude error (SO(3) error)'''
    res = So3ToVec(Rd.T @ R - R.T @ Rd) * 0.5
    return res.astype(np.float32)

def rotErrorFun(R, Rd) -> float:
    '''Attitude error metric (Psi function)'''
    res = np.trace(np.identity(3) - Rd.T @ R) * 0.5
    return res.astype(np.float32)

def cal_eOmega(R, Rd, Omega, Omega_d) -> np.ndarray:
    '''Angular velocity error'''
    res = Omega - R.T @ Rd @ Omega_d
    return res.astype(np.float32)

# Derivative calculation
def safe_time_diff(t: float, t_last: float, epsilon: float = 1e-6) -> float:
    """Safe time difference (avoid division by zero)"""
    dt = t - t_last
    if dt < epsilon:
        return epsilon
    return dt

def RDerive(R: np.ndarray, R_last: np.ndarray, t: float, t_last: float):
    """Numerical derivative of rotation matrix"""
    dt = safe_time_diff(t, t_last)
    res = (R - R_last) / dt
    return res.astype(np.float32)

def getOmega(R: np.ndarray, RD: np.ndarray):
    """Calculate angular velocity from rotation matrix derivative (RD = dR/dt)"""
    res = So3ToVec(R.T @ RD)
    return res.astype(np.float32)
###########################################################################

# UAV control parameters
###########################################################################
@dataclass
class UAVControlParameter:
    '''UAV controller parameters (default values from paper)'''
    kx: float = 69.44
    kv: float = 24.304
    kR: float = 8.81
    kOmega: float = 2.54
###########################################################################

# UAV model parameters
###########################################################################
@dataclass
class UAVModel:
    '''UAV model parameters (default values from paper)'''
    J: np.ndarray = field(default_factory=lambda: np.diag([0.0820, 0.0845, 0.1377]))  # Inertia matrix (kg·m²)
    m: float = 4.34                                                                    # Mass (kg)
    d: float = 0.315                                                                   # Propeller distance (m)
    ctf: float = 8.004e-4                                                              # Thrust-torque coefficient

    # Force/torque conversion matrix (FM) and its inverse (FM2f)
    FM: np.ndarray = field(init=False) 
    FM2f: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.FM = np.array([
            [1, 1, 1, 1],
            [0, -self.d, 0, self.d],
            [self.d, 0, -self.d, 0],
            [-self.ctf, self.ctf, -self.ctf, self.ctf]
        ])
        self.FM2f = np.linalg.inv(self.FM)
###########################################################################

# UAV trajectory definition
###########################################################################
@dataclass
class UAVTrajectory:
    '''UAV trajectory definition'''
    xd: Callable[[float], np.ndarray]          # Position trajectory
    b1d: Callable[[float], np.ndarray]         # Attitude trajectory (b1 direction)
    vd: Callable[[float], np.ndarray] | None = None  # Velocity (computed via difference if None)
    ad: Callable[[float], np.ndarray] | None = None  # Acceleration (computed via difference if None)

    def get_vd(self, t: float, t_last: float):
        """Get desired velocity (use provided or numerical derivative)"""
        if self.vd is not None:
            res = self.vd(t)
        else:
            dt = safe_time_diff(t, t_last)
            res = (self.xd(t) - self.xd(t_last)) / dt
        return res.astype(np.float32)
    
    def get_ad(self, t: float, t_last: float, t_ll: float):
        """Get desired acceleration (use provided or numerical derivative)"""
        if self.ad is not None:
            res =  self.ad(t)
        else:
            dt = safe_time_diff(t, t_last)
            res =(self.get_vd(t, t_last) - self.get_vd(t_last, t_ll)) / dt
        return res.astype(np.float32)
###########################################################################