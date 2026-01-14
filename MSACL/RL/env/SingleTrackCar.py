"""
Implementation of the Singletrackcar model
Reference:
Safe model-based reinforcement learning with stability guarantees
(https://github.com/MIT-REALM/neural_clbf)

The model is designed for high-fidelity ground vehicle trajectory . 
The core equations are consistent with the above reference.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class singletrackcar(gym.Env):
    """
    Single-track car trajectory tracking environment (from TRO paper)
    State: [x error, y error, steering angle, longitudinal speed error, 
            heading error angle, heading error rate, lateral deflection angle]
    Control: [steering angular velocity, longitudinal acceleration]
    """

    N_DIMS = 7
    N_CONTROLS = 2
    __g = 9.81  # Gravitational acceleration
    
    # State indices
    SXE = 0     # X position error
    SYE = 1     # Y position error
    DELTA = 2   # Steering angle [rad]
    VE = 3      # Longitudinal speed error [m/s]
    PSI_E = 4   # Heading error angle [rad]
    PSI_E_DOT = 5 # Heading error rate [rad/s]
    BETA = 6    # Lateral deflection angle [rad]

    # Control indices
    VDELTA = 0  # Steering angular velocity [rad/s]
    ALONG = 1   # Longitudinal acceleration [m/s²]

    def __init__(self, dt=0.005):
        super().__init__()

        # State/step initialization
        self.obs = None
        self.current_step = 0

        # Vehicle physical limits
        self.steering_min = -1.066  # Min steering angle [rad]
        self.steering_max = 1.066  # Max steering angle [rad]
        self.steering_v_min = -0.4  # Min steering velocity [rad/s]
        self.steering_v_max = 0.4  # Max steering velocity [rad/s]
        self.longitudinal_a_max = 5  # Max longitudinal acceleration [m/s²]

        # Tire parameters
        self.tire_p_dy1 = 1.0489   # Lateral friction coefficient
        self.tire_p_ky1 = -21.92   # Max stiffness (Kfy/Fznom)

        # Vehicle geometric/inertial parameters
        self.a = 0.3048 * 3.793293  # CG to front axle [m]
        self.b = 0.3048 * 4.667707  # CG to rear axle [m]
        self.h_s = 0.3048 * 2.01355 # CG height [m]
        self.m = 4.4482216152605 / 0.3048 * (74.91452)  # Mass [kg]
        self.I_z = 4.4482216152605 * 0.3048 * (1321.416) # Yaw moment of inertia [kg·m²]
        
        # Reference trajectory parameters (unit circle)
        self.nominal_params = {
            "v_ref": 1.0,      # Reference speed [m/s]
            "omega_ref": 0.0,  # Reference angular velocity [rad/s]
            "a_ref": 0.0,      # Reference acceleration [m/s²]
            "psi_ref": 0.0,     # Reference heading [rad]
            "mu_scale": 0.1,    # Friction scale (nonlinearity level)
        }

        # Observation space bounds (from original TRO code)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, self.steering_min, -1.0, -np.pi/2, -np.pi/2, -np.pi/3]),
            high=np.array([1.0, 1.0, self.steering_max, 1.0, np.pi/2, np.pi/2, np.pi/3]),
            dtype=np.float32
        )

        # Action space bounds
        self.action_space = spaces.Box(
            low=np.array([-5.0, -self.longitudinal_a_max]),
            high=np.array([5.0, self.longitudinal_a_max]),
            dtype=np.float32
        )

        # Random seed range
        self.min_seed_value = np.iinfo(np.uint32).min
        self.max_seed_value = np.iinfo(np.uint32).max

        # State dimensions and reset perturbation
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]
        self.reset_noise = 0.5  # Initial state perturbation (small neighborhood of origin)

        # Reward function weights
        self.Q = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # State error weights
        self.R = np.array([0.1, 0.1], dtype=np.float32)  # Control cost weights

        # Convergence threshold
        self.origin_radius = 1e-2 

        # Simulation parameters
        self.dt = 0.01          # Time step
        self.control_step = 5   # Control update interval
        self.max_step = 1000    # Max steps per episode


    def reset(self, seed=None, options=None):
        """Reset environment to random initial state (small perturbation around origin)"""
        # Set random seed
        if seed == None:
            seed = random.randint(self.min_seed_value, self.max_seed_value)
        elif not isinstance(seed, int):
            seed = int(seed)
        super().reset(seed=seed)

        # Random initial state (small perturbation)
        self.obs = self.np_random.uniform(
            low=-self.reset_noise * np.ones(self.obs_dim, dtype=np.float32),
            high=self.reset_noise * np.ones(self.obs_dim, dtype=np.float32)
        ).astype(np.float32)

        self.current_step = 0  # Reset step counter

        return self.obs.copy(), {}
    

    def _f(self, x):
        """Compute f(x) in dx/dt = f(x) + g(x)u (control-independent dynamics)"""
        f = np.zeros(self.obs_dim)

        # Extract reference parameters
        params = self.nominal_params
        v_ref = params["v_ref"]
        a_ref = params["a_ref"]
        omega_ref = params["omega_ref"]
        mu_scale = params.get("mu_scale", 0.1)

        # Extract state variables
        sxe = x[self.SXE]
        sye = x[self.SYE]
        delta = x[self.DELTA]
        ve = x[self.VE]
        psi_e = x[self.PSI_E]
        psi_e_dot = x[self.PSI_E_DOT]
        beta = x[self.BETA]

        # Derived vehicle states
        v = ve + v_ref
        psi_dot = psi_e_dot + omega_ref

        # Tire/vehicle dynamic parameters
        mu = mu_scale * self.tire_p_dy1
        C_Sf = -self.tire_p_ky1 / self.tire_p_dy1
        C_Sr = -self.tire_p_ky1 / self.tire_p_dy1
        lf = self.a
        lr = self.b
        m = self.m
        Iz = self.I_z

        # Position error dynamics (reference frame)
        dsxe_r = v * np.cos(psi_e + beta) - v_ref + omega_ref * sye
        dsye_r = v * np.sin(psi_e + beta) - omega_ref * sxe

        # Base dynamics
        f[self.SXE] = dsxe_r
        f[self.SYE] = dsye_r
        f[self.VE] = -a_ref
        f[self.DELTA] = 0.0

        # Switch between kinematic/dynamic model (based on speed)
        use_kinematic_model = np.abs(v) < 0.1
        if not use_kinematic_model:
            """Dynamic model (high speed)"""
            f[self.PSI_E] = psi_e_dot
            f[self.PSI_E_DOT] = (
                -(mu * m / (v * Iz * (lr + lf)))
                * (lf ** 2 * C_Sf * self.__g * lr + lr ** 2 * C_Sr * self.__g * lf)
                * psi_dot
                + (mu * m / (Iz * (lr + lf)))
                * (lr * C_Sr * self.__g * lf - lf * C_Sf * self.__g * lr)
                * beta
                + (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * self.__g * lr) * delta
            )
            f[self.BETA] = (
                (
                    (mu / (v ** 2 * (lr + lf))) * (C_Sr * self.__g * lf * lr - C_Sf * self.__g * lr * lf)
                    - 1
                )
                * psi_dot
                - (mu / (v * (lr + lf))) * (C_Sr * self.__g * lf + C_Sf * self.__g * lr) * beta
                + mu / (v * (lr + lf)) * (C_Sf * self.__g * lr) * delta
            )
        else:
            """Kinematic model (low speed)"""
            lwb = lf + lr
            f[self.PSI_E] = (
                v * np.cos(beta) / lwb * np.tan(delta) - omega_ref
            )
            f[self.PSI_E_DOT] = 0.0
            f[self.BETA] = 0.0

        return f


    def _g(self, x):
        """Compute g(x) in dx/dt = f(x) + g(x)u (control-dependent dynamics)"""
        g = np.zeros((self.obs_dim, self.act_dim))

        # Extract reference parameters
        params = self.nominal_params
        v_ref = params["v_ref"]
        omega_ref = params["omega_ref"]
        mu_scale = params.get("mu_scale", 0.1)

        # Extract state variables
        delta = x[self.DELTA]
        ve = x[self.VE]
        psi_e_dot = x[self.PSI_E_DOT]
        beta = x[self.BETA]

        # Derived vehicle states
        v = ve + v_ref
        psi_dot = psi_e_dot + omega_ref

        # Tire/vehicle dynamic parameters
        mu = mu_scale * self.tire_p_dy1
        C_Sf = -self.tire_p_ky1 / self.tire_p_dy1
        C_Sr = -self.tire_p_ky1 / self.tire_p_dy1
        lf = self.a
        lr = self.b
        h = self.h_s
        m = self.m
        Iz = self.I_z

        # Switch between kinematic/dynamic model (based on speed)
        use_kinematic_model = np.abs(v) < 0.1
        if not use_kinematic_model:
            """Dynamic model (high speed)"""
            g[self.DELTA, self.VDELTA] = 1.0
            g[self.VE, self.ALONG] = 1.0
            g[self.PSI_E_DOT, self.ALONG] = (
                -(mu * m / (v * Iz * (lr + lf)))
                * (-(lf ** 2) * C_Sf * h + lr ** 2 * C_Sr * h)
                * psi_dot
                + (mu * m / (Iz * (lr + lf))) * (lr * C_Sr * h + lf * C_Sf * h) * beta
                - (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * h) * delta
            )
            g[self.BETA, self.ALONG] = (
                (mu / (v ** 2 * (lr + lf))) * (C_Sr * h * lr + C_Sf * h * lf) * psi_dot
                - (mu / (v * (lr + lf))) * (C_Sr * h - C_Sf * h) * beta
                - mu / (v * (lr + lf)) * C_Sf * h * delta
            )
        else:
            """Kinematic model (low speed)"""
            lwb = lf + lr
            beta_dot = (
                1
                / (1 + (np.tan(delta) * lr / lwb) ** 2)
                * lr
                / (lwb * np.cos(delta) ** 2)
            )
            g[self.PSI_E_DOT, self.ALONG] = (
                1 / lwb * (np.cos(beta) * np.tan(delta))
            )
            g[self.PSI_E_DOT, self.VDELTA] = (
                1
                / lwb
                * (
                    -v * np.sin(beta) * np.tan(delta) * beta_dot
                    + v * np.cos(beta) / np.cos(delta) ** 2
                )
            )
            g[self.BETA, 0] = beta_dot

        return g
    

    def _dynamics(self, obs, action):
        """Compute state derivative: dx/dt = f(x) + g(x)u"""
        f = self._f(obs)    # Control-independent term
        g = self._g(obs)    # Control-dependent matrix
        u = action          # Control input
        
        return f + g @ u


    def step(self, action):
        """Interact with environment (single step)"""
        # Update state with Euler integration
        for _ in range(self.control_step):
            dot_obs = self._dynamics(self.obs, action)
            self.obs += dot_obs * self.dt

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

        return next_obs, reward, terminated, truncated, info
