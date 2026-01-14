"""
Base class for sampler classes
"""

from abc import ABCMeta, abstractmethod
from RL.create_pkg.create_envs import create_envs
from RL.create_pkg.create_alg import create_approx_contrainer
from RL.utils.explore_noise import GaussNoise

from typing import List, NamedTuple, Union, Tuple
from collections import deque
import numpy as np
import torch

import time
from RL.utils.tensorboard_setup import tb_tags
from RL.utils.rew_plus_cost import rew_plus_cost

class Experience(NamedTuple):
    """
    NamedTuple for single-step RL experience (state transition).
    Immutable data structure with named fields for easy access.
    """
    obs: np.ndarray
    action: np.ndarray
    reward: float
    cost: float
    next_obs: np.ndarray
    done: bool
    log_prob: float

# N-step experience class for storing multi-step interaction data
class nStepExperience(NamedTuple):
    """
    NamedTuple for n-step RL experience (multi-step state transitions).
    All fields store n-step data as numpy arrays (converted to tensors during training).
    """
    n_step_obs: np.ndarray
    n_step_act: np.ndarray
    n_step_rew: float
    n_step_cost: float
    # n_step_gain: float
    n_step_obs2: np.ndarray
    n_step_done: bool
    n_step_log_prob: float


class BaseSampler(metaclass=ABCMeta):
    """Abstract base class for Sampler (OnSampler/OffSampler)."""
    
    def __init__(
        self,
        **kwargs,
    ):
        self.env_id = kwargs["env_name"]
        self.num_envs = kwargs["env_num"]
        self.envs = create_envs(**kwargs)

        # Get observation/action space dimensions
        self.obs_dim = self.envs.single_observation_space.shape
        self.act_dim = self.envs.single_action_space.shape

        # Create function approximators (Q/policy networks)
        self.networks = create_approx_contrainer(**kwargs)

        # Sampling configuration
        self.sample_batch_size = kwargs["sample_batch_size"] * self.num_envs
        self.action_type = kwargs["action_type"]
        self.reward_scale = kwargs["reward_scale"]
        self.cost_scale = kwargs["cost_scale"]

        # Exploration noise configuration
        self.noise_params = kwargs["noise_params"]
        if self.noise_params is not None:
            if self.action_type == "continu":
                self.noise_processor = GaussNoise(**self.noise_params)
            else:
                raise RuntimeError("Only continuous action space is supported!")
            
        # Target state/value for reward/cost calculation
        self.target_value = kwargs["target_value"]
            
        # Total sampled steps counter
        self.total_sample_number = 0

        # Interaction horizon per environment
        self.horizon = self.sample_batch_size // self.num_envs

        # N-step configuration (default: 1 step)
        self.n_step = kwargs.get("n_step", 1)
        self.gamma = kwargs["gamma"]  # Discount factor for Retrace operator
        self.td_lambda = kwargs.get("retrace_lambda", 0.95)  # Lambda for Retrace(lambda)

        # N-step buffers (one deque per environment, max length = n_step)
        self.n_step_buffers = [deque(maxlen=self.n_step) for _ in range(self.num_envs)]

        # Initialize environment (no fixed seed for random initial state)
        self.obs, _ = self.envs.reset(seed=None)
        self.obs = np.float32(self.obs)


    def get_total_sample_num(self) -> int:
        """Return total number of sampled steps."""
        return self.total_sample_number
    

    def get_total_sample_number(self) -> int:
        """Duplicate method for total sampled steps (compatibility)."""
        return self.total_sample_number
    

    def load_state_dict(self, state_dict):
        """Load pre-trained network parameters from state dict."""
        self.networks.load_state_dict(state_dict)


    # N-step interaction function for storing multi-step data
    def _n_step(self,) -> List[nStepExperience]:
        """
        Single step environment interaction for collecting n-step experience.
        Returns list of nStepExperience (empty if buffer not filled to n-step).
        """
        # Convert current observation to tensor
        obs_tensor = torch.from_numpy(self.obs)

        # Sample action from policy distribution
        logits = self.networks.policy(obs_tensor)
        action_distribution = self.networks.create_action_distributions(logits)
        actions, log_probs = action_distribution.sample()

        # Detach gradients and convert to numpy float32
        actions = actions.detach().numpy().astype("float32")
        log_probs = log_probs.detach().numpy().astype("float32")

        # Add exploration noise if enabled
        if self.noise_params is not None:
            actions = self.noise_processor.sample(actions)

        # Clip actions to valid range (continuous space only)
        if self.action_type == "continu":
            actions_clip = actions.clip(
                self.envs.action_space.low, self.envs.action_space.high
            )
        else:
            actions_clip = actions

        # Step environment (single step interaction)
        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions_clip)

        # Type conversion to float32
        next_obs = np.float32(next_obs)
        original_rewards = np.float32(rewards)
        obs = np.float32(self.obs)

        # Get true next observation (handle termination/truncation)
        dones = np.logical_or(terminations, truncations)
        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Calculate scaled rewards and costs
        rewards, costs = rew_plus_cost(
            self.env_id,
            obs,
            real_next_obs,
            self.target_value,
            self.cost_scale,
            self.reward_scale,
            original_rewards,
            terminations
        )

        # Collect n-step experiences
        n_step_experiences = []

        # Process each environment's n-step buffer
        for i in range(self.num_envs):
            # Add single-step data to environment's deque buffer
            self.n_step_buffers[i].append({
                "obs": obs[i],
                "act": actions_clip[i],
                "rew": rewards[i],
                "cost": costs[i],
                "obs2": real_next_obs[i],
                "done": dones[i],
                "log_prob": log_probs[i]
            })

            buffer_i = self.n_step_buffers[i]

            # Create nStepExperience only when buffer is full (n-step collected)
            if len(buffer_i) == self.n_step:
                # Concatenate n-step data from buffer
                buffer_i_obs = np.array([item["obs"] for item in buffer_i], dtype=np.float32)
                buffer_i_act = np.array([item["act"] for item in buffer_i], dtype=np.float32)
                buffer_i_rew = np.array([item["rew"] for item in buffer_i], dtype=np.float32)
                buffer_i_cost = np.array([item["cost"] for item in buffer_i], dtype=np.float32)
                buffer_i_obs2 = np.array([item["obs2"] for item in buffer_i], dtype=np.float32)
                buffer_i_done = np.array([item["done"] for item in buffer_i], dtype=np.float32)
                buffer_i_log_prob = np.array([item["log_prob"] for item in buffer_i], dtype=np.float32)

                # Create nStepExperience object
                n_step_experience = nStepExperience(
                    n_step_obs=buffer_i_obs,
                    n_step_act=buffer_i_act,
                    n_step_rew=buffer_i_rew,
                    n_step_cost=buffer_i_cost,
                    n_step_obs2=buffer_i_obs2,
                    n_step_done=buffer_i_done,
                    n_step_log_prob=buffer_i_log_prob,
                )
                n_step_experiences.append(n_step_experience)

            # Clear buffer if episode ends (ensure complete trajectory collection)
            if buffer_i and buffer_i[-1]["done"]:
                self.n_step_buffers[i].clear()

        # Update current observation (use next_obs, not real_next_obs)
        self.obs = next_obs

        return n_step_experiences
        

    def _step(self,) -> List[Experience]:
        """
        Single step environment interaction for collecting single-step experience.
        Returns list of Experience objects (one per environment).
        """
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(self.obs)

        # Sample action from policy distribution
        logits = self.networks.policy(obs_tensor)
        action_distribution = self.networks.create_action_distributions(logits)
        actions, log_probs = action_distribution.sample()

        # Detach gradients and convert to numpy float32
        actions = actions.detach().numpy().astype("float32")
        log_probs = log_probs.detach().numpy().astype("float32")

        # Add exploration noise if enabled
        if self.noise_params is not None:
            actions = self.noise_processor.sample(actions)

        # Clip actions to valid range (continuous space only)
        if self.action_type == "continu":
            actions_clip = actions.clip(
                self.envs.action_space.low, self.envs.action_space.high
            )
        else:
            actions_clip = actions

        # Step environment
        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions_clip)

        # Type conversion to float32
        next_obs = np.float32(next_obs)
        original_rewards = np.float32(rewards)
        obs = np.float32(self.obs)

        # Get true next observation (handle termination/truncation)
        dones = np.logical_or(terminations, truncations)
        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Calculate scaled rewards and costs
        rewards, costs = rew_plus_cost(
            self.env_id,
            obs,
            real_next_obs,
            self.target_value,
            self.cost_scale,
            self.reward_scale,
            original_rewards,
            terminations
        )

        # Collect single-step experiences
        experiences = []
        for i in range(0, self.num_envs):
            experience = Experience(
                obs=obs[i],
                action=actions_clip[i],
                reward=rewards[i],
                cost=costs[i],
                next_obs=real_next_obs[i],
                done=dones[i],
                log_prob=log_probs[i],
            )
            experiences.append(experience)

        # Update current observation
        self.obs = next_obs

        return experiences

    @abstractmethod
    def _sample(self) -> Union[List[Experience], dict]:
        """
        Abstract sampling method (must be implemented by subclasses).
        Returns: List of Experience objects or dict of sampled data.
        """
        pass

    def sample(self) -> Tuple[Union[List[Experience], List[nStepExperience], dict], dict]:
        """
        Main sampling interface (measures sampling time for logging).
        Returns: Sampled data (Experience/nStepExperience list or dict) + tensorboard info.
        """
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()

        start_time = time.perf_counter()
        data = self._sample()
        end_time = time.perf_counter()

        # Log sampling time (ms)
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000

        return data, tb_info