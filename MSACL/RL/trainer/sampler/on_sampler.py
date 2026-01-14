"""
On-policy sampler for algorithms like PPO, POLYC
"""
from typing import List

import numpy as np
import torch

from RL.trainer.sampler.base import BaseSampler, Experience

class OnSampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gamma = kwargs["gamma"]
        self.gae_lambda = kwargs["gae_lambda"]

        # Obs/action space dimensions (single env for vectorized envs)
        self.obs_dim = self.envs.single_observation_space.shape
        self.act_dim = self.envs.single_action_space.shape

        # Mini-batch buffers (shape: [env_num, horizon, *dim])
        self.mb_obs = np.zeros(
            (self.num_envs, self.horizon, *self.obs_dim), dtype=np.float32
        )
        self.mb_obs2 = np.zeros(
            (self.num_envs, self.horizon, *self.obs_dim), dtype=np.float32
        )
        self.mb_act = np.zeros(
            (self.num_envs, self.horizon, *self.act_dim), dtype=np.float32
        )

        # Additional data buffers
        self.mb_rew = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_cost = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_logp = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_done = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)

        # Value/advantage/return buffers for GAE calculation
        self.mb_val = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_adv = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_ret = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        

    def _sample(self) -> dict:
        # Reset pointers for each sampling call (critical for data alignment)
        self.ptr = np.zeros(self.num_envs, dtype=np.int32)
        self.last_ptr = -np.ones(self.num_envs, dtype=np.int32)

        ########## Environment Interaction ##########
        # Interact with environment for horizon steps and store data
        for t in range(self.horizon):
            batch_obs = torch.from_numpy(self.obs.astype("float32"))
            experiences = self._step()
            self._process_experiences(experiences, batch_obs, t)

        ########## Data Preparation for Training ##########
        # Flatten env dimension (shape: [env_num*horizon, *dim])
        mb_data = {
            "obs": torch.from_numpy(self.mb_obs.reshape(-1, *self.obs_dim)),
            "obs2": torch.from_numpy(self.mb_obs2.reshape(-1, *self.obs_dim)),
            "act": torch.from_numpy(self.mb_act.reshape(-1, *self.act_dim)),
            "rew": torch.from_numpy(self.mb_rew.reshape(-1)),
            "cost": torch.from_numpy(self.mb_cost.reshape(-1)),
            "done": torch.from_numpy(self.mb_done.reshape(-1)),
            "logp": torch.from_numpy(self.mb_logp.reshape(-1)),
        }
        # Add GAE-derived advantages, returns and value estimates
        mb_data.update({
            "adv": torch.from_numpy(self.mb_adv.reshape(-1)),
            "ret": torch.from_numpy(self.mb_ret.reshape(-1)),
            "val": torch.from_numpy(self.mb_val.reshape(-1))
        })

        return mb_data
    

    def sample_with_replay_format(self):
        """Return sampled data in replay buffer format (data + tensorboard info)"""
        return self.sample()


    def _process_experiences(
        self,
        experiences: List[Experience],
        batch_obs: np.ndarray,
        t: int
    ):
        # Compute state values (for GAE calculation)
        value = self.networks.value(batch_obs).detach()
        self.mb_val[:, t] = value

        # Process each environment's experience individually
        for i in np.arange(self.num_envs):
            obs, action, reward, cost, next_obs, done, logp = experiences[i]

            # Store experience data to mini-batch buffers
            self.mb_obs[i,t,...] = obs
            self.mb_act[i,t,...] = action
            self.mb_rew[i,t,...] = reward
            self.mb_cost[i,t,...] = cost
            self.mb_obs2[i,t,...] = next_obs
            self.mb_done[i,t,...] = done
            self.mb_logp[i,t,...] = logp

            # Calculate returns/advantages when episode ends or horizon is reached
            if done or t == self.horizon - 1:
                # Estimate last state value (0 if episode terminates)
                last_obs_expand = torch.from_numpy(
                    np.expand_dims(next_obs, axis=0).astype("float32")
                )
                est_last_value = self.networks.value(last_obs_expand).detach().item() * (1-done)

                self.ptr[i] = t  # Update trajectory end position
                self._finish_trajs(i, est_last_value)  # Compute GAE and returns
                self.last_ptr[i] = self.ptr[i]  # Update last trajectory end position

    def _finish_trajs(
            self, 
            env_index: int, 
            est_last_val: float,
        ):
        """
        Calculate returns (for value network update) and advantages (for policy network update)
        using Generalized Advantage Estimation (GAE).
        """
        # Define trajectory slice (from last end to current end)
        path_slice = slice(self.last_ptr[env_index] + 1, self.ptr[env_index] + 1)

        # Prepare value predictions (add last state value) and rewards
        value_preds_slice = np.append(self.mb_val[env_index, path_slice], est_last_val)
        rews_slice = self.mb_rew[env_index, path_slice]
        length = len(rews_slice)

        # Initialize return/advantage arrays
        ret = np.zeros(length)
        adv = np.zeros(length)
        gae = 0.0
        G_t = 0
        
        # Calculate GAE and discounted returns (backward pass)
        for i in reversed(range(length)):
            # TD error: r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rews_slice[i] + self.gamma * value_preds_slice[i + 1] - value_preds_slice[i]
            # Accumulated GAE
            gae = delta + self.gamma * self.gae_lambda * gae
            adv[i] = gae

            # Discounted cumulative return
            G_t = rews_slice[i] + self.gamma * G_t
            ret[i] = G_t

        # Store computed advantages and returns to buffers
        self.mb_adv[env_index, path_slice] = adv
        self.mb_ret[env_index, path_slice] = ret