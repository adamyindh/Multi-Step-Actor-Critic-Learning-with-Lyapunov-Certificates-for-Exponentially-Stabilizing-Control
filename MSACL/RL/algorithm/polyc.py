"""
The POLYC algorithm
Reference:
This algorithm is implemented based on the work Stabilizing neural control using self-learned almost Lyapunov critics.
url={https://doi.org/10.1109/ICRA48506.2021.9560886}
"""

__all__ = ["ApproxContainer", "POLYC"] 

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from typing import Any, Optional, Tuple, Dict
from torch import Tensor
import time

from RL.utils.common_utils import get_apprfunc_dict
from RL.create_pkg.create_apprfunc import create_apprfunc 

from RL.utils.tensorboard_setup import tb_tags



class ApproxContainer(nn.Module):
    """POLYC networks: Value function, Lyapunov function, Policy network"""
    def __init__(self, **kwargs):
        super().__init__()

        # Value network
        value_args = get_apprfunc_dict("value", **kwargs)
        self.value: nn.Module = create_apprfunc(**value_args)

        # Lyapunov function network
        lyapunov_args = get_apprfunc_dict("lyapunov", **kwargs)
        self.lyapunov: nn.Module = create_apprfunc(**kwargs)

        # Policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)

        # Optimizers (separate for each network)
        self.value_optimizer = Adam(self.value.parameters(), lr=kwargs["learning_rate"])
        self.lyapunov_optimizer = Adam(self.lyapunov.parameters(), lr=kwargs["learning_rate"])
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs["policy_learning_rate"])

    def create_action_distributions(self, logits):
        """Create action distribution (e.g., TanhGaussian) from policy logits"""
        return self.policy.get_act_dist_cls(logits)
    
class POLYC:
    """POLYC algorithm (PPO-based with Lyapunov stability constraint)"""
    def __init__(
        self,
        *,  # Force keyword arguments
        max_iteration: int,
        num_repeat: int,
        num_mini_batch: int,
        mini_batch_size: int,
        sample_batch_size: int,
        env_num: int,
        index: int = 0,
        gamma: float = 0.99,
        clip: float = 0.1,
        beta: float = 0.2,  # Weight for hybrid advantage
        lya_diff_scale: float = 1.0,  # Weight for Lyapunov difference term (1/delta_t)
        lya_zero_scale: float = 20.0,  # Weight for Lyapunov zero term
        lya_positive_scale: float = 1.0,  # Weight for Lyapunov positive term
        advantage_norm: bool = True,  # Normalize advantage
        loss_value_clip: bool = True,  # Clip value loss
        value_clip: float = 0.2,  # Value clip range
        loss_value_norm: bool = False,  # Normalize value loss
        loss_coefficient_kl: float = 0.2,  # Unused
        loss_coefficient_value: float = 1.0,  # Value loss weight
        loss_coefficient_entropy: float = 0.01,  # Entropy loss weight
        schedule_adam: str = "None",  # Adam lr annealing
        schedule_clip: str = "None",  # Clip coefficient annealing
        **kwargs
    ):
        self.max_iteration = max_iteration
        self.num_repeat = num_repeat
        self.num_mini_batch = num_mini_batch

        # Batch size settings (scale with env num)
        self.env_num = env_num
        self.sample_batch_size = sample_batch_size * self.env_num
        self.indices = np.arange(self.sample_batch_size)
        self.mini_batch_size = mini_batch_size * self.env_num

        # PPO core params
        self.gamma = gamma
        self.clip = clip
        self.clip_now = self.clip
        self.advantage_norm = advantage_norm
        self.loss_value_clip = loss_value_clip
        self.value_clip = value_clip
        self.loss_value_norm = loss_value_norm
        self.loss_coefficient_kl = loss_coefficient_kl
        self.loss_coefficient_value = loss_coefficient_value
        self.loss_coefficient_entropy = loss_coefficient_entropy
        self.schedule_adam = schedule_adam
        self.schedule_clip = schedule_clip

        # Lyapunov loss weights
        self.lya_diff_scale = lya_diff_scale
        self.lya_zero_scale = lya_zero_scale
        self.lya_positive_scale = lya_positive_scale

        # Env settings
        self.env_name = kwargs["env_name"]
        self.env_id = self.env_name
        self.target_value = kwargs["target_value"]
        
        # Network init
        self.networks = ApproxContainer(**kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.policy_learning_rate = kwargs["policy_learning_rate"]
        self.EPS = 1e-8  # Avoid division by zero
        self.beta = beta  # Hybrid advantage weight

        # Global iteration counter
        self.global_iteration = 0

    
    @property
    def adjustable_parameters(self):
        return (
            "gamma", "clip", "advantage_norm", "loss_value_clip",
            "value_clip", "loss_value_norm", "loss_coefficient_kl",
            "loss_coefficient_value", "loss_coefficient_entropy",
            "schedule_adam", "schedule_clip"
        )


    def model_update(self, data: Dict[str, Tensor]):
        """POLYC model update pipeline (PPO + Lyapunov)"""
        start_time = time.perf_counter()

        #################### Env-specific state correction (HalfCheetah-v4) ####################
        if self.env_id == "HalfCheetah-v4":
            data["obs"][:, 8] -= self.target_value
            data["obs2"][:, 8] -= self.target_value

        # Precompute policy logits (KL divergence calculation)
        with torch.no_grad():
            data["logits"] = self.networks.policy(data["obs"])

        # Normalize PPO advantage
        data["adv"] = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + self.EPS)

        #################### Lyapunov network update (batch-level) ####################
        # Update Lyapunov network with Lyapunov Risk loss
        loss_lya = self._compute_loss_lya(data)
        self.networks.lyapunov_optimizer.zero_grad()
        loss_lya.backward()
        self.networks.lyapunov_optimizer.step()

        # Compute hybrid advantage (PPO adv + Lyapunov stability adv)
        with torch.no_grad():
            data["lya"] = self.networks.lyapunov(data["obs"])
            data["lya2"] = self.networks.lyapunov(data["obs2"])

        # Normalize Lyapunov difference (stability constraint)
        data["diff_lya"] = data["lya2"] - data["lya"]
        data["diff_lya"] = (data["diff_lya"] - data["diff_lya"].mean()) / (data["diff_lya"].std() + self.EPS)
        data["diff_lya"] = torch.min(-data["diff_lya"], torch.zeros_like(data["diff_lya"]))

        # Hybrid advantage (weighted by beta)
        data["adv"] = (1-self.beta) * data["adv"] + self.beta * data["diff_lya"]

        #################### PPO repeat training (on-policy reuse) ####################
        for _ in range(self.num_repeat):
            np.random.shuffle(self.indices)  # Shuffle batch indices

            for n in range(self.num_mini_batch):
                self.global_iteration += 1

                # Get mini-batch indices
                mb_start = self.mini_batch_size * n
                mb_end = self.mini_batch_size * (n + 1)
                mb_indices = self.indices[mb_start:mb_end]
                mb_sample = {k: v[mb_indices] for k, v in data.items()}

                ########## Policy update ##########
                loss_policy = self._compute_loss_policy(mb_sample, self.global_iteration)
                self.networks.policy_optimizer.zero_grad()
                loss_policy.backward()
                self.networks.policy_optimizer.step()

                ########## Value network update ##########
                loss_value = self._compute_loss_value(mb_sample)
                self.networks.value_optimizer.zero_grad()
                loss_value.backward()
                self.networks.value_optimizer.step()

                ########## Adam lr annealing##########
                if self.schedule_adam == "linear":
                    decay_rate = max(0.0, 1 - (self.global_iteration / self.max_iteration))
                    lr_now = self.learning_rate * decay_rate
                    self.networks.lyapunov_optimizer.param_groups[0]["lr"] = lr_now
                    self.networks.value_optimizer.param_groups[0]["lr"] = lr_now
                    self.networks.policy_optimizer.param_groups[0]["lr"] = self.policy_learning_rate * decay_rate

        # Training time calculation
        end_time = time.perf_counter()

        # TensorBoard logging
        tb_info = dict()
        tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        tb_info[tb_tags["loss_critic"]] = loss_value.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000

        return tb_info, self.global_iteration
    

    def _compute_loss_lya(self, data: Dict[str, Tensor]):
        """Compute Lyapunov Risk loss (zero term + difference term)"""
        obs, obs2 = data["obs"], data["obs2"]

        # Term 1: Lyapunov value at origin = 0
        if self.env_id == "HalfCheetah-v4":
            obs_zero = data["obs"].clone()
            obs_zero[:, 8] = 0
        else:
            obs_zero = torch.zeros_like(obs)
        loss_lya1 = torch.pow(self.networks.lyapunov(obs_zero), 2).mean() * self.lya_zero_scale

        # Term 3: Lyapunov difference ≤ 0
        diff_lyapunov = self.networks.lyapunov(obs2) - self.networks.lyapunov(obs)
        loss_lya3 = torch.max(diff_lyapunov, torch.zeros_like(diff_lyapunov)).mean() * self.lya_diff_scale

        # Total Lyapunov loss
        loss_lya = loss_lya1 + loss_lya3

        return loss_lya


    def _compute_loss_policy(self, data: Dict[str, Tensor], global_iteration: int):
        """Compute PPO policy loss (surrogate + entropy + KL)"""
        obs, act, logp = data["obs"], data["act"], data["logp"]
        advantages = data["adv"]
        logits = data["logits"]

        # New/old policy log prob
        mb_new_logits = self.networks.policy(obs)
        mb_new_act_dist = self.networks.create_action_distributions(mb_new_logits)
        mb_new_log_pro = mb_new_act_dist.log_prob(act)
        mb_old_act_dist = self.networks.create_action_distributions(logits)

        # PPO surrogate loss (clip)
        mb_advantage = advantages.detach()
        ratio = torch.exp(mb_new_log_pro - logp)
        sur1 = ratio * mb_advantage
        sur2 = ratio.clamp(1 - self.clip_now, 1 + self.clip_now) * mb_advantage
        loss_surrogate = -torch.mean(torch.min(sur1, sur2))

        # Entropy loss (exploration)
        loss_entropy = -torch.mean(mb_new_act_dist.entropy()) * self.loss_coefficient_entropy

        # KL divergence loss (unused)
        loss_kl = torch.mean(mb_old_act_dist.kl_divergence(mb_new_act_dist)) * self.loss_coefficient_kl

        # Total policy loss
        loss_policy = loss_surrogate + loss_entropy + loss_kl

        # Linear clip schedule
        if self.schedule_clip == "linear":
            decay_rate = 1 - (global_iteration / self.max_iteration)
            self.clip_now = self.clip * decay_rate

        return loss_policy 


    def _compute_loss_value(self, data: Dict[str, Tensor]):
        """Compute PPO value loss (clipped/non-clipped)"""
        obs, returns, values = data["obs"], data["ret"], data["val"]

        # Value prediction
        mb_new_value = self.networks.value(obs)
        mb_old_value = values
        mb_return = returns.detach()

        # Clipped value loss
        if self.loss_value_clip:
            value_losses1 = torch.pow(mb_new_value - mb_return, 2)
            mb_new_value_clipped = mb_old_value + (mb_new_value - mb_old_value).clamp(-self.value_clip, self.value_clip)
            value_losses2 = torch.pow(mb_new_value_clipped - mb_return, 2)
            value_losses = torch.max(value_losses1, value_losses2)
        else:
            value_losses = torch.pow(mb_new_value - mb_return, 2)

        # Value loss normalization (optional)
        if self.loss_value_norm:
            mb_return_6std = 6 * mb_return.std()
            loss_value = torch.mean(value_losses) / mb_return_6std
        else:
            loss_value = torch.mean(value_losses)

        return loss_value