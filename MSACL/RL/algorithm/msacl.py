"""
The MSACL algorithm
"""

__all__ = ["ApproxContainer", "MSACL"] 

import torch
import torch.nn as nn

from RL.utils.common_utils import get_apprfunc_dict
from RL.create_pkg.create_apprfunc import create_apprfunc

from copy import deepcopy
from torch.optim import Adam

import math
from typing import Any, Optional, Tuple, Dict
from torch import Tensor
import time
from RL.utils.tensorboard_setup import tb_tags


class ApproxContainer(nn.Module):
    """
    Define all function approximators for MOL algorithm: 1 policy function + 2 value functions (2 Q-values) + 1 Lyapunov function
    """
    def __init__(self, **kwargs):
        super().__init__()

        # Create Q-networks (state-action value functions)
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)

        # Create target Q-networks (no gradient update, manual soft update)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Create Lyapunov function network (params prefixed with "Lyapunov")
        lyapunov_args = get_apprfunc_dict("lyapunov", **kwargs)
        self.lyapunov: nn.Module = create_apprfunc(**lyapunov_args)

        # Create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)

        # Entropy coefficient (log form for positive constraint, initial value=1.0)
        self.log_alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # Create optimizers for all networks/parameters
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])
        self.lyapunov_optimizer = Adam(
            self.lyapunov.parameters(), lr=kwargs["lyapunov_learning_rate"]
        )
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])


    def create_action_distributions(self, logits):
        """Create action distribution instance from policy network logits (e.g., TanhGaussDistribution)"""
        return self.policy.get_act_dist_cls(logits)
    

class MSACL:
    """
    MSACL algorithm class using the defined function approximators
    """
    def __init__(
        self,
        gamma: float = 0.99,
        retrace_lambda: float = 0.95,  # Weighting coefficient for multi-step Lyapunov difference
        lya_eta: float = 0.15,
        tau: float = 0.005,
        alpha: float = math.e,  # Stability coefficient
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        policy_frequency: int = 2,  # Delayed policy update frequency
        target_network_frequency: int = 1,  # Target network update frequency
        lya_diff_scale: float = 1.0,  # Weight for 3rd term in Lyapunov Risk (analogous to 1/delta_t)
        lya_zero_scale: float = 1.0,
        lya_positive_scale: float = 1.0,
        **kwargs: Any,
    ):
        #################### Network Training Params Initialization ####################
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.retrace_lambda = retrace_lambda
        self.lya_eta = lya_eta
        self.tau = tau

        # Learning rates for 4 trainable components (Q, Lyapunov, policy, alpha)
        self.q_learning_rate = kwargs["q_learning_rate"]
        self.lyapunov_learning_rate = kwargs["lyapunov_learning_rate"]
        self.policy_learning_rate = kwargs["policy_learning_rate"]
        self.alpha_learning_rate = kwargs["alpha_learning_rate"]

        # Update frequencies for policy/target networks
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency

        # Multi-step data length and replay batch size
        self.n_step = kwargs["n_step"]
        self.batch_size = kwargs["replay_batch_size"]

        # Learning rate annealing settings
        self.anneal_lr = kwargs["anneal_lr"]
        self.max_iteration = kwargs["max_iteration"]

        #################### Entropy Term Settings ####################
        # Auto alpha adjustment flag
        self.disable_auto_alpha = kwargs["disable_auto_alpha"]
        self.auto_alpha = not self.disable_auto_alpha
        print("Auto alpha adjustment enabled:", self.auto_alpha)

        # Alpha boundary settings
        self.set_alpha_bound = kwargs["set_alpha_bound"]
        self.alpha_bound = kwargs["alpha_bound"]
        print("Alpha boundary enabled:", self.set_alpha_bound)
        print("Alpha boundary value:", self.alpha_bound)

        # Initialize log_alpha with initial alpha value
        self.networks.log_alpha.data.fill_(math.log(alpha))

        # Target entropy (custom for QuadTracking environment)
        if target_entropy is None:
            target_entropy = -kwargs["act_dim"]
            if kwargs["env_name"] == "QuadTracking":
                target_entropy -= 1
        self.target_entropy = target_entropy
        print("Target entropy for QuadTracking environment:", self.target_entropy)

        #################### Lyapunov-related Params ####################
        # Weight coefficients for 3 terms in Lyapunov Loss
        self.lya_diff_scale = lya_diff_scale
        self.lya_zero_scale = lya_zero_scale
        self.lya_positive_scale = lya_positive_scale

        # Coefficients for exponential stability label construction
        self.alpha1 = kwargs.get("alpha1", 1.0)
        self.alpha2 = kwargs.get("alpha2", 2.0)

        # Clipping coefficient
        self.clip_coef = kwargs.get("clip_coef", 0.1)

        # Initial observation normalization coefficient (shape=[1, n_step], CUDA)
        self.start_obs_norm_coef = (
            (torch.tensor(1-self.lya_eta) ** torch.arange(1, self.n_step + 1) 
            * torch.tensor(self.alpha2/self.alpha1)) ** 0.5
        ).unsqueeze(0).cuda()

        # Normalized Retrace(lambda) coefficients for Lyapunov difference (shape=[1, n_step], CUDA)
        self.lya_diff_coef = torch.pow(self.retrace_lambda, torch.arange(self.n_step)).cuda()
        self.lya_diff_coef = self.lya_diff_coef / torch.sum(self.lya_diff_coef)
        self.lya_diff_coef = self.lya_diff_coef.unsqueeze(0)

        # Coefficient for (1-eta)^n in Lyapunov difference condition (shape=[1, n_step], CUDA)
        self.start_lya_coef = torch.pow((1-self.lya_eta), torch.arange(self.n_step)+1).unsqueeze(0).cuda()


    @property
    def adjustable_parameters(self):
        """Read-only hyperparameters adjustable during training (no re-instantiation needed)"""
        return ("gamma", "tau", "auto_alpha", "alpha",
                "target_entropy", "policy_frequency", "target_network_frequency")
    

    def model_update(self, data: Dict[str, Tensor], global_iteration: int):
        """
        Model update (data shape differs from LAC/DBLAC: multi-step data with shape [batch_size, n_step, dim])
        data: Dictionary with keys "obs", "act", etc., values are multi-step tensors (GPU-based)
        """
        start_time = time.time()

        #################### Learning Rate Annealing ####################
        if self.anneal_lr:
            frac = max(0, 1.0 - (global_iteration - 0) / self.max_iteration)
            self.networks.q1_optimizer.param_groups[0]['lr'] = self.q_learning_rate * frac
            self.networks.q2_optimizer.param_groups[0]['lr'] = self.q_learning_rate * frac
            self.networks.lyapunov_optimizer.param_groups[0]['lr'] = self.lyapunov_learning_rate * frac
            self.networks.policy_optimizer.param_groups[0]['lr'] = self.policy_learning_rate * frac
            self.networks.alpha_optimizer.param_groups[0]['lr'] = self.alpha_learning_rate * frac
            self.networks.beta_optimizer.param_groups[0]['lr'] = self.beta_learning_rate * frac
            
        #################### Step1: Update Q-networks (Retrace(lambda) operator) ####################
        # Q-networks updated immediately (no delay), policy/Lagrangian multipliers use delayed update
        loss_q, q1_mean, q2_mean = self._q_update(data)

        # Soft update target Q-networks at specified frequency (default=1)
        if global_iteration % self.target_network_frequency == 0:
            self._target_update()

        #################### Step2: Update Lyapunov Network ####################
        loss_lya = self._lyapunov_update(data)

        #################### Step3: Update Policy Network ####################
        if global_iteration % self.policy_frequency == 0:
            # Compensate delayed update with multiple policy updates
            for _ in range(self.policy_frequency):
                loss_policy, entropy = self._policy_update(data=data)

                #################### Step4: Update Lagrangian Multiplier (alpha) ####################
                if self.auto_alpha:
                    self._alpha_update(entropy=entropy)

            # Collect TensorBoard logging information
            tb_info = {
                "MSACL/entropy-RL iter": entropy.item(),
                "MSACL/alpha-RL iter": self._get_alpha(),
                "MSACL/q1_mean-RL iter": q1_mean.item(),
                "MSACL/q2_mean-RL iter": q2_mean.item(),
                tb_tags["loss_critic"]: loss_q.item(),
                tb_tags["loss_lyapunov"]: loss_lya.item(),
                tb_tags["loss_actor"]: loss_policy.item(),
                tb_tags["alg_time"]: (time.time() - start_time) * 1000,
            }

            return tb_info


    def _q_update(self, data: Dict[str, Tensor]):
        """Update Q-networks with multi-step TD target (shape: [batch_size, n_step, dim] or [batch_size, n_step])"""
        # Extract training data
        obs, act, rew, obs2, done = (
            data["obs"],  # [batch_size, n_step, obs_dim]
            data["act"],  # [batch_size, n_step, act_dim]
            data["rew"],  # [batch_size, n_step]
            data["obs2"], # [batch_size, n_step, obs_dim]
            data["done"], # [batch_size, n_step]
        )

        # Compute current Q-values for (s,a)
        q1 = self.networks.q1(obs, act)  # [batch_size, n_step]
        q2 = self.networks.q2(obs, act)  # [batch_size, n_step]

        # Compute multi-step TD target (no gradient)
        with torch.no_grad():
            next_logits = self.networks.policy(obs2)  # [batch_size, n_step, 2]
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, next_logp = next_act_dist.rsample()
            next_q1 = self.networks.q1_target(obs2, next_act)  # [batch_size, n_step]
            next_q2 = self.networks.q2_target(obs2, next_act)  # [batch_size, n_step]
            next_q = torch.min(next_q1, next_q2)
            backup = rew + (1 - done) * self.gamma * (
                next_q - self._get_alpha() * next_logp
            )

        # Calculate Q-network loss (MSE)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Optimize Q-networks
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q.backward()
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        return loss_q.detach(), q1.detach().mean(), q2.detach().mean()


    def _lyapunov_update(self, data: Dict[str, Tensor]):
        """Update Lyapunov network (uses clipped importance sampling from Q-update)"""
        # Extract data for Lyapunov loss calculation
        obs, obs2, act, old_logp = (
            data["obs"],   # [batch_size, n_step, obs_dim]
            data["obs2"],  # [batch_size, n_step, obs_dim]
            data["act"],   # [batch_size, n_step, act_dim]
            data["logp"],  # [batch_size, n_step]
        )

        ########## 0、Compute clipped importance sampling for Lyapunov update ##########
        with torch.no_grad():
            logits = self.networks.policy(obs)  # [batch_size, n_step, act_dim*2]
            act_dist = self.networks.create_action_distributions(logits)
            logp = act_dist.log_prob(act)  # [batch_size, n_step]
            ratio = torch.exp(logp - old_logp)  # [batch_size, n_step]
            clip_ratio = torch.clamp(ratio, 0 ,1)
            is_clip_ratio = torch.cumprod(clip_ratio, dim=1)  # [batch_size, n_step]

        ########## 2、Lyapunov Loss Term 2: Boundedness (alpha1||x||² ≤ V(x) ≤ alpha2||x||²) ##########
        lya_obs = self.networks.lyapunov(obs)  # [batch_size, n_step]
        obs_pow2 = torch.sum(obs ** 2, dim=-1)  # [batch_size, n_step]

        # Loss for lower bound (V(x) ≥ alpha1||x||²)
        loss_lya2_1 = torch.max(
            self.alpha1 * obs_pow2 - lya_obs,
            torch.zeros_like(lya_obs)
        ).mean()
        # Loss for upper bound (V(x) ≤ alpha2||x||²)
        loss_lya2_2 = torch.max(
            lya_obs - self.alpha2 * obs_pow2,
            torch.zeros_like(lya_obs)
        ).mean()

        loss_lya2 = (loss_lya2_1 + loss_lya2_2) * self.lya_positive_scale

        ########## 3、Lyapunov Loss Term 3: Weighted sum of multi-step differences ≤ 0 ##########
        # Compute exponential stability label (ESL) for sample selection
        start_obs = obs[:, 0, :]  # [batch_size, obs_dim]
        start_obs_norm = torch.norm(start_obs, p=2, dim=-1)  # [batch_size]
        expanded_start_obs_norm = start_obs_norm.unsqueeze(1) * self.start_obs_norm_coef  # [batch_size, n_step]
        obs2_norm = torch.norm(obs2, p=2, dim=-1)  # [batch_size, n_step]

        # ESL=1 (positive sample: satisfies exponential stability), ESL=-1 (negative sample)
        diff = expanded_start_obs_norm - obs2_norm  # [batch_size, n_step]
        ESL = torch.where(diff >= 0, torch.tensor(1.0), torch.tensor(-1.0))  # [batch_size, n_step]
        
        # Lyapunov values at current/next states
        lya_obs = self.networks.lyapunov(obs)  # [batch_size, n_step]
        lya_obs2 = self.networks.lyapunov(obs2)  # [batch_size, n_step]
        start_lya = lya_obs[:, 0]  # [batch_size]
        start_lya_extended = start_lya.unsqueeze(1).repeat(1, self.n_step)  # [batch_size, n_step]
        start_lya_extended_coef = start_lya_extended * self.start_lya_coef  # [batch_size, n_step]

        # Weighted Lyapunov difference term (importance sampling corrected)
        lya_diff_term = is_clip_ratio * torch.max(
            ESL * (lya_obs2 - start_lya_extended_coef), torch.zeros_like(lya_obs2)
        )
        lya_diff_term_lambda = self.lya_diff_coef * lya_diff_term  # [batch_size, n_step]
        lya_diff = lya_diff_term_lambda.sum(dim=1)  # [batch_size]
        loss_lya3 = lya_diff.mean() * self.lya_diff_scale

        ########## 4、Total Lyapunov Loss and Network Update ##########
        loss_lya = loss_lya2 + loss_lya3
        self.networks.lyapunov_optimizer.zero_grad()
        loss_lya.backward()
        self.networks.lyapunov_optimizer.step()

        return loss_lya.detach()
    

    def _get_alpha(self, requires_grad: bool=False):
        """Get entropy coefficient alpha (tensor with grad / scalar without grad)"""
        alpha = self.networks.log_alpha.exp()
        
        if requires_grad:
            return alpha
        else:
            return alpha.item()
    

    def _policy_update(self, data: Dict[str, Tensor]):
        """
        Policy update with:
        1. Soft Q-value loss (min_q_pi - alpha*log_p)
        2. Clipped Lyapunov stability advantage loss
        """
        # Extract training data
        obs, old_act, obs2, old_logp = (
            data["obs"],   # [batch_size, n_step, obs_dim]
            data["act"],   # [batch_size, n_step, act_dim]
            data["obs2"],  # [batch_size, n_step, obs_dim]
            data["logp"],  # [batch_size, n_step]
        )

        # Disable gradient for Q-networks (already updated)
        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False

        ########## 1、Create new action distribution from current policy ##########
        new_logits = self.networks.policy(obs)
        new_act_dist = self.networks.create_action_distributions(new_logits)
        new_act, new_act_logp = new_act_dist.rsample()  # [batch_size, n_step, act_dim], [batch_size, n_step]

        ########## 2、Compute Soft Q-value loss ##########
        q1 = self.networks.q1(obs, new_act)  # [batch_size, n_step]
        q2 = self.networks.q2(obs, new_act)
        min_q_pi = torch.min(q1, q2)  # [batch_size, n_step]
        loss_policy_q = (min_q_pi - self._get_alpha() * new_act_logp).mean()

        ########## 4、Compute Lyapunov stability advantage loss ##########
        # Importance sampling ratio (first step only, unclipped)
        new_logp = new_act_dist.log_prob(old_act)  # [batch_size, n_step]
        ratio = torch.exp(new_logp - old_logp)
        is_ratio = ratio[:, 0]  # [batch_size]

        # Compute Lyapunov stability advantage (no gradient for Lyapunov values)
        with torch.no_grad():
            start_lya = self.networks.lyapunov(obs)[:, 0]  # [batch_size]
            lya_obs2 = self.networks.lyapunov(obs2)  # [batch_size, n_step]

        # Stability advantage: A = (1-η)^n V(x₀) - V(xₜ)
        start_lya_extended = start_lya.unsqueeze(1).repeat(1, self.n_step)  # [batch_size, n_step]
        start_lya_extended_coef = start_lya_extended * self.start_lya_coef  # [batch_size, n_step]
        stability_adv = start_lya_extended_coef - lya_obs2  # [batch_size, n_step]

        # Weighted & normalized stability advantage
        stability_adv_lambda = self.lya_diff_coef * stability_adv  # [batch_size, n_step]
        mb_stability_adv = stability_adv_lambda.sum(dim=1)  # [batch_size]
        mb_stability_adv = (mb_stability_adv - mb_stability_adv.mean()) / (mb_stability_adv.std() + 1e-8)

        # PPO-style clipped advantage loss
        surr1 = is_ratio * mb_stability_adv  # [batch_size]
        surr2 = torch.clamp(is_ratio, 1-self.clip_coef, 1+self.clip_coef) * mb_stability_adv
        loss_policy_lya = torch.min(surr1, surr2).mean()

        ########## 5、Total Policy Loss ##########
        loss_policy = (
            - loss_policy_q  # Maximize Soft Q-value (minimize negative loss)
            - loss_policy_lya  # Maximize stability advantage (minimize negative loss)
        )

        # Optimize policy network
        self.networks.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.networks.policy_optimizer.step()

        # Compute policy entropy (detached from graph)
        entropy = -new_act_logp.mean().detach()
        
        # Re-enable gradient for Q-networks
        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True

        return loss_policy.detach(), entropy
    
    def _alpha_update(self, entropy: Tensor):
        """Auto update entropy coefficient alpha (with boundary constraint if enabled)"""
        alpha = self._get_alpha(requires_grad=True)
        loss_alpha = alpha * (entropy - self.target_entropy)

        # Optimize alpha
        self.networks.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.networks.alpha_optimizer.step()

        # Apply alpha boundary constraint (no gradient)
        if self.set_alpha_bound:
            with torch.no_grad():
                self.networks.log_alpha.clamp_(max=math.log(self.alpha_bound))


    def _target_update(self):
        """Soft update (Polyak averaging) for target Q-networks (no gradient)"""
        with torch.no_grad():
            polyak = 1 - self.tau
            # Update q1 target network
            for p, p_targ in zip(
                self.networks.q1.parameters(), self.networks.q1_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            # Update q2 target network
            for p, p_targ in zip(
                self.networks.q2.parameters(), self.networks.q2_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)