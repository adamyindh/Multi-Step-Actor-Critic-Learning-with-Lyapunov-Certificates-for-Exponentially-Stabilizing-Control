"""
The SAC algorithm
Reference:
This algorithm is implemented based on the work Soft actor-critic algorithms and applications.
url={https://doi.org/10.48550/arXiv.1812.05905}
"""

__all__ = ["ApproxContainer", "SAC"] 

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
    """SAC networks: q1/q2 (with target), policy, alpha (log_alpha)"""
    def __init__(self, **kwargs):
        super().__init__()

        # Q networks (double Q)
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)

        # Target Q networks (no grad)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        # Policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)

        # Entropy coefficient (log_alpha for stable optimization)
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # Optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs["policy_learning_rate"])
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

    def create_action_distributions(self, logits):
        """Create action distribution (e.g., TanhGaussian) from logits"""
        return self.policy.get_act_dist_cls(logits)
    

class SAC:
    """SAC algorithm (off-policy, entropy-regularized RL)"""
    def __init__(
        self,
        gamma: float = 0.99,  # Discount factor
        tau: float = 0.005,  # Target network soft update coeff
        alpha: float = math.e,  # Initial entropy coeff
        auto_alpha: bool = True,  # Auto-tune alpha
        target_entropy: Optional[float] = None,  # Target entropy for alpha
        policy_frequency: int = 2,  # Policy delayed update freq
        target_network_frequency: int = 1,  # Target Q update freq
        **kwargs: Any,
    ):
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.networks.log_alpha.data.fill_(math.log(alpha))
        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy if target_entropy else -kwargs["act_dim"]
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency


    @property
    def adjustable_parameters(self):
        """Tunable hyperparameters (read-only)"""
        return ("gamma", "tau", "alpha", "auto_alpha", "target_entropy")
    

    def model_update(self, data: Dict[str, Tensor], global_iteration: int):
        """SAC core update: Q -> target Q -> policy (delayed) -> alpha (auto)"""
        start_time = time.time()

        # Update Q networks
        loss_q, q1, q2 = self._q_update(data)

        # Update target Q networks (freq-based)
        if global_iteration % self.target_network_frequency == 0:
            self._target_update()

        # Update policy (delayed, TD3-style)
        if global_iteration % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                # Sample new actions/logp for policy/alpha update
                obs = data["obs"]
                logits = self.networks.policy(obs)
                act_dist = self.networks.create_action_distributions(logits)
                new_act, new_logp = act_dist.rsample()
                data.update({"new_act": new_act, "new_logp": new_logp})
                
                # Update policy
                loss_policy, entropy = self._policy_update(data)

                # Auto-update alpha
                if self.auto_alpha:
                    self._alpha_update(data)

            # TensorBoard logging
            tb_info = {
                "SAC/critic_q1-RL iter": q1.item(),
                "SAC/critic_q2-RL iter": q2.item(),
                "SAC/entropy-RL iter": entropy.item(),
                "SAC/alpha-RL iter": self._get_alpha(),
                tb_tags["loss_critic"]: loss_q.item(),
                tb_tags["loss_actor"]: loss_policy.item(),
                tb_tags["alg_time"]: (time.time() - start_time) * 1000,
            }

            return tb_info

    
    def _get_alpha(self, requires_grad: bool = False):
        """Get alpha (exp(log_alpha)), with/without grad"""
        alpha = self.networks.log_alpha.exp()
        return alpha if requires_grad else alpha.item()

        
    def _q_update(self, data: Dict[str, Tensor]):
        """Update Q networks with TD target (1-step)"""
        obs, act, rew, obs2, done = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]
        
        # Current Q values
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)

        # TD target (no grad)
        with torch.no_grad():
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, next_logp = next_act_dist.rsample()
            next_q = torch.min(self.networks.q1_target(obs2, next_act), self.networks.q2_target(obs2, next_act))
            backup = rew + (1 - done) * self.gamma * (next_q - self._get_alpha() * next_logp)

        # Q loss (MSE)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Update Q networks
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q.backward()
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        return loss_q, q1.detach().mean(), q2.detach().mean()

    
    def _policy_update(self, data: Dict[str, Tensor]):
        """Update policy (max Q + max entropy)"""
        # Freeze Q networks (no grad)
        for p in self.networks.q1.parameters(): p.requires_grad = False
        for p in self.networks.q2.parameters(): p.requires_grad = False

        # Policy loss (max min(Q) - alpha*logp)
        obs, new_act, new_logp = data["obs"], data["new_act"], data["new_logp"]
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        loss_policy = (self._get_alpha(True) * new_logp - torch.min(q1, q2)).mean()

        # Update policy
        self.networks.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.networks.policy_optimizer.step()

        # Entropy (detach logp to avoid grad)
        entropy = -new_logp.detach().mean()

        # Unfreeze Q networks
        for p in self.networks.q1.parameters(): p.requires_grad = True
        for p in self.networks.q2.parameters(): p.requires_grad = True

        return loss_policy, entropy


    def _alpha_update(self, data: Dict[str, Tensor]):
        """Auto-tune alpha (match target entropy)"""
        new_logp = data["new_logp"]
        alpha = self._get_alpha(True)
        loss_alpha = -alpha * (new_logp.detach() + self.target_entropy).mean()

        # Update alpha
        self.networks.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.networks.alpha_optimizer.step()


    def _target_update(self):
        """Soft update target Q networks (Polyak averaging)"""
        with torch.no_grad():
            polyak = 1 - self.tau
            # Update q1 target
            for p, p_targ in zip(self.networks.q1.parameters(), self.networks.q1_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            # Update q2 target
            for p, p_targ in zip(self.networks.q2.parameters(), self.networks.q2_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)