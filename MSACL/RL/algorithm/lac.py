"""
The LAC algorithm
Reference:
This algorithm is implemented based on the work Actor-Critic Reinforcement Learning for Control with Stability Guarantee.
url={https://doi.org/10.1109/LRA.2020.3011351}
"""

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
    Define function approximators for LAC algorithm: 1 policy function + 2 value functions (2 L-values)
    L-functions are learned based on Bellman equation for cost, minimizing L-values (reduce overestimation) like SAC
    """
    def __init__(self, **kwargs):
        # Initialize parent class nn.Module
        super().__init__()

        # Get L-network params (type, layer sizes, activation) and create 2 L-networks
        l_args = get_apprfunc_dict("value", **kwargs)
        self.l1: nn.Module = create_apprfunc(**l_args)
        self.l2: nn.Module = create_apprfunc(**l_args)

        # Create target L-networks (no gradient update, manual soft update)
        self.l1_target = deepcopy(self.l1)
        self.l2_target = deepcopy(self.l2)
        for p in self.l1_target.parameters():
            p.requires_grad = False
        for p in self.l2_target.parameters():
            p.requires_grad = False

        # Get policy network params and create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)

        # Entropy/stability coefficients (log form for positive constraint)
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))  # Stability coefficient (lambda in LAC)
        self.log_beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))   # Entropy coefficient (beta in LAC)

        # Create optimizers for all networks (q→l in SAC-LAC adaptation)
        self.l1_optimizer = Adam(self.l1.parameters(), lr=kwargs["l_learning_rate"])
        self.l2_optimizer = Adam(self.l2.parameters(), lr=kwargs["l_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])
        self.beta_optimizer = Adam([self.log_beta], lr=kwargs["beta_learning_rate"])

    def create_action_distributions(self, logits):
        """Create action distribution instance from policy network logits (e.g., TanhGaussDistribution)"""
        return self.policy.get_act_dist_cls(logits)
    

class LAC:
    """
    LAC algorithm class (no inheritance for now)
    kwargs contains args excluding specified keys (e.g., "gamma", "tau")
    """
    def __init__(
        self,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha3: float = 0.05,
        alpha: float = math.e,  # Stability coefficient (lambda in LAC paper)
        beta: float = math.e,    # Entropy coefficient (beta in LAC paper)
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        policy_frequency: int = 2,  # Delayed policy update frequency
        target_network_frequency: int = 1,  # Target network update frequency
        **kwargs: Any,
    ):
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.alpha3 = alpha3
        # Initialize log_alpha/log_beta with initial alpha/beta values
        self.networks.log_alpha.data.fill_(math.log(alpha))
        self.networks.log_beta.data.fill_(math.log(beta))
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -kwargs["act_dim"]
        self.target_entropy = target_entropy
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency

    @property
    def adjustable_parameters(self):
        """Read-only hyperparameters adjustable during training (no re-instantiation needed)"""
        return ("gamma", "tau", "alpha", "auto_alpha", "target_entropy")
    
    def model_update(self, data: Dict[str, Tensor], global_iteration: int):
        """
        Update model (2 L-networks, 1 policy network, alpha/beta parameters)
        data: Dictionary containing training data
        """
        start_time = time.time()

        # Update L-networks first
        loss_l, l1, l2 = self._l_update(data)

        # Soft update target L-networks at specified frequency
        if global_iteration % self.target_network_frequency == 0:
            self._target_update()

        # Delayed policy update (TD3 style)
        if global_iteration % self.policy_frequency == 0:
            # Compensate delayed update with multiple policy updates
            for _ in range(self.policy_frequency):
                # Generate next action/distribution for policy update (add to data for alpha/beta update)
                obs2 = data["obs2"]
                next_logits = self.networks.policy(obs2)
                next_act_dist = self.networks.create_action_distributions(next_logits)
                next_act, next_logp = next_act_dist.rsample()
                data.update({"next_act": next_act, "next_logp": next_logp})
                loss_policy, entropy = self._policy_update(data)

                # Auto update alpha (stability) and beta (entropy) coefficients
                if self.auto_alpha:
                    self._alpha_update(data)
                    self._beta_update(data)

            # Collect TensorBoard logging info
            tb_info = {
                "LAC/critic_l1-RL iter": l1.item(),
                "LAC/critic_l2-RL iter": l2.item(),
                "LAC/entropy-RL iter": entropy.item(),
                "LAC/alpha-RL iter": self._get_alpha(),
                "LAC/beta-RL iter": self._get_beta(),
                tb_tags["loss_critic"]: loss_l.item(),
                tb_tags["loss_actor"]: loss_policy.item(),
                tb_tags["alg_time"]: (time.time() - start_time) * 1000,
            }

            return tb_info


    def _get_alpha(self, requires_grad: bool = False):
        """Get alpha (stability coefficient) - scalar if no grad, tensor if grad needed"""
        alpha = self.networks.log_alpha.exp()
        if requires_grad:
            return alpha
        else:
            return alpha.item()
        
    
    def _get_beta(self, requires_grad: bool = False):
        """Get beta (entropy coefficient) - scalar if no grad, tensor if grad needed"""
        beta = self.networks.log_beta.exp()
        if requires_grad:
            return beta
        else:
            return beta.item()
        
    
    def _l_update(self, data: Dict[str, Tensor]):
        """Update L-networks with cost-based Bellman equation (1-step TD target)"""
        # Extract training data (only cost is used for LAC, no reward)
        obs, act, cost, obs2, done = (
            data["obs"],
            data["act"],
            data["cost"],
            data["obs2"],
            data["done"],
        )
        
        # Current L-values for (s,a)
        l1 = self.networks.l1(obs, act)
        l2 = self.networks.l2(obs, act)

        # Compute TD target (no gradient for target calculation)
        with torch.no_grad():
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, _ = next_act_dist.rsample()
            next_l1 = self.networks.l1_target(obs2, next_act)
            next_l2 = self.networks.l2_target(obs2, next_act)
            next_l = torch.min(next_l1, next_l2)  # Minimize L-values to reduce overestimation (SAC style)
            backup = cost + (1 - done) * self.gamma * next_l  # 1-step TD target for LAC

        # Calculate L-network loss (MSE)
        loss_l1 = ((l1 - backup) ** 2).mean()
        loss_l2 = ((l2 - backup) ** 2).mean()
        loss_l = loss_l1 + loss_l2

        # Optimize L-networks
        self.networks.l1_optimizer.zero_grad()
        self.networks.l2_optimizer.zero_grad()
        loss_l.backward()
        self.networks.l1_optimizer.step()
        self.networks.l2_optimizer.step()

        return loss_l, l1.detach().mean(), l2.detach().mean()
    

    def _policy_update(self, data: Dict[str, Tensor]):
        """Update policy network based on Lyapunov descent condition (stability + entropy loss)"""
        # Disable gradient for L-networks (only used for value calculation)
        for p in self.networks.l1.parameters():
            p.requires_grad = False
        for p in self.networks.l2.parameters():
            p.requires_grad = False

        # Extract data (next_act/next_logp generated in model_update)
        obs, act, cost, obs2, next_act, next_logp = (
            data["obs"],
            data["act"],
            data["cost"],
            data["obs2"],
            data["next_act"],
            data["next_logp"]
        )

        # Calculate L-values for policy loss
        l1 = self.networks.l1(obs, act)
        l2 = self.networks.l2(obs, act)
        next_l1 = self.networks.l1(obs2, next_act)
        next_l2 = self.networks.l2(obs2, next_act)

        # Policy loss: stability loss + entropy loss
        loss_stability = self._get_alpha() * (
            torch.min(next_l1, next_l2) - torch.min(l1, l2) + self.alpha3 * cost
            ).mean()
        loss_entropy = self._get_beta() * (next_logp.mean() + self.target_entropy)
        loss_policy = loss_stability + loss_entropy

        # Calculate policy entropy (detach gradient for logging)
        entropy = -next_logp.detach().mean()

        # Update data dict with L-values/entropy (no grad) for alpha/beta update
        data.update({
            "l_val": torch.min(l1, l2).detach(),
            "next_l_val": torch.min(next_l1, next_l2).detach(),
            "entropy": entropy
        })

        # Optimize policy network
        self.networks.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.networks.policy_optimizer.step()

        # Re-enable gradient for L-networks after policy update
        for p in self.networks.l1.parameters():
            p.requires_grad = True
        for p in self.networks.l2.parameters():
            p.requires_grad = True

        return loss_policy, entropy
    

    def _alpha_update(self, data: Dict[str, Tensor]):
        """Update alpha (stability coefficient) with adaptive optimization"""
        # Extract precomputed L-values/cost (no grad)
        l_val, next_l_val, cost = (
            data["l_val"],
            data["next_l_val"],
            data["cost"]
        )

        # Alpha loss (retain gradient for log_alpha)
        alpha = self._get_alpha(True)
        loss_alpha = -alpha * (
            next_l_val - l_val + self.alpha3 * cost
        ).mean()

        # Optimize alpha
        self.networks.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.networks.alpha_optimizer.step()


    def _beta_update(self, data: Dict[str, Tensor]):
        """Update beta (entropy coefficient) with adaptive optimization"""
        # Extract precomputed entropy (no grad)
        entropy = data["entropy"]

        # Beta loss (retain gradient for log_beta)
        beta = self._get_beta(True)
        loss_beta = beta * (entropy - self.target_entropy)  # Minimize beta*(entropy - target_entropy) per LAC

        # Optimize beta
        self.networks.beta_optimizer.zero_grad()
        loss_beta.backward()
        self.networks.beta_optimizer.step()


    def _target_update(self):
        """Soft update (Polyak averaging) for target L-networks"""
        with torch.no_grad():
            polyak = 1 - self.tau
            # Update l1 target network
            for p, p_targ in zip(
                self.networks.l1.parameters(), self.networks.l1_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            # Update l2 target network
            for p, p_targ in zip(
                self.networks.l2.parameters(), self.networks.l2_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)