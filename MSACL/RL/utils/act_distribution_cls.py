"""
Defines action distribution classes used by get_apprfunc_dict in common_utils.py
var["action_distribution_cls"] = GaussDistribution
"""

import torch
EPS = 1e-6

#################### Action Distribution Base Class ####################
class Action_Distribution_Cls:
    """Base class to get action distribution for policy network"""
    def __init__(self):
        super().__init__()

    def get_act_dist_cls(self, logits):
        # Get action distribution class reference from policy network
        action_distribution_cls = getattr(self, "action_distribution_cls")
        # Check if action limits exist in policy network
        has_act_lim = hasattr(self, "act_high_lim") and hasattr(self, "act_low_lim")
        # Create action distribution instance from logits
        act_dist_cls = action_distribution_cls(logits)
        # Assign env action limits to distribution if available
        if has_act_lim:
            act_dist_cls.act_high_lim = getattr(self, "act_high_lim")
            act_dist_cls.act_low_lim = getattr(self, "act_low_lim")
        return act_dist_cls


#################### Specific Action Distributions ####################
class TanhGaussDistribution:
    """Tanh-constrained Gaussian distribution (action range [-1, 1])"""
    def __init__(self, logits):
        # logits: [batch_size, 2*action_dim] (mean + std)
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)  # [batch_size, action_dim] each
        # Independent multivariate normal distribution (batch_dim: batch_size, event_dim: action_dim)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        # Default normalized action limits
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        # Sample + tanh constraint + log prob (Jacobian corrected)
        action = self.gauss_distribution.sample()
        action_limited = (
            (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(action) 
            + (self.act_high_lim + self.act_low_lim) / 2
        )
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
            - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def rsample(self):
        # Reparameterized sampling (gradient-friendly) + tanh constraint
        action = self.gauss_distribution.rsample()
        action_limited = (
            (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(action) 
            + (self.act_high_lim + self.act_low_lim) / 2
        )
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
            - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def log_prob(self, action_limited) -> torch.Tensor:
        # Compute log prob from constrained action (inverse tanh)
        action = torch.atanh(
            (1 - EPS)
            * (2 * action_limited - (self.act_high_lim + self.act_low_lim))
            / (self.act_high_lim - self.act_low_lim)
        )
        log_prob = self.gauss_distribution.log_prob(action) - torch.log(
            (self.act_high_lim - self.act_low_lim) / 2
            * (1 + EPS - torch.pow(torch.tanh(action), 2))
        ).sum(-1)
        return log_prob

    def entropy(self):
        # Entropy of base Gaussian distribution
        return self.gauss_distribution.entropy()

    def mode(self):
        # Mode (mean with tanh constraint)
        return (
            (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) 
            + (self.act_high_lim + self.act_low_lim) / 2
        )

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        # KL divergence between two Gaussian distributions
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class GaussDistribution:
    """Unconstrained Gaussian distribution (no tanh transform)"""
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        # Sample from unconstrained Gaussian
        action = self.gauss_distribution.sample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def rsample(self):
        # Reparameterized sampling (gradient-friendly)
        action = self.gauss_distribution.rsample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def log_prob(self, action) -> torch.Tensor:
        # Log prob of unconstrained action
        log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        # Entropy of Gaussian distribution
        return self.gauss_distribution.entropy()

    def mode(self):
        # Mode (clamped mean to [-1, 1])
        return torch.clamp(self.mean, self.act_low_lim, self.act_high_lim)

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        # KL divergence between two Gaussian distributions
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )
    

class DiracDistribution:
    """Dirac distribution for deterministic policies (logits = direct action output)"""
    def __init__(self, logits):
        # logits: [batch_size, action_dim] (deterministic action)
        self.logits = logits

    def sample(self):
        # Return deterministic action + zero log prob
        return self.logits, torch.zeros_like(self.logits).sum(-1)

    def mode(self):
        # Mode = deterministic action
        return self.logits