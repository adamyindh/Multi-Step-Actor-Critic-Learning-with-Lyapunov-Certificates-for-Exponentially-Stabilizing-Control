"""
Derived from off_sampler.py, for collecting n-step interaction data with environment
"""
from typing import List

# Import base class and n-step experience class
from RL.trainer.sampler.base import BaseSampler, nStepExperience

class NstepOffSampler(BaseSampler):
    def __init__(self, **kwargs):
        # Initialize base class with input parameters
        super().__init__(**kwargs)


    def _sample(self,) -> List[nStepExperience]:
        """
        Interact with environment to collect n-step experience data.
        Returns: List of nStepExperience objects
        """
        batch_data = []

        # Interact with environment for horizon steps per environment
        for _ in range(self.horizon):
            # Collect n-step experience from single environment step
            n_step_experiences = self._n_step()
            # Add all nStepExperience objects to batch (empty list adds nothing)
            batch_data.extend(n_step_experiences)
        
        return batch_data

