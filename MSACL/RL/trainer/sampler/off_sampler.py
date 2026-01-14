"""
Defines OffSampler class that inherits from abstract BaseSampler (base.py)
"""
from typing import List
from RL.trainer.sampler.base import BaseSampler, Experience

class OffSampler(BaseSampler):
    def __init__(self, **kwargs):
        # Initialize base class with input parameters
        super().__init__(**kwargs)


    def _sample(self,) -> List[Experience]:
        """
        Interact with environment to collect single-step experience data.
        Returns: List of Experience objects
        """
        batch_data = []

        # Interact with environment for horizon steps per environment
        for _ in range(self.horizon):
            # Collect single-step experience (one per environment)
            experiences = self._step()
            # Add all Experience objects to batch (empty list adds nothing)
            batch_data.extend(experiences)
        
        return batch_data