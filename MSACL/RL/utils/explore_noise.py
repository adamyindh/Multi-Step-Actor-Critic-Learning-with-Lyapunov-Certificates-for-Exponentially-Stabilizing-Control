
import numpy as np

class GaussNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, action):
        return action + np.random.normal(self.mean, self.std)