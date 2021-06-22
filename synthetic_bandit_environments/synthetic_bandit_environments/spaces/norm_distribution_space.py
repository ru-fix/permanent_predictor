import numpy as np
from gym.spaces import Space


class NormDistSpace(Space):
    def __init__(self, mu, sigma, shape, dtype=np.float32):
        super(NormDistSpace, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        sample = self.np_random.normal(self.mu, self.sigma, self.shape)
        return sample.astype(self.dtype)

    def __repr__(self):
        return "Normal Distribution ({}, {}, {}, {})"\
            .format(self.mu, self.sigma, self.shape, self.dtype)
