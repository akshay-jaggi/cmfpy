import numpy as np
from ..common import cmf_predict


class Synthetic(object):
    """Synthetic data."""
    def __init__(self,
                 n_components=3,
                 n_features=100,
                 n_lags=100,
                 n_timebins=10000,
                 H_sparsity=0.9,
                 noise_scale=1.0,
                 seed=None):

        # Set data name and random state.
        self.name = "synthetic"
        self.rs = np.random.RandomState(seed)
        self.n_components = n_components
        self.n_features = n_features
        self.n_lags = n_lags
        self.H_sparsity = H_sparsity

        self.H = self.create_H()
        self.W = self.create_W()

        # Determine noise
        self.noise = noise_scale * self.rs.rand(n_features, n_timebins)

        # Add noise to model prediction
        self.data = cmf_predict(self.W, self.H) + self.noise

    def create_W(self):
        W = np.zeros((self.n_lags, self.n_features, self.n_components))
        # Add structure to motifs
        for i, j in enumerate(np.random.choice(self.n_components, size=self.n_features)):
            W[:, i, j] += _gauss_plus_delay(self.n_lags)
        return W

    def create_H(self):
        H = self.rs.rand(self.n_components, self.n_timebins)
        return H * self.rs.binomial(1, 1 - self.H_sparsity, size=H.shape)

    def generate(self):
        return self.data + self.noise


class NotNonNegSynthetic(Synthetic):
    def create_W(self):
        W = np.zeros((self.n_lags, self.n_features, self.n_components))
        # Add structure to motifs
        for i, j in enumerate(np.random.choice(self.n_components, size=self.n_features)):
            W[:, i, j] += _sin_gauss_plus_delay(self.n_lags)
        return W


def _gauss_plus_delay(n_steps):
    tau = np.random.uniform(-1.5, 1.5)
    x = np.linspace(-3-tau, 3-tau, n_steps)
    y = np.exp(-x**2)
    return y / y.max()


def _sin_gauss_plus_delay(n_steps):
    tau = np.random.uniform(-1.5, 1.5)
    x = np.linspace(-3-tau, 3-tau, n_steps)
    y = np.sin(x) * np.exp(-x**2)
    return y / y.max()
