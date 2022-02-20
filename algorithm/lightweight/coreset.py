import numpy as np

class LightweightCoreset:
    def __init__(self, X, k, eps):
        self.X = X
        self.k = k
        self.eps = eps

    def set_k(self, k):
        self.k = k

    def set_X(self, X):
        self.X = X

    def compute_m(self):
        # Computing coreset size
        self.m = np.int64(self.X.shape[1] * self.k * np.log2(self.k) / np.power(self.eps, 2))
        if self.m > self.X.shape[0] * 0.2:
            self.m = int(self.k * 0.01 * self.X.shape[0])

    def compute_coreset(self):
        # Lightweight coreset construction
        dist = np.power(self.X - self.X.mean(axis=0), 2).sum(axis=1)
        q = 0.5 / self.X.shape[0] + 0.5 * dist / dist.sum()
        indices = np.random.choice(self.X.shape[0], size=self.m, replace=True)
        X_cs = self.X[indices, :]
        w_cs = 1.0 / (self.m * q[indices])
        return X_cs, w_cs

    def compute(self):
        self.compute_m()
        print("Coreset size: ", self.m)
        return self.compute_coreset()