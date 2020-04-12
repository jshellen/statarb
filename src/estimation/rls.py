
import numpy as np

class FilterRLS:

    def __init__(self, n, mu=0.99, eps=0.1, w_0="random"):

        self.kind = "RLS filter"

        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer')

        self.mu = mu
        self.eps = eps
        self.w = w_0.copy()
        self.R = 1/self.eps * np.eye(n)
        self.w_history = False


    def update(self, d, x):
        """
        Adapt weights according one desired value and its input.

        """

        x = x.reshape(-1, 1)
        y = np.dot(self.w.T, x).flatten()[0]
        e = d - y
        R1 = np.dot(np.dot(np.dot(self.R, x), x.T), self.R)
        R2 = self.mu + np.dot(np.dot(x.T, self.R), x)
        self.R = 1.0/self.mu * (self.R - R1/R2)
        self.R = 0.5 * (self.R + self.R.T)  # Ensure symmetry
        dw = np.dot(self.R, x) * e
        self.w += dw
